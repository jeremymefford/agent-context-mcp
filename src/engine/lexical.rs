use anyhow::{Context, Result, anyhow};
use serde::Serialize;
use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tantivy::collector::{DocSetCollector, TopDocs};
use tantivy::query::{BooleanQuery, FuzzyTermQuery, QueryParser, TermQuery};
use tantivy::schema::{
    FAST, Field, IndexRecordOption, STORED, STRING, Schema, TEXT, TantivyDocument, Value as _,
};
use tantivy::{Index, IndexReader, IndexWriter, Order, ReloadPolicy, Term, doc};

use super::relationships::{CONFIDENCE_LEXICAL, RelationKind, ResolvedRelation};

const CHUNK_SEGMENT_COMPACTION_THRESHOLD: usize = 64;
const SYMBOL_SEGMENT_COMPACTION_THRESHOLD: usize = 64;
// Relationship documents share many long exact keys. Keeping fewer segments avoids
// repeating their term dictionaries and also improves cold-open frontier latency.
const RELATION_SEGMENT_COMPACTION_THRESHOLD: usize = 64;
const DELETED_DOC_COMPACTION_MIN: u64 = 256;
const DELETED_DOC_COMPACTION_RATIO: f64 = 0.20;
const RELATION_DELETED_DOC_COMPACTION_RATIO: f64 = 0.10;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum QueryFlavor {
    NaturalLanguage,
    Identifier,
    Path,
    Mixed,
}

#[derive(Debug, Clone)]
pub struct ChunkIndexDoc {
    pub id: String,
    pub relative_path: String,
    pub basename: String,
    pub extension: String,
    pub language: String,
    pub content: String,
    pub start_line: u64,
    pub end_line: u64,
    pub indexed_at: String,
    pub file_hash: String,
}

#[derive(Debug, Clone)]
pub struct SymbolIndexDoc {
    pub symbol_id: String,
    pub relative_path: String,
    pub basename: String,
    pub name: String,
    pub kind: String,
    pub container: Option<String>,
    pub language: String,
    pub start_line: u64,
    pub end_line: u64,
    pub indexed_at: String,
    pub file_hash: String,
}

#[derive(Debug, Clone)]
pub struct ChunkSearchRequest {
    pub query: String,
    pub limit: usize,
    pub flavor: QueryFlavor,
    pub path_prefix: Option<String>,
    pub language: Option<String>,
    pub file: Option<String>,
    pub extension_filter: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct SymbolSearchRequest {
    pub query: String,
    pub limit: usize,
    pub flavor: QueryFlavor,
    pub path_prefix: Option<String>,
    pub language: Option<String>,
    pub kind: Option<String>,
    pub container: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ChunkSearchHit {
    pub id: String,
    pub relative_path: String,
    pub basename: String,
    pub extension: String,
    pub language: String,
    pub content: String,
    pub start_line: u64,
    pub end_line: u64,
    pub indexed_at: String,
    pub file_hash: String,
    pub score: f64,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ChunkIndexCoverage {
    pub indexed_files: u64,
    pub total_chunks: u64,
}

#[derive(Debug, Clone)]
pub struct SymbolSearchHit {
    pub symbol_id: String,
    pub relative_path: String,
    pub basename: String,
    pub name: String,
    pub kind: String,
    pub container: Option<String>,
    pub language: String,
    pub start_line: u64,
    pub end_line: u64,
    pub indexed_at: String,
    pub file_hash: String,
    pub score: f64,
}

#[derive(Clone)]
pub struct LocalIndexStore {
    root: PathBuf,
    max_warm_repos: usize,
    cache: Arc<Mutex<RepoCacheState>>,
}

#[derive(Default)]
struct RepoCacheState {
    access_tick: u64,
    repos: HashMap<PathBuf, RepoIndexCache>,
}

#[derive(Default)]
struct RepoIndexCache {
    chunk: Option<Arc<CachedIndex>>,
    symbol: Option<Arc<CachedIndex>>,
    relation: Option<Arc<CachedIndex>>,
    last_access_tick: u64,
}

struct CachedIndex {
    index: Index,
    reader: IndexReader,
}

#[derive(Clone, Copy)]
enum CachedIndexKind {
    Chunk,
    Symbol,
    Relation,
}

impl LocalIndexStore {
    pub fn new(root: PathBuf, max_warm_repos: usize) -> Self {
        Self {
            root,
            max_warm_repos,
            cache: Arc::new(Mutex::new(RepoCacheState::default())),
        }
    }

    pub fn clear_repo(&self, repo: &Path) -> Result<()> {
        self.remove_cached_repo(repo)?;
        let repo_root = self.repo_root(repo);
        if repo_root.exists() {
            std::fs::remove_dir_all(&repo_root)
                .with_context(|| format!("removing repo index dir {}", repo_root.display()))?;
        }
        Ok(())
    }

    pub fn delete_paths(&self, repo: &Path, relative_paths: &[String]) -> Result<()> {
        if relative_paths.is_empty() {
            return Ok(());
        }
        if let Some(handle) = self.open_existing_chunk_index(repo)? {
            let schema = ChunkSchema::from_index(&handle.index)?;
            let writer = handle
                .index
                .writer::<TantivyDocument>(32_000_000)
                .context("opening chunk index writer")?;
            for relative_path in relative_paths {
                writer.delete_term(Term::from_field_text(
                    schema.relative_path_raw,
                    relative_path,
                ));
            }
            finish_writer(
                &handle.index,
                writer,
                CachedIndexKind::Chunk,
                &self.chunk_index_dir(repo),
                "committing chunk deletes",
            )?;
            handle
                .reader
                .reload()
                .context("reloading chunk reader after delete")?;
        }
        if let Some(handle) = self.open_existing_symbol_index(repo)? {
            let schema = SymbolSchema::from_index(&handle.index)?;
            let writer = handle
                .index
                .writer::<TantivyDocument>(16_000_000)
                .context("opening symbol index writer")?;
            for relative_path in relative_paths {
                writer.delete_term(Term::from_field_text(
                    schema.relative_path_raw,
                    relative_path,
                ));
            }
            finish_writer(
                &handle.index,
                writer,
                CachedIndexKind::Symbol,
                &self.symbol_index_dir(repo),
                "committing symbol deletes",
            )?;
            handle
                .reader
                .reload()
                .context("reloading symbol reader after delete")?;
        }
        Ok(())
    }

    pub fn index_chunks(&self, repo: &Path, documents: &[ChunkIndexDoc]) -> Result<()> {
        if documents.is_empty() {
            return Ok(());
        }
        let handle = self.open_or_create_chunk_index(repo)?;
        let schema = ChunkSchema::from_index(&handle.index)?;
        let writer = handle
            .index
            .writer::<TantivyDocument>(64_000_000)
            .context("opening chunk index writer")?;
        for document in documents {
            writer
                .add_document(doc!(
                schema.id => document.id.clone(),
                schema.relative_path_raw => document.relative_path.clone(),
                schema.relative_path_text => tokenize_path(&document.relative_path),
                schema.basename_raw => document.basename.clone(),
                schema.basename_text => tokenize_path(&document.basename),
                schema.extension => document.extension.clone(),
                schema.language => document.language.clone(),
                schema.content => document.content.clone(),
                schema.identifiers => tokenize_identifiers(&format!("{} {}", document.relative_path, document.content)),
                schema.start_line => document.start_line,
                schema.end_line => document.end_line,
                schema.indexed_at => document.indexed_at.clone(),
                schema.file_hash => document.file_hash.clone(),
            ))
                .context("adding chunk document to Tantivy")?;
        }
        finish_writer(
            &handle.index,
            writer,
            CachedIndexKind::Chunk,
            &self.chunk_index_dir(repo),
            "committing chunk documents",
        )?;
        handle
            .reader
            .reload()
            .context("reloading chunk reader after commit")?;
        Ok(())
    }

    #[cfg(test)]
    pub fn replace_symbol_docs(
        &self,
        repo: &Path,
        relative_path: &str,
        documents: &[SymbolIndexDoc],
    ) -> Result<()> {
        self.replace_symbol_docs_batch(repo, &[(relative_path.to_string(), documents.to_vec())])
    }

    pub fn replace_symbol_docs_batch(
        &self,
        repo: &Path,
        replacements: &[(String, Vec<SymbolIndexDoc>)],
    ) -> Result<()> {
        if replacements.is_empty() {
            return Ok(());
        }
        let handle = self.open_or_create_symbol_index(repo)?;
        let schema = SymbolSchema::from_index(&handle.index)?;
        let writer = handle
            .index
            .writer::<TantivyDocument>(16_000_000)
            .context("opening symbol index writer")?;
        for (relative_path, documents) in replacements {
            writer.delete_term(Term::from_field_text(
                schema.relative_path_raw,
                relative_path,
            ));
            for document in documents {
                writer
                    .add_document(doc!(
                        schema.symbol_id => document.symbol_id.clone(),
                        schema.relative_path_raw => document.relative_path.clone(),
                        schema.relative_path_text => tokenize_path(&document.relative_path),
                        schema.basename_raw => document.basename.clone(),
                        schema.basename_text => tokenize_path(&document.basename),
                        schema.name_raw => document.name.clone(),
                        schema.name_text => tokenize_identifiers(&document.name),
                        schema.kind => document.kind.clone(),
                        schema.container_text => document.container.clone().unwrap_or_default(),
                        schema.language => document.language.clone(),
                        schema.start_line => document.start_line,
                        schema.end_line => document.end_line,
                        schema.indexed_at => document.indexed_at.clone(),
                        schema.file_hash => document.file_hash.clone(),
                    ))
                    .context("adding symbol document to Tantivy")?;
            }
        }
        finish_writer(
            &handle.index,
            writer,
            CachedIndexKind::Symbol,
            &self.symbol_index_dir(repo),
            "committing symbol documents",
        )?;
        handle
            .reader
            .reload()
            .context("reloading symbol reader after commit")?;
        Ok(())
    }

    pub fn replace_relation_docs_for_paths(
        &self,
        repo: &Path,
        source_paths: &[String],
        documents: &[ResolvedRelation],
    ) -> Result<()> {
        if source_paths.is_empty() && documents.is_empty() {
            return Ok(());
        }
        let handle = self.open_or_create_relation_index(repo)?;
        let schema = RelationSchema::from_index(&handle.index)?;
        let writer = handle
            .index
            .writer::<TantivyDocument>(32_000_000)
            .context("opening relation index writer")?;
        for source_path in source_paths {
            writer.delete_term(Term::from_field_text(schema.source_path_raw, source_path));
        }
        add_relation_documents(&writer, &schema, documents)?;
        finish_writer(
            &handle.index,
            writer,
            CachedIndexKind::Relation,
            &self.relation_index_dir(repo),
            "committing relation documents",
        )?;
        handle
            .reader
            .reload()
            .context("reloading relation reader")?;
        Ok(())
    }

    pub fn search_relations(
        &self,
        repo: &Path,
        query_text: &str,
        limit: usize,
    ) -> Result<Vec<ResolvedRelation>> {
        let Some(handle) = self.open_existing_relation_index(repo)? else {
            return Ok(Vec::new());
        };
        let schema = RelationSchema::from_index(&handle.index)?;
        let searcher = handle.reader.searcher();
        let parser = QueryParser::for_index(
            &handle.index,
            vec![
                schema.target_name_text,
                schema.target_qualified_name,
                schema.evidence,
            ],
        );
        let mut queries: Vec<Box<dyn tantivy::query::Query>> = Vec::new();
        if let Ok(query) = parser.parse_query(query_text) {
            queries.push(query);
        }
        for term in normalized_terms(query_text) {
            queries.push(Box::new(FuzzyTermQuery::new_prefix(
                Term::from_field_text(schema.target_name_text, &term),
                1,
                true,
            )));
        }
        if queries.is_empty() {
            return Ok(Vec::new());
        }
        let query = BooleanQuery::union(queries);
        let docs = searcher.search(&query, &TopDocs::with_limit(limit.clamp(1, 100)))?;
        docs.into_iter()
            .map(|(_, address)| {
                let document: TantivyDocument = searcher.doc(address)?;
                let mut relation = relation_from_document(&schema, &document)?;
                // A text or fuzzy match is candidate evidence, regardless of the
                // structural confidence carried by the indexed occurrence.
                relation.confidence = CONFIDENCE_LEXICAL;
                relation.resolution = "lexical_fallback".to_string();
                Ok(relation)
            })
            .collect()
    }

    pub fn relation_frontier(
        &self,
        repo: &Path,
        keys: &[String],
        min_confidence: u64,
        limit: usize,
        reverse: bool,
    ) -> Result<Vec<ResolvedRelation>> {
        if keys.is_empty() {
            return Ok(Vec::new());
        }
        let Some(handle) = self.open_existing_relation_index(repo)? else {
            return Ok(Vec::new());
        };
        let schema = RelationSchema::from_index(&handle.index)?;
        let key_field = if reverse {
            schema.target_key
        } else {
            schema.source_key
        };
        let key_query = BooleanQuery::union(
            keys.iter()
                .map(|key| {
                    Box::new(TermQuery::new(
                        Term::from_field_text(key_field, key),
                        IndexRecordOption::Basic,
                    )) as Box<dyn tantivy::query::Query>
                })
                .collect(),
        );
        let searcher = handle.reader.searcher();
        let docs = searcher.search(
            &key_query,
            &TopDocs::with_limit(limit.clamp(1, 1_000))
                .order_by_fast_field::<u64>("confidence", Order::Desc),
        )?;
        let mut output = docs
            .into_iter()
            .map(|(_, address)| {
                let document: TantivyDocument = searcher.doc(address)?;
                relation_from_document(&schema, &document)
            })
            .collect::<Result<Vec<_>>>()?;
        output.retain(|relation| relation.confidence >= min_confidence);
        output.sort_by(|left, right| {
            right
                .confidence
                .cmp(&left.confidence)
                .then(left.source_path.cmp(&right.source_path))
                .then(left.start_line.cmp(&right.start_line))
                .then(left.reference_id.cmp(&right.reference_id))
        });
        Ok(output)
    }

    pub fn search_chunks(
        &self,
        repo: &Path,
        request: &ChunkSearchRequest,
    ) -> Result<Vec<ChunkSearchHit>> {
        let Some(handle) = self.open_existing_chunk_index(repo)? else {
            return Ok(Vec::new());
        };
        let schema = ChunkSchema::from_index(&handle.index)?;
        let searcher = handle.reader.searcher();
        let query = build_chunk_query(&handle.index, &schema, request)?;
        let fetch_limit = (request.limit.max(5) * 12).min(256);
        let docs = searcher
            .search(&query, &TopDocs::with_limit(fetch_limit))
            .context("executing chunk lexical search")?;

        let mut hits = Vec::new();
        for (score, address) in docs {
            let document: TantivyDocument = searcher.doc(address).context("loading chunk doc")?;
            let hit = chunk_hit_from_document(&schema, &document, score as f64)?;
            if !matches_chunk_filters(&hit, request) {
                continue;
            }
            hits.push(ChunkSearchHit {
                score: hit.score + lexical_boost_for_hit(&hit, request),
                ..hit
            });
            if hits.len() >= request.limit {
                break;
            }
        }
        hits.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(hits)
    }

    pub fn search_symbols(
        &self,
        repo: &Path,
        request: &SymbolSearchRequest,
    ) -> Result<Vec<SymbolSearchHit>> {
        let Some(handle) = self.open_existing_symbol_index(repo)? else {
            return Ok(Vec::new());
        };
        let schema = SymbolSchema::from_index(&handle.index)?;
        let searcher = handle.reader.searcher();
        let query = build_symbol_query(&handle.index, &schema, request)?;
        let fetch_limit = (request.limit.max(5) * 10).min(256);
        let docs = searcher
            .search(&query, &TopDocs::with_limit(fetch_limit))
            .context("executing symbol lexical search")?;

        let mut hits = Vec::new();
        for (score, address) in docs {
            let document: TantivyDocument = searcher.doc(address).context("loading symbol doc")?;
            let hit = symbol_hit_from_document(&schema, &document, score as f64)?;
            if !matches_symbol_filters(&hit, request) {
                continue;
            }
            hits.push(SymbolSearchHit {
                score: hit.score + symbol_boost_for_hit(&hit, request),
                ..hit
            });
            if hits.len() >= request.limit {
                break;
            }
        }
        hits.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(hits)
    }

    pub fn chunks_for_file(&self, repo: &Path, relative_path: &str) -> Result<Vec<ChunkSearchHit>> {
        let Some(handle) = self.open_existing_chunk_index(repo)? else {
            return Ok(Vec::new());
        };
        let schema = ChunkSchema::from_index(&handle.index)?;
        let searcher = handle.reader.searcher();
        let query = TermQuery::new(
            Term::from_field_text(schema.relative_path_raw, relative_path),
            IndexRecordOption::Basic,
        );
        let docs = searcher
            .search(&query, &DocSetCollector)
            .context("searching chunks by file")?;

        let mut hits = docs
            .into_iter()
            .map(|address| -> Result<_> {
                let document: TantivyDocument =
                    searcher.doc(address).context("loading chunk doc")?;
                chunk_hit_from_document(&schema, &document, 0.0)
            })
            .collect::<Result<Vec<_>>>()?;
        hits.sort_by(|left, right| {
            left.start_line
                .cmp(&right.start_line)
                .then(left.end_line.cmp(&right.end_line))
        });
        Ok(hits)
    }

    pub fn relation_document_count(&self, repo: &Path) -> Result<u64> {
        let Some(handle) = self.open_existing_relation_index(repo)? else {
            return Ok(0);
        };
        Ok(handle.reader.searcher().num_docs())
    }

    pub fn chunk_coverage(&self, repo: &Path) -> Result<ChunkIndexCoverage> {
        let Some(handle) = self.open_existing_chunk_index(repo)? else {
            return Ok(ChunkIndexCoverage::default());
        };
        let schema = ChunkSchema::from_index(&handle.index)?;
        let searcher = handle.reader.searcher();
        let mut files = BTreeSet::new();
        for segment in searcher.segment_readers() {
            let store = segment
                .get_store_reader(1)
                .context("opening chunk store reader")?;
            for doc_id in 0..segment.max_doc() {
                if segment.is_deleted(doc_id) {
                    continue;
                }
                let document: TantivyDocument = store.get(doc_id).context("loading chunk doc")?;
                if let Some(path) = document
                    .get_first(schema.relative_path_raw)
                    .and_then(|value| value.as_str())
                {
                    files.insert(path.to_string());
                }
            }
        }
        Ok(ChunkIndexCoverage {
            indexed_files: files.len() as u64,
            total_chunks: searcher.num_docs(),
        })
    }

    fn open_or_create_chunk_index(&self, repo: &Path) -> Result<Arc<CachedIndex>> {
        self.open_cached_index(repo, CachedIndexKind::Chunk, true)?
            .ok_or_else(|| anyhow!("chunk index unexpectedly missing after create"))
    }

    fn open_or_create_symbol_index(&self, repo: &Path) -> Result<Arc<CachedIndex>> {
        self.open_cached_index(repo, CachedIndexKind::Symbol, true)?
            .ok_or_else(|| anyhow!("symbol index unexpectedly missing after create"))
    }

    fn open_or_create_relation_index(&self, repo: &Path) -> Result<Arc<CachedIndex>> {
        self.open_cached_index(repo, CachedIndexKind::Relation, true)?
            .ok_or_else(|| anyhow!("relation index unexpectedly missing after create"))
    }

    fn open_existing_chunk_index(&self, repo: &Path) -> Result<Option<Arc<CachedIndex>>> {
        self.open_cached_index(repo, CachedIndexKind::Chunk, false)
    }

    fn open_existing_symbol_index(&self, repo: &Path) -> Result<Option<Arc<CachedIndex>>> {
        self.open_cached_index(repo, CachedIndexKind::Symbol, false)
    }

    fn open_existing_relation_index(&self, repo: &Path) -> Result<Option<Arc<CachedIndex>>> {
        self.open_cached_index(repo, CachedIndexKind::Relation, false)
    }

    fn open_cached_index(
        &self,
        repo: &Path,
        kind: CachedIndexKind,
        create: bool,
    ) -> Result<Option<Arc<CachedIndex>>> {
        let repo_root = self.repo_root(repo);
        if let Some(existing) = self.cached_index(&repo_root, kind)? {
            return Ok(Some(existing));
        }

        let path = match kind {
            CachedIndexKind::Chunk => self.chunk_index_dir(repo),
            CachedIndexKind::Symbol => self.symbol_index_dir(repo),
            CachedIndexKind::Relation => self.relation_index_dir(repo),
        };
        if !path.exists() {
            if !create {
                return Ok(None);
            }
            std::fs::create_dir_all(&path)
                .with_context(|| format!("creating lexical index dir {}", path.display()))?;
        }

        let index = if path
            .read_dir()
            .with_context(|| format!("reading lexical index dir {}", path.display()))?
            .next()
            .is_none()
        {
            let schema = match kind {
                CachedIndexKind::Chunk => chunk_schema(),
                CachedIndexKind::Symbol => symbol_schema(),
                CachedIndexKind::Relation => relation_schema(),
            };
            Index::create_in_dir(&path, schema)
                .with_context(|| format!("creating Tantivy index {}", path.display()))?
        } else {
            Index::open_in_dir(&path)
                .with_context(|| format!("opening Tantivy index {}", path.display()))?
        };
        maintain_existing_index(&index, kind, &path)?;
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()
            .with_context(|| format!("opening Tantivy reader {}", path.display()))?;
        let cached = Arc::new(CachedIndex { index, reader });
        Ok(Some(self.insert_cached_index(repo_root, kind, cached)?))
    }

    fn cached_index(
        &self,
        repo_root: &Path,
        kind: CachedIndexKind,
    ) -> Result<Option<Arc<CachedIndex>>> {
        let cache = self
            .cache
            .lock()
            .map_err(|_| anyhow!("local lexical index cache poisoned"))?;
        let mut cache = cache;
        let tick = next_access_tick(&mut cache);
        let Some(repo_cache) = cache.repos.get_mut(repo_root) else {
            return Ok(None);
        };
        repo_cache.last_access_tick = tick;
        Ok(match kind {
            CachedIndexKind::Chunk => repo_cache.chunk.clone(),
            CachedIndexKind::Symbol => repo_cache.symbol.clone(),
            CachedIndexKind::Relation => repo_cache.relation.clone(),
        })
    }

    fn insert_cached_index(
        &self,
        repo_root: PathBuf,
        kind: CachedIndexKind,
        cached: Arc<CachedIndex>,
    ) -> Result<Arc<CachedIndex>> {
        let mut cache = self
            .cache
            .lock()
            .map_err(|_| anyhow!("local lexical index cache poisoned"))?;
        let tick = next_access_tick(&mut cache);
        let repo_cache = cache.repos.entry(repo_root.clone()).or_default();
        repo_cache.last_access_tick = tick;
        let slot = match kind {
            CachedIndexKind::Chunk => &mut repo_cache.chunk,
            CachedIndexKind::Symbol => &mut repo_cache.symbol,
            CachedIndexKind::Relation => &mut repo_cache.relation,
        };
        if let Some(existing) = slot {
            return Ok(existing.clone());
        }
        *slot = Some(cached.clone());
        evict_lru_repos(&mut cache, self.max_warm_repos, Some(&repo_root));
        Ok(cached)
    }

    fn remove_cached_repo(&self, repo: &Path) -> Result<()> {
        let mut cache = self
            .cache
            .lock()
            .map_err(|_| anyhow!("local lexical index cache poisoned"))?;
        cache.repos.remove(&self.repo_root(repo));
        Ok(())
    }

    fn repo_root(&self, repo: &Path) -> PathBuf {
        let digest = md5::compute(repo.display().to_string());
        self.root.join("repos").join(format!("{digest:x}"))
    }

    fn chunk_index_dir(&self, repo: &Path) -> PathBuf {
        self.repo_root(repo).join("chunks")
    }

    fn symbol_index_dir(&self, repo: &Path) -> PathBuf {
        self.repo_root(repo).join("symbols")
    }

    fn relation_index_dir(&self, repo: &Path) -> PathBuf {
        self.repo_root(repo).join("relations")
    }

    #[cfg(test)]
    fn cached_repo_count(&self) -> Result<usize> {
        let cache = self
            .cache
            .lock()
            .map_err(|_| anyhow!("local lexical index cache poisoned"))?;
        Ok(cache.repos.len())
    }

    #[cfg(test)]
    fn cached_repo_roots(&self) -> Result<Vec<PathBuf>> {
        let cache = self
            .cache
            .lock()
            .map_err(|_| anyhow!("local lexical index cache poisoned"))?;
        Ok(cache.repos.keys().cloned().collect())
    }

    #[cfg(test)]
    fn symbol_segment_count(&self, repo: &Path) -> Result<usize> {
        let Some(handle) = self.open_existing_symbol_index(repo)? else {
            return Ok(0);
        };
        Ok(handle.index.searchable_segment_metas()?.len())
    }
}

fn next_access_tick(state: &mut RepoCacheState) -> u64 {
    state.access_tick = state.access_tick.saturating_add(1);
    state.access_tick
}

fn finish_writer(
    index: &Index,
    mut writer: IndexWriter,
    kind: CachedIndexKind,
    path: &Path,
    commit_context: &'static str,
) -> Result<()> {
    writer.commit().context(commit_context)?;
    compact_index_if_needed(index, &mut writer, kind, path)?;
    writer
        .wait_merging_threads()
        .with_context(|| format!("waiting for Tantivy maintenance {}", path.display()))?;
    Ok(())
}

fn maintain_existing_index(index: &Index, kind: CachedIndexKind, path: &Path) -> Result<()> {
    if !index_needs_compaction(index, kind, path)? {
        return Ok(());
    }

    let mut writer = index
        .writer::<TantivyDocument>(64_000_000)
        .with_context(|| format!("opening Tantivy maintenance writer {}", path.display()))?;
    compact_index_if_needed(index, &mut writer, kind, path)?;
    writer
        .wait_merging_threads()
        .with_context(|| format!("waiting for Tantivy maintenance {}", path.display()))?;
    Ok(())
}

fn compact_index_if_needed(
    index: &Index,
    writer: &mut IndexWriter,
    kind: CachedIndexKind,
    path: &Path,
) -> Result<()> {
    let segment_metas = index
        .searchable_segment_metas()
        .with_context(|| format!("reading Tantivy segments {}", path.display()))?;
    if !segment_metas_need_compaction(&segment_metas, kind) {
        return Ok(());
    }

    let segment_ids = segment_metas
        .iter()
        .map(|segment| segment.id())
        .collect::<Vec<_>>();
    if segment_ids.is_empty() {
        return Ok(());
    }
    writer
        .merge(&segment_ids)
        .wait()
        .with_context(|| format!("compacting Tantivy index {}", path.display()))?;
    Ok(())
}

fn index_needs_compaction(index: &Index, kind: CachedIndexKind, path: &Path) -> Result<bool> {
    let segment_metas = index
        .searchable_segment_metas()
        .with_context(|| format!("reading Tantivy segments {}", path.display()))?;
    Ok(segment_metas_need_compaction(&segment_metas, kind))
}

fn segment_metas_need_compaction(
    segment_metas: &[tantivy::index::SegmentMeta],
    kind: CachedIndexKind,
) -> bool {
    if segment_metas.len() > compaction_segment_threshold(kind) {
        return true;
    }

    let deleted_docs = segment_metas
        .iter()
        .map(|segment| segment.num_deleted_docs() as u64)
        .sum::<u64>();
    if deleted_docs < DELETED_DOC_COMPACTION_MIN {
        return false;
    }

    let max_docs = segment_metas
        .iter()
        .map(|segment| segment.max_doc() as u64)
        .sum::<u64>();
    deleted_docs_need_compaction(deleted_docs, max_docs, kind)
}

fn deleted_docs_need_compaction(deleted_docs: u64, max_docs: u64, kind: CachedIndexKind) -> bool {
    deleted_docs >= DELETED_DOC_COMPACTION_MIN
        && max_docs > 0
        && (deleted_docs as f64 / max_docs as f64) >= deleted_doc_compaction_ratio(kind)
}

fn deleted_doc_compaction_ratio(kind: CachedIndexKind) -> f64 {
    match kind {
        CachedIndexKind::Relation => RELATION_DELETED_DOC_COMPACTION_RATIO,
        CachedIndexKind::Chunk | CachedIndexKind::Symbol => DELETED_DOC_COMPACTION_RATIO,
    }
}

fn compaction_segment_threshold(kind: CachedIndexKind) -> usize {
    match kind {
        CachedIndexKind::Chunk => CHUNK_SEGMENT_COMPACTION_THRESHOLD,
        CachedIndexKind::Symbol => SYMBOL_SEGMENT_COMPACTION_THRESHOLD,
        CachedIndexKind::Relation => RELATION_SEGMENT_COMPACTION_THRESHOLD,
    }
}

fn evict_lru_repos(state: &mut RepoCacheState, max_warm_repos: usize, preserve: Option<&Path>) {
    if max_warm_repos == 0 {
        state.repos.clear();
        return;
    }

    while state.repos.len() > max_warm_repos {
        let Some((repo_root, _)) = state
            .repos
            .iter()
            .filter(|(_, cache)| {
                cache.chunk.is_some() || cache.symbol.is_some() || cache.relation.is_some()
            })
            .filter(|(repo_root, _)| preserve.is_none_or(|value| *repo_root != value))
            .min_by_key(|(_, cache)| cache.last_access_tick)
            .map(|(repo_root, cache)| (repo_root.clone(), cache.last_access_tick))
        else {
            break;
        };
        state.repos.remove(&repo_root);
    }
}

#[derive(Clone, Copy)]
struct ChunkSchema {
    id: Field,
    relative_path_raw: Field,
    relative_path_text: Field,
    basename_raw: Field,
    basename_text: Field,
    extension: Field,
    language: Field,
    content: Field,
    identifiers: Field,
    start_line: Field,
    end_line: Field,
    indexed_at: Field,
    file_hash: Field,
}

#[derive(Clone, Copy)]
struct SymbolSchema {
    symbol_id: Field,
    relative_path_raw: Field,
    relative_path_text: Field,
    basename_raw: Field,
    basename_text: Field,
    name_raw: Field,
    name_text: Field,
    kind: Field,
    container_text: Field,
    language: Field,
    start_line: Field,
    end_line: Field,
    indexed_at: Field,
    file_hash: Field,
}

#[derive(Clone, Copy)]
struct RelationSchema {
    source_key: Field,
    source_path_raw: Field,
    target_key: Field,
    target_path_raw: Field,
    target_name_raw: Field,
    target_name_text: Field,
    target_qualified_name: Field,
    kind: Field,
    confidence: Field,
    resolution: Field,
    start_line: Field,
    end_line: Field,
    evidence: Field,
    source_role: Field,
    language: Field,
}

impl ChunkSchema {
    fn from_index(index: &Index) -> Result<Self> {
        let schema = index.schema();
        Ok(Self {
            id: field(&schema, "id")?,
            relative_path_raw: field(&schema, "relative_path_raw")?,
            relative_path_text: field(&schema, "relative_path_text")?,
            basename_raw: field(&schema, "basename_raw")?,
            basename_text: field(&schema, "basename_text")?,
            extension: field(&schema, "extension")?,
            language: field(&schema, "language")?,
            content: field(&schema, "content")?,
            identifiers: field(&schema, "identifiers")?,
            start_line: field(&schema, "start_line")?,
            end_line: field(&schema, "end_line")?,
            indexed_at: field(&schema, "indexed_at")?,
            file_hash: field(&schema, "file_hash")?,
        })
    }
}

impl SymbolSchema {
    fn from_index(index: &Index) -> Result<Self> {
        let schema = index.schema();
        Ok(Self {
            symbol_id: field(&schema, "symbol_id")?,
            relative_path_raw: field(&schema, "relative_path_raw")?,
            relative_path_text: field(&schema, "relative_path_text")?,
            basename_raw: field(&schema, "basename_raw")?,
            basename_text: field(&schema, "basename_text")?,
            name_raw: field(&schema, "name_raw")?,
            name_text: field(&schema, "name_text")?,
            kind: field(&schema, "kind")?,
            container_text: field(&schema, "container_text")?,
            language: field(&schema, "language")?,
            start_line: field(&schema, "start_line")?,
            end_line: field(&schema, "end_line")?,
            indexed_at: field(&schema, "indexed_at")?,
            file_hash: field(&schema, "file_hash")?,
        })
    }
}

impl RelationSchema {
    fn from_index(index: &Index) -> Result<Self> {
        let schema = index.schema();
        Ok(Self {
            source_key: field(&schema, "source_key")?,
            source_path_raw: field(&schema, "source_path_raw")?,
            target_key: field(&schema, "target_key")?,
            target_path_raw: field(&schema, "target_path_raw")?,
            target_name_raw: field(&schema, "target_name_raw")?,
            target_name_text: field(&schema, "target_name_text")?,
            target_qualified_name: field(&schema, "target_qualified_name")?,
            kind: field(&schema, "kind")?,
            confidence: field(&schema, "confidence")?,
            resolution: field(&schema, "resolution")?,
            start_line: field(&schema, "start_line")?,
            end_line: field(&schema, "end_line")?,
            evidence: field(&schema, "evidence")?,
            source_role: field(&schema, "source_role")?,
            language: field(&schema, "language")?,
        })
    }
}

fn add_relation_documents(
    writer: &IndexWriter<TantivyDocument>,
    schema: &RelationSchema,
    documents: &[ResolvedRelation],
) -> Result<()> {
    for relation in documents {
        let mut document = doc!(
            schema.source_key => relation.source_key.clone(),
            schema.source_path_raw => relation.source_path.clone(),
            schema.target_name_raw => relation.target_name.clone(),
            schema.target_name_text => tokenize_identifiers(&relation.target_name),
            schema.kind => relation.kind.as_str(),
            schema.confidence => relation.confidence,
            schema.resolution => relation.resolution.clone(),
            schema.start_line => relation.start_line,
            schema.end_line => relation.end_line,
            schema.evidence => relation.evidence.clone(),
            schema.source_role => relation.source_role.clone(),
            schema.language => relation.language.clone(),
        );
        if let Some(target_key) = relation.target_key.as_deref() {
            document.add_text(schema.target_key, target_key);
        }
        if let Some(target_path) = relation.target_path.as_deref() {
            document.add_text(schema.target_path_raw, target_path);
        }
        if let Some(target_qualified_name) = relation.target_qualified_name.as_deref() {
            document.add_text(schema.target_qualified_name, target_qualified_name);
        }
        writer.add_document(document)?;
    }
    Ok(())
}

fn chunk_schema() -> Schema {
    let mut builder = Schema::builder();
    builder.add_text_field("id", STRING | STORED);
    builder.add_text_field("relative_path_raw", STRING | STORED);
    builder.add_text_field("relative_path_text", TEXT);
    builder.add_text_field("basename_raw", STRING | STORED);
    builder.add_text_field("basename_text", TEXT);
    builder.add_text_field("extension", STRING | STORED);
    builder.add_text_field("language", STRING | STORED);
    builder.add_text_field("content", TEXT | STORED);
    builder.add_text_field("identifiers", TEXT);
    builder.add_u64_field("start_line", FAST | STORED);
    builder.add_u64_field("end_line", FAST | STORED);
    builder.add_text_field("indexed_at", STRING | STORED);
    builder.add_text_field("file_hash", STRING | STORED);
    builder.build()
}

fn symbol_schema() -> Schema {
    let mut builder = Schema::builder();
    builder.add_text_field("symbol_id", STRING | STORED);
    builder.add_text_field("relative_path_raw", STRING | STORED);
    builder.add_text_field("relative_path_text", TEXT);
    builder.add_text_field("basename_raw", STRING | STORED);
    builder.add_text_field("basename_text", TEXT);
    builder.add_text_field("name_raw", STRING | STORED);
    builder.add_text_field("name_text", TEXT | STORED);
    builder.add_text_field("kind", STRING | STORED);
    builder.add_text_field("container_text", TEXT | STORED);
    builder.add_text_field("language", STRING | STORED);
    builder.add_u64_field("start_line", FAST | STORED);
    builder.add_u64_field("end_line", FAST | STORED);
    builder.add_text_field("indexed_at", STRING | STORED);
    builder.add_text_field("file_hash", STRING | STORED);
    builder.build()
}

fn relation_schema() -> Schema {
    let mut builder = Schema::builder();
    builder.add_text_field("source_key", STRING | STORED);
    builder.add_text_field("source_path_raw", STRING | STORED);
    builder.add_text_field("target_key", STRING | STORED);
    builder.add_text_field("target_path_raw", STRING);
    builder.add_text_field("target_name_raw", STRING | STORED);
    builder.add_text_field("target_name_text", TEXT);
    builder.add_text_field("target_qualified_name", TEXT);
    builder.add_text_field("kind", STRING | STORED);
    builder.add_u64_field("confidence", FAST | STORED);
    builder.add_text_field("resolution", STRING | STORED);
    builder.add_u64_field("start_line", FAST | STORED);
    builder.add_u64_field("end_line", FAST | STORED);
    builder.add_text_field("evidence", TEXT | STORED);
    builder.add_text_field("source_role", STRING | STORED);
    builder.add_text_field("language", STRING);
    builder.build()
}

fn build_chunk_query(
    index: &Index,
    schema: &ChunkSchema,
    request: &ChunkSearchRequest,
) -> Result<Box<dyn tantivy::query::Query>> {
    let (fields, query_text) = match request.flavor {
        QueryFlavor::NaturalLanguage => (
            vec![
                schema.content,
                schema.identifiers,
                schema.basename_text,
                schema.relative_path_text,
            ],
            request.query.trim().to_string(),
        ),
        QueryFlavor::Identifier => (
            vec![
                schema.identifiers,
                schema.basename_text,
                schema.relative_path_text,
                schema.content,
            ],
            tokenize_identifiers(&request.query),
        ),
        QueryFlavor::Path => (
            vec![
                schema.relative_path_text,
                schema.basename_text,
                schema.identifiers,
            ],
            tokenize_path(&request.query),
        ),
        QueryFlavor::Mixed => (
            vec![
                schema.content,
                schema.identifiers,
                schema.relative_path_text,
                schema.basename_text,
            ],
            format!(
                "{} {}",
                request.query.trim(),
                tokenize_identifiers(&request.query)
            ),
        ),
    };

    if query_text.trim().is_empty() {
        anyhow::bail!("query is empty after normalization");
    }

    let mut parser = QueryParser::for_index(index, fields);
    parser.set_field_boost(schema.content, 1.0);
    parser.set_field_boost(schema.identifiers, 2.0);
    parser.set_field_boost(schema.basename_text, 2.5);
    parser.set_field_boost(schema.relative_path_text, 2.2);
    let (query, _) = parser.parse_query_lenient(&query_text);
    Ok(query)
}

fn build_symbol_query(
    index: &Index,
    schema: &SymbolSchema,
    request: &SymbolSearchRequest,
) -> Result<Box<dyn tantivy::query::Query>> {
    let query_text = match request.flavor {
        QueryFlavor::Path => tokenize_path(&request.query),
        QueryFlavor::Identifier => tokenize_identifiers(&request.query),
        QueryFlavor::Mixed => format!(
            "{} {}",
            request.query.trim(),
            tokenize_identifiers(&request.query)
        ),
        QueryFlavor::NaturalLanguage => request.query.trim().to_string(),
    };
    if query_text.trim().is_empty() {
        anyhow::bail!("symbol query is empty after normalization");
    }

    let mut parser = QueryParser::for_index(
        index,
        vec![
            schema.name_text,
            schema.container_text,
            schema.basename_text,
            schema.relative_path_text,
        ],
    );
    parser.set_field_boost(schema.name_text, 3.0);
    parser.set_field_boost(schema.container_text, 1.8);
    parser.set_field_boost(schema.basename_text, 1.5);
    parser.set_field_boost(schema.relative_path_text, 1.5);
    let (query, _) = parser.parse_query_lenient(&query_text);
    Ok(query)
}

fn lexical_boost_for_hit(hit: &ChunkSearchHit, request: &ChunkSearchRequest) -> f64 {
    let normalized_query = request.query.trim().to_lowercase();
    let basename = hit.basename.to_lowercase();
    let relative_path = hit.relative_path.to_lowercase();
    let content = hit.content.to_lowercase();
    let identifier_terms = normalized_terms(&request.query);
    let exact_identifier = identifier_terms
        .iter()
        .any(|term| !term.is_empty() && content.contains(term));

    let mut boost = 0.0;
    if relative_path == normalized_query || basename == normalized_query {
        boost += 6.0;
    }
    if relative_path.contains(&normalized_query) && !normalized_query.is_empty() {
        boost += 2.5;
    }
    if basename.contains(&normalized_query) && !normalized_query.is_empty() {
        boost += 2.0;
    }
    if exact_identifier {
        boost += 1.4;
    }
    boost
}

fn symbol_boost_for_hit(hit: &SymbolSearchHit, request: &SymbolSearchRequest) -> f64 {
    let normalized_query = request.query.trim().to_lowercase();
    let mut boost = 0.0;
    if hit.name.to_lowercase() == normalized_query {
        boost += 8.0;
    }
    if hit.basename.to_lowercase() == normalized_query {
        boost += 3.0;
    }
    if hit.relative_path.to_lowercase().contains(&normalized_query) && !normalized_query.is_empty()
    {
        boost += 2.0;
    }
    if let Some(container) = &hit.container
        && container.to_lowercase().contains(&normalized_query)
        && !normalized_query.is_empty()
    {
        boost += 1.0;
    }
    boost
}

fn matches_chunk_filters(hit: &ChunkSearchHit, request: &ChunkSearchRequest) -> bool {
    if let Some(file) = &request.file
        && normalize_relative_path(&hit.relative_path) != normalize_relative_path(file)
    {
        return false;
    }
    if let Some(path_prefix) = &request.path_prefix
        && !normalize_relative_path(&hit.relative_path)
            .starts_with(&normalize_relative_path(path_prefix))
    {
        return false;
    }
    if let Some(language) = &request.language
        && hit.language != language.to_lowercase()
    {
        return false;
    }
    if !request.extension_filter.is_empty() && !request.extension_filter.contains(&hit.extension) {
        return false;
    }
    true
}

fn matches_symbol_filters(hit: &SymbolSearchHit, request: &SymbolSearchRequest) -> bool {
    if let Some(path_prefix) = &request.path_prefix
        && !normalize_relative_path(&hit.relative_path)
            .starts_with(&normalize_relative_path(path_prefix))
    {
        return false;
    }
    if let Some(language) = &request.language
        && hit.language != language.to_lowercase()
    {
        return false;
    }
    if let Some(kind) = &request.kind
        && hit.kind != kind.to_lowercase()
    {
        return false;
    }
    if let Some(container) = &request.container
        && !hit
            .container
            .as_deref()
            .unwrap_or_default()
            .to_lowercase()
            .contains(&container.to_lowercase())
    {
        return false;
    }
    true
}

fn chunk_hit_from_document(
    schema: &ChunkSchema,
    document: &TantivyDocument,
    score: f64,
) -> Result<ChunkSearchHit> {
    Ok(ChunkSearchHit {
        id: string_value(document, schema.id)?,
        relative_path: string_value(document, schema.relative_path_raw)?,
        basename: string_value(document, schema.basename_raw)?,
        extension: string_value(document, schema.extension)?,
        language: string_value(document, schema.language)?,
        content: string_value(document, schema.content)?,
        start_line: u64_value(document, schema.start_line)?,
        end_line: u64_value(document, schema.end_line)?,
        indexed_at: string_value(document, schema.indexed_at)?,
        file_hash: string_value(document, schema.file_hash)?,
        score,
    })
}

fn symbol_hit_from_document(
    schema: &SymbolSchema,
    document: &TantivyDocument,
    score: f64,
) -> Result<SymbolSearchHit> {
    Ok(SymbolSearchHit {
        symbol_id: string_value(document, schema.symbol_id)?,
        relative_path: string_value(document, schema.relative_path_raw)?,
        basename: string_value(document, schema.basename_raw)?,
        name: string_value(document, schema.name_raw)?,
        kind: string_value(document, schema.kind)?,
        container: optional_string_value(document, schema.container_text),
        language: string_value(document, schema.language)?,
        start_line: u64_value(document, schema.start_line)?,
        end_line: u64_value(document, schema.end_line)?,
        indexed_at: string_value(document, schema.indexed_at)?,
        file_hash: string_value(document, schema.file_hash)?,
        score,
    })
}

fn relation_from_document(
    schema: &RelationSchema,
    document: &TantivyDocument,
) -> Result<ResolvedRelation> {
    let source_path = string_value(document, schema.source_path_raw)?;
    let target_name = string_value(document, schema.target_name_raw)?;
    let kind = RelationKind::parse(&string_value(document, schema.kind)?);
    let start_line = u64_value(document, schema.start_line)?;
    Ok(ResolvedRelation {
        reference_id: format!("{source_path}:{start_line}:{}:{target_name}", kind.as_str()),
        repo: String::new(),
        source_key: string_value(document, schema.source_key)?,
        source_symbol_id: None,
        source_path,
        target_key: optional_string_value(document, schema.target_key),
        target_symbol_id: None,
        target_path: None,
        target_name,
        target_qualified_name: None,
        kind,
        confidence: u64_value(document, schema.confidence)?,
        resolution: string_value(document, schema.resolution)?,
        start_line,
        end_line: u64_value(document, schema.end_line)?,
        evidence: string_value(document, schema.evidence)?,
        source_role: string_value(document, schema.source_role)?,
        language: String::new(),
        file_hash: String::new(),
    })
}

fn tokenize_identifiers(text: &str) -> String {
    normalized_terms(text).join(" ")
}

fn tokenize_path(text: &str) -> String {
    normalized_terms(&text.replace(['/', '\\', '.', ':', '-'], " ")).join(" ")
}

fn normalized_terms(text: &str) -> Vec<String> {
    let mut normalized = String::new();
    let mut previous_lowercase = false;
    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            if ch.is_uppercase() && previous_lowercase {
                normalized.push(' ');
            }
            normalized.push(ch.to_ascii_lowercase());
            previous_lowercase = ch.is_lowercase();
        } else {
            normalized.push(' ');
            previous_lowercase = false;
        }
    }
    normalized
        .split_whitespace()
        .filter(|term| !term.is_empty())
        .map(ToString::to_string)
        .collect()
}

fn normalize_relative_path(path: &str) -> String {
    path.replace('\\', "/").trim_matches('/').to_string()
}

fn field(schema: &Schema, name: &str) -> Result<Field> {
    schema
        .get_field(name)
        .with_context(|| format!("missing Tantivy field `{name}`"))
}

fn string_value(document: &TantivyDocument, field: Field) -> Result<String> {
    document
        .get_first(field)
        .and_then(|value| value.as_str())
        .map(ToString::to_string)
        .with_context(|| format!("missing string field {}", field.field_id()))
}

fn optional_string_value(document: &TantivyDocument, field: Field) -> Option<String> {
    document
        .get_first(field)
        .and_then(|value| value.as_str())
        .map(ToString::to_string)
        .filter(|value| !value.is_empty())
}

fn u64_value(document: &TantivyDocument, field: Field) -> Result<u64> {
    document
        .get_first(field)
        .and_then(|value| value.as_u64())
        .with_context(|| format!("missing u64 field {}", field.field_id()))
}

#[cfg(test)]
mod tests {
    use super::{
        CachedIndexKind, ChunkIndexDoc, ChunkSearchRequest, LocalIndexStore, QueryFlavor,
        SYMBOL_SEGMENT_COMPACTION_THRESHOLD, SymbolIndexDoc, SymbolSearchRequest,
        deleted_docs_need_compaction,
    };
    use crate::engine::relationships::{CONFIDENCE_LEXICAL, RelationKind, ResolvedRelation};
    use std::path::Path;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn relationship_tombstones_compact_before_general_search_indexes() {
        assert!(deleted_docs_need_compaction(
            256,
            2_560,
            CachedIndexKind::Relation
        ));
        assert!(!deleted_docs_need_compaction(
            256,
            2_560,
            CachedIndexKind::Chunk
        ));
        assert!(!deleted_docs_need_compaction(
            255,
            1_000,
            CachedIndexKind::Relation
        ));
    }

    fn temp_path(name: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time went backwards")
            .as_nanos();
        std::env::temp_dir().join(format!("agent-context-lexical-{name}-{nanos}"))
    }

    #[test]
    fn chunk_search_prioritizes_exact_path_like_hits() {
        let root = temp_path("chunks");
        let store = LocalIndexStore::new(root.clone(), 4);
        let repo = std::path::Path::new("/tmp/example");
        store
            .index_chunks(
                repo,
                &[
                    ChunkIndexDoc {
                        id: "chunk_a".to_string(),
                        relative_path: "src/graphql/schema.rs".to_string(),
                        basename: "schema.rs".to_string(),
                        extension: ".rs".to_string(),
                        language: "rust".to_string(),
                        content: "pub struct Schema {}".to_string(),
                        start_line: 1,
                        end_line: 4,
                        indexed_at: "2026-01-01T00:00:00Z".to_string(),
                        file_hash: "hash-a".to_string(),
                    },
                    ChunkIndexDoc {
                        id: "chunk_b".to_string(),
                        relative_path: "src/search/index.rs".to_string(),
                        basename: "index.rs".to_string(),
                        extension: ".rs".to_string(),
                        language: "rust".to_string(),
                        content: "pub fn search_index() {}".to_string(),
                        start_line: 1,
                        end_line: 4,
                        indexed_at: "2026-01-01T00:00:00Z".to_string(),
                        file_hash: "hash-b".to_string(),
                    },
                ],
            )
            .unwrap();

        let hits = store
            .search_chunks(
                repo,
                &ChunkSearchRequest {
                    query: "graphql/schema.rs".to_string(),
                    limit: 3,
                    flavor: QueryFlavor::Path,
                    path_prefix: None,
                    language: None,
                    file: None,
                    extension_filter: Vec::new(),
                },
            )
            .unwrap();

        assert_eq!(
            hits.first().map(|hit| hit.relative_path.as_str()),
            Some("src/graphql/schema.rs")
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn symbol_search_finds_exact_name() {
        let root = temp_path("symbols");
        let store = LocalIndexStore::new(root.clone(), 4);
        let repo = std::path::Path::new("/tmp/example");
        store
            .replace_symbol_docs(
                repo,
                "src/graphql/schema.rs",
                &[SymbolIndexDoc {
                    symbol_id: "sym_a".to_string(),
                    relative_path: "src/graphql/schema.rs".to_string(),
                    basename: "schema.rs".to_string(),
                    name: "Schema".to_string(),
                    kind: "struct".to_string(),
                    container: Some("graphql".to_string()),
                    language: "rust".to_string(),
                    start_line: 1,
                    end_line: 3,
                    indexed_at: "2026-01-01T00:00:00Z".to_string(),
                    file_hash: "hash-a".to_string(),
                }],
            )
            .unwrap();

        let hits = store
            .search_symbols(
                repo,
                &SymbolSearchRequest {
                    query: "Schema".to_string(),
                    limit: 3,
                    flavor: QueryFlavor::Identifier,
                    path_prefix: None,
                    language: None,
                    kind: None,
                    container: None,
                },
            )
            .unwrap();
        assert_eq!(hits.first().map(|hit| hit.name.as_str()), Some("Schema"));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn symbol_index_maintenance_compacts_many_small_commits() {
        let root = temp_path("symbol-maintenance");
        let store = LocalIndexStore::new(root.clone(), 4);
        let repo = std::path::Path::new("/tmp/example");

        for index in 0..(SYMBOL_SEGMENT_COMPACTION_THRESHOLD + 8) {
            let relative_path = format!("src/generated_{index}.rs");
            store
                .replace_symbol_docs(
                    repo,
                    &relative_path,
                    &[SymbolIndexDoc {
                        symbol_id: format!("sym_{index}"),
                        relative_path: relative_path.clone(),
                        basename: format!("generated_{index}.rs"),
                        name: format!("Generated{index}"),
                        kind: "struct".to_string(),
                        container: None,
                        language: "rust".to_string(),
                        start_line: 1,
                        end_line: 3,
                        indexed_at: "2026-01-01T00:00:00Z".to_string(),
                        file_hash: format!("hash-{index}"),
                    }],
                )
                .unwrap();
        }

        assert!(
            store.symbol_segment_count(repo).unwrap() <= SYMBOL_SEGMENT_COMPACTION_THRESHOLD,
            "symbol index should stay below the maintenance segment threshold"
        );

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn file_chunk_lookups_are_not_capped_by_top_docs_limit() {
        let root = temp_path("many-chunks");
        let store = LocalIndexStore::new(root.clone(), 4);
        let repo = std::path::Path::new("/tmp/example");
        let docs = (0..96)
            .map(|index| ChunkIndexDoc {
                id: format!("chunk_{index}"),
                relative_path: "src/generated.rs".to_string(),
                basename: "generated.rs".to_string(),
                extension: ".rs".to_string(),
                language: "rust".to_string(),
                content: format!("fn symbol_{index}() {{}}"),
                start_line: (index as u64 * 3) + 1,
                end_line: (index as u64 * 3) + 3,
                indexed_at: "2026-01-01T00:00:00Z".to_string(),
                file_hash: "hash-generated".to_string(),
            })
            .collect::<Vec<_>>();
        store.index_chunks(repo, &docs).unwrap();

        let all_chunks = store.chunks_for_file(repo, "src/generated.rs").unwrap();
        assert_eq!(all_chunks.len(), docs.len());

        assert_eq!(
            all_chunks.last().map(|chunk| chunk.start_line),
            docs.last().map(|chunk| chunk.start_line)
        );

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn warm_repo_cache_evicts_least_recently_used_repo() {
        let root = temp_path("lru-eviction");
        let store = LocalIndexStore::new(root.clone(), 2);
        let repo_a = Path::new("/tmp/repo-a");
        let repo_b = Path::new("/tmp/repo-b");
        let repo_c = Path::new("/tmp/repo-c");

        for (repo, symbol) in [(repo_a, "alpha"), (repo_b, "beta"), (repo_c, "gamma")] {
            store
                .index_chunks(
                    repo,
                    &[ChunkIndexDoc {
                        id: format!("chunk-{symbol}"),
                        relative_path: "src/lib.rs".to_string(),
                        basename: "lib.rs".to_string(),
                        extension: ".rs".to_string(),
                        language: "rust".to_string(),
                        content: format!("fn {symbol}() {{}}"),
                        start_line: 1,
                        end_line: 3,
                        indexed_at: "2026-01-01T00:00:00Z".to_string(),
                        file_hash: format!("hash-{symbol}"),
                    }],
                )
                .unwrap();
        }

        let _ = store
            .search_chunks(
                repo_a,
                &ChunkSearchRequest {
                    query: "alpha".to_string(),
                    limit: 1,
                    flavor: QueryFlavor::Identifier,
                    path_prefix: None,
                    language: None,
                    file: None,
                    extension_filter: Vec::new(),
                },
            )
            .unwrap();

        store
            .index_chunks(
                repo_c,
                &[ChunkIndexDoc {
                    id: "chunk-delta".to_string(),
                    relative_path: "src/other.rs".to_string(),
                    basename: "other.rs".to_string(),
                    extension: ".rs".to_string(),
                    language: "rust".to_string(),
                    content: "fn delta() {}".to_string(),
                    start_line: 4,
                    end_line: 6,
                    indexed_at: "2026-01-01T00:00:00Z".to_string(),
                    file_hash: "hash-delta".to_string(),
                }],
            )
            .unwrap();

        let cached_roots = store.cached_repo_roots().unwrap();
        assert_eq!(store.cached_repo_count().unwrap(), 2);
        assert!(cached_roots.contains(&store.repo_root(repo_a)));
        assert!(cached_roots.contains(&store.repo_root(repo_c)));
        assert!(!cached_roots.contains(&store.repo_root(repo_b)));

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn active_reader_handles_remain_valid_after_eviction() {
        let root = temp_path("lru-active-handle");
        let store = LocalIndexStore::new(root.clone(), 1);
        let repo_a = Path::new("/tmp/repo-active-a");
        let repo_b = Path::new("/tmp/repo-active-b");

        for (repo, symbol) in [(repo_a, "alpha"), (repo_b, "beta")] {
            store
                .index_chunks(
                    repo,
                    &[ChunkIndexDoc {
                        id: format!("chunk-{symbol}"),
                        relative_path: "src/lib.rs".to_string(),
                        basename: "lib.rs".to_string(),
                        extension: ".rs".to_string(),
                        language: "rust".to_string(),
                        content: format!("fn {symbol}() {{}}"),
                        start_line: 1,
                        end_line: 3,
                        indexed_at: "2026-01-01T00:00:00Z".to_string(),
                        file_hash: format!("hash-{symbol}"),
                    }],
                )
                .unwrap();
        }

        let handle = store.open_existing_chunk_index(repo_a).unwrap().unwrap();
        store
            .index_chunks(
                repo_b,
                &[ChunkIndexDoc {
                    id: "chunk-gamma".to_string(),
                    relative_path: "src/other.rs".to_string(),
                    basename: "other.rs".to_string(),
                    extension: ".rs".to_string(),
                    language: "rust".to_string(),
                    content: "fn gamma() {}".to_string(),
                    start_line: 4,
                    end_line: 6,
                    indexed_at: "2026-01-01T00:00:00Z".to_string(),
                    file_hash: "hash-gamma".to_string(),
                }],
            )
            .unwrap();

        let searcher = handle.reader.searcher();
        assert_eq!(searcher.num_docs(), 1);
        assert_eq!(store.cached_repo_count().unwrap(), 1);

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn relationship_index_supports_fuzzy_search_and_path_replacement() {
        let root = temp_path("relationships");
        let store = LocalIndexStore::new(root.clone(), 4);
        let repo = Path::new("/tmp/repo-relationships");
        let relation = ResolvedRelation {
            reference_id: "ref-1".to_string(),
            repo: repo.display().to_string(),
            source_key: "source-key".to_string(),
            source_symbol_id: Some("source-id".to_string()),
            source_path: "src/source.rs".to_string(),
            target_key: Some("target-key".to_string()),
            target_symbol_id: Some("target-id".to_string()),
            target_path: Some("src/target.rs".to_string()),
            target_name: "TargetService".to_string(),
            target_qualified_name: Some("crate::TargetService".to_string()),
            kind: RelationKind::Calls,
            confidence: 900,
            resolution: "same_module_unique".to_string(),
            start_line: 12,
            end_line: 12,
            evidence: "TargetService::run()".to_string(),
            source_role: "production".to_string(),
            language: "rust".to_string(),
            file_hash: "hash".to_string(),
        };
        store
            .replace_relation_docs_for_paths(
                repo,
                &["src/source.rs".to_string()],
                std::slice::from_ref(&relation),
            )
            .unwrap();
        let fuzzy = store.search_relations(repo, "TargetServce", 5).unwrap();
        assert!(!fuzzy.is_empty());
        assert_eq!(fuzzy[0].confidence, CONFIDENCE_LEXICAL);
        assert_eq!(fuzzy[0].resolution, "lexical_fallback");
        let reverse = store
            .relation_frontier(repo, &["target-key".to_string()], 650, 5, true)
            .unwrap();
        assert_eq!(reverse.len(), 1);
        assert_eq!(reverse[0].source_key, "source-key");
        let forward = store
            .relation_frontier(repo, &["source-key".to_string()], 650, 5, false)
            .unwrap();
        assert_eq!(forward.len(), 1);
        assert_eq!(forward[0].target_key.as_deref(), Some("target-key"));

        let mut replacement = relation.clone();
        replacement.reference_id = "ref-2".to_string();
        replacement.target_key = Some("new-target-key".to_string());
        store
            .replace_relation_docs_for_paths(
                repo,
                &["src/source.rs".to_string()],
                std::slice::from_ref(&replacement),
            )
            .unwrap();
        let replacement_hit = store.search_relations(repo, "TargetServce", 5).unwrap();
        assert_eq!(replacement_hit.len(), 1);
        assert_eq!(
            replacement_hit[0].target_key.as_deref(),
            Some("new-target-key")
        );
        let mut unresolved = relation.clone();
        unresolved.reference_id = "ref-3".to_string();
        unresolved.target_key = None;
        unresolved.target_symbol_id = None;
        unresolved.target_path = None;
        unresolved.confidence = 0;
        unresolved.resolution = "unresolved".to_string();
        store
            .replace_relation_docs_for_paths(
                repo,
                &["src/source.rs".to_string()],
                std::slice::from_ref(&unresolved),
            )
            .unwrap();
        let unresolved_hit = store.search_relations(repo, "TargetServce", 5).unwrap();
        assert_eq!(unresolved_hit.len(), 1);
        assert!(unresolved_hit[0].target_key.is_none());
        assert_eq!(unresolved_hit[0].confidence, CONFIDENCE_LEXICAL);
        store
            .replace_relation_docs_for_paths(repo, &["src/source.rs".to_string()], &[])
            .unwrap();
        assert!(
            store
                .search_relations(repo, "TargetServce", 5)
                .unwrap()
                .is_empty()
        );
        let _ = std::fs::remove_dir_all(root);
    }
}
