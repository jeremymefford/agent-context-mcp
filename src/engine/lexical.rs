use anyhow::{Context, Result};
use serde::Serialize;
use std::path::{Path, PathBuf};
use tantivy::collector::{DocSetCollector, TopDocs};
use tantivy::query::{QueryParser, TermQuery};
use tantivy::schema::{
    FAST, Field, IndexRecordOption, STORED, STRING, Schema, TEXT, TantivyDocument, Value as _,
};
use tantivy::{Index, ReloadPolicy, Term, doc};

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
}

impl LocalIndexStore {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    pub fn clear_repo(&self, repo: &Path) -> Result<()> {
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
        if let Some(index) = self.open_existing_chunk_index(repo)? {
            let schema = ChunkSchema::from_index(&index)?;
            let mut writer = index
                .writer::<TantivyDocument>(32_000_000)
                .context("opening chunk index writer")?;
            for relative_path in relative_paths {
                writer.delete_term(Term::from_field_text(
                    schema.relative_path_raw,
                    relative_path,
                ));
            }
            writer.commit().context("committing chunk deletes")?;
        }
        if let Some(index) = self.open_existing_symbol_index(repo)? {
            let schema = SymbolSchema::from_index(&index)?;
            let mut writer = index
                .writer::<TantivyDocument>(16_000_000)
                .context("opening symbol index writer")?;
            for relative_path in relative_paths {
                writer.delete_term(Term::from_field_text(
                    schema.relative_path_raw,
                    relative_path,
                ));
            }
            writer.commit().context("committing symbol deletes")?;
        }
        Ok(())
    }

    pub fn index_chunks(&self, repo: &Path, documents: &[ChunkIndexDoc]) -> Result<()> {
        if documents.is_empty() {
            return Ok(());
        }
        let index = self.open_or_create_chunk_index(repo)?;
        let schema = ChunkSchema::from_index(&index)?;
        let mut writer = index
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
        writer.commit().context("committing chunk documents")?;
        Ok(())
    }

    pub fn replace_symbol_docs(
        &self,
        repo: &Path,
        relative_path: &str,
        documents: &[SymbolIndexDoc],
    ) -> Result<()> {
        let index = self.open_or_create_symbol_index(repo)?;
        let schema = SymbolSchema::from_index(&index)?;
        let mut writer = index
            .writer::<TantivyDocument>(16_000_000)
            .context("opening symbol index writer")?;
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
        writer.commit().context("committing symbol documents")?;
        Ok(())
    }

    pub fn search_chunks(
        &self,
        repo: &Path,
        request: &ChunkSearchRequest,
    ) -> Result<Vec<ChunkSearchHit>> {
        let Some(index) = self.open_existing_chunk_index(repo)? else {
            return Ok(Vec::new());
        };
        let schema = ChunkSchema::from_index(&index)?;
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()
            .context("opening chunk index reader")?;
        reader.reload().context("reloading chunk reader")?;
        let searcher = reader.searcher();
        let query = build_chunk_query(&index, &schema, request)?;
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
        let Some(index) = self.open_existing_symbol_index(repo)? else {
            return Ok(Vec::new());
        };
        let schema = SymbolSchema::from_index(&index)?;
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()
            .context("opening symbol index reader")?;
        reader.reload().context("reloading symbol reader")?;
        let searcher = reader.searcher();
        let query = build_symbol_query(&index, &schema, request)?;
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

    pub fn chunk_for_file_line(
        &self,
        repo: &Path,
        relative_path: &str,
        line: u64,
    ) -> Result<Option<ChunkSearchHit>> {
        let Some(index) = self.open_existing_chunk_index(repo)? else {
            return Ok(None);
        };
        let schema = ChunkSchema::from_index(&index)?;
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()
            .context("opening chunk index reader")?;
        reader.reload().context("reloading chunk reader")?;
        let searcher = reader.searcher();
        let query = TermQuery::new(
            Term::from_field_text(schema.relative_path_raw, relative_path),
            IndexRecordOption::Basic,
        );
        let docs = searcher
            .search(&query, &DocSetCollector)
            .context("searching chunk by file")?;

        let mut best: Option<ChunkSearchHit> = None;
        for address in docs {
            let document: TantivyDocument = searcher.doc(address).context("loading chunk doc")?;
            let hit = chunk_hit_from_document(&schema, &document, 0.0)?;
            if hit.start_line <= line && hit.end_line >= line {
                return Ok(Some(hit));
            }
            let replace = best.as_ref().is_none_or(|current| {
                distance_to_line(hit.start_line, hit.end_line, line)
                    < distance_to_line(current.start_line, current.end_line, line)
            });
            if replace {
                best = Some(hit);
            }
        }
        Ok(best)
    }

    pub fn chunks_for_file(&self, repo: &Path, relative_path: &str) -> Result<Vec<ChunkSearchHit>> {
        let Some(index) = self.open_existing_chunk_index(repo)? else {
            return Ok(Vec::new());
        };
        let schema = ChunkSchema::from_index(&index)?;
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()
            .context("opening chunk index reader")?;
        reader.reload().context("reloading chunk reader")?;
        let searcher = reader.searcher();
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

    fn open_or_create_chunk_index(&self, repo: &Path) -> Result<Index> {
        let path = self.chunk_index_dir(repo);
        if path.exists() {
            return Index::open_in_dir(&path)
                .with_context(|| format!("opening chunk index {}", path.display()));
        }
        std::fs::create_dir_all(&path)
            .with_context(|| format!("creating chunk index dir {}", path.display()))?;
        Index::create_in_dir(&path, chunk_schema())
            .with_context(|| format!("creating chunk index {}", path.display()))
    }

    fn open_or_create_symbol_index(&self, repo: &Path) -> Result<Index> {
        let path = self.symbol_index_dir(repo);
        if path.exists() {
            return Index::open_in_dir(&path)
                .with_context(|| format!("opening symbol index {}", path.display()));
        }
        std::fs::create_dir_all(&path)
            .with_context(|| format!("creating symbol index dir {}", path.display()))?;
        Index::create_in_dir(&path, symbol_schema())
            .with_context(|| format!("creating symbol index {}", path.display()))
    }

    fn open_existing_chunk_index(&self, repo: &Path) -> Result<Option<Index>> {
        let path = self.chunk_index_dir(repo);
        if !path.exists() {
            return Ok(None);
        }
        Ok(Some(Index::open_in_dir(&path).with_context(|| {
            format!("opening chunk index {}", path.display())
        })?))
    }

    fn open_existing_symbol_index(&self, repo: &Path) -> Result<Option<Index>> {
        let path = self.symbol_index_dir(repo);
        if !path.exists() {
            return Ok(None);
        }
        Ok(Some(Index::open_in_dir(&path).with_context(|| {
            format!("opening symbol index {}", path.display())
        })?))
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

fn distance_to_line(start_line: u64, end_line: u64, line: u64) -> u64 {
    if start_line <= line && end_line >= line {
        0
    } else if end_line < line {
        line - end_line
    } else {
        start_line - line
    }
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
        ChunkIndexDoc, ChunkSearchRequest, LocalIndexStore, QueryFlavor, SymbolIndexDoc,
        SymbolSearchRequest,
    };
    use std::time::{SystemTime, UNIX_EPOCH};

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
        let store = LocalIndexStore::new(root.clone());
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
        let store = LocalIndexStore::new(root.clone());
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
    fn file_chunk_lookups_are_not_capped_by_top_docs_limit() {
        let root = temp_path("many-chunks");
        let store = LocalIndexStore::new(root.clone());
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

        let tail_line = docs.last().unwrap().start_line;
        let tail_chunk = store
            .chunk_for_file_line(repo, "src/generated.rs", tail_line)
            .unwrap()
            .expect("tail chunk should be found");
        assert_eq!(tail_chunk.start_line, tail_line);

        let _ = std::fs::remove_dir_all(root);
    }
}
