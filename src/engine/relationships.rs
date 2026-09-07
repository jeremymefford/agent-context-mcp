use anyhow::{Context, Result};
use rusqlite::{Connection, ErrorCode, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
#[cfg(test)]
use tree_sitter::Parser;
use tree_sitter::{Node, Tree};
use xxhash_rust::xxh3::xxh3_128;

use super::symbols::{IndexedSymbol, source_role_for_path};

pub const CONFIDENCE_EXACT_QUALIFIED: u64 = 1000;
pub const CONFIDENCE_IMPORTED_ALIAS: u64 = 950;
pub const CONFIDENCE_SAME_MODULE: u64 = 900;
pub const CONFIDENCE_UNIQUE_REPO: u64 = 750;
pub const CONFIDENCE_AMBIGUOUS: u64 = 450;
pub const CONFIDENCE_LEXICAL: u64 = 300;
const MAX_AMBIGUOUS_CANDIDATES: usize = 8;
const MAX_NAME_RESOLUTION_CANDIDATES: usize = 256;
const SQLITE_VACUUM_MIN_FREE_BYTES: u64 = 64 * 1024 * 1024;
const SQLITE_VACUUM_MIN_FREE_RATIO_PERCENT: u64 = 20;
const RELATION_SELECT: &str =
    "SELECT gr.reference_id, rp.repo, e.source_key, e.source_symbol_id, e.source_path,
            e.target_key, e.target_symbol_id, e.target_path,
            gr.target_name, gr.target_qualified_name,
            e.kind, e.confidence, e.resolution, gr.start_line, gr.end_line,
            gr.evidence, gr.source_role, e.language, e.file_hash
     FROM graph_edges_v5 e
     JOIN graph_repositories_v5 rp ON rp.id = e.repo_id
     JOIN graph_references_v5 gr ON gr.id = e.reference_id";
const RAW_REFERENCE_SELECT: &str =
    "SELECT gr.id, gr.source_key_id, gr.reference_id, rp.repo, sk.key, sk.symbol_id,
            fc.relative_path, gr.target_name, gr.target_qualified_name, gr.alias,
            gr.kind, gr.start_line, gr.end_line, gr.evidence, gr.source_role,
            fc.language, fc.file_hash
     FROM graph_references_v5 gr
     JOIN graph_repositories_v5 rp ON rp.id = gr.repo_id
     JOIN graph_keys_v5 sk ON sk.id = gr.source_key_id
     JOIN graph_file_coverage fc ON fc.rowid = gr.file_id";

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RelationKind {
    Calls,
    Imports,
    Reexports,
    TypeUses,
    ValueUses,
    Implements,
    Inherits,
}

impl RelationKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Calls => "calls",
            Self::Imports => "imports",
            Self::Reexports => "reexports",
            Self::TypeUses => "type_uses",
            Self::ValueUses => "value_uses",
            Self::Implements => "implements",
            Self::Inherits => "inherits",
        }
    }

    pub fn parse(value: &str) -> Self {
        match value {
            "imports" => Self::Imports,
            "reexports" => Self::Reexports,
            "type_uses" => Self::TypeUses,
            "value_uses" => Self::ValueUses,
            "implements" => Self::Implements,
            "inherits" => Self::Inherits,
            _ => Self::Calls,
        }
    }

    fn code(self) -> i64 {
        match self {
            Self::Calls => 0,
            Self::Imports => 1,
            Self::Reexports => 2,
            Self::TypeUses => 3,
            Self::Implements => 4,
            Self::Inherits => 5,
            Self::ValueUses => 6,
        }
    }

    fn from_code(value: i64) -> Self {
        match value {
            1 => Self::Imports,
            2 => Self::Reexports,
            3 => Self::TypeUses,
            4 => Self::Implements,
            5 => Self::Inherits,
            6 => Self::ValueUses,
            _ => Self::Calls,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RawReference {
    #[serde(skip)]
    pub storage_id: Option<i64>,
    #[serde(skip)]
    pub storage_source_key_id: Option<i64>,
    pub reference_id: String,
    pub repo: String,
    pub source_key: String,
    pub source_symbol_id: Option<String>,
    pub source_path: String,
    pub target_name: String,
    pub target_qualified_name: Option<String>,
    pub alias: Option<String>,
    pub kind: RelationKind,
    pub start_line: u64,
    pub end_line: u64,
    pub evidence: String,
    pub source_role: String,
    pub language: String,
    pub file_hash: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ResolvedRelation {
    pub reference_id: String,
    pub repo: String,
    pub source_key: String,
    pub source_symbol_id: Option<String>,
    pub source_path: String,
    pub target_key: Option<String>,
    pub target_symbol_id: Option<String>,
    pub target_path: Option<String>,
    pub target_name: String,
    pub target_qualified_name: Option<String>,
    pub kind: RelationKind,
    pub confidence: u64,
    pub resolution: String,
    pub start_line: u64,
    pub end_line: u64,
    pub evidence: String,
    pub source_role: String,
    pub language: String,
    pub file_hash: String,
}

#[derive(Debug, Clone)]
pub struct AnalysisRelation {
    pub source_key: String,
    pub source_path: String,
    pub target_key: Option<String>,
    pub target_name: String,
    pub kind: RelationKind,
    pub confidence: u64,
    pub resolution: String,
    pub start_line: u64,
    pub source_role: String,
    pub language: String,
}

#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FileRelationshipCoverage {
    pub supported: bool,
    pub language: String,
    pub definitions: u64,
    pub references: u64,
    pub unstable_identities: u64,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RepoRelationshipCoverage {
    pub repo: String,
    pub graph_status: String,
    pub root_hash: Option<String>,
    pub supported_files: u64,
    pub unsupported_files: u64,
    pub definitions: u64,
    pub references: u64,
    pub definite: u64,
    pub probable: u64,
    pub possible: u64,
    pub unresolved: u64,
    pub unstable_identities: u64,
    pub resolution_percentage: f64,
    pub stale_files: Vec<String>,
    pub unsupported_paths: Vec<String>,
    pub by_language: Vec<LanguageRelationshipCoverage>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LanguageRelationshipCoverage {
    pub language: String,
    pub files: u64,
    pub definitions: u64,
    pub references: u64,
    pub definite: u64,
    pub probable: u64,
    pub possible: u64,
    pub unresolved: u64,
    pub resolution_percentage: f64,
}

#[derive(Debug, Clone, Default)]
pub struct ExtractedRelationships {
    pub references: Vec<RawReference>,
    pub coverage: FileRelationshipCoverage,
}

#[derive(Debug, Clone)]
pub struct GraphFileReplacement {
    pub relative_path: String,
    pub references: Vec<RawReference>,
    pub coverage: FileRelationshipCoverage,
    pub file_hash: String,
}

#[derive(Clone)]
pub struct GraphStore {
    path: PathBuf,
    schema_initialized: Arc<SchemaInitialization>,
}

#[derive(Default)]
struct SchemaInitialization {
    ready: AtomicBool,
    lock: Mutex<()>,
}

impl GraphStore {
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            schema_initialized: Arc::new(SchemaInitialization::default()),
        }
    }

    #[cfg(test)]
    pub fn replace_file(
        &self,
        repo: &str,
        relative_path: &str,
        references: &[RawReference],
        coverage: &FileRelationshipCoverage,
        file_hash: &str,
    ) -> Result<()> {
        self.replace_files(
            repo,
            &[GraphFileReplacement {
                relative_path: relative_path.to_string(),
                references: references.to_vec(),
                coverage: coverage.clone(),
                file_hash: file_hash.to_string(),
            }],
        )
    }

    pub fn replace_files(&self, repo: &str, replacements: &[GraphFileReplacement]) -> Result<()> {
        if replacements.is_empty() {
            return Ok(());
        }
        let mut connection = self.open()?;
        let transaction = connection
            .transaction()
            .context("starting batched graph transaction")?;
        let repo_id = ensure_graph_repo(&transaction, repo)?;
        let mut delete_references =
            transaction.prepare_cached("DELETE FROM graph_references_v5 WHERE file_id = ?1")?;
        let mut delete_edges =
            transaction.prepare_cached("DELETE FROM graph_edges_v5 WHERE source_file_id = ?1")?;
        let mut insert_reference = transaction.prepare_cached(
            "INSERT INTO graph_references_v5 (
                repo_id, file_id, reference_id, source_key_id, target_name,
                target_qualified_name, alias, kind, start_line, end_line,
                evidence, source_role
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
        )?;
        let mut upsert_coverage = transaction.prepare_cached(
            "INSERT INTO graph_file_coverage (
                repo, relative_path, supported, definitions, reference_count,
                unstable_identities, file_hash, language
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
             ON CONFLICT(repo, relative_path) DO UPDATE SET
                supported = excluded.supported,
                definitions = excluded.definitions,
                reference_count = excluded.reference_count,
                unstable_identities = excluded.unstable_identities,
                file_hash = excluded.file_hash,
                language = excluded.language,
                definite_count = 0,
                probable_count = 0,
                possible_count = 0,
                unresolved_count = 0",
        )?;
        let mut key_ids = HashMap::<String, i64>::new();
        for replacement in replacements {
            upsert_coverage.execute(params![
                repo,
                replacement.relative_path,
                replacement.coverage.supported as i64,
                replacement.coverage.definitions as i64,
                replacement.coverage.references as i64,
                replacement.coverage.unstable_identities as i64,
                replacement.file_hash,
                replacement.coverage.language,
            ])?;
            let file_id = transaction.query_row(
                "SELECT rowid FROM graph_file_coverage WHERE repo = ?1 AND relative_path = ?2",
                params![repo, replacement.relative_path],
                |row| row.get::<_, i64>(0),
            )?;
            delete_edges.execute(params![file_id])?;
            delete_references.execute(params![file_id])?;
            for reference in &replacement.references {
                let source_key_id = if let Some(id) = key_ids.get(&reference.source_key) {
                    *id
                } else {
                    let id = ensure_graph_key(
                        &transaction,
                        repo_id,
                        &reference.source_key,
                        reference.source_symbol_id.as_deref(),
                        Some(&reference.source_path),
                        None,
                        None,
                    )?;
                    key_ids.insert(reference.source_key.clone(), id);
                    id
                };
                insert_reference.execute(params![
                    repo_id,
                    file_id,
                    reference.reference_id,
                    source_key_id,
                    reference.target_name,
                    reference.target_qualified_name,
                    reference.alias,
                    reference.kind.code(),
                    reference.start_line as i64,
                    reference.end_line as i64,
                    reference.evidence,
                    reference.source_role,
                ])?;
            }
        }
        drop(delete_references);
        drop(delete_edges);
        drop(insert_reference);
        drop(upsert_coverage);
        transaction
            .commit()
            .context("committing batched graph transaction")
    }

    pub fn delete_file(&self, repo: &str, relative_path: &str) -> Result<()> {
        let mut connection = self.open()?;
        let transaction = connection.transaction()?;
        if let Some(file_id) = graph_file_id(&transaction, repo, relative_path)? {
            transaction.execute(
                "DELETE FROM graph_edges_v5 WHERE source_file_id = ?1",
                params![file_id],
            )?;
            transaction.execute(
                "DELETE FROM graph_references_v5 WHERE file_id = ?1",
                params![file_id],
            )?;
        }
        transaction.execute(
            "DELETE FROM graph_file_coverage WHERE repo = ?1 AND relative_path = ?2",
            params![repo, relative_path],
        )?;
        transaction.commit()?;
        Ok(())
    }

    pub fn clear_repo(&self, repo: &str) -> Result<()> {
        let mut connection = self.open()?;
        let transaction = connection.transaction()?;
        if let Some(repo_id) = graph_repo_id(&transaction, repo)? {
            transaction.execute(
                "DELETE FROM graph_edges_v5 WHERE repo_id = ?1",
                params![repo_id],
            )?;
            transaction.execute(
                "DELETE FROM graph_references_v5 WHERE repo_id = ?1",
                params![repo_id],
            )?;
            transaction.execute(
                "DELETE FROM graph_keys_v5 WHERE repo_id = ?1",
                params![repo_id],
            )?;
        }
        transaction.execute(
            "DELETE FROM graph_file_coverage WHERE repo = ?1",
            params![repo],
        )?;
        transaction.execute(
            "DELETE FROM graph_coverage_cache WHERE repo = ?1",
            params![repo],
        )?;
        transaction.execute("DELETE FROM graph_state WHERE repo = ?1", params![repo])?;
        transaction.execute(
            "DELETE FROM graph_repositories_v5 WHERE repo = ?1",
            params![repo],
        )?;
        transaction.commit()?;
        Ok(())
    }

    pub fn set_state(&self, repo: &str, status: &str, root_hash: Option<&str>) -> Result<()> {
        let connection = self.open()?;
        let document_count = if status == "ready" {
            calculate_relation_document_count(&connection, repo)?
        } else {
            0
        };
        connection.execute(
            "INSERT INTO graph_state(
                repo, status, root_hash, graph_format, relationship_document_count
             ) VALUES (?1, ?2, ?3, 5, ?4)
             ON CONFLICT(repo) DO UPDATE SET
                status = excluded.status,
                root_hash = excluded.root_hash,
                graph_format = excluded.graph_format,
                relationship_document_count = excluded.relationship_document_count",
            params![repo, status, root_hash, document_count as i64],
        )?;
        Ok(())
    }

    #[cfg(test)]
    pub fn state(&self, repo: &str) -> Result<Option<(String, Option<String>)>> {
        let connection = self.open()?;
        Self::state_on(&connection, repo)
    }

    fn state_on(connection: &Connection, repo: &str) -> Result<Option<(String, Option<String>)>> {
        connection
            .query_row(
                "SELECT status, root_hash, graph_format FROM graph_state WHERE repo = ?1",
                params![repo],
                |row| {
                    let status = if row.get::<_, i64>(2)? == 5 {
                        row.get(0)?
                    } else {
                        "incompatible".to_string()
                    };
                    Ok((status, row.get(1)?))
                },
            )
            .optional()
            .context("loading graph state")
    }

    pub fn resolve_repo(&self, repo: &str) -> Result<()> {
        let mut connection = self.open()?;
        let symbols = load_symbols(&connection, repo)?;
        self.resolve_repo_with_symbols(&mut connection, repo, symbols, None)?;
        self.refresh_coverage_cache(repo)
    }

    pub fn resolve_repo_paths(&self, repo: &str, refresh_paths: &[String]) -> Result<Vec<String>> {
        if refresh_paths.is_empty() {
            self.resolve_repo(repo)?;
            return self.source_paths(repo);
        }
        let mut connection = self.open()?;
        let symbols = load_symbols(&connection, repo)?;
        let affected = affected_source_paths(&connection, repo, refresh_paths, &symbols)?;
        self.resolve_repo_with_symbols(&mut connection, repo, symbols, Some(&affected))?;
        self.refresh_coverage_cache(repo)?;
        Ok(affected.into_iter().collect())
    }

    pub fn resolve_repo_with_fallback(
        &self,
        repo: &str,
        fallback_repo: &str,
        suppressed_paths: &BTreeSet<String>,
    ) -> Result<()> {
        let mut connection = self.open()?;
        let mut symbols = load_symbols(&connection, fallback_repo)?
            .into_iter()
            .filter(|symbol| !suppressed_paths.contains(&symbol.relative_path))
            .collect::<Vec<_>>();
        let overlay_symbols = load_symbols(&connection, repo)?;
        let overlay_keys = overlay_symbols
            .iter()
            .map(|symbol| symbol.logical_key.clone())
            .collect::<BTreeSet<_>>();
        symbols.retain(|symbol| !overlay_keys.contains(&symbol.logical_key));
        symbols.extend(overlay_symbols);
        self.resolve_repo_with_symbols(&mut connection, repo, symbols, None)?;
        self.refresh_coverage_cache(repo)
    }

    fn resolve_repo_with_symbols(
        &self,
        connection: &mut Connection,
        repo: &str,
        symbols: Vec<IndexedSymbol>,
        source_paths: Option<&BTreeSet<String>>,
    ) -> Result<()> {
        let references = if let Some(source_paths) = source_paths {
            load_raw_references_for_paths(connection, repo, source_paths)?
        } else {
            load_raw_references(connection, repo)?
        };
        let aliases = load_import_aliases(connection, repo)?;
        let by_name = symbols_by_name(&symbols);
        let repo_id = ensure_graph_repo(connection, repo)?;
        let transaction = connection.transaction()?;
        if let Some(source_paths) = source_paths {
            let mut delete_edges = transaction
                .prepare_cached("DELETE FROM graph_edges_v5 WHERE source_file_id = ?1")?;
            for source_path in source_paths {
                if let Some(file_id) = graph_file_id(&transaction, repo, source_path)? {
                    delete_edges.execute(params![file_id])?;
                }
            }
        } else {
            transaction.execute(
                "DELETE FROM graph_edges_v5 WHERE repo_id = ?1",
                params![repo_id],
            )?;
        }
        let mut insert_edge = transaction.prepare_cached(
            "INSERT INTO graph_edges_v5 (
                repo_id, source_key_id, target_key_id, kind, reference_id,
                source_file_id, confidence, resolution, source_key, source_symbol_id,
                source_path, target_key, target_symbol_id, target_path, language, file_hash
             ) VALUES (
                ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8,
                ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16
             )
             ON CONFLICT(repo_id, source_key_id, target_key_id, kind) DO UPDATE SET
                reference_id = excluded.reference_id,
                source_file_id = excluded.source_file_id,
                confidence = excluded.confidence,
                resolution = excluded.resolution,
                source_key = excluded.source_key,
                source_symbol_id = excluded.source_symbol_id,
                source_path = excluded.source_path,
                target_key = excluded.target_key,
                target_symbol_id = excluded.target_symbol_id,
                target_path = excluded.target_path,
                language = excluded.language,
                file_hash = excluded.file_hash
             WHERE excluded.confidence > graph_edges_v5.confidence
                OR (excluded.confidence = graph_edges_v5.confidence
                    AND excluded.reference_id < graph_edges_v5.reference_id)",
        )?;
        let mut resolution_by_path = BTreeMap::<String, [u64; 4]>::new();
        let mut cached_source_path = "";
        let mut resolution_cache = HashMap::<
            (&str, Option<&str>, Option<&str>, RelationKind),
            Vec<ResolutionCandidate<'_>>,
        >::new();
        let mut target_key_ids = HashMap::<String, i64>::new();
        let mut source_file_ids = HashMap::<String, i64>::new();
        let mut mark_reference_resolved = transaction
            .prepare_cached("UPDATE graph_references_v5 SET resolved = ?1 WHERE id = ?2")?;
        for reference in &references {
            if cached_source_path != reference.source_path {
                resolution_cache.clear();
                cached_source_path = &reference.source_path;
            }
            let cache_key = (
                reference.target_name.as_str(),
                reference.target_qualified_name.as_deref(),
                reference.alias.as_deref(),
                reference.kind,
            );
            let candidates = resolution_cache
                .entry(cache_key)
                .or_insert_with(|| resolve_reference(reference, &by_name, &aliases))
                .clone();
            let best_confidence = candidates
                .iter()
                .map(|candidate| candidate.confidence)
                .max()
                .unwrap_or(0);
            let counts = resolution_by_path
                .entry(reference.source_path.clone())
                .or_default();
            match best_confidence {
                900.. => counts[0] += 1,
                650..=899 => counts[1] += 1,
                1..=649 => counts[2] += 1,
                _ => counts[3] += 1,
            }
            mark_reference_resolved.execute(params![
                (!candidates.is_empty()) as i64,
                reference
                    .storage_id
                    .context("raw reference missing storage id")?,
            ])?;
            let source_file_id = if let Some(id) = source_file_ids.get(&reference.source_path) {
                *id
            } else {
                let id = graph_file_id(&transaction, repo, &reference.source_path)?
                    .context("raw reference missing source file")?;
                source_file_ids.insert(reference.source_path.clone(), id);
                id
            };
            for candidate in candidates {
                let target_key_id =
                    if let Some(id) = target_key_ids.get(&candidate.symbol.logical_key) {
                        *id
                    } else {
                        let id = ensure_graph_key(
                            &transaction,
                            repo_id,
                            &candidate.symbol.logical_key,
                            Some(&candidate.symbol.symbol_id),
                            Some(&candidate.symbol.relative_path),
                            Some(&candidate.symbol.name),
                            Some(&candidate.symbol.qualified_name),
                        )?;
                        target_key_ids.insert(candidate.symbol.logical_key.clone(), id);
                        id
                    };
                insert_edge.execute(params![
                    repo_id,
                    reference
                        .storage_source_key_id
                        .context("raw reference missing source key id")?,
                    target_key_id,
                    reference.kind.code(),
                    reference
                        .storage_id
                        .context("raw reference missing storage id")?,
                    source_file_id,
                    candidate.confidence as i64,
                    resolution_code(candidate.resolution),
                    reference.source_key,
                    reference.source_symbol_id,
                    reference.source_path,
                    candidate.symbol.logical_key,
                    candidate.symbol.symbol_id,
                    candidate.symbol.relative_path,
                    reference.language,
                    reference.file_hash,
                ])?;
            }
        }
        drop(insert_edge);
        drop(mark_reference_resolved);
        let mut update_coverage = transaction.prepare_cached(
            "UPDATE graph_file_coverage SET
                definite_count = ?1, probable_count = ?2,
                possible_count = ?3, unresolved_count = ?4
             WHERE repo = ?5 AND relative_path = ?6",
        )?;
        for (source_path, counts) in resolution_by_path {
            update_coverage.execute(params![
                counts[0] as i64,
                counts[1] as i64,
                counts[2] as i64,
                counts[3] as i64,
                repo,
                source_path,
            ])?;
        }
        drop(update_coverage);
        transaction.commit()?;
        Ok(())
    }

    pub fn file_hashes(&self, repo: &str) -> Result<BTreeMap<String, String>> {
        let connection = self.open()?;
        let mut statement = connection
            .prepare("SELECT relative_path, file_hash FROM graph_file_coverage WHERE repo = ?1")?;
        let rows = statement.query_map(params![repo], |row| Ok((row.get(0)?, row.get(1)?)))?;
        rows.collect::<rusqlite::Result<BTreeMap<_, _>>>()
            .context("loading graph file hashes")
    }

    pub fn source_paths(&self, repo: &str) -> Result<Vec<String>> {
        Ok(self.file_hashes(repo)?.into_keys().collect())
    }

    pub fn compact_storage(&self) -> Result<()> {
        let connection = self.open()?;
        connection
            .execute_batch(
                "DROP INDEX IF EXISTS idx_graph_refs_source;
                 DROP TABLE IF EXISTS graph_edges;
                 DROP TABLE IF EXISTS graph_references;
                 DELETE FROM graph_keys_v5
                  WHERE id NOT IN (SELECT source_key_id FROM graph_references_v5)
                    AND id NOT IN (SELECT target_key_id FROM graph_edges_v5);
                 DELETE FROM graph_repositories_v5
                  WHERE id NOT IN (SELECT repo_id FROM graph_keys_v5)
                    AND id NOT IN (SELECT repo_id FROM graph_references_v5)
                    AND id NOT IN (SELECT repo_id FROM graph_edges_v5);
                 PRAGMA wal_checkpoint(TRUNCATE);
                 VACUUM;",
            )
            .context("compacting relationship storage")?;
        Ok(())
    }

    /// Performs bounded maintenance after an incremental graph generation.
    ///
    /// Orphan keys are cheap historical residue from renamed, moved, or deleted
    /// declarations. SQLite normally reuses their pages. A full VACUUM is only
    /// worthwhile after a material contraction, so it is gated by both an
    /// absolute byte threshold and a free-page ratio.
    pub fn maintain_incremental_storage(&self, repo: &str, graph_changed: bool) -> Result<()> {
        let mut connection = self.open()?;
        if graph_changed && let Some(repo_id) = graph_repo_id(&connection, repo)? {
            let transaction = connection
                .transaction()
                .context("starting incremental graph maintenance")?;
            transaction.execute(
                "WITH live_keys(id) AS (
                    SELECT source_key_id FROM graph_references_v5 WHERE repo_id = ?1
                    UNION
                    SELECT source_key_id FROM graph_edges_v5 WHERE repo_id = ?1
                    UNION
                    SELECT target_key_id FROM graph_edges_v5 WHERE repo_id = ?1
                 )
                 DELETE FROM graph_keys_v5
                  WHERE repo_id = ?1
                    AND id NOT IN (SELECT id FROM live_keys)",
                params![repo_id],
            )?;
            transaction
                .commit()
                .context("committing incremental graph maintenance")?;
        }

        connection
            .execute_batch("PRAGMA optimize;")
            .context("optimizing relationship query plans")?;
        let _checkpoint = connection
            .query_row("PRAGMA wal_checkpoint(PASSIVE)", [], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                ))
            })
            .context("checkpointing relationship database")?;

        let page_size = pragma_u64(&connection, "PRAGMA page_size")?;
        let page_count = pragma_u64(&connection, "PRAGMA page_count")?;
        let free_pages = pragma_u64(&connection, "PRAGMA freelist_count")?;
        if sqlite_storage_needs_vacuum(page_size, page_count, free_pages) {
            match connection.execute_batch("VACUUM; PRAGMA wal_checkpoint(TRUNCATE);") {
                Ok(()) => {}
                Err(error) if sqlite_maintenance_is_busy(&error) => {
                    // An active reader can defer physical reclamation safely. The
                    // unchanged freelist causes the next incremental run to retry.
                }
                Err(error) => return Err(error).context("vacuuming relationship database"),
            }
        }
        Ok(())
    }

    pub fn relation_document_count(&self, repo: &str) -> Result<u64> {
        let connection = self.open()?;
        connection
            .query_row(
                "SELECT relationship_document_count FROM graph_state WHERE repo = ?1",
                params![repo],
                |row| Ok(row.get::<_, i64>(0)? as u64),
            )
            .optional()
            .map(|count| count.unwrap_or(0))
            .context("counting canonical relationship documents")
    }

    pub fn all_relations(&self, repo: &str) -> Result<Vec<ResolvedRelation>> {
        let connection = self.open()?;
        let sql = format!(
            "{RELATION_SELECT} WHERE rp.repo = ?1
             ORDER BY e.source_path, gr.start_line, gr.reference_id, e.confidence DESC"
        );
        let mut statement = connection.prepare(&sql)?;
        let rows = statement.query_map(params![repo], map_relation_row)?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .context("collecting graph relations")
    }

    pub fn all_relation_documents(&self, repo: &str) -> Result<Vec<ResolvedRelation>> {
        let mut documents = self.all_relations(repo)?;
        documents.extend(self.unresolved_relation_documents(repo, None)?);
        Ok(documents)
    }

    pub fn all_analysis_relations(&self, repo: &str) -> Result<Vec<AnalysisRelation>> {
        let connection = self.open()?;
        let mut documents = Vec::new();
        let resolved_sql = "SELECT e.source_key, e.source_path, e.target_key, gr.target_name,
                    e.kind, e.confidence, e.resolution, gr.start_line, gr.source_role, e.language
             FROM graph_edges_v5 e
             JOIN graph_repositories_v5 rp ON rp.id = e.repo_id
             JOIN graph_references_v5 gr ON gr.id = e.reference_id
             WHERE rp.repo = ?1
             ORDER BY e.source_path, gr.start_line, gr.reference_id, e.confidence DESC";
        let mut resolved = connection.prepare(resolved_sql)?;
        let rows = resolved.query_map(params![repo], |row| {
            Ok(AnalysisRelation {
                source_key: row.get(0)?,
                source_path: row.get(1)?,
                target_key: row.get(2)?,
                target_name: row.get(3)?,
                kind: RelationKind::from_code(row.get(4)?),
                confidence: row.get::<_, i64>(5)? as u64,
                resolution: resolution_from_code(row.get(6)?).to_string(),
                start_line: row.get::<_, i64>(7)? as u64,
                source_role: row.get(8)?,
                language: row.get(9)?,
            })
        })?;
        documents.extend(rows.collect::<rusqlite::Result<Vec<_>>>()?);

        let unresolved_sql =
            "SELECT sk.key, fc.relative_path, gr.target_name, gr.target_qualified_name,
                    gr.kind, gr.start_line, gr.source_role, fc.language
             FROM graph_references_v5 gr
             JOIN graph_repositories_v5 rp ON rp.id = gr.repo_id
             JOIN graph_keys_v5 sk ON sk.id = gr.source_key_id
             JOIN graph_file_coverage fc ON fc.rowid = gr.file_id
             WHERE rp.repo = ?1 AND gr.resolved = 0
             ORDER BY fc.relative_path, gr.start_line, gr.reference_id";
        let mut unresolved = connection.prepare(unresolved_sql)?;
        let rows = unresolved.query_map(params![repo], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, Option<String>>(3)?,
                RelationKind::from_code(row.get(4)?),
                row.get::<_, i64>(5)? as u64,
                row.get::<_, String>(6)?,
                row.get::<_, String>(7)?,
            ))
        })?;
        let mut seen = BTreeSet::new();
        for row in rows {
            let (
                source_key,
                source_path,
                target_name,
                target_qualified_name,
                kind,
                start_line,
                source_role,
                language,
            ) = row?;
            if !seen.insert((
                source_key.clone(),
                target_name.clone(),
                target_qualified_name,
                kind.as_str(),
            )) {
                continue;
            }
            documents.push(AnalysisRelation {
                source_key,
                source_path,
                target_key: None,
                target_name,
                kind,
                confidence: 0,
                resolution: "unresolved".to_string(),
                start_line,
                source_role,
                language,
            });
        }
        Ok(documents)
    }

    pub fn relations_for_source_paths(
        &self,
        repo: &str,
        source_paths: &[String],
    ) -> Result<Vec<ResolvedRelation>> {
        if source_paths.is_empty() {
            return Ok(Vec::new());
        }
        let connection = self.open()?;
        let sql = format!(
            "{RELATION_SELECT} WHERE e.source_file_id = ?1
             ORDER BY gr.start_line, gr.reference_id, e.confidence DESC"
        );
        let mut statement = connection.prepare(&sql)?;
        let mut output = Vec::new();
        for source_path in source_paths {
            let Some(file_id) = graph_file_id(&connection, repo, source_path)? else {
                continue;
            };
            let rows = statement.query_map(params![file_id], map_relation_row)?;
            for row in rows {
                output.push(row?);
            }
        }
        Ok(output)
    }

    pub fn relation_documents_for_source_paths(
        &self,
        repo: &str,
        source_paths: &[String],
    ) -> Result<Vec<ResolvedRelation>> {
        let mut documents = self.relations_for_source_paths(repo, source_paths)?;
        documents.extend(self.unresolved_relation_documents(repo, Some(source_paths))?);
        Ok(documents)
    }

    fn unresolved_relation_documents(
        &self,
        repo: &str,
        source_paths: Option<&[String]>,
    ) -> Result<Vec<ResolvedRelation>> {
        let connection = self.open()?;
        let sql = format!(
            "{RAW_REFERENCE_SELECT} WHERE rp.repo = ?1 AND gr.resolved = 0{}\n\
             ORDER BY fc.relative_path, gr.start_line, gr.reference_id",
            if source_paths.is_some() {
                " AND fc.relative_path = ?2"
            } else {
                ""
            }
        );
        let mut statement = connection.prepare(&sql)?;
        let mut documents = Vec::new();
        let mut seen = BTreeSet::new();
        let mut append = |reference: RawReference| {
            let identity = (
                reference.source_key.clone(),
                reference.target_name.clone(),
                reference.target_qualified_name.clone(),
                reference.kind.as_str(),
            );
            if !seen.insert(identity) {
                return;
            }
            documents.push(ResolvedRelation {
                reference_id: reference.reference_id,
                repo: reference.repo,
                source_key: reference.source_key,
                source_symbol_id: reference.source_symbol_id,
                source_path: reference.source_path,
                target_key: None,
                target_symbol_id: None,
                target_path: None,
                target_name: reference.target_name,
                target_qualified_name: reference.target_qualified_name,
                kind: reference.kind,
                confidence: 0,
                resolution: "unresolved".to_string(),
                start_line: reference.start_line,
                end_line: reference.end_line,
                evidence: reference.evidence,
                source_role: reference.source_role,
                language: reference.language,
                file_hash: reference.file_hash,
            });
        };
        if let Some(source_paths) = source_paths {
            for source_path in source_paths {
                let rows =
                    statement.query_map(params![repo, source_path], map_raw_reference_row)?;
                for row in rows {
                    append(row?);
                }
            }
        } else {
            let rows = statement.query_map(params![repo], map_raw_reference_row)?;
            for row in rows {
                append(row?);
            }
        }
        Ok(documents)
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn relations_to(
        &self,
        repo: &str,
        target_keys: &[String],
        min_confidence: u64,
        limit: usize,
    ) -> Result<Vec<ResolvedRelation>> {
        self.relations_for_keys(repo, target_keys, min_confidence, limit, true)
    }

    #[cfg_attr(not(test), allow(dead_code))]
    fn relations_for_keys(
        &self,
        repo: &str,
        keys: &[String],
        min_confidence: u64,
        limit: usize,
        reverse: bool,
    ) -> Result<Vec<ResolvedRelation>> {
        if keys.is_empty() {
            return Ok(Vec::new());
        }
        let connection = self.open()?;
        let Some(repo_id) = graph_repo_id(&connection, repo)? else {
            return Ok(Vec::new());
        };
        let key_column = if reverse {
            "e.target_key_id"
        } else {
            "e.source_key_id"
        };
        let sql = format!(
            "{RELATION_SELECT}
             WHERE e.repo_id = ?1 AND {key_column} = ?2 AND e.confidence >= ?3
             ORDER BY e.confidence DESC, e.source_path, gr.start_line, gr.reference_id
             LIMIT ?4"
        );
        let mut lookup_key = connection
            .prepare_cached("SELECT id FROM graph_keys_v5 WHERE repo_id = ?1 AND key = ?2")?;
        let mut statement = connection.prepare(&sql)?;
        let mut output = Vec::new();
        let mut seen = BTreeSet::new();
        for key in keys {
            let Some(key_id) = lookup_key
                .query_row(params![repo_id, key], |row| row.get::<_, i64>(0))
                .optional()?
            else {
                continue;
            };
            let rows = statement.query_map(
                params![repo_id, key_id, min_confidence as i64, limit.max(1) as i64],
                map_relation_row,
            )?;
            for row in rows {
                let relation = row?;
                let identity = (
                    relation.source_key.clone(),
                    relation.target_key.clone(),
                    relation.kind.as_str(),
                );
                if seen.insert(identity) {
                    output.push(relation);
                }
            }
        }
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

    pub fn coverage(&self, repo: &str) -> Result<RepoRelationshipCoverage> {
        let connection = self.open()?;
        let (graph_status, root_hash) =
            Self::state_on(&connection, repo)?.unwrap_or_else(|| ("missing".to_string(), None));
        let mut coverage = RepoRelationshipCoverage {
            repo: repo.to_string(),
            graph_status,
            root_hash,
            ..RepoRelationshipCoverage::default()
        };
        connection.query_row(
            "SELECT
                COALESCE(SUM(CASE WHEN supported = 1 THEN 1 ELSE 0 END), 0),
                COALESCE(SUM(CASE WHEN supported = 0 THEN 1 ELSE 0 END), 0),
                COALESCE(SUM(definitions), 0), COALESCE(SUM(reference_count), 0),
                COALESCE(SUM(unstable_identities), 0)
             FROM graph_file_coverage WHERE repo = ?1",
            params![repo],
            |row| {
                coverage.supported_files = row.get::<_, i64>(0)? as u64;
                coverage.unsupported_files = row.get::<_, i64>(1)? as u64;
                coverage.definitions = row.get::<_, i64>(2)? as u64;
                coverage.references = row.get::<_, i64>(3)? as u64;
                coverage.unstable_identities = row.get::<_, i64>(4)? as u64;
                Ok(())
            },
        )?;
        let mut unsupported = connection.prepare(
            "SELECT relative_path FROM graph_file_coverage
             WHERE repo = ?1 AND supported = 0 ORDER BY relative_path",
        )?;
        coverage.unsupported_paths = unsupported
            .query_map(params![repo], |row| row.get(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?;

        let mut by_language = connection.prepare(
            "SELECT language, COUNT(*), COALESCE(SUM(definitions), 0),
                    COALESCE(SUM(reference_count), 0),
                    COALESCE(SUM(definite_count), 0),
                    COALESCE(SUM(probable_count), 0),
                    COALESCE(SUM(possible_count), 0),
                    COALESCE(SUM(unresolved_count), 0)
             FROM graph_file_coverage WHERE repo = ?1
             GROUP BY language ORDER BY language",
        )?;
        let rows = by_language.query_map(params![repo], |row| {
            let language = row.get::<_, String>(0)?;
            let files = row.get::<_, i64>(1)? as u64;
            let definitions = row.get::<_, i64>(2)? as u64;
            let references = row.get::<_, i64>(3)? as u64;
            let definite = row.get::<_, i64>(4)? as u64;
            let probable = row.get::<_, i64>(5)? as u64;
            Ok(LanguageRelationshipCoverage {
                language,
                files,
                definitions,
                references,
                definite,
                probable,
                possible: row.get::<_, i64>(6)? as u64,
                unresolved: row.get::<_, i64>(7)? as u64,
                resolution_percentage: resolution_percentage(definite + probable, references),
            })
        })?;
        coverage.by_language = rows.collect::<rusqlite::Result<Vec<_>>>()?;
        for language in &coverage.by_language {
            coverage.definite += language.definite;
            coverage.probable += language.probable;
            coverage.possible += language.possible;
            coverage.unresolved += language.unresolved;
        }
        coverage.resolution_percentage =
            resolution_percentage(coverage.definite + coverage.probable, coverage.references);
        Ok(coverage)
    }

    pub fn coverage_cached(&self, repo: &str) -> Result<RepoRelationshipCoverage> {
        let connection = self.open()?;
        let cached = connection
            .query_row(
                "SELECT summary_json FROM graph_coverage_cache WHERE repo = ?1",
                params![repo],
                |row| row.get::<_, String>(0),
            )
            .optional()?;
        let mut coverage: RepoRelationshipCoverage = if let Some(summary) = cached {
            serde_json::from_str(&summary).context("decoding cached graph coverage")?
        } else {
            return self.coverage(repo);
        };
        let (status, root_hash) =
            Self::state_on(&connection, repo)?.unwrap_or_else(|| ("missing".to_string(), None));
        coverage.graph_status = status;
        coverage.root_hash = root_hash;
        Ok(coverage)
    }

    fn refresh_coverage_cache(&self, repo: &str) -> Result<()> {
        let coverage = self.coverage(repo)?;
        let summary = serde_json::to_string(&coverage).context("encoding graph coverage")?;
        let connection = self.open()?;
        connection.execute(
            "INSERT INTO graph_coverage_cache(repo, summary_json)
             VALUES (?1, ?2)
             ON CONFLICT(repo) DO UPDATE SET summary_json = excluded.summary_json",
            params![repo, summary],
        )?;
        Ok(())
    }

    fn open(&self) -> Result<Connection> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let connection = Connection::open(&self.path)?;
        if self.schema_initialized.ready.load(Ordering::Acquire) {
            connection.execute_batch("PRAGMA synchronous = NORMAL; PRAGMA foreign_keys = ON;")?;
            return Ok(connection);
        }
        let _guard = self
            .schema_initialized
            .lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if self.schema_initialized.ready.load(Ordering::Acquire) {
            connection.execute_batch("PRAGMA synchronous = NORMAL; PRAGMA foreign_keys = ON;")?;
            return Ok(connection);
        }
        connection.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA synchronous = NORMAL;
             PRAGMA foreign_keys = ON;
             CREATE TABLE IF NOT EXISTS graph_repositories_v5 (
                id INTEGER PRIMARY KEY, repo TEXT NOT NULL UNIQUE
             );
             CREATE TABLE IF NOT EXISTS graph_keys_v5 (
                id INTEGER PRIMARY KEY, repo_id INTEGER NOT NULL, key TEXT NOT NULL,
                symbol_id TEXT, path TEXT, name TEXT, qualified_name TEXT,
                UNIQUE(repo_id, key)
             );
             CREATE TABLE IF NOT EXISTS graph_references_v5 (
                id INTEGER PRIMARY KEY, repo_id INTEGER NOT NULL, file_id INTEGER NOT NULL,
                reference_id TEXT NOT NULL, source_key_id INTEGER NOT NULL,
                target_name TEXT NOT NULL, target_qualified_name TEXT, alias TEXT,
                kind INTEGER NOT NULL, start_line INTEGER NOT NULL, end_line INTEGER NOT NULL,
                evidence TEXT NOT NULL, source_role TEXT NOT NULL,
                resolved INTEGER NOT NULL DEFAULT 0
             );
             CREATE TABLE IF NOT EXISTS graph_edges_v5 (
                repo_id INTEGER NOT NULL, source_key_id INTEGER NOT NULL,
                target_key_id INTEGER NOT NULL, kind INTEGER NOT NULL,
                reference_id INTEGER NOT NULL, source_file_id INTEGER NOT NULL,
                confidence INTEGER NOT NULL, resolution INTEGER NOT NULL,
                source_key TEXT NOT NULL, source_symbol_id TEXT, source_path TEXT NOT NULL,
                target_key TEXT NOT NULL, target_symbol_id TEXT NOT NULL,
                target_path TEXT NOT NULL, language TEXT NOT NULL, file_hash TEXT NOT NULL,
                PRIMARY KEY(repo_id, source_key_id, target_key_id, kind)
             ) WITHOUT ROWID;
             CREATE TABLE IF NOT EXISTS graph_file_coverage (
                repo TEXT NOT NULL, relative_path TEXT NOT NULL, supported INTEGER NOT NULL,
                definitions INTEGER NOT NULL, reference_count INTEGER NOT NULL,
                unstable_identities INTEGER NOT NULL, file_hash TEXT NOT NULL,
                language TEXT NOT NULL DEFAULT 'unknown',
                definite_count INTEGER NOT NULL DEFAULT 0,
                probable_count INTEGER NOT NULL DEFAULT 0,
                possible_count INTEGER NOT NULL DEFAULT 0,
                unresolved_count INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(repo, relative_path)
             );
             CREATE TABLE IF NOT EXISTS graph_state (
                repo TEXT PRIMARY KEY, status TEXT NOT NULL, root_hash TEXT,
                graph_format INTEGER NOT NULL DEFAULT 1,
                relationship_document_count INTEGER NOT NULL DEFAULT 0
             );
             CREATE TABLE IF NOT EXISTS graph_coverage_cache (
                repo TEXT PRIMARY KEY, summary_json TEXT NOT NULL
             );
             CREATE INDEX IF NOT EXISTS idx_graph_refs_v5_file
                ON graph_references_v5(file_id);
             CREATE INDEX IF NOT EXISTS idx_graph_refs_v5_name
                ON graph_references_v5(repo_id, target_name, kind, file_id);
             CREATE INDEX IF NOT EXISTS idx_graph_edges_v5_source
                ON graph_edges_v5(repo_id, source_key_id, confidence, kind);
             CREATE INDEX IF NOT EXISTS idx_graph_edges_v5_target
                ON graph_edges_v5(repo_id, target_key_id, confidence, kind);
             CREATE INDEX IF NOT EXISTS idx_graph_edges_v5_file
                ON graph_edges_v5(source_file_id);",
        )?;
        if !table_has_column(&connection, "graph_file_coverage", "language")? {
            connection.execute(
                "ALTER TABLE graph_file_coverage ADD COLUMN language TEXT NOT NULL DEFAULT 'unknown'",
                [],
            )?;
        }
        if !table_has_column(&connection, "graph_state", "graph_format")? {
            connection.execute(
                "ALTER TABLE graph_state ADD COLUMN graph_format INTEGER NOT NULL DEFAULT 1",
                [],
            )?;
        }
        if !table_has_column(&connection, "graph_state", "relationship_document_count")? {
            connection.execute(
                "ALTER TABLE graph_state ADD COLUMN relationship_document_count INTEGER NOT NULL DEFAULT 0",
                [],
            )?;
        }
        if !table_has_column(&connection, "graph_references_v5", "resolved")? {
            connection.execute(
                "ALTER TABLE graph_references_v5 ADD COLUMN resolved INTEGER NOT NULL DEFAULT 0",
                [],
            )?;
        }
        for column in [
            "definite_count",
            "probable_count",
            "possible_count",
            "unresolved_count",
        ] {
            if !table_has_column(&connection, "graph_file_coverage", column)? {
                connection.execute(
                    &format!(
                        "ALTER TABLE graph_file_coverage ADD COLUMN {column} INTEGER NOT NULL DEFAULT 0"
                    ),
                    [],
                )?;
            }
        }
        self.schema_initialized.ready.store(true, Ordering::Release);
        Ok(connection)
    }
}

#[cfg(test)]
pub fn extract_relationships(
    repo: &str,
    relative_path: &str,
    path: &Path,
    text: &str,
    symbols: &[IndexedSymbol],
    file_hash: &str,
) -> Result<ExtractedRelationships> {
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or_default();
    let grammar = match extension {
        "rs" => tree_sitter_rust::LANGUAGE.into(),
        "ts" | "mts" | "cts" => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
        "tsx" => tree_sitter_typescript::LANGUAGE_TSX.into(),
        _ => {
            return Ok(ExtractedRelationships {
                references: Vec::new(),
                coverage: FileRelationshipCoverage {
                    supported: false,
                    language: "unsupported".to_string(),
                    definitions: symbols.len() as u64,
                    unstable_identities: symbols
                        .iter()
                        .filter(|symbol| !symbol.identity_stable)
                        .count() as u64,
                    ..FileRelationshipCoverage::default()
                },
            });
        }
    };
    let mut parser = Parser::new();
    parser.set_language(&grammar)?;
    let Some(tree) = parser.parse(text, None) else {
        return Ok(ExtractedRelationships::default());
    };
    extract_relationships_from_tree(
        repo,
        relative_path,
        path,
        text,
        symbols,
        file_hash,
        Some(&tree),
    )
}

pub fn extract_relationships_from_tree(
    repo: &str,
    relative_path: &str,
    path: &Path,
    text: &str,
    symbols: &[IndexedSymbol],
    file_hash: &str,
    tree: Option<&Tree>,
) -> Result<ExtractedRelationships> {
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or_default();
    let language = match extension {
        "rs" => "rust",
        "ts" | "mts" | "cts" | "tsx" => "typescript",
        _ => {
            return Ok(ExtractedRelationships {
                references: Vec::new(),
                coverage: FileRelationshipCoverage {
                    supported: false,
                    language: "unsupported".to_string(),
                    definitions: symbols.len() as u64,
                    unstable_identities: symbols
                        .iter()
                        .filter(|symbol| !symbol.identity_stable)
                        .count() as u64,
                    ..FileRelationshipCoverage::default()
                },
            });
        }
    };
    let Some(tree) = tree else {
        return Ok(ExtractedRelationships::default());
    };
    let mut references = Vec::new();
    let context = RelationshipExtractionContext {
        repo,
        path: relative_path,
        text: text.as_bytes(),
        language,
        symbols,
        file_hash,
    };
    collect_references(tree.root_node(), &context, &mut references);
    dedupe_references(&mut references);
    Ok(ExtractedRelationships {
        coverage: FileRelationshipCoverage {
            supported: true,
            language: language.to_string(),
            definitions: symbols.len() as u64,
            references: references.len() as u64,
            unstable_identities: symbols
                .iter()
                .filter(|symbol| !symbol.identity_stable)
                .count() as u64,
        },
        references,
    })
}

struct RelationshipExtractionContext<'a> {
    repo: &'a str,
    path: &'a str,
    text: &'a [u8],
    language: &'a str,
    symbols: &'a [IndexedSymbol],
    file_hash: &'a str,
}

fn collect_references(
    node: Node<'_>,
    context: &RelationshipExtractionContext<'_>,
    output: &mut Vec<RawReference>,
) {
    for (kind, target, alias) in references_from_node(node, context.text, context.language) {
        let target = normalize_target(&target);
        if !target.is_empty() {
            let start_line = node.start_position().row as u64 + 1;
            let end_line = node.end_position().row as u64 + 1;
            let owner = narrowest_owner(context.symbols, start_line);
            let source_key = owner
                .map(|symbol| symbol.logical_key.clone())
                .unwrap_or_else(|| file_logical_key(context.language, context.path));
            let evidence = node
                .utf8_text(context.text)
                .unwrap_or_default()
                .trim()
                .to_string();
            let target_name = terminal_name(&target);
            let digest = xxh3_128(
                format!(
                    "{}:{start_line}:{end_line}:{}:{target}:{alias:?}",
                    context.path,
                    kind.as_str(),
                )
                .as_bytes(),
            );
            output.push(RawReference {
                storage_id: None,
                storage_source_key_id: None,
                reference_id: format!("ref_{digest:032x}"),
                repo: context.repo.to_string(),
                source_key,
                source_symbol_id: owner.map(|symbol| symbol.symbol_id.clone()),
                source_path: context.path.to_string(),
                target_name,
                target_qualified_name: target.contains([':', '.', '/']).then_some(target),
                alias,
                kind,
                start_line,
                end_line,
                evidence,
                source_role: owner
                    .map(|symbol| symbol.source_role.clone())
                    .unwrap_or_else(|| source_role_for_path(context.path).to_string()),
                language: context.language.to_string(),
                file_hash: context.file_hash.to_string(),
            });
        }
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor).filter(|child| child.is_named()) {
        collect_references(child, context, output);
    }
}

fn references_from_node(
    node: Node<'_>,
    text: &[u8],
    language: &str,
) -> Vec<(RelationKind, String, Option<String>)> {
    let raw = || node.utf8_text(text).ok().map(str::to_string);
    match (language, node.kind()) {
        ("rust", "use_declaration") => {
            let Some(value) = raw() else {
                return Vec::new();
            };
            let body = value
                .trim()
                .trim_start_matches("pub ")
                .trim_start_matches("use ")
                .trim_end_matches(';');
            let kind = if value.trim_start().starts_with("pub use") {
                RelationKind::Reexports
            } else {
                RelationKind::Imports
            };
            expand_rust_use(body)
                .into_iter()
                .map(|(target, alias)| (kind, target, alias))
                .collect()
        }
        ("rust", "identifier" | "scoped_identifier") => {
            if let Some(value) = rust_function_value_argument(node, text) {
                vec![(RelationKind::Calls, value, None)]
            } else {
                rust_value_use(node, text)
                    .map(|value| vec![(RelationKind::ValueUses, value, None)])
                    .unwrap_or_default()
            }
        }
        ("rust", "call_expression") => node
            .child_by_field_name("function")
            .and_then(|child| child.utf8_text(text).ok())
            .map(|value| vec![(RelationKind::Calls, value.to_string(), None)])
            .unwrap_or_default(),
        ("rust", "method_call_expression") => node
            .child_by_field_name("method")
            .and_then(|child| child.utf8_text(text).ok())
            .map(|value| vec![(RelationKind::Calls, value.to_string(), None)])
            .unwrap_or_default(),
        ("rust", "struct_expression") => node
            .child_by_field_name("name")
            .or_else(|| node.child_by_field_name("type"))
            .and_then(|child| child.utf8_text(text).ok())
            .map(|value| vec![(RelationKind::Calls, value.to_string(), None)])
            .unwrap_or_default(),
        ("rust", "macro_invocation") => node
            .child_by_field_name("macro")
            .and_then(|child| child.utf8_text(text).ok())
            .map(|value| vec![(RelationKind::Calls, value.to_string(), None)])
            .unwrap_or_default(),
        ("rust", "macro_definition") => raw()
            .map(|value| {
                rust_macro_call_targets(&value)
                    .into_iter()
                    .map(|target| (RelationKind::Calls, target, None))
                    .collect()
            })
            .unwrap_or_default(),
        ("rust", "impl_item") => raw()
            .and_then(|value| parse_rust_impl(&value))
            .into_iter()
            .collect(),
        ("rust", "trait_item") => raw()
            .map(|value| parse_rust_trait_inheritance(&value))
            .unwrap_or_default(),
        ("rust", "type_identifier") if !is_definition_name(node) => raw()
            .map(|value| vec![(RelationKind::TypeUses, value, None)])
            .unwrap_or_default(),
        ("typescript", "import_statement") => raw()
            .map(|value| parse_typescript_import(&value))
            .unwrap_or_default(),
        ("typescript", "export_statement") => raw()
            .filter(|value| value.contains(" from "))
            .map(|value| parse_typescript_reexport(&value))
            .unwrap_or_default(),
        ("typescript", "lexical_declaration" | "variable_declaration") => raw()
            .and_then(|value| parse_commonjs_import(&value))
            .into_iter()
            .collect(),
        ("typescript", "call_expression") => node
            .child_by_field_name("function")
            .and_then(|child| child.utf8_text(text).ok())
            .map(|value| vec![(RelationKind::Calls, value.to_string(), None)])
            .unwrap_or_default(),
        ("typescript", "new_expression") => node
            .child_by_field_name("constructor")
            .or_else(|| node.named_child(0))
            .and_then(|child| child.utf8_text(text).ok())
            .map(|value| vec![(RelationKind::Calls, value.to_string(), None)])
            .unwrap_or_default(),
        ("typescript", "jsx_opening_element" | "jsx_self_closing_element") => node
            .child_by_field_name("name")
            .or_else(|| node.named_child(0))
            .and_then(|child| child.utf8_text(text).ok())
            .map(|value| vec![(RelationKind::Calls, value.to_string(), None)])
            .unwrap_or_default(),
        ("typescript", "type_identifier") if !is_definition_name(node) => raw()
            .map(|value| vec![(RelationKind::TypeUses, value, None)])
            .unwrap_or_default(),
        ("typescript", "extends_clause") => raw()
            .map(|value| {
                split_relationship_list(
                    RelationKind::Inherits,
                    value.trim_start_matches("extends "),
                )
            })
            .unwrap_or_default(),
        ("typescript", "implements_clause") => raw()
            .map(|value| {
                split_relationship_list(
                    RelationKind::Implements,
                    value.trim_start_matches("implements "),
                )
            })
            .unwrap_or_default(),
        _ => Vec::new(),
    }
}

fn rust_function_value_argument(node: Node<'_>, text: &[u8]) -> Option<String> {
    let arguments = node
        .parent()
        .filter(|parent| parent.kind() == "arguments")?;
    let invocation = arguments.parent()?;
    let callable = match invocation.kind() {
        "call_expression" => invocation.child_by_field_name("function"),
        "method_call_expression" => invocation.child_by_field_name("method"),
        _ => None,
    }?;
    let callable = callable.utf8_text(text).ok()?;
    let callable_name = terminal_name(callable);
    if !matches!(
        callable_name.as_str(),
        "all"
            | "and_then"
            | "any"
            | "delete"
            | "fallback"
            | "fallback_service"
            | "binary_search_by"
            | "binary_search_by_key"
            | "filter"
            | "filter_map"
            | "find"
            | "find_map"
            | "flat_map"
            | "fold"
            | "for_each"
            | "get"
            | "inspect"
            | "map"
            | "map_err"
            | "map_or_else"
            | "map_while"
            | "ok_or_else"
            | "or_else"
            | "on"
            | "patch"
            | "position"
            | "post"
            | "put"
            | "reduce"
            | "register"
            | "retain"
            | "retain_mut"
            | "rposition"
            | "scan"
            | "skip_while"
            | "sort_by"
            | "sort_by_key"
            | "sort_unstable_by"
            | "sort_unstable_by_key"
            | "service_fn"
            | "set_handler"
            | "spawn"
            | "spawn_blocking"
            | "take_while"
            | "try_fold"
            | "try_for_each"
            | "unwrap_or_else"
    ) {
        return None;
    }
    node.utf8_text(text).ok().map(str::to_string)
}

fn rust_value_use(node: Node<'_>, text: &[u8]) -> Option<String> {
    if is_definition_name(node) {
        return None;
    }
    let parent = node.parent()?;
    if parent.kind() == "scoped_identifier" {
        return None;
    }
    let mut ancestor = Some(parent);
    while let Some(current) = ancestor {
        if current.kind() == "use_declaration" {
            return None;
        }
        ancestor = current.parent();
    }
    if (parent.kind() == "call_expression"
        && parent
            .child_by_field_name("function")
            .is_some_and(|child| child.id() == node.id()))
        || (parent.kind() == "macro_invocation"
            && parent
                .child_by_field_name("macro")
                .is_some_and(|child| child.id() == node.id()))
    {
        return None;
    }
    let value = node.utf8_text(text).ok()?.trim();
    let constant_like = value.bytes().any(|byte| byte.is_ascii_alphabetic())
        && value
            .bytes()
            .all(|byte| byte.is_ascii_uppercase() || byte.is_ascii_digit() || byte == b'_');
    (value.contains("::") || constant_like).then(|| value.to_string())
}

fn rust_macro_call_targets(value: &str) -> Vec<String> {
    let bytes = value.as_bytes();
    let mut output = Vec::new();
    let mut index = 0;
    let mut previous_identifier = None::<String>;
    while index < bytes.len() {
        if !(bytes[index].is_ascii_alphabetic() || bytes[index] == b'_') {
            index += 1;
            continue;
        }
        let start = index;
        index += 1;
        while index < bytes.len()
            && (bytes[index].is_ascii_alphanumeric() || matches!(bytes[index], b'_' | b':' | b'#'))
        {
            index += 1;
        }
        let token = &value[start..index];
        let mut next = index;
        while next < bytes.len() && bytes[next].is_ascii_whitespace() {
            next += 1;
        }
        let declaration_keyword = previous_identifier.as_deref().is_some_and(|previous| {
            matches!(
                previous,
                "fn" | "struct" | "enum" | "trait" | "type" | "mod" | "macro_rules"
            )
        });
        if next < bytes.len()
            && bytes[next] == b'('
            && !declaration_keyword
            && !matches!(
                token,
                "if" | "while" | "for" | "loop" | "match" | "Some" | "Ok" | "Err"
            )
        {
            output.push(token.trim_start_matches("r#").to_string());
        }
        previous_identifier = Some(token.to_string());
    }
    output.sort();
    output.dedup();
    output
}

fn parse_rust_impl(value: &str) -> Option<(RelationKind, String, Option<String>)> {
    let header = value.split('{').next()?.trim().trim_start_matches("impl ");
    let (trait_name, _) = header.split_once(" for ")?;
    Some((
        RelationKind::Implements,
        trait_name.trim().to_string(),
        None,
    ))
}

fn parse_rust_trait_inheritance(value: &str) -> Vec<(RelationKind, String, Option<String>)> {
    let Some(header) = value.split('{').next() else {
        return Vec::new();
    };
    let Some((_, bounds)) = header.split_once(':') else {
        return Vec::new();
    };
    bounds
        .split('+')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| (RelationKind::Inherits, value.to_string(), None))
        .collect()
}

fn expand_rust_use(value: &str) -> Vec<(String, Option<String>)> {
    let value = value.trim();
    if let Some(open) = value.find('{')
        && value.ends_with('}')
    {
        let prefix = value[..open].trim().trim_end_matches("::");
        return split_top_level(&value[open + 1..value.len() - 1], ',')
            .into_iter()
            .flat_map(|item| {
                let item = item.trim();
                let expanded = if item == "self" {
                    prefix.to_string()
                } else if prefix.is_empty() {
                    item.to_string()
                } else {
                    format!("{prefix}::{item}")
                };
                expand_rust_use(&expanded)
            })
            .collect();
    }
    let (target, alias) = split_alias(value, " as ");
    vec![(target.trim_end_matches("::*").to_string(), alias)]
}

fn parse_typescript_import(value: &str) -> Vec<(RelationKind, String, Option<String>)> {
    let Some(module) = quoted_value(value) else {
        return Vec::new();
    };
    let mut output = vec![(RelationKind::Imports, module.clone(), None)];
    let clause = value
        .trim()
        .trim_start_matches("import ")
        .split(" from ")
        .next()
        .unwrap_or_default()
        .trim();
    if clause.starts_with(['\'', '"']) {
        return output;
    }
    if let Some(alias) = clause.strip_prefix("* as ").map(str::trim) {
        output.push((RelationKind::Imports, module, Some(alias.to_string())));
        return output;
    }
    let named_start = clause.find('{');
    if let Some(start) = named_start
        && let Some(end) = clause.rfind('}')
    {
        for part in split_top_level(&clause[start + 1..end], ',') {
            let (target, alias) = split_alias(part.trim(), " as ");
            output.push((
                RelationKind::Imports,
                target.clone(),
                alias.or(Some(target)),
            ));
        }
    }
    let default = clause
        .split([',', '{'])
        .next()
        .map(str::trim)
        .filter(|value| !value.is_empty() && *value != "type");
    if let Some(default) = default {
        output.push((
            RelationKind::Imports,
            default.to_string(),
            Some(default.to_string()),
        ));
    }
    output
}

fn parse_typescript_reexport(value: &str) -> Vec<(RelationKind, String, Option<String>)> {
    parse_typescript_import(&value.replacen("export", "import", 1))
        .into_iter()
        .map(|(_, target, alias)| (RelationKind::Reexports, target, alias))
        .collect()
}

fn parse_commonjs_import(value: &str) -> Option<(RelationKind, String, Option<String>)> {
    if !value.contains("require(") {
        return None;
    }
    let module = quoted_value(value)?;
    let alias = value
        .split_once('=')?
        .0
        .split_whitespace()
        .last()
        .map(|value| value.trim().to_string());
    Some((RelationKind::Imports, module, alias))
}

fn split_relationship_list(
    kind: RelationKind,
    value: &str,
) -> Vec<(RelationKind, String, Option<String>)> {
    split_top_level(value, ',')
        .into_iter()
        .map(|target| (kind, target.trim().to_string(), None))
        .collect()
}

fn split_top_level(value: &str, delimiter: char) -> Vec<&str> {
    let mut depth = 0usize;
    let mut start = 0usize;
    let mut output = Vec::new();
    for (index, ch) in value.char_indices() {
        match ch {
            '{' | '(' | '<' | '[' => depth += 1,
            '}' | ')' | '>' | ']' => depth = depth.saturating_sub(1),
            _ if ch == delimiter && depth == 0 => {
                output.push(&value[start..index]);
                start = index + ch.len_utf8();
            }
            _ => {}
        }
    }
    output.push(&value[start..]);
    output
}

fn quoted_value(value: &str) -> Option<String> {
    let start = value.find(['\'', '"'])?;
    let quote = value.as_bytes()[start] as char;
    let rest = &value[start + 1..];
    let end = rest.find(quote)?;
    Some(rest[..end].to_string())
}

fn split_alias(value: &str, delimiter: &str) -> (String, Option<String>) {
    value
        .split_once(delimiter)
        .map(|(target, alias)| (target.trim().to_string(), Some(alias.trim().to_string())))
        .unwrap_or_else(|| (value.trim().to_string(), None))
}

fn normalize_target(value: &str) -> String {
    value
        .trim()
        .trim_matches(|ch: char| matches!(ch, '&' | '*' | '!' | '(' | ')' | '{' | '}' | ';'))
        .split('<')
        .next()
        .unwrap_or_default()
        .trim()
        .to_string()
}

fn terminal_name(value: &str) -> String {
    value
        .rsplit([':', '.', '/'])
        .find(|part| !part.is_empty())
        .unwrap_or(value)
        .trim_matches(|ch: char| !ch.is_alphanumeric() && ch != '_' && ch != '$')
        .to_string()
}

fn is_definition_name(node: Node<'_>) -> bool {
    node.parent()
        .is_some_and(|parent| parent.child_by_field_name("name") == Some(node))
}

fn narrowest_owner(symbols: &[IndexedSymbol], line: u64) -> Option<&IndexedSymbol> {
    symbols
        .iter()
        .filter(|symbol| symbol.start_line <= line && line <= symbol.end_line)
        .min_by_key(|symbol| symbol.end_line.saturating_sub(symbol.start_line))
}

fn file_logical_key(language: &str, path: &str) -> String {
    let digest = xxh3_128(format!("{language}:{path}:file").as_bytes());
    format!("filekey_{digest:032x}")
}

fn dedupe_references(references: &mut Vec<RawReference>) {
    let mut seen = BTreeSet::new();
    references.retain(|reference| seen.insert(reference.reference_id.clone()));
}

#[derive(Clone, Copy)]
struct ResolutionCandidate<'a> {
    symbol: &'a IndexedSymbol,
    confidence: u64,
    resolution: &'static str,
}

fn symbols_by_name(symbols: &[IndexedSymbol]) -> HashMap<String, Vec<&IndexedSymbol>> {
    let mut output: HashMap<String, Vec<&IndexedSymbol>> = HashMap::new();
    for symbol in symbols {
        output.entry(symbol.name.clone()).or_default().push(symbol);
    }
    output
}

fn resolve_reference<'a>(
    reference: &RawReference,
    by_name: &HashMap<String, Vec<&'a IndexedSymbol>>,
    aliases: &HashMap<String, HashMap<String, String>>,
) -> Vec<ResolutionCandidate<'a>> {
    let normalized_reference = reference
        .target_qualified_name
        .as_deref()
        .map(|value| value.replace('.', "::"));
    let alias_name = normalized_reference
        .as_deref()
        .and_then(|value| value.split("::").next())
        .unwrap_or(&reference.target_name);
    let alias_target = aliases
        .get(&reference.source_path)
        .and_then(|file_aliases| file_aliases.get(alias_name));
    let qualified = alias_target
        .map(|target| {
            let remainder = normalized_reference
                .as_deref()
                .and_then(|value| value.split_once("::"))
                .map(|(_, remainder)| remainder);
            remainder
                .map(|remainder| format!("{target}::{remainder}"))
                .unwrap_or_else(|| target.clone())
        })
        .or_else(|| normalized_reference.clone());
    if let Some(qualified) = qualified {
        let normalized = qualified.trim_start_matches("crate::").to_string();
        let matches = by_name
            .get(&terminal_name(&normalized))
            .into_iter()
            .flatten()
            .copied()
            .filter(|symbol| {
                symbol.language == reference.language
                    && compatible_symbol(reference.kind, symbol)
                    && (symbol.qualified_name == normalized
                        || (symbol.qualified_name.contains("::")
                            && normalized.ends_with(&format!("::{}", symbol.qualified_name))))
            })
            .collect::<Vec<_>>();
        if matches.len() == 1 {
            return vec![ResolutionCandidate {
                symbol: matches[0],
                confidence: if alias_target.is_some() {
                    CONFIDENCE_IMPORTED_ALIAS
                } else {
                    CONFIDENCE_EXACT_QUALIFIED
                },
                resolution: if alias_target.is_some() {
                    "imported_alias"
                } else {
                    "exact_qualified"
                },
            }];
        }
    }
    let Some(matches) = by_name.get(&reference.target_name) else {
        return Vec::new();
    };
    let matches = matches
        .iter()
        .copied()
        .filter(|symbol| {
            symbol.language == reference.language && compatible_symbol(reference.kind, symbol)
        })
        .collect::<Vec<_>>();
    if matches.is_empty() {
        return Vec::new();
    }
    if matches.len() > MAX_NAME_RESOLUTION_CANDIDATES {
        return Vec::new();
    }
    let same_module = matches
        .iter()
        .copied()
        .filter(|symbol| symbol.relative_path == reference.source_path)
        .collect::<Vec<_>>();
    if same_module.len() == 1 {
        return vec![ResolutionCandidate {
            symbol: same_module[0],
            confidence: CONFIDENCE_SAME_MODULE,
            resolution: "same_module_unique",
        }];
    }
    if matches.len() == 1 {
        return vec![ResolutionCandidate {
            symbol: matches[0],
            confidence: CONFIDENCE_UNIQUE_REPO,
            resolution: "unique_repository_symbol",
        }];
    }
    let mut candidates = matches
        .into_iter()
        .map(|symbol| ResolutionCandidate {
            symbol,
            confidence: CONFIDENCE_AMBIGUOUS,
            resolution: "ambiguous_name_candidate",
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| {
        path_affinity(&reference.source_path, &right.symbol.relative_path)
            .cmp(&path_affinity(
                &reference.source_path,
                &left.symbol.relative_path,
            ))
            .then(left.symbol.relative_path.cmp(&right.symbol.relative_path))
            .then(left.symbol.qualified_name.cmp(&right.symbol.qualified_name))
            .then(left.symbol.logical_key.cmp(&right.symbol.logical_key))
    });
    candidates.truncate(MAX_AMBIGUOUS_CANDIDATES);
    candidates
}

fn path_affinity(source: &str, target: &str) -> usize {
    source
        .split('/')
        .zip(target.split('/'))
        .take_while(|(left, right)| left == right)
        .count()
}

pub(crate) fn compatible_symbol(kind: RelationKind, symbol: &IndexedSymbol) -> bool {
    match kind {
        RelationKind::Imports | RelationKind::Reexports => symbol.kind != "impl",
        RelationKind::Calls => matches!(
            symbol.kind.as_str(),
            "function" | "method" | "constructor" | "macro" | "struct" | "class" | "enum_variant"
        ),
        RelationKind::TypeUses => !matches!(
            symbol.kind.as_str(),
            "function" | "method" | "constructor" | "macro" | "module" | "impl"
        ),
        RelationKind::ValueUses => !matches!(
            symbol.kind.as_str(),
            "impl" | "module" | "trait" | "interface" | "type" | "type_alias"
        ),
        RelationKind::Implements | RelationKind::Inherits => matches!(
            symbol.kind.as_str(),
            "trait" | "interface" | "class" | "struct" | "type" | "type_alias"
        ),
    }
}

fn load_symbols(connection: &Connection, repo: &str) -> Result<Vec<IndexedSymbol>> {
    let mut statement = connection.prepare(
        "SELECT symbol_id, logical_key, repo, relative_path, name, kind, container,
                language, start_line, end_line, indexed_at, file_hash, parent_symbol_id,
                parent_logical_key, qualified_name, signature, source_role, identity_stable
         FROM symbols WHERE repo = ?1",
    )?;
    let rows = statement.query_map(params![repo], |row| {
        Ok(IndexedSymbol {
            symbol_id: row.get(0)?,
            logical_key: row.get(1)?,
            repo: row.get(2)?,
            relative_path: row.get(3)?,
            name: row.get(4)?,
            kind: row.get(5)?,
            container: row.get(6)?,
            language: row.get(7)?,
            start_line: row.get::<_, i64>(8)? as u64,
            end_line: row.get::<_, i64>(9)? as u64,
            indexed_at: row.get(10)?,
            file_hash: row.get(11)?,
            parent_symbol_id: row.get(12)?,
            parent_logical_key: row.get(13)?,
            qualified_name: row.get(14)?,
            signature: row.get(15)?,
            source_role: row.get(16)?,
            identity_stable: row.get::<_, i64>(17)? != 0,
        })
    })?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .context("loading graph symbols")
}

fn ensure_graph_repo(connection: &Connection, repo: &str) -> Result<i64> {
    connection.execute(
        "INSERT OR IGNORE INTO graph_repositories_v5(repo) VALUES (?1)",
        params![repo],
    )?;
    connection
        .query_row(
            "SELECT id FROM graph_repositories_v5 WHERE repo = ?1",
            params![repo],
            |row| row.get(0),
        )
        .context("loading compact graph repository id")
}

fn graph_repo_id(connection: &Connection, repo: &str) -> Result<Option<i64>> {
    connection
        .query_row(
            "SELECT id FROM graph_repositories_v5 WHERE repo = ?1",
            params![repo],
            |row| row.get(0),
        )
        .optional()
        .context("loading compact graph repository id")
}

fn calculate_relation_document_count(connection: &Connection, repo: &str) -> Result<u64> {
    connection
        .query_row(
            "SELECT
                (SELECT COUNT(*) FROM graph_edges_v5 e
                 JOIN graph_repositories_v5 rp ON rp.id = e.repo_id
                 WHERE rp.repo = ?1)
                +
                (SELECT COUNT(*) FROM (
                    SELECT 1 FROM graph_references_v5 gr
                    JOIN graph_repositories_v5 rp ON rp.id = gr.repo_id
                    WHERE rp.repo = ?1 AND gr.resolved = 0
                    GROUP BY gr.source_key_id, gr.target_name,
                             COALESCE(gr.target_qualified_name, ''), gr.kind
                ))",
            params![repo],
            |row| Ok(row.get::<_, i64>(0)? as u64),
        )
        .context("calculating canonical relationship document count")
}

fn graph_file_id(connection: &Connection, repo: &str, path: &str) -> Result<Option<i64>> {
    connection
        .query_row(
            "SELECT rowid FROM graph_file_coverage WHERE repo = ?1 AND relative_path = ?2",
            params![repo, path],
            |row| row.get(0),
        )
        .optional()
        .context("loading compact graph file id")
}

#[allow(clippy::too_many_arguments)]
fn ensure_graph_key(
    connection: &Connection,
    repo_id: i64,
    key: &str,
    symbol_id: Option<&str>,
    path: Option<&str>,
    name: Option<&str>,
    qualified_name: Option<&str>,
) -> Result<i64> {
    connection.execute(
        "INSERT INTO graph_keys_v5 (
            repo_id, key, symbol_id, path, name, qualified_name
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6)
         ON CONFLICT(repo_id, key) DO UPDATE SET
            symbol_id = COALESCE(excluded.symbol_id, graph_keys_v5.symbol_id),
            path = COALESCE(excluded.path, graph_keys_v5.path),
            name = COALESCE(excluded.name, graph_keys_v5.name),
            qualified_name = COALESCE(excluded.qualified_name, graph_keys_v5.qualified_name)",
        params![repo_id, key, symbol_id, path, name, qualified_name],
    )?;
    connection
        .query_row(
            "SELECT id FROM graph_keys_v5 WHERE repo_id = ?1 AND key = ?2",
            params![repo_id, key],
            |row| row.get(0),
        )
        .context("loading compact graph key id")
}

fn resolution_code(value: &str) -> i64 {
    match value {
        "exact_qualified" => 0,
        "imported_alias" => 1,
        "same_module_unique" => 2,
        "unique_repository_symbol" => 3,
        "ambiguous_name_candidate" => 4,
        _ => 5,
    }
}

fn resolution_from_code(value: i64) -> &'static str {
    match value {
        0 => "exact_qualified",
        1 => "imported_alias",
        2 => "same_module_unique",
        3 => "unique_repository_symbol",
        4 => "ambiguous_name_candidate",
        _ => "lexical_fallback",
    }
}

fn affected_source_paths(
    connection: &Connection,
    repo: &str,
    refresh_paths: &[String],
    symbols: &[IndexedSymbol],
) -> Result<BTreeSet<String>> {
    let refreshed = refresh_paths.iter().cloned().collect::<BTreeSet<_>>();
    let mut affected = refreshed.clone();
    let mut target_names = symbols
        .iter()
        .filter(|symbol| refreshed.contains(&symbol.relative_path))
        .map(|symbol| symbol.name.clone())
        .collect::<BTreeSet<_>>();
    let mut prior_targets = connection.prepare(
        "SELECT DISTINCT e.source_path, gr.target_name
         FROM graph_edges_v5 e
         JOIN graph_repositories_v5 rp ON rp.id = e.repo_id
         JOIN graph_references_v5 gr ON gr.id = e.reference_id
         WHERE rp.repo = ?1 AND e.target_path = ?2",
    )?;
    for path in refresh_paths {
        let rows = prior_targets.query_map(params![repo, path], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?;
        for row in rows {
            let (source_path, target_name) = row?;
            affected.insert(source_path);
            target_names.insert(target_name);
        }
    }
    let mut named_sources = connection.prepare(
        "SELECT DISTINCT fc.relative_path
         FROM graph_references_v5 gr
         JOIN graph_repositories_v5 rp ON rp.id = gr.repo_id
         JOIN graph_file_coverage fc ON fc.rowid = gr.file_id
         WHERE rp.repo = ?1 AND gr.target_name = ?2",
    )?;
    for target_name in target_names {
        let rows = named_sources.query_map(params![repo, target_name], |row| row.get(0))?;
        for row in rows {
            affected.insert(row?);
        }
    }
    Ok(affected)
}

fn load_import_aliases(
    connection: &Connection,
    repo: &str,
) -> Result<HashMap<String, HashMap<String, String>>> {
    let mut statement = connection.prepare(
        "SELECT fc.relative_path, gr.alias,
                COALESCE(gr.target_qualified_name, gr.target_name)
         FROM graph_references_v5 gr
         JOIN graph_repositories_v5 rp ON rp.id = gr.repo_id
         JOIN graph_file_coverage fc ON fc.rowid = gr.file_id
         WHERE rp.repo = ?1 AND gr.alias IS NOT NULL
           AND gr.kind IN (1, 2)",
    )?;
    let rows = statement.query_map(params![repo], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
        ))
    })?;
    let mut aliases = HashMap::<String, HashMap<String, String>>::new();
    for row in rows {
        let (source_path, alias, target) = row?;
        aliases
            .entry(source_path)
            .or_default()
            .insert(alias, target);
    }
    Ok(aliases)
}

fn load_raw_references(connection: &Connection, repo: &str) -> Result<Vec<RawReference>> {
    let mut statement = connection.prepare(&format!(
        "{} WHERE rp.repo = ?1
         ORDER BY fc.relative_path, gr.start_line, gr.reference_id",
        RAW_REFERENCE_SELECT
    ))?;
    let rows = statement.query_map(params![repo], map_raw_reference_row)?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .context("loading raw graph references")
}

fn load_raw_references_for_paths(
    connection: &Connection,
    repo: &str,
    source_paths: &BTreeSet<String>,
) -> Result<Vec<RawReference>> {
    let mut statement = connection.prepare(&format!(
        "{} WHERE rp.repo = ?1 AND fc.relative_path = ?2
         ORDER BY gr.start_line, gr.reference_id",
        RAW_REFERENCE_SELECT
    ))?;
    let mut references = Vec::new();
    for source_path in source_paths {
        let rows = statement.query_map(params![repo, source_path], map_raw_reference_row)?;
        for row in rows {
            references.push(row?);
        }
    }
    Ok(references)
}

fn map_raw_reference_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<RawReference> {
    Ok(RawReference {
        storage_id: Some(row.get(0)?),
        storage_source_key_id: Some(row.get(1)?),
        reference_id: row.get(2)?,
        repo: row.get(3)?,
        source_key: row.get(4)?,
        source_symbol_id: row.get(5)?,
        source_path: row.get(6)?,
        target_name: row.get(7)?,
        target_qualified_name: row.get(8)?,
        alias: row.get(9)?,
        kind: RelationKind::from_code(row.get(10)?),
        start_line: row.get::<_, i64>(11)? as u64,
        end_line: row.get::<_, i64>(12)? as u64,
        evidence: row.get(13)?,
        source_role: row.get(14)?,
        language: row.get(15)?,
        file_hash: row.get(16)?,
    })
}

fn map_relation_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<ResolvedRelation> {
    Ok(ResolvedRelation {
        reference_id: row.get(0)?,
        repo: row.get(1)?,
        source_key: row.get(2)?,
        source_symbol_id: row.get(3)?,
        source_path: row.get(4)?,
        target_key: row.get(5)?,
        target_symbol_id: row.get(6)?,
        target_path: row.get(7)?,
        target_name: row.get(8)?,
        target_qualified_name: row.get(9)?,
        kind: RelationKind::from_code(row.get(10)?),
        confidence: row.get::<_, i64>(11)? as u64,
        resolution: resolution_from_code(row.get(12)?).to_string(),
        start_line: row.get::<_, i64>(13)? as u64,
        end_line: row.get::<_, i64>(14)? as u64,
        evidence: row.get(15)?,
        source_role: row.get(16)?,
        language: row.get(17)?,
        file_hash: row.get(18)?,
    })
}

fn resolution_percentage(resolved: u64, total: u64) -> f64 {
    if total == 0 {
        100.0
    } else {
        resolved as f64 * 100.0 / total as f64
    }
}

fn pragma_u64(connection: &Connection, pragma: &str) -> Result<u64> {
    connection
        .query_row(pragma, [], |row| row.get::<_, i64>(0))
        .map(|value| value.max(0) as u64)
        .with_context(|| format!("reading {pragma}"))
}

fn sqlite_storage_needs_vacuum(page_size: u64, page_count: u64, free_pages: u64) -> bool {
    if page_count == 0 || free_pages > page_count {
        return false;
    }
    let free_bytes = page_size.saturating_mul(free_pages);
    free_bytes >= SQLITE_VACUUM_MIN_FREE_BYTES
        && free_pages.saturating_mul(100)
            >= page_count.saturating_mul(SQLITE_VACUUM_MIN_FREE_RATIO_PERCENT)
}

fn sqlite_maintenance_is_busy(error: &rusqlite::Error) -> bool {
    matches!(
        error,
        rusqlite::Error::SqliteFailure(failure, _)
            if matches!(failure.code, ErrorCode::DatabaseBusy | ErrorCode::DatabaseLocked)
    )
}

fn table_has_column(connection: &Connection, table: &str, column: &str) -> Result<bool> {
    let mut statement = connection.prepare(&format!("PRAGMA table_info({table})"))?;
    let names = statement
        .query_map([], |row| row.get::<_, String>(1))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(names.iter().any(|name| name == column))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::symbols::{SymbolStore, extract_symbols};
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_db(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("agent-context-graph-{label}-{nonce}.sqlite3"))
    }

    #[test]
    fn rust_relationships_resolve_calls_types_and_impls() {
        let source = r#"
            use crate::service::Service;
            trait Runner { fn run(&self); }
            struct Service;
            impl Runner for Service { fn run(&self) { helper(); } }
            fn helper() {}
            fn caller(value: Service) { helper(); value.run(); }
        "#;
        let path = Path::new("src/lib.rs");
        let symbols = extract_symbols("/repo", "src/lib.rs", path, source, "now", "hash").unwrap();
        let extracted =
            extract_relationships("/repo", "src/lib.rs", path, source, &symbols, "hash").unwrap();
        assert!(
            extracted
                .references
                .iter()
                .any(|value| value.kind == RelationKind::Calls && value.target_name == "helper")
        );
        assert!(extracted.references.iter().any(|value| value.kind == RelationKind::TypeUses && value.target_name == "Service"));
        assert!(
            extracted.references.iter().any(
                |value| value.kind == RelationKind::Implements && value.target_name == "Runner"
            )
        );

        let db = temp_db("rust");
        let symbols_store = SymbolStore::new(db.clone());
        symbols_store
            .replace_file_symbols("/repo", "src/lib.rs", &symbols)
            .unwrap();
        let graph = GraphStore::new(db.clone());
        graph
            .replace_file(
                "/repo",
                "src/lib.rs",
                &extracted.references,
                &extracted.coverage,
                "hash",
            )
            .unwrap();
        graph.resolve_repo("/repo").unwrap();
        let relations = graph.all_relations("/repo").unwrap();
        assert!(relations.iter().any(
            |value| value.target_name == "helper" && value.confidence >= CONFIDENCE_SAME_MODULE
        ));
        let coverage = graph.coverage("/repo").unwrap();
        assert_eq!(coverage.supported_files, 1);
        assert!(coverage.references >= 3);
        let _ = fs::remove_file(db);
    }

    #[test]
    fn rust_relationships_capture_function_items_used_as_callbacks() {
        let source = r#"
            struct Parser;
            impl Parser {
                fn parse(value: &str) -> usize { value.len() }
                fn caller(values: &[&str]) -> Vec<usize> {
                    values.iter().map(Self::parse).collect()
                }
            }
            fn report(_: std::num::ParseIntError) {}
            fn fallible(value: &str) { let _ = value.parse::<u8>().map_err(report); }
            fn live_test_connection() {}
            fn passes_value(value: fn()) { value(); }
            fn not_a_callback() { passes_value(live_test_connection); }
            fn route_handler() {}
            fn ui_fallback() {}
            fn routes() { post(route_handler); Router::new().fallback(ui_fallback); }
            fn build_notification(_: usize) {}
            macro_rules! mapping { () => { build_notification(1) } }
            const LOOKUP_TABLE: &[u8] = &[];
            fn reads_values() { let _ = LOOKUP_TABLE; let _ = catalog::DEFAULT_VALUE; }
        "#;
        let path = Path::new("src/lib.rs");
        let symbols = extract_symbols("/repo", "src/lib.rs", path, source, "now", "hash").unwrap();
        let extracted =
            extract_relationships("/repo", "src/lib.rs", path, source, &symbols, "hash").unwrap();

        assert!(extracted.references.iter().any(|reference| {
            reference.kind == RelationKind::Calls
                && reference.target_qualified_name.as_deref() == Some("Self::parse")
        }));
        assert!(extracted.references.iter().any(|reference| {
            reference.kind == RelationKind::Calls && reference.target_name == "report"
        }));
        assert!(!extracted.references.iter().any(|reference| {
            reference.kind == RelationKind::Calls
                && reference.target_name == "live_test_connection"
                && reference.evidence == "live_test_connection"
        }));
        for target in ["route_handler", "ui_fallback", "build_notification"] {
            assert!(extracted.references.iter().any(|reference| {
                reference.kind == RelationKind::Calls && reference.target_name == target
            }));
        }
        for target in ["LOOKUP_TABLE", "DEFAULT_VALUE"] {
            assert!(extracted.references.iter().any(|reference| {
                reference.kind == RelationKind::ValueUses && reference.target_name == target
            }));
        }
    }

    #[test]
    fn typescript_relationships_capture_import_call_jsx_and_inheritance() {
        let source = r#"
            import Widget from './widget';
            interface Base {}
            class Panel extends Base { render(): Widget { return makeWidget(); } }
            function makeWidget(): Widget { return <Widget />; }
        "#;
        let path = Path::new("src/panel.tsx");
        let symbols =
            extract_symbols("/repo", "src/panel.tsx", path, source, "now", "hash").unwrap();
        let extracted =
            extract_relationships("/repo", "src/panel.tsx", path, source, &symbols, "hash")
                .unwrap();
        assert!(
            extracted
                .references
                .iter()
                .any(|value| value.kind == RelationKind::Imports)
        );
        assert!(extracted.references.iter().any(|value| value.kind == RelationKind::Calls && value.target_name == "makeWidget"));
        assert!(
            extracted
                .references
                .iter()
                .any(|value| value.kind == RelationKind::Calls && value.target_name == "Widget")
        );
        assert!(
            extracted
                .references
                .iter()
                .any(|value| value.kind == RelationKind::Inherits && value.target_name == "Base")
        );
    }

    #[test]
    fn rust_relationships_expand_aliases_globs_macros_and_trait_inheritance() {
        let source = r#"
            use crate::service::{Service as S, helper, *};
            trait Parent {}
            trait Child: Parent + Send {}
            struct Service;
            fn caller(value: S) { helper(); tracing::info!("called"); let _ = S {}; }
        "#;
        let path = Path::new("src/lib.rs");
        let symbols = extract_symbols("/repo", "src/lib.rs", path, source, "now", "hash").unwrap();
        let extracted =
            extract_relationships("/repo", "src/lib.rs", path, source, &symbols, "hash").unwrap();
        assert!(extracted.references.iter().any(|reference| {
            reference.kind == RelationKind::Imports
                && reference.target_name == "Service"
                && reference.alias.as_deref() == Some("S")
        }));
        assert!(extracted.references.iter().any(|reference| {
            reference.kind == RelationKind::Calls && reference.target_name == "info"
        }));
        assert!(extracted.references.iter().any(|reference| {
            reference.kind == RelationKind::Inherits && reference.target_name == "Parent"
        }));
    }

    #[test]
    fn typescript_relationships_expand_named_namespace_and_commonjs_imports() {
        let source = r#"
            import Widget, { makeWidget as build, Model } from './widget';
            import * as tools from './tools';
            const legacy = require('./legacy');
            interface Base<T> {}
            class Panel extends Base<Model> implements Widget { render() { return build(tools.value); } }
        "#;
        let path = Path::new("src/panel.ts");
        let symbols =
            extract_symbols("/repo", "src/panel.ts", path, source, "now", "hash").unwrap();
        let extracted =
            extract_relationships("/repo", "src/panel.ts", path, source, &symbols, "hash").unwrap();
        assert!(extracted.references.iter().any(|reference| {
            reference.kind == RelationKind::Imports
                && reference.target_name == "makeWidget"
                && reference.alias.as_deref() == Some("build")
        }));
        assert!(extracted.references.iter().any(|reference| {
            reference.kind == RelationKind::Imports && reference.alias.as_deref() == Some("tools")
        }));
        assert!(extracted.references.iter().any(|reference| {
            reference.kind == RelationKind::Imports && reference.alias.as_deref() == Some("legacy")
        }));
        assert!(extracted.references.iter().any(|reference| {
            reference.kind == RelationKind::Implements && reference.target_name == "Widget"
        }));
    }

    #[test]
    fn resolution_bounds_ambiguity_and_rejects_cross_language_candidates() {
        let source = "fn duplicate() {}";
        let mut base = extract_symbols(
            "/repo",
            "src/base.rs",
            Path::new("src/base.rs"),
            source,
            "now",
            "hash",
        )
        .unwrap()
        .into_iter()
        .find(|symbol| symbol.name == "duplicate")
        .unwrap();
        let mut symbols = Vec::new();
        for index in 0..10 {
            let mut symbol = base.clone();
            symbol.symbol_id = format!("rust-symbol-{index}");
            symbol.logical_key = format!("rust-key-{index}");
            symbol.relative_path = format!("src/module_{index}.rs");
            symbol.qualified_name = format!("module_{index}::duplicate");
            symbols.push(symbol);
        }
        base.symbol_id = "typescript-symbol".to_string();
        base.logical_key = "typescript-key".to_string();
        base.language = "typescript".to_string();
        base.relative_path = "src/duplicate.ts".to_string();
        symbols.push(base);
        let reference = RawReference {
            storage_id: None,
            storage_source_key_id: None,
            reference_id: "reference".to_string(),
            repo: "/repo".to_string(),
            source_key: "source".to_string(),
            source_symbol_id: None,
            source_path: "src/caller.rs".to_string(),
            target_name: "duplicate".to_string(),
            target_qualified_name: None,
            alias: None,
            kind: RelationKind::Calls,
            start_line: 1,
            end_line: 1,
            evidence: "duplicate()".to_string(),
            source_role: "production".to_string(),
            language: "rust".to_string(),
            file_hash: "hash".to_string(),
        };
        let by_name = symbols_by_name(&symbols);
        let candidates = resolve_reference(&reference, &by_name, &HashMap::new());
        assert_eq!(candidates.len(), MAX_AMBIGUOUS_CANDIDATES);
        assert!(
            candidates
                .iter()
                .all(|candidate| candidate.symbol.language == "rust")
        );

        let mut excessive = Vec::new();
        for index in 0..=MAX_NAME_RESOLUTION_CANDIDATES {
            let mut symbol = symbols[0].clone();
            symbol.symbol_id = format!("excessive-symbol-{index}");
            symbol.logical_key = format!("excessive-key-{index}");
            symbol.relative_path = format!("src/excessive_{index}.rs");
            excessive.push(symbol);
        }
        assert!(
            resolve_reference(&reference, &symbols_by_name(&excessive), &HashMap::new()).is_empty()
        );

        let mut nominal = symbols[0].clone();
        nominal.symbol_id = "app-use-case-struct".to_string();
        nominal.logical_key = "app-use-case-struct-key".to_string();
        nominal.name = "AppUseCase".to_string();
        nominal.kind = "struct".to_string();
        nominal.qualified_name = "AppUseCase".to_string();
        let mut with_impls = vec![nominal.clone()];
        for index in 0..=MAX_NAME_RESOLUTION_CANDIDATES {
            let mut implementation = nominal.clone();
            implementation.symbol_id = format!("app-use-case-impl-{index}");
            implementation.logical_key = format!("app-use-case-impl-key-{index}");
            implementation.kind = "impl".to_string();
            with_impls.push(implementation);
        }
        let mut import = reference.clone();
        import.target_name = "AppUseCase".to_string();
        import.target_qualified_name = Some("crate::AppUseCase".to_string());
        import.kind = RelationKind::Imports;
        let resolved = resolve_reference(&import, &symbols_by_name(&with_impls), &HashMap::new());
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].symbol.kind, "struct");
        assert_eq!(resolved[0].confidence, CONFIDENCE_EXACT_QUALIFIED);

        import.kind = RelationKind::TypeUses;
        let resolved = resolve_reference(&import, &symbols_by_name(&with_impls), &HashMap::new());
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].symbol.kind, "struct");
    }

    #[test]
    fn qualified_resolution_requires_the_declared_container() {
        let mut symbol = extract_symbols(
            "/repo",
            "src/base.rs",
            Path::new("src/base.rs"),
            "struct BasePath; impl BasePath { fn join() {} }",
            "now",
            "hash",
        )
        .unwrap()
        .into_iter()
        .find(|symbol| symbol.name == "join")
        .unwrap();
        symbol.qualified_name = "BasePath::join".to_string();
        let mut reference = RawReference {
            storage_id: None,
            storage_source_key_id: None,
            reference_id: "reference".to_string(),
            repo: "/repo".to_string(),
            source_key: "source".to_string(),
            source_symbol_id: None,
            source_path: "src/caller.rs".to_string(),
            target_name: "join".to_string(),
            target_qualified_name: Some("PathBuf::join".to_string()),
            alias: None,
            kind: RelationKind::Calls,
            start_line: 1,
            end_line: 1,
            evidence: "PathBuf::join()".to_string(),
            source_role: "production".to_string(),
            language: "rust".to_string(),
            file_hash: "hash".to_string(),
        };
        let symbols = vec![symbol];
        let by_name = symbols_by_name(&symbols);
        let mismatched = resolve_reference(&reference, &by_name, &HashMap::new());
        assert_eq!(mismatched.len(), 1);
        assert_eq!(mismatched[0].confidence, CONFIDENCE_UNIQUE_REPO);
        assert_eq!(mismatched[0].resolution, "unique_repository_symbol");

        reference.target_qualified_name = Some("BasePath::join".to_string());
        let exact = resolve_reference(&reference, &by_name, &HashMap::new());
        assert_eq!(exact.len(), 1);
        assert_eq!(exact[0].confidence, CONFIDENCE_EXACT_QUALIFIED);
        assert_eq!(exact[0].resolution, "exact_qualified");
    }

    #[test]
    fn ambiguous_candidates_are_counted_as_possible_not_resolved() {
        fn prepare(
            repo: &str,
            path: &str,
            source: &str,
        ) -> (Vec<IndexedSymbol>, GraphFileReplacement) {
            let symbols =
                extract_symbols(repo, path, Path::new(path), source, "now", "hash").unwrap();
            let extracted =
                extract_relationships(repo, path, Path::new(path), source, &symbols, "hash")
                    .unwrap();
            (
                symbols,
                GraphFileReplacement {
                    relative_path: path.to_string(),
                    references: extracted.references,
                    coverage: extracted.coverage,
                    file_hash: "hash".to_string(),
                },
            )
        }

        let repo = "/repo";
        let db = temp_db("ambiguous-coverage");
        let symbol_store = SymbolStore::new(db.clone());
        let graph = GraphStore::new(db.clone());
        let (left_symbols, left_graph) = prepare(repo, "src/left.rs", "pub fn duplicate() {}");
        let (right_symbols, right_graph) = prepare(repo, "src/right.rs", "pub fn duplicate() {}");
        let (caller_symbols, caller_graph) =
            prepare(repo, "src/caller.rs", "pub fn caller() { duplicate(); }");
        symbol_store
            .replace_files_symbols(
                repo,
                &[
                    ("src/left.rs".to_string(), left_symbols),
                    ("src/right.rs".to_string(), right_symbols),
                    ("src/caller.rs".to_string(), caller_symbols),
                ],
            )
            .unwrap();
        graph
            .replace_files(repo, &[left_graph, right_graph, caller_graph])
            .unwrap();
        graph.resolve_repo(repo).unwrap();

        let relations = graph.all_relations(repo).unwrap();
        assert_eq!(relations.len(), 2);
        assert!(relations.iter().all(|relation| {
            relation.confidence == CONFIDENCE_AMBIGUOUS
                && relation.resolution == "ambiguous_name_candidate"
        }));
        let coverage = graph.coverage_cached(repo).unwrap();
        assert_eq!(coverage.references, 1);
        assert_eq!(coverage.definite, 0);
        assert_eq!(coverage.probable, 0);
        assert_eq!(coverage.possible, 1);
        assert_eq!(coverage.unresolved, 0);
        assert_eq!(coverage.resolution_percentage, 0.0);
        let _ = fs::remove_file(db);
    }

    #[test]
    fn batched_replacement_deduplicates_edges_and_partial_resolution_repairs_dependents() {
        fn prepare(
            repo: &str,
            path: &str,
            source: &str,
        ) -> (Vec<IndexedSymbol>, GraphFileReplacement) {
            let symbols =
                extract_symbols(repo, path, Path::new(path), source, "now", "hash").unwrap();
            let extracted =
                extract_relationships(repo, path, Path::new(path), source, &symbols, "hash")
                    .unwrap();
            (
                symbols,
                GraphFileReplacement {
                    relative_path: path.to_string(),
                    references: extracted.references,
                    coverage: extracted.coverage,
                    file_hash: "hash".to_string(),
                },
            )
        }

        let repo = "/repo";
        let db = temp_db("batched-partial");
        let symbol_store = SymbolStore::new(db.clone());
        let graph = GraphStore::new(db.clone());
        let (target_symbols, target_graph) = prepare(repo, "src/target.rs", "pub fn target() {}");
        let (caller_symbols, caller_graph) = prepare(
            repo,
            "src/caller.rs",
            "pub fn caller() {\n target();\n target();\n}",
        );
        let symbol_batch = vec![
            ("src/target.rs".to_string(), target_symbols.clone()),
            ("src/caller.rs".to_string(), caller_symbols.clone()),
        ];
        let graph_batch = vec![target_graph.clone(), caller_graph.clone()];
        symbol_store
            .replace_files_symbols(repo, &symbol_batch)
            .unwrap();
        graph.replace_files(repo, &graph_batch).unwrap();
        symbol_store
            .replace_files_symbols(repo, &symbol_batch)
            .unwrap();
        graph.replace_files(repo, &graph_batch).unwrap();
        graph.resolve_repo(repo).unwrap();

        let target_key = target_symbols
            .iter()
            .find(|symbol| symbol.name == "target")
            .unwrap()
            .logical_key
            .clone();
        let inbound = graph
            .relations_to(repo, std::slice::from_ref(&target_key), 650, 100)
            .unwrap();
        assert_eq!(inbound.len(), 1, "repeat call sites must share one edge");
        let analysis_documents = graph.all_analysis_relations(repo).unwrap();
        assert_eq!(analysis_documents.len(), 1);
        assert_eq!(
            analysis_documents[0].target_key.as_deref(),
            Some(&*target_key)
        );
        assert_eq!(analysis_documents[0].confidence, 750);
        let coverage = graph.coverage_cached(repo).unwrap();
        assert_eq!(coverage.references, 2);
        assert_eq!(coverage.definite, 0);
        assert_eq!(coverage.probable, 2);

        let (renamed_symbols, renamed_graph) =
            prepare(repo, "src/target.rs", "pub fn renamed() {}");
        symbol_store
            .replace_files_symbols(
                repo,
                &[("src/target.rs".to_string(), renamed_symbols.clone())],
            )
            .unwrap();
        graph.replace_files(repo, &[renamed_graph]).unwrap();
        let affected = graph
            .resolve_repo_paths(repo, &["src/target.rs".to_string()])
            .unwrap();
        assert!(affected.contains(&"src/caller.rs".to_string()));
        assert!(
            graph
                .relations_to(repo, &[target_key], 0, 100)
                .unwrap()
                .is_empty()
        );
        let coverage = graph.coverage_cached(repo).unwrap();
        assert_eq!(coverage.unresolved, 2);
        let documents = graph
            .relation_documents_for_source_paths(repo, &["src/caller.rs".to_string()])
            .unwrap();
        assert_eq!(documents.len(), 1);
        assert!(documents.iter().all(|document| {
            document.target_key.is_none()
                && document.confidence == 0
                && document.resolution == "unresolved"
        }));
        let analysis_documents = graph.all_analysis_relations(repo).unwrap();
        assert_eq!(analysis_documents.len(), 1);
        assert_eq!(analysis_documents[0].target_name, "target");
        assert!(analysis_documents[0].target_key.is_none());
        assert_eq!(analysis_documents[0].confidence, 0);
        assert_eq!(analysis_documents[0].resolution, "unresolved");

        let (updated_caller_symbols, updated_caller_graph) =
            prepare(repo, "src/caller.rs", "pub fn caller() { renamed(); }");
        symbol_store
            .replace_files_symbols(
                repo,
                &[("src/caller.rs".to_string(), updated_caller_symbols)],
            )
            .unwrap();
        graph.replace_files(repo, &[updated_caller_graph]).unwrap();
        graph
            .resolve_repo_paths(repo, &["src/caller.rs".to_string()])
            .unwrap();
        let renamed_key = renamed_symbols
            .iter()
            .find(|symbol| symbol.name == "renamed")
            .unwrap()
            .logical_key
            .clone();
        assert_eq!(
            graph
                .relations_to(repo, &[renamed_key], 650, 100)
                .unwrap()
                .len(),
            1
        );
        let _ = fs::remove_file(db);
    }

    #[test]
    fn overlay_resolution_uses_canonical_symbols_and_honors_suppression() {
        let db = temp_db("overlay-fallback");
        let symbol_store = SymbolStore::new(db.clone());
        let graph = GraphStore::new(db.clone());
        let canonical = "/canonical";
        let overlay = "/overlay";
        let target_symbols = extract_symbols(
            canonical,
            "src/target.rs",
            Path::new("src/target.rs"),
            "pub fn target() {}",
            "now",
            "target-hash",
        )
        .unwrap();
        symbol_store
            .replace_files_symbols(
                canonical,
                &[("src/target.rs".to_string(), target_symbols.clone())],
            )
            .unwrap();

        let caller_symbols = extract_symbols(
            overlay,
            "src/caller.rs",
            Path::new("src/caller.rs"),
            "pub fn caller() { target(); }",
            "now",
            "caller-hash",
        )
        .unwrap();
        let caller_relationships = extract_relationships(
            overlay,
            "src/caller.rs",
            Path::new("src/caller.rs"),
            "pub fn caller() { target(); }",
            &caller_symbols,
            "caller-hash",
        )
        .unwrap();
        symbol_store
            .replace_files_symbols(overlay, &[("src/caller.rs".to_string(), caller_symbols)])
            .unwrap();
        graph
            .replace_files(
                overlay,
                &[GraphFileReplacement {
                    relative_path: "src/caller.rs".to_string(),
                    references: caller_relationships.references,
                    coverage: caller_relationships.coverage,
                    file_hash: "caller-hash".to_string(),
                }],
            )
            .unwrap();

        graph
            .resolve_repo_with_fallback(overlay, canonical, &BTreeSet::new())
            .unwrap();
        let target_key = target_symbols
            .iter()
            .find(|symbol| symbol.name == "target")
            .unwrap()
            .logical_key
            .clone();
        assert_eq!(
            graph
                .relations_to(overlay, std::slice::from_ref(&target_key), 650, 10)
                .unwrap()
                .len(),
            1
        );

        graph
            .resolve_repo_with_fallback(
                overlay,
                canonical,
                &BTreeSet::from(["src/target.rs".to_string()]),
            )
            .unwrap();
        assert!(
            graph
                .relations_to(overlay, &[target_key], 0, 10)
                .unwrap()
                .is_empty()
        );
        assert_eq!(graph.coverage_cached(overlay).unwrap().unresolved, 1);
        let _ = fs::remove_file(db);
    }

    #[test]
    fn incremental_maintenance_bounds_renamed_graph_key_churn() {
        let db = temp_db("incremental-maintenance");
        let graph = GraphStore::new(db.clone());
        let mut repo_id = 0;
        for generation in 0..32 {
            let connection = graph.open().unwrap();
            repo_id = ensure_graph_repo(&connection, "/repo").unwrap();
            connection
                .execute(
                    "DELETE FROM graph_references_v5 WHERE repo_id = ?1",
                    params![repo_id],
                )
                .unwrap();
            let key = format!("renamed-key-{generation}");
            let live_key = ensure_graph_key(
                &connection,
                repo_id,
                &key,
                None,
                Some("src/live.rs"),
                None,
                None,
            )
            .unwrap();
            connection
                .execute(
                    "INSERT INTO graph_references_v5 (
                        repo_id, file_id, reference_id, source_key_id, target_name,
                        kind, start_line, end_line, evidence, source_role
                     ) VALUES (?1, 1, ?2, ?3, 'target', 0, 1, 1, 'target()', 'production')",
                    params![repo_id, format!("reference-{generation}"), live_key],
                )
                .unwrap();
            drop(connection);
            graph.maintain_incremental_storage("/repo", true).unwrap();
        }

        let connection = graph.open().unwrap();
        let keys = connection
            .prepare("SELECT key FROM graph_keys_v5 WHERE repo_id = ?1 ORDER BY key")
            .unwrap()
            .query_map(params![repo_id], |row| row.get::<_, String>(0))
            .unwrap()
            .collect::<rusqlite::Result<Vec<_>>>()
            .unwrap();
        assert_eq!(keys, vec!["renamed-key-31"]);
        drop(connection);
        let _ = fs::remove_file(db);
    }

    #[test]
    fn sqlite_vacuum_requires_material_absolute_and_relative_reclaim() {
        let page_size = 4096;
        let threshold_pages = SQLITE_VACUUM_MIN_FREE_BYTES / page_size;
        assert!(!sqlite_storage_needs_vacuum(
            page_size,
            threshold_pages * 10,
            threshold_pages - 1
        ));
        assert!(!sqlite_storage_needs_vacuum(
            page_size,
            threshold_pages * 10,
            threshold_pages
        ));
        assert!(sqlite_storage_needs_vacuum(
            page_size,
            threshold_pages * 5,
            threshold_pages
        ));
        assert!(!sqlite_storage_needs_vacuum(page_size, 10, 11));
    }

    #[test]
    fn incompatible_graph_format_fails_closed() {
        let db = temp_db("format");
        let graph = GraphStore::new(db.clone());
        let connection = graph.open().unwrap();
        connection
            .execute(
                "INSERT INTO graph_state(repo, status, root_hash, graph_format)
                 VALUES ('/repo', 'ready', 'root', 1)",
                [],
            )
            .unwrap();
        assert_eq!(graph.state("/repo").unwrap().unwrap().0, "incompatible");
        let _ = fs::remove_file(db);
    }

    #[test]
    fn storage_compaction_drops_legacy_unused_index_and_preserves_state() {
        let db = temp_db("compact");
        let graph = GraphStore::new(db.clone());
        let connection = graph.open().unwrap();
        connection
            .execute_batch(
                "CREATE TABLE graph_references (
                    repo TEXT NOT NULL, source_key TEXT NOT NULL, kind TEXT NOT NULL
                 );
                 CREATE TABLE graph_edges (repo TEXT NOT NULL);
                 CREATE INDEX idx_graph_refs_source
                    ON graph_references(repo, source_key, kind);",
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO graph_state(repo, status, root_hash, graph_format)
                 VALUES ('/repo', 'ready', 'root', 5)",
                [],
            )
            .unwrap();
        drop(connection);

        graph.compact_storage().unwrap();

        let connection = Connection::open(&db).unwrap();
        let index_count = connection
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master
                 WHERE type = 'index' AND name = 'idx_graph_refs_source'",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap();
        assert_eq!(index_count, 0);
        assert_eq!(graph.state("/repo").unwrap().unwrap().0, "ready");
        drop(connection);
        let _ = fs::remove_file(db);
    }
}
