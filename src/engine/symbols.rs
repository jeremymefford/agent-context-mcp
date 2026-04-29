use anyhow::{Context, Result};
use rusqlite::{Connection, OptionalExtension, params};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use tree_sitter::{Language, Node, Parser};
use xxhash_rust::xxh3::xxh3_128;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct IndexedSymbol {
    pub symbol_id: String,
    pub repo: String,
    pub relative_path: String,
    pub name: String,
    pub kind: String,
    pub container: Option<String>,
    pub language: String,
    pub start_line: u64,
    pub end_line: u64,
    pub indexed_at: String,
    pub file_hash: String,
    pub parent_symbol_id: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OutlineNode {
    pub symbol_id: String,
    pub name: String,
    pub kind: String,
    pub container: Option<String>,
    pub language: String,
    pub start_line: u64,
    pub end_line: u64,
    pub children: Vec<OutlineNode>,
}

#[derive(Clone)]
pub struct SymbolStore {
    path: PathBuf,
}

#[derive(Clone, Copy)]
struct SymbolRule {
    node_kind: &'static str,
    symbol_kind: &'static str,
}

struct SymbolConfig {
    language: Language,
    language_name: &'static str,
    rules: &'static [SymbolRule],
}

struct SymbolExtractionContext<'a> {
    text: &'a [u8],
    relative_path: &'a str,
    repo: &'a str,
    config: &'a SymbolConfig,
    indexed_at: &'a str,
    file_hash: &'a str,
}

const IDENTIFIER_NODE_KINDS: &[&str] = &[
    "identifier",
    "type_identifier",
    "field_identifier",
    "property_identifier",
    "namespace_identifier",
    "simple_identifier",
    "constant",
    "name",
];

const JS_SYMBOLS: &[SymbolRule] = &[
    SymbolRule {
        node_kind: "class_declaration",
        symbol_kind: "class",
    },
    SymbolRule {
        node_kind: "function_declaration",
        symbol_kind: "function",
    },
    SymbolRule {
        node_kind: "method_definition",
        symbol_kind: "method",
    },
    SymbolRule {
        node_kind: "interface_declaration",
        symbol_kind: "interface",
    },
    SymbolRule {
        node_kind: "type_alias_declaration",
        symbol_kind: "type",
    },
];

const PYTHON_SYMBOLS: &[SymbolRule] = &[
    SymbolRule {
        node_kind: "class_definition",
        symbol_kind: "class",
    },
    SymbolRule {
        node_kind: "function_definition",
        symbol_kind: "function",
    },
    SymbolRule {
        node_kind: "async_function_definition",
        symbol_kind: "function",
    },
];

const JAVA_SYMBOLS: &[SymbolRule] = &[
    SymbolRule {
        node_kind: "class_declaration",
        symbol_kind: "class",
    },
    SymbolRule {
        node_kind: "interface_declaration",
        symbol_kind: "interface",
    },
    SymbolRule {
        node_kind: "enum_declaration",
        symbol_kind: "enum",
    },
    SymbolRule {
        node_kind: "method_declaration",
        symbol_kind: "method",
    },
    SymbolRule {
        node_kind: "constructor_declaration",
        symbol_kind: "constructor",
    },
];

const CPP_SYMBOLS: &[SymbolRule] = &[
    SymbolRule {
        node_kind: "namespace_definition",
        symbol_kind: "namespace",
    },
    SymbolRule {
        node_kind: "class_specifier",
        symbol_kind: "class",
    },
    SymbolRule {
        node_kind: "struct_specifier",
        symbol_kind: "struct",
    },
    SymbolRule {
        node_kind: "enum_specifier",
        symbol_kind: "enum",
    },
    SymbolRule {
        node_kind: "function_definition",
        symbol_kind: "function",
    },
];

const GO_SYMBOLS: &[SymbolRule] = &[
    SymbolRule {
        node_kind: "function_declaration",
        symbol_kind: "function",
    },
    SymbolRule {
        node_kind: "method_declaration",
        symbol_kind: "method",
    },
    SymbolRule {
        node_kind: "type_declaration",
        symbol_kind: "type",
    },
];

const RUST_SYMBOLS: &[SymbolRule] = &[
    SymbolRule {
        node_kind: "mod_item",
        symbol_kind: "module",
    },
    SymbolRule {
        node_kind: "struct_item",
        symbol_kind: "struct",
    },
    SymbolRule {
        node_kind: "enum_item",
        symbol_kind: "enum",
    },
    SymbolRule {
        node_kind: "trait_item",
        symbol_kind: "trait",
    },
    SymbolRule {
        node_kind: "impl_item",
        symbol_kind: "impl",
    },
    SymbolRule {
        node_kind: "function_item",
        symbol_kind: "function",
    },
    SymbolRule {
        node_kind: "type_item",
        symbol_kind: "type",
    },
    SymbolRule {
        node_kind: "const_item",
        symbol_kind: "const",
    },
];

const CSHARP_SYMBOLS: &[SymbolRule] = &[
    SymbolRule {
        node_kind: "class_declaration",
        symbol_kind: "class",
    },
    SymbolRule {
        node_kind: "interface_declaration",
        symbol_kind: "interface",
    },
    SymbolRule {
        node_kind: "struct_declaration",
        symbol_kind: "struct",
    },
    SymbolRule {
        node_kind: "enum_declaration",
        symbol_kind: "enum",
    },
    SymbolRule {
        node_kind: "method_declaration",
        symbol_kind: "method",
    },
    SymbolRule {
        node_kind: "constructor_declaration",
        symbol_kind: "constructor",
    },
];

const SCALA_SYMBOLS: &[SymbolRule] = &[
    SymbolRule {
        node_kind: "object_definition",
        symbol_kind: "object",
    },
    SymbolRule {
        node_kind: "class_definition",
        symbol_kind: "class",
    },
    SymbolRule {
        node_kind: "trait_definition",
        symbol_kind: "trait",
    },
    SymbolRule {
        node_kind: "function_definition",
        symbol_kind: "function",
    },
];

const KOTLIN_SYMBOLS: &[SymbolRule] = &[
    SymbolRule {
        node_kind: "class_declaration",
        symbol_kind: "class",
    },
    SymbolRule {
        node_kind: "object_declaration",
        symbol_kind: "object",
    },
    SymbolRule {
        node_kind: "function_declaration",
        symbol_kind: "function",
    },
    SymbolRule {
        node_kind: "property_declaration",
        symbol_kind: "property",
    },
    SymbolRule {
        node_kind: "secondary_constructor",
        symbol_kind: "constructor",
    },
];

const PHP_SYMBOLS: &[SymbolRule] = &[
    SymbolRule {
        node_kind: "class_declaration",
        symbol_kind: "class",
    },
    SymbolRule {
        node_kind: "interface_declaration",
        symbol_kind: "interface",
    },
    SymbolRule {
        node_kind: "trait_declaration",
        symbol_kind: "trait",
    },
    SymbolRule {
        node_kind: "function_definition",
        symbol_kind: "function",
    },
    SymbolRule {
        node_kind: "method_declaration",
        symbol_kind: "method",
    },
];

const RUBY_SYMBOLS: &[SymbolRule] = &[
    SymbolRule {
        node_kind: "class",
        symbol_kind: "class",
    },
    SymbolRule {
        node_kind: "module",
        symbol_kind: "module",
    },
    SymbolRule {
        node_kind: "method",
        symbol_kind: "method",
    },
    SymbolRule {
        node_kind: "singleton_method",
        symbol_kind: "method",
    },
];

const SWIFT_SYMBOLS: &[SymbolRule] = &[
    SymbolRule {
        node_kind: "class_declaration",
        symbol_kind: "class",
    },
    SymbolRule {
        node_kind: "struct_declaration",
        symbol_kind: "struct",
    },
    SymbolRule {
        node_kind: "enum_declaration",
        symbol_kind: "enum",
    },
    SymbolRule {
        node_kind: "protocol_declaration",
        symbol_kind: "protocol",
    },
    SymbolRule {
        node_kind: "function_declaration",
        symbol_kind: "function",
    },
];

impl SymbolStore {
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }

    pub fn replace_file_symbols(
        &self,
        repo: &str,
        relative_path: &str,
        symbols: &[IndexedSymbol],
    ) -> Result<()> {
        let mut connection = self.open()?;
        let transaction = connection
            .transaction()
            .context("starting symbol transaction")?;
        transaction
            .execute(
                "DELETE FROM symbols WHERE repo = ?1 AND relative_path = ?2",
                params![repo, relative_path],
            )
            .context("deleting stale symbols for file")?;
        {
            let mut statement = transaction
                .prepare_cached(
                    "INSERT INTO symbols (
                        symbol_id,
                        repo,
                        relative_path,
                        name,
                        kind,
                        container,
                        language,
                        start_line,
                        end_line,
                        indexed_at,
                        file_hash,
                        parent_symbol_id
                    ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
                )
                .context("preparing symbol insert")?;
            let mut inserted_ids = BTreeSet::new();
            for symbol in symbols {
                // Keep refreshes resilient if the extractor emits duplicate IDs
                // for one file batch.
                if !inserted_ids.insert(symbol.symbol_id.clone()) {
                    continue;
                }
                statement
                    .execute(params![
                        symbol.symbol_id,
                        symbol.repo,
                        symbol.relative_path,
                        symbol.name,
                        symbol.kind,
                        symbol.container,
                        symbol.language,
                        symbol.start_line as i64,
                        symbol.end_line as i64,
                        symbol.indexed_at,
                        symbol.file_hash,
                        symbol.parent_symbol_id,
                    ])
                    .with_context(|| format!("inserting symbol {}", symbol.symbol_id))?;
            }
        }
        transaction
            .commit()
            .context("committing symbol transaction")
    }

    pub fn delete_file(&self, repo: &str, relative_path: &str) -> Result<()> {
        let connection = self.open()?;
        connection
            .execute(
                "DELETE FROM symbols WHERE repo = ?1 AND relative_path = ?2",
                params![repo, relative_path],
            )
            .context("deleting file symbols")?;
        Ok(())
    }

    pub fn clear_repo(&self, repo: &str) -> Result<()> {
        let connection = self.open()?;
        connection
            .execute("DELETE FROM symbols WHERE repo = ?1", params![repo])
            .context("clearing repo symbols")?;
        Ok(())
    }

    pub fn symbol_by_id(&self, repo: &str, symbol_id: &str) -> Result<Option<IndexedSymbol>> {
        let connection = self.open()?;
        connection
            .query_row(
                "SELECT
                    symbol_id,
                    repo,
                    relative_path,
                    name,
                    kind,
                    container,
                    language,
                    start_line,
                    end_line,
                    indexed_at,
                    file_hash,
                    parent_symbol_id
                 FROM symbols
                 WHERE repo = ?1 AND symbol_id = ?2",
                params![repo, symbol_id],
                map_symbol_row,
            )
            .optional()
            .with_context(|| format!("loading symbol {repo}:{symbol_id}"))
    }

    pub fn symbols_by_repo_and_ids(
        &self,
        repo: &str,
        ids: &[String],
    ) -> Result<Vec<IndexedSymbol>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let connection = self.open()?;
        let mut symbols = Vec::new();
        let mut statement = connection
            .prepare_cached(
                "SELECT
                    symbol_id,
                    repo,
                    relative_path,
                    name,
                    kind,
                    container,
                    language,
                    start_line,
                    end_line,
                    indexed_at,
                    file_hash,
                    parent_symbol_id
                FROM symbols
                WHERE repo = ?1 AND symbol_id = ?2",
            )
            .context("preparing symbol lookup")?;
        for id in ids {
            if let Some(symbol) = statement
                .query_row(params![repo, id], map_symbol_row)
                .optional()
                .with_context(|| format!("loading symbol {repo}:{id}"))?
            {
                symbols.push(symbol);
            }
        }
        Ok(symbols)
    }

    pub fn file_symbols(&self, repo: &str, relative_path: &str) -> Result<Vec<IndexedSymbol>> {
        let connection = self.open()?;
        let mut statement = connection
            .prepare_cached(
                "SELECT
                    symbol_id,
                    repo,
                    relative_path,
                    name,
                    kind,
                    container,
                    language,
                    start_line,
                    end_line,
                    indexed_at,
                    file_hash,
                    parent_symbol_id
                FROM symbols
                WHERE repo = ?1 AND relative_path = ?2
                ORDER BY start_line ASC, end_line ASC, name ASC",
            )
            .context("preparing outline query")?;
        let rows = statement
            .query_map(params![repo, relative_path], map_symbol_row)
            .context("querying file symbols")?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .context("collecting file symbols")
    }

    fn open(&self) -> Result<Connection> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating symbol db dir {}", parent.display()))?;
        }
        let connection = Connection::open(&self.path)
            .with_context(|| format!("opening symbol db {}", self.path.display()))?;
        migrate_symbol_schema(&connection)?;
        connection
            .execute_batch(
                "PRAGMA journal_mode = WAL;
                 PRAGMA synchronous = NORMAL;
                 CREATE TABLE IF NOT EXISTS symbols (
                    symbol_id TEXT NOT NULL,
                    repo TEXT NOT NULL,
                    relative_path TEXT NOT NULL,
                    name TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    container TEXT,
                    language TEXT NOT NULL,
                    start_line INTEGER NOT NULL,
                    end_line INTEGER NOT NULL,
                    indexed_at TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    parent_symbol_id TEXT,
                    PRIMARY KEY (repo, symbol_id)
                 );
                 CREATE INDEX IF NOT EXISTS idx_symbols_repo_path ON symbols(repo, relative_path, start_line, end_line);
                 CREATE INDEX IF NOT EXISTS idx_symbols_repo_name ON symbols(repo, name, kind);",
            )
            .context("initializing symbol db schema")?;
        Ok(connection)
    }
}

fn migrate_symbol_schema(connection: &Connection) -> Result<()> {
    connection
        .execute_batch("PRAGMA journal_mode = WAL; PRAGMA synchronous = NORMAL;")
        .context("initializing symbol db pragmas")?;

    let existing_sql = connection
        .query_row(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'symbols'",
            [],
            |row| row.get::<_, Option<String>>(0),
        )
        .optional()
        .context("loading symbol table schema")?
        .flatten();

    let Some(existing_sql) = existing_sql else {
        return Ok(());
    };
    if existing_sql.contains("PRIMARY KEY (repo, symbol_id)") {
        return Ok(());
    }

    connection
        .execute_batch(
            "BEGIN IMMEDIATE;
             ALTER TABLE symbols RENAME TO symbols_legacy;
             CREATE TABLE symbols (
                symbol_id TEXT NOT NULL,
                repo TEXT NOT NULL,
                relative_path TEXT NOT NULL,
                name TEXT NOT NULL,
                kind TEXT NOT NULL,
                container TEXT,
                language TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                indexed_at TEXT NOT NULL,
                file_hash TEXT NOT NULL,
                parent_symbol_id TEXT,
                PRIMARY KEY (repo, symbol_id)
             );
             INSERT INTO symbols (
                symbol_id,
                repo,
                relative_path,
                name,
                kind,
                container,
                language,
                start_line,
                end_line,
                indexed_at,
                file_hash,
                parent_symbol_id
             )
             SELECT
                symbol_id,
                repo,
                relative_path,
                name,
                kind,
                container,
                language,
                start_line,
                end_line,
                indexed_at,
                file_hash,
                parent_symbol_id
             FROM symbols_legacy;
             DROP TABLE symbols_legacy;
             COMMIT;",
        )
        .context("migrating symbol table schema")?;
    Ok(())
}

pub fn extract_symbols(
    repo: &str,
    relative_path: &str,
    path: &Path,
    text: &str,
    indexed_at: &str,
    file_hash: &str,
) -> Result<Vec<IndexedSymbol>> {
    let Some(config) = symbol_config(path) else {
        return Ok(Vec::new());
    };
    if should_skip_symbol_extraction(path, text, &config) {
        return Ok(Vec::new());
    }

    let mut parser = Parser::new();
    parser
        .set_language(&config.language)
        .context("setting tree-sitter language for symbol extraction")?;
    let Some(tree) = parser.parse(text, None) else {
        return Ok(Vec::new());
    };

    let mut output = Vec::new();
    let mut stack = Vec::new();
    let context = SymbolExtractionContext {
        text: text.as_bytes(),
        relative_path,
        repo,
        config: &config,
        indexed_at,
        file_hash,
    };
    collect_symbols(tree.root_node(), &context, &mut stack, &mut output)?;
    dedupe_symbols_in_place(&mut output);
    Ok(output)
}

pub fn build_outline(symbols: &[IndexedSymbol]) -> Vec<OutlineNode> {
    let mut by_parent: BTreeMap<Option<String>, Vec<IndexedSymbol>> = BTreeMap::new();
    for symbol in symbols {
        by_parent
            .entry(symbol.parent_symbol_id.clone())
            .or_default()
            .push(symbol.clone());
    }

    for children in by_parent.values_mut() {
        children.sort_by(|left, right| {
            left.start_line
                .cmp(&right.start_line)
                .then(left.end_line.cmp(&right.end_line))
                .then(left.name.cmp(&right.name))
        });
    }

    build_outline_children(None, &by_parent)
}

fn build_outline_children(
    parent: Option<&str>,
    by_parent: &BTreeMap<Option<String>, Vec<IndexedSymbol>>,
) -> Vec<OutlineNode> {
    by_parent
        .get(&parent.map(ToString::to_string))
        .into_iter()
        .flat_map(|symbols| symbols.iter())
        .map(|symbol| OutlineNode {
            symbol_id: symbol.symbol_id.clone(),
            name: symbol.name.clone(),
            kind: symbol.kind.clone(),
            container: symbol.container.clone(),
            language: symbol.language.clone(),
            start_line: symbol.start_line,
            end_line: symbol.end_line,
            children: build_outline_children(Some(&symbol.symbol_id), by_parent),
        })
        .collect()
}

fn collect_symbols(
    node: Node<'_>,
    context: &SymbolExtractionContext<'_>,
    stack: &mut Vec<(String, String)>,
    output: &mut Vec<IndexedSymbol>,
) -> Result<()> {
    let mut pushed = false;

    if let Some(kind) = context
        .config
        .rules
        .iter()
        .find(|rule| rule.node_kind == node.kind())
        .map(|rule| rule.symbol_kind)
        && should_index_symbol_node(node, context)
        && let Some(name) = extract_symbol_name(node, context.text)
    {
        let start_line = node.start_position().row as u64 + 1;
        let end_line = node.end_position().row as u64 + 1;
        let parent_symbol_id = stack.last().map(|(symbol_id, _)| symbol_id.clone());
        let container = (!stack.is_empty()).then(|| {
            stack
                .iter()
                .map(|(_, name)| name.as_str())
                .collect::<Vec<_>>()
                .join("::")
        });
        let symbol_id = symbol_id(context.relative_path, kind, &name, start_line, end_line);
        output.push(IndexedSymbol {
            symbol_id: symbol_id.clone(),
            repo: context.repo.to_string(),
            relative_path: context.relative_path.to_string(),
            name: name.clone(),
            kind: kind.to_string(),
            container,
            language: context.config.language_name.to_string(),
            start_line,
            end_line,
            indexed_at: context.indexed_at.to_string(),
            file_hash: context.file_hash.to_string(),
            parent_symbol_id,
        });
        stack.push((symbol_id, name));
        pushed = true;
    }

    let mut cursor = node.walk();
    for child in node.children(&mut cursor).filter(|child| child.is_named()) {
        collect_symbols(child, context, stack, output)?;
    }

    if pushed {
        let _ = stack.pop();
    }

    Ok(())
}

fn dedupe_symbols_in_place(symbols: &mut Vec<IndexedSymbol>) {
    let mut seen = BTreeSet::new();
    symbols.retain(|symbol| seen.insert(symbol.symbol_id.clone()));
}

fn should_skip_symbol_extraction(path: &Path, text: &str, config: &SymbolConfig) -> bool {
    match config.language_name {
        "javascript" | "typescript" => is_probably_minified_javascript_like(path, text),
        _ => false,
    }
}

fn is_probably_minified_javascript_like(path: &Path, text: &str) -> bool {
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or_default();
    if !matches!(extension, "js" | "jsx" | "ts" | "tsx") {
        return false;
    }

    let line_count = text.lines().take(21).count();
    if line_count == 0 {
        return false;
    }

    let longest_line = text.lines().map(str::len).max().unwrap_or_default();
    (line_count <= 5 && longest_line >= 2_000) || (line_count <= 20 && longest_line >= 4_000)
}

fn should_index_symbol_node(node: Node<'_>, context: &SymbolExtractionContext<'_>) -> bool {
    match context.config.language_name {
        "cpp" => should_index_cpp_symbol_node(node),
        _ => true,
    }
}

fn should_index_cpp_symbol_node(node: Node<'_>) -> bool {
    match node.kind() {
        "struct_specifier" | "class_specifier" => {
            node.child_by_field_name("body").is_some()
                || has_named_child_kind(node, "field_declaration_list")
        }
        "enum_specifier" => {
            node.child_by_field_name("body").is_some()
                || has_named_child_kind(node, "enumerator_list")
        }
        _ => true,
    }
}

fn has_named_child_kind(node: Node<'_>, kind: &str) -> bool {
    let mut cursor = node.walk();
    node.children(&mut cursor)
        .filter(|child| child.is_named())
        .any(|child| child.kind() == kind)
}

fn extract_symbol_name(node: Node<'_>, text: &[u8]) -> Option<String> {
    for field in ["name", "declarator", "type", "function", "body"] {
        if let Some(child) = node.child_by_field_name(field)
            && let Some(name) = extract_name_from_node(child, text, 0)
        {
            return Some(name);
        }
    }
    extract_name_from_node(node, text, 0)
}

fn extract_name_from_node(node: Node<'_>, text: &[u8], depth: usize) -> Option<String> {
    if depth > 6 {
        return None;
    }

    if IDENTIFIER_NODE_KINDS.contains(&node.kind()) {
        let raw = node.utf8_text(text).ok()?.trim();
        if raw.is_empty() || raw.len() > 128 {
            return None;
        }
        return Some(raw.to_string());
    }

    let mut cursor = node.walk();
    for child in node.children(&mut cursor).filter(|child| child.is_named()) {
        if let Some(name) = extract_name_from_node(child, text, depth + 1) {
            return Some(name);
        }
    }
    None
}

fn symbol_id(
    relative_path: &str,
    kind: &str,
    name: &str,
    start_line: u64,
    end_line: u64,
) -> String {
    let digest =
        xxh3_128(format!("{relative_path}:{kind}:{name}:{start_line}:{end_line}").as_bytes());
    format!("sym_{digest:032x}").chars().take(22).collect()
}

fn symbol_config(path: &Path) -> Option<SymbolConfig> {
    let extension = path.extension().and_then(|value| value.to_str())?;
    match extension {
        "js" | "jsx" => Some(SymbolConfig {
            language: tree_sitter_javascript::LANGUAGE.into(),
            language_name: "javascript",
            rules: JS_SYMBOLS,
        }),
        "ts" => Some(SymbolConfig {
            language: tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
            language_name: "typescript",
            rules: JS_SYMBOLS,
        }),
        "tsx" => Some(SymbolConfig {
            language: tree_sitter_typescript::LANGUAGE_TSX.into(),
            language_name: "typescript",
            rules: JS_SYMBOLS,
        }),
        "py" => Some(SymbolConfig {
            language: tree_sitter_python::LANGUAGE.into(),
            language_name: "python",
            rules: PYTHON_SYMBOLS,
        }),
        "java" => Some(SymbolConfig {
            language: tree_sitter_java::LANGUAGE.into(),
            language_name: "java",
            rules: JAVA_SYMBOLS,
        }),
        "cpp" | "c" | "hpp" | "h" => Some(SymbolConfig {
            language: tree_sitter_cpp::LANGUAGE.into(),
            language_name: "cpp",
            rules: CPP_SYMBOLS,
        }),
        "go" => Some(SymbolConfig {
            language: tree_sitter_go::LANGUAGE.into(),
            language_name: "go",
            rules: GO_SYMBOLS,
        }),
        "rs" => Some(SymbolConfig {
            language: tree_sitter_rust::LANGUAGE.into(),
            language_name: "rust",
            rules: RUST_SYMBOLS,
        }),
        "cs" => Some(SymbolConfig {
            language: tree_sitter_c_sharp::LANGUAGE.into(),
            language_name: "csharp",
            rules: CSHARP_SYMBOLS,
        }),
        "scala" => Some(SymbolConfig {
            language: tree_sitter_scala::LANGUAGE.into(),
            language_name: "scala",
            rules: SCALA_SYMBOLS,
        }),
        "kt" => Some(SymbolConfig {
            language: tree_sitter_kotlin_ng::LANGUAGE.into(),
            language_name: "kotlin",
            rules: KOTLIN_SYMBOLS,
        }),
        "php" => Some(SymbolConfig {
            language: tree_sitter_php::LANGUAGE_PHP.into(),
            language_name: "php",
            rules: PHP_SYMBOLS,
        }),
        "rb" => Some(SymbolConfig {
            language: tree_sitter_ruby::LANGUAGE.into(),
            language_name: "ruby",
            rules: RUBY_SYMBOLS,
        }),
        "swift" => Some(SymbolConfig {
            language: tree_sitter_swift::LANGUAGE.into(),
            language_name: "swift",
            rules: SWIFT_SYMBOLS,
        }),
        _ => None,
    }
}

fn map_symbol_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<IndexedSymbol> {
    Ok(IndexedSymbol {
        symbol_id: row.get(0)?,
        repo: row.get(1)?,
        relative_path: row.get(2)?,
        name: row.get(3)?,
        kind: row.get(4)?,
        container: row.get(5)?,
        language: row.get(6)?,
        start_line: row.get::<_, i64>(7)? as u64,
        end_line: row.get::<_, i64>(8)? as u64,
        indexed_at: row.get(9)?,
        file_hash: row.get(10)?,
        parent_symbol_id: row.get(11)?,
    })
}

#[cfg(test)]
mod tests {
    use super::{IndexedSymbol, SymbolStore, build_outline, extract_symbols};
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_path(name: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time went backwards")
            .as_nanos();
        std::env::temp_dir().join(format!("agent-context-symbols-{name}-{nanos}"))
    }

    #[test]
    fn extracts_nested_rust_symbols() {
        let path = std::path::Path::new("src/lib.rs");
        let text = r#"
            pub mod graphql {
                pub struct Schema;
                impl Schema {
                    pub fn build() {}
                }
            }
        "#;
        let symbols = extract_symbols(
            "/tmp/repo",
            "src/lib.rs",
            path,
            text,
            "2026-01-01T00:00:00Z",
            "abc",
        )
        .unwrap();

        let names = symbols
            .iter()
            .map(|symbol| symbol.name.as_str())
            .collect::<Vec<_>>();
        assert!(names.contains(&"graphql"));
        assert!(names.contains(&"Schema"));
        assert!(names.contains(&"build"));

        let outline = build_outline(&symbols);
        assert!(!outline.is_empty());
    }

    #[test]
    fn persists_and_reads_file_symbols() {
        let db_path = temp_path("db");
        let store = SymbolStore::new(db_path.clone());
        let symbols = extract_symbols(
            "/tmp/repo",
            "src/main.kt",
            std::path::Path::new("src/main.kt"),
            "class Engine { fun search() {} }",
            "2026-01-01T00:00:00Z",
            "hash1",
        )
        .unwrap();

        store
            .replace_file_symbols("/tmp/repo", "src/main.kt", &symbols)
            .unwrap();
        let loaded = store.file_symbols("/tmp/repo", "src/main.kt").unwrap();
        assert_eq!(loaded.len(), symbols.len());

        let ids = loaded
            .iter()
            .map(|symbol| symbol.symbol_id.clone())
            .collect::<Vec<_>>();
        let by_id = store.symbols_by_repo_and_ids("/tmp/repo", &ids).unwrap();
        assert_eq!(by_id.len(), symbols.len());

        let _ = fs::remove_file(db_path);
    }

    #[test]
    fn repo_scoped_symbol_ids_do_not_collide() {
        let db_path = temp_path("repo-scoped");
        let store = SymbolStore::new(db_path.clone());
        let left = extract_symbols(
            "/tmp/repo-a",
            "src/main.kt",
            std::path::Path::new("src/main.kt"),
            "class Engine { fun search() {} }",
            "2026-01-01T00:00:00Z",
            "hash-a",
        )
        .unwrap();
        let mut right = extract_symbols(
            "/tmp/repo-b",
            "src/main.kt",
            std::path::Path::new("src/main.kt"),
            "class Engine { fun search() {} }",
            "2026-01-01T00:00:00Z",
            "hash-b",
        )
        .unwrap();

        let duplicate_id = left[0].symbol_id.clone();
        right[0].symbol_id = duplicate_id.clone();

        store
            .replace_file_symbols("/tmp/repo-a", "src/main.kt", &left)
            .unwrap();
        store
            .replace_file_symbols("/tmp/repo-b", "src/main.kt", &right)
            .unwrap();

        let left_symbol = store.symbol_by_id("/tmp/repo-a", &duplicate_id).unwrap();
        let right_symbol = store.symbol_by_id("/tmp/repo-b", &duplicate_id).unwrap();

        assert_eq!(left_symbol.unwrap().file_hash, "hash-a");
        assert_eq!(right_symbol.unwrap().file_hash, "hash-b");

        let _ = fs::remove_file(db_path);
    }

    #[test]
    fn c_struct_references_are_not_indexed_as_definitions() {
        let path = std::path::Path::new("spellfix.c");
        let text = r#"struct spellfix1_row { int iScore; };
static int spellfix1_row_compare(const void *A, const void *B){
  const struct spellfix1_row *a = (const struct spellfix1_row*)A;
  const struct spellfix1_row *b = (const struct spellfix1_row*)B;
  return a->iScore - b->iScore;
}
"#;
        let symbols = extract_symbols(
            "/tmp/repo",
            "spellfix.c",
            path,
            text,
            "2026-01-01T00:00:00Z",
            "hash-c",
        )
        .unwrap();

        let spellfix_structs = symbols
            .iter()
            .filter(|symbol| symbol.name == "spellfix1_row" && symbol.kind == "struct")
            .collect::<Vec<_>>();
        assert_eq!(spellfix_structs.len(), 1);
        assert_eq!(spellfix_structs[0].start_line, 1);
        assert_eq!(spellfix_structs[0].end_line, 1);
    }

    #[test]
    fn replace_file_symbols_skips_duplicate_ids_in_same_batch() {
        let db_path = temp_path("duplicates");
        let store = SymbolStore::new(db_path.clone());
        let symbol = IndexedSymbol {
            symbol_id: "sym_duplicate".to_string(),
            repo: "/tmp/repo".to_string(),
            relative_path: "src/main.rs".to_string(),
            name: "duplicate".to_string(),
            kind: "function".to_string(),
            container: None,
            language: "rust".to_string(),
            start_line: 10,
            end_line: 10,
            indexed_at: "2026-01-01T00:00:00Z".to_string(),
            file_hash: "hash-1".to_string(),
            parent_symbol_id: None,
        };

        store
            .replace_file_symbols("/tmp/repo", "src/main.rs", &[symbol.clone(), symbol])
            .unwrap();

        let loaded = store.file_symbols("/tmp/repo", "src/main.rs").unwrap();
        assert_eq!(loaded.len(), 1);

        let _ = fs::remove_file(db_path);
    }

    #[test]
    fn minified_javascript_bundle_skips_symbol_extraction() {
        let path = std::path::Path::new("bundle.min.js");
        let text = format!(
            "class A{{constructor(){{}}render(){{}}}}class B{{constructor(){{}}render(){{}}}}{}",
            "x".repeat(4_100)
        );
        let symbols = extract_symbols(
            "/tmp/repo",
            "bundle.min.js",
            path,
            &text,
            "2026-01-01T00:00:00Z",
            "hash-js",
        )
        .unwrap();

        assert!(symbols.is_empty());
    }
}
