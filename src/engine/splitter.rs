use anyhow::{Context, Result};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use text_splitter::{
    ChunkCharIndex, ChunkConfig, CodeSplitter as SemanticCodeSplitter, MarkdownSplitter,
    TextSplitter,
};
use tree_sitter::{Language, Node, Parser};

const CHUNK_SIZE: usize = 2500;
const CHUNK_OVERLAP: usize = 300;
const LANGCHAIN_CHUNK_SIZE: usize = 1000;
const LANGCHAIN_CHUNK_OVERLAP: usize = 200;

const JS_NODES: &[&str] = &[
    "function_declaration",
    "arrow_function",
    "class_declaration",
    "method_definition",
    "export_statement",
];
const TS_NODES: &[&str] = &[
    "function_declaration",
    "arrow_function",
    "class_declaration",
    "method_definition",
    "export_statement",
    "interface_declaration",
    "type_alias_declaration",
];
const PYTHON_NODES: &[&str] = &[
    "function_definition",
    "class_definition",
    "decorated_definition",
    "async_function_definition",
];
const JAVA_NODES: &[&str] = &[
    "method_declaration",
    "class_declaration",
    "interface_declaration",
    "constructor_declaration",
];
const CPP_NODES: &[&str] = &[
    "function_definition",
    "class_specifier",
    "namespace_definition",
    "declaration",
];
const GO_NODES: &[&str] = &[
    "function_declaration",
    "method_declaration",
    "type_declaration",
    "var_declaration",
    "const_declaration",
];
const RUST_NODES: &[&str] = &[
    "function_item",
    "impl_item",
    "struct_item",
    "enum_item",
    "trait_item",
    "mod_item",
];
const CSHARP_NODES: &[&str] = &[
    "method_declaration",
    "class_declaration",
    "interface_declaration",
    "struct_declaration",
    "enum_declaration",
];
const SCALA_NODES: &[&str] = &[
    "function_definition",
    "class_definition",
    "trait_definition",
    "object_definition",
];
const KOTLIN_NODES: &[&str] = &[
    "function_declaration",
    "class_declaration",
    "object_declaration",
    "property_declaration",
    "secondary_constructor",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitterKind {
    Ast,
    LangChain,
}

#[derive(Debug, Clone)]
pub struct CodeChunk {
    pub content: String,
    pub file_path: PathBuf,
    pub language: String,
    pub start_line: u64,
    pub end_line: u64,
}

pub fn default_supported_extensions() -> BTreeSet<String> {
    [
        ".ts",
        ".tsx",
        ".js",
        ".jsx",
        ".py",
        ".java",
        ".cpp",
        ".c",
        ".h",
        ".hpp",
        ".cs",
        ".go",
        ".rs",
        ".php",
        ".rb",
        ".swift",
        ".kt",
        ".scala",
        ".m",
        ".mm",
        ".md",
        ".markdown",
        ".ipynb",
    ]
    .into_iter()
    .map(ToString::to_string)
    .collect()
}

pub fn split_text(path: &Path, text: &str, kind: SplitterKind) -> Result<Vec<CodeChunk>> {
    let extension = extension_for(path);
    let language = language_for_extension(&extension).to_string();

    if matches!(kind, SplitterKind::Ast)
        && let Some(config) = ast_config(path)
    {
        let mut parser = Parser::new();
        if parser.set_language(&config.language).is_ok()
            && let Some(tree) = parser.parse(text, None)
        {
            let mut chunks = Vec::new();
            collect_ast_chunks(
                tree.root_node(),
                text,
                config.node_types,
                &mut chunks,
                path,
                config.language_name,
            );
            if !chunks.is_empty() {
                return Ok(add_overlap(refine_chunks(
                    chunks,
                    path,
                    config.language_name,
                )));
            }
        }
    }

    langchain_chunks(path, text, &language)
}

pub fn language_for_extension(extension: &str) -> &'static str {
    match extension {
        ".ts" | ".tsx" => "typescript",
        ".js" | ".jsx" => "javascript",
        ".py" => "python",
        ".java" => "java",
        ".cpp" | ".hpp" => "cpp",
        ".c" | ".h" => "c",
        ".go" => "go",
        ".rs" => "rust",
        ".cs" => "csharp",
        ".scala" => "scala",
        ".php" => "php",
        ".rb" => "ruby",
        ".swift" => "swift",
        ".kt" => "kotlin",
        ".m" | ".mm" => "objective-c",
        ".ipynb" => "jupyter",
        ".md" | ".markdown" => "markdown",
        _ => "text",
    }
}

fn extension_for(path: &Path) -> String {
    match path.extension().and_then(|value| value.to_str()) {
        Some(ext) => format!(".{ext}"),
        None => String::new(),
    }
}

fn langchain_chunks(path: &Path, text: &str, language: &str) -> Result<Vec<CodeChunk>> {
    let config = ChunkConfig::new(LANGCHAIN_CHUNK_SIZE)
        .with_overlap(LANGCHAIN_CHUNK_OVERLAP)
        .context("building text-splitter chunk config")?;
    let line_starts = line_start_offsets(text);

    if language == "markdown" {
        let splitter = MarkdownSplitter::new(config);
        return Ok(build_chunks_from_indices(
            splitter.chunk_char_indices(text),
            path,
            language,
            &line_starts,
        ));
    }

    if let Some(ts_language) = semantic_language_for_path(path) {
        let splitter = SemanticCodeSplitter::new(ts_language, config)
            .context("creating semantic code splitter")?;
        return Ok(build_chunks_from_indices(
            splitter.chunk_char_indices(text),
            path,
            language,
            &line_starts,
        ));
    }

    let splitter = TextSplitter::new(config);
    Ok(build_chunks_from_indices(
        splitter.chunk_char_indices(text),
        path,
        language,
        &line_starts,
    ))
}

struct AstConfig {
    language: Language,
    node_types: &'static [&'static str],
    language_name: &'static str,
}

fn ast_config(path: &Path) -> Option<AstConfig> {
    let ext = extension_for(path);
    match ext.as_str() {
        ".js" | ".jsx" => Some(AstConfig {
            language: tree_sitter_javascript::LANGUAGE.into(),
            node_types: JS_NODES,
            language_name: "javascript",
        }),
        ".ts" => Some(AstConfig {
            language: tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
            node_types: TS_NODES,
            language_name: "typescript",
        }),
        ".tsx" => Some(AstConfig {
            language: tree_sitter_typescript::LANGUAGE_TSX.into(),
            node_types: TS_NODES,
            language_name: "typescript",
        }),
        ".py" => Some(AstConfig {
            language: tree_sitter_python::LANGUAGE.into(),
            node_types: PYTHON_NODES,
            language_name: "python",
        }),
        ".java" => Some(AstConfig {
            language: tree_sitter_java::LANGUAGE.into(),
            node_types: JAVA_NODES,
            language_name: "java",
        }),
        ".cpp" | ".c" | ".hpp" | ".h" => Some(AstConfig {
            language: tree_sitter_cpp::LANGUAGE.into(),
            node_types: CPP_NODES,
            language_name: "cpp",
        }),
        ".go" => Some(AstConfig {
            language: tree_sitter_go::LANGUAGE.into(),
            node_types: GO_NODES,
            language_name: "go",
        }),
        ".rs" => Some(AstConfig {
            language: tree_sitter_rust::LANGUAGE.into(),
            node_types: RUST_NODES,
            language_name: "rust",
        }),
        ".cs" => Some(AstConfig {
            language: tree_sitter_c_sharp::LANGUAGE.into(),
            node_types: CSHARP_NODES,
            language_name: "csharp",
        }),
        ".scala" => Some(AstConfig {
            language: tree_sitter_scala::LANGUAGE.into(),
            node_types: SCALA_NODES,
            language_name: "scala",
        }),
        ".kt" => Some(AstConfig {
            language: tree_sitter_kotlin_ng::LANGUAGE.into(),
            node_types: KOTLIN_NODES,
            language_name: "kotlin",
        }),
        _ => None,
    }
}

fn semantic_language_for_path(path: &Path) -> Option<Language> {
    match extension_for(path).as_str() {
        ".js" | ".jsx" => Some(tree_sitter_javascript::LANGUAGE.into()),
        ".ts" => Some(tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()),
        ".tsx" => Some(tree_sitter_typescript::LANGUAGE_TSX.into()),
        ".py" => Some(tree_sitter_python::LANGUAGE.into()),
        ".java" => Some(tree_sitter_java::LANGUAGE.into()),
        ".cpp" | ".c" | ".hpp" | ".h" => Some(tree_sitter_cpp::LANGUAGE.into()),
        ".go" => Some(tree_sitter_go::LANGUAGE.into()),
        ".rs" => Some(tree_sitter_rust::LANGUAGE.into()),
        ".cs" => Some(tree_sitter_c_sharp::LANGUAGE.into()),
        ".scala" => Some(tree_sitter_scala::LANGUAGE.into()),
        ".kt" => Some(tree_sitter_kotlin_ng::LANGUAGE.into()),
        ".php" => Some(tree_sitter_php::LANGUAGE_PHP.into()),
        ".rb" => Some(tree_sitter_ruby::LANGUAGE.into()),
        ".swift" => Some(tree_sitter_swift::LANGUAGE.into()),
        _ => None,
    }
}

fn build_chunks_from_indices<'text>(
    indices: impl Iterator<Item = ChunkCharIndex<'text>>,
    path: &Path,
    language: &str,
    line_starts: &[usize],
) -> Vec<CodeChunk> {
    indices
        .filter_map(|chunk| {
            if chunk.chunk.trim().is_empty() {
                return None;
            }

            let start_line = line_number_for_offset(line_starts, chunk.byte_offset);
            let end_offset = chunk.byte_offset + chunk.chunk.len().saturating_sub(1);
            let end_line = line_number_for_offset(line_starts, end_offset);

            Some(CodeChunk {
                content: chunk.chunk.to_string(),
                file_path: path.to_path_buf(),
                language: language.to_string(),
                start_line,
                end_line,
            })
        })
        .collect()
}

fn line_start_offsets(text: &str) -> Vec<usize> {
    let mut offsets = vec![0usize];
    for (index, byte) in text.bytes().enumerate() {
        if byte == b'\n' && index + 1 < text.len() {
            offsets.push(index + 1);
        }
    }
    offsets
}

fn line_number_for_offset(line_starts: &[usize], byte_offset: usize) -> u64 {
    match line_starts.binary_search(&byte_offset) {
        Ok(index) => index as u64 + 1,
        Err(index) => index as u64,
    }
}

fn collect_ast_chunks(
    node: Node<'_>,
    text: &str,
    node_types: &[&str],
    chunks: &mut Vec<CodeChunk>,
    path: &Path,
    language: &str,
) {
    if node_types.contains(&node.kind()) {
        let snippet = &text[node.byte_range()];
        if !snippet.trim().is_empty() {
            chunks.push(CodeChunk {
                content: snippet.to_string(),
                file_path: path.to_path_buf(),
                language: language.to_string(),
                start_line: node.start_position().row as u64 + 1,
                end_line: node.end_position().row as u64 + 1,
            });
        }
    }

    let child_count = node.child_count();
    for index in 0..child_count {
        if let Some(child) = node.child(index as u32) {
            collect_ast_chunks(child, text, node_types, chunks, path, language);
        }
    }
}

fn refine_chunks(chunks: Vec<CodeChunk>, path: &Path, language: &str) -> Vec<CodeChunk> {
    let mut refined = Vec::new();
    for chunk in chunks {
        if chunk.content.len() <= CHUNK_SIZE {
            refined.push(chunk);
            continue;
        }

        let mut current = String::new();
        let mut current_start = chunk.start_line;
        let mut line_count = 0u64;
        for line in chunk.content.lines() {
            let rendered = format!("{line}\n");
            if !current.is_empty() && current.len() + rendered.len() > CHUNK_SIZE {
                refined.push(CodeChunk {
                    content: current.trim_end().to_string(),
                    file_path: path.to_path_buf(),
                    language: language.to_string(),
                    start_line: current_start,
                    end_line: current_start + line_count.saturating_sub(1),
                });
                current.clear();
                current_start += line_count;
                line_count = 0;
            }
            current.push_str(&rendered);
            line_count += 1;
        }
        if !current.trim().is_empty() {
            refined.push(CodeChunk {
                content: current.trim_end().to_string(),
                file_path: path.to_path_buf(),
                language: language.to_string(),
                start_line: current_start,
                end_line: current_start + line_count.saturating_sub(1),
            });
        }
    }
    refined
}

fn add_overlap(chunks: Vec<CodeChunk>) -> Vec<CodeChunk> {
    if chunks.len() <= 1 || CHUNK_OVERLAP == 0 {
        return chunks;
    }

    let mut output = Vec::with_capacity(chunks.len());
    for (index, chunk) in chunks.iter().enumerate() {
        if index == 0 {
            output.push(chunk.clone());
            continue;
        }

        let previous = &chunks[index - 1];
        let overlap = previous
            .content
            .chars()
            .rev()
            .take(CHUNK_OVERLAP)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<String>();
        let overlap_lines = overlap.lines().count() as u64;

        output.push(CodeChunk {
            content: format!("{overlap}\n{}", chunk.content),
            file_path: chunk.file_path.clone(),
            language: chunk.language.clone(),
            start_line: chunk.start_line.saturating_sub(overlap_lines),
            end_line: chunk.end_line,
        });
    }
    output
}

#[cfg(test)]
mod tests {
    use super::{SplitterKind, split_text};
    use std::path::Path;

    fn repeated_block(prefix: &str, count: usize) -> String {
        (0..count)
            .map(|index| format!("{prefix}_{index} {}\n", "x".repeat(120)))
            .collect::<Vec<_>>()
            .join("")
    }

    #[test]
    fn ast_mode_falls_back_for_php() {
        let code = format!(
            "<?php\n{}\nfunction real_code() {{ return 1; }}\n",
            repeated_block("$value", 40)
        );
        let chunks = split_text(Path::new("example.php"), &code, SplitterKind::Ast).unwrap();

        assert!(!chunks.is_empty());
        assert!(chunks.iter().all(|chunk| chunk.language == "php"));
        assert!(
            chunks
                .iter()
                .any(|chunk| chunk.content.contains("real_code"))
        );
    }

    #[test]
    fn langchain_mode_supports_ruby() {
        let code = format!(
            "{}\ndef greet(name)\n  \"hello #{{name}}\"\nend\n",
            repeated_block("value", 40)
        );
        let chunks = split_text(Path::new("example.rb"), &code, SplitterKind::LangChain).unwrap();

        assert!(!chunks.is_empty());
        assert!(chunks.iter().all(|chunk| chunk.language == "ruby"));
        assert!(chunks.iter().any(|chunk| chunk.content.contains("greet")));
    }

    #[test]
    fn langchain_mode_supports_swift() {
        let code = format!(
            "{}\nfunc greet(name: String) -> String {{\n    return \"hello \\(name)\"\n}}\n",
            repeated_block("let value", 40)
        );
        let chunks =
            split_text(Path::new("example.swift"), &code, SplitterKind::LangChain).unwrap();

        assert!(!chunks.is_empty());
        assert!(chunks.iter().all(|chunk| chunk.language == "swift"));
        assert!(
            chunks
                .iter()
                .any(|chunk| chunk.content.contains("func greet"))
        );
    }

    #[test]
    fn ast_mode_supports_kotlin() {
        let code = format!(
            "{}\nclass Greeter {{\n    fun greet(name: String): String {{\n        return \"hello $name\"\n    }}\n}}\n",
            repeated_block("val value", 40)
        );
        let chunks = split_text(Path::new("example.kt"), &code, SplitterKind::Ast).unwrap();

        assert!(!chunks.is_empty());
        assert!(chunks.iter().all(|chunk| chunk.language == "kotlin"));
        assert!(
            chunks
                .iter()
                .any(|chunk| chunk.content.contains("fun greet"))
        );
    }
}
