use anyhow::{Context, Result, bail};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::process::Command;

use super::impact::{ImpactEvidence, ImpactRequest, ImpactResponse};
use super::relationships::RepoRelationshipCoverage;
use super::symbols::{IndexedSymbol, extract_symbols};
use super::{Engine, ResolvedScope, run_low_priority_blocking};

#[derive(Debug, Clone)]
pub struct AnalyzeChangesRequest {
    pub repo: Option<String>,
    pub base_ref: String,
    pub include_untracked: bool,
    pub max_depth: usize,
    pub max_nodes: usize,
    pub include_tests: bool,
    pub min_confidence: u64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ChangeAnalysisResponse {
    pub scope: String,
    pub repo: String,
    pub base_ref: String,
    pub resolved_base: Option<String>,
    pub needs_index: bool,
    pub invalid_base: bool,
    pub reason: Option<String>,
    pub files: Vec<ChangedFile>,
    pub symbols: Vec<ChangedSymbol>,
    pub impacts: Vec<ChangedSymbolImpact>,
    pub coverage: Option<RepoRelationshipCoverage>,
    pub truncated: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ChangedFile {
    pub status: String,
    pub old_path: Option<String>,
    pub path: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ChangedSymbol {
    pub change: String,
    pub old: Option<IndexedSymbol>,
    pub current: Option<IndexedSymbol>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ChangedSymbolImpact {
    pub change: String,
    pub logical_key: String,
    pub qualified_name: String,
    pub analysis: Option<ImpactResponse>,
    pub possible_evidence: Vec<ImpactEvidence>,
    pub note: Option<String>,
}

struct GitChanges {
    resolved_base: String,
    files: Vec<ChangedFile>,
}

impl Engine {
    pub async fn analyze_changes(
        &self,
        scope: ResolvedScope,
        request: AnalyzeChangesRequest,
    ) -> Result<ChangeAnalysisResponse> {
        let repo = self
            .resolve_prepare_repo(&scope, request.repo.as_deref(), true)?
            .context("change analysis requires one repository")?;
        let repo_text = repo.display().to_string();
        let base_ref = if request.base_ref.trim().is_empty() {
            "HEAD".to_string()
        } else {
            request.base_ref.clone()
        };
        let repo_for_git = repo.clone();
        let base_for_git = base_ref.clone();
        let include_untracked = request.include_untracked;
        let git = match run_low_priority_blocking("analyze_git_changes", move || {
            collect_git_changes(&repo_for_git, &base_for_git, include_untracked)
        })
        .await
        {
            Ok(changes) => changes,
            Err(error) => {
                return Ok(ChangeAnalysisResponse {
                    scope: scope.id,
                    repo: repo_text,
                    base_ref,
                    resolved_base: None,
                    needs_index: false,
                    invalid_base: true,
                    reason: Some(error.to_string()),
                    files: Vec::new(),
                    symbols: Vec::new(),
                    impacts: Vec::new(),
                    coverage: None,
                    truncated: false,
                });
            }
        };

        let mut coverage = self
            .relationship_coverage(scope.clone(), Some(&repo_text))
            .await?
            .into_iter()
            .next();
        if coverage
            .as_ref()
            .is_none_or(|coverage| coverage.graph_status != "ready")
        {
            return Ok(ChangeAnalysisResponse {
                scope: scope.id,
                repo: repo_text,
                base_ref,
                resolved_base: Some(git.resolved_base),
                needs_index: true,
                invalid_base: false,
                reason: Some(
                    "the current relationship index does not match the working tree; refresh the index before change analysis"
                        .to_string(),
                ),
                files: git.files,
                symbols: Vec::new(),
                impacts: Vec::new(),
                coverage,
                truncated: false,
            });
        }

        let repo_for_parse = repo.clone();
        let resolved_base = git.resolved_base.clone();
        let files_for_parse = git.files.clone();
        let symbols = run_low_priority_blocking("parse_changed_symbols", move || {
            changed_symbols(&repo_for_parse, &resolved_base, &files_for_parse)
        })
        .await?;

        let max_impacts = request.max_nodes.clamp(1, 250);
        let current_roots = symbols
            .iter()
            .take(max_impacts)
            .filter_map(|change| change.current.clone())
            .collect::<Vec<_>>();
        let impact_request = ImpactRequest {
            repo: Some(repo_text.clone()),
            symbol_id: None,
            file: None,
            line: None,
            max_depth: request.max_depth,
            max_nodes: request.max_nodes,
            include_tests: request.include_tests,
            min_confidence: request.min_confidence,
            include_possible: false,
        };
        let analyses = self
            .analyze_impacts_for_symbols(scope.id.clone(), &repo, &current_roots, &impact_request)
            .await?;
        let mut analyses_by_key = current_roots
            .iter()
            .map(|symbol| symbol.logical_key.clone())
            .zip(analyses)
            .collect::<BTreeMap<_, _>>();
        let mut impacts = Vec::new();
        let mut truncated = false;
        for symbol_change in &symbols {
            if impacts.len() >= max_impacts {
                truncated = true;
                break;
            }
            let selected = symbol_change
                .current
                .as_ref()
                .or(symbol_change.old.as_ref())
                .expect("classified symbol change has one side");
            if let Some(current) = symbol_change.current.as_ref() {
                let analysis = analyses_by_key
                    .remove(&current.logical_key)
                    .context("missing batched impact analysis")?;
                let possible_evidence = if let Some(old) = symbol_change
                    .old
                    .as_ref()
                    .filter(|old| old.logical_key != current.logical_key)
                {
                    self.change_candidate_evidence(&repo, &old.qualified_name, &old.name)
                        .await?
                } else {
                    Vec::new()
                };
                impacts.push(ChangedSymbolImpact {
                    change: symbol_change.change.clone(),
                    logical_key: selected.logical_key.clone(),
                    qualified_name: selected.qualified_name.clone(),
                    analysis: Some(analysis),
                    note: (!possible_evidence.is_empty()).then(|| {
                        "references to the previous declaration identity remain possible breakage evidence"
                            .to_string()
                    }),
                    possible_evidence,
                });
            } else {
                let possible_evidence = self
                    .change_candidate_evidence(&repo, &selected.qualified_name, &selected.name)
                    .await?;
                impacts.push(ChangedSymbolImpact {
                    change: symbol_change.change.clone(),
                    logical_key: selected.logical_key.clone(),
                    qualified_name: selected.qualified_name.clone(),
                    analysis: None,
                    note: Some(if possible_evidence.is_empty() {
                        "deleted declarations have no current graph node and no qualified-name candidates were found"
                            .to_string()
                    } else {
                        "deleted declaration impact is possible evidence from qualified-name and fuzzy relationship lookup, not a definite structural edge"
                            .to_string()
                    }),
                    possible_evidence,
                });
            }
        }
        if let Some(value) = coverage.as_mut() {
            value.repo = repo_text.clone();
        }
        Ok(ChangeAnalysisResponse {
            scope: scope.id,
            repo: repo_text,
            base_ref,
            resolved_base: Some(git.resolved_base),
            needs_index: false,
            invalid_base: false,
            reason: None,
            files: git.files,
            symbols,
            impacts,
            coverage,
            truncated,
        })
    }

    async fn change_candidate_evidence(
        &self,
        repo: &Path,
        qualified_name: &str,
        name: &str,
    ) -> Result<Vec<ImpactEvidence>> {
        let ctx = self.repo_context(repo)?;
        let mut sources = Vec::new();
        if let Some(overlay) = ctx.overlay.as_ref() {
            let state = self.load_overlay_state(overlay).await?;
            sources.push((overlay.storage_root.clone(), BTreeSet::new()));
            sources.push((
                ctx.canonical_root.clone(),
                state
                    .as_ref()
                    .map(super::overlay_suppressed_paths)
                    .unwrap_or_default(),
            ));
        } else {
            sources.push((ctx.canonical_root, BTreeSet::new()));
        }
        let mut output = Vec::new();
        let mut seen = BTreeSet::new();
        for (storage_repo, suppressed) in sources {
            let index = self.inner.local_index.clone();
            let query = if qualified_name.is_empty() {
                name.to_string()
            } else {
                qualified_name.to_string()
            };
            let relations = self
                .run_search_lexical_blocking("change_relationship_candidates", move || {
                    index.search_relations(&storage_repo, &query, 25)
                })
                .await?;
            for relation in relations {
                if suppressed.contains(&relation.source_path)
                    || !relation.target_name.eq_ignore_ascii_case(name)
                    || !seen.insert((relation.reference_id.clone(), relation.target_key.clone()))
                {
                    continue;
                }
                output.push(ImpactEvidence {
                    relation: relation.kind,
                    confidence: relation.confidence,
                    resolution: if relation.confidence >= 650 {
                        relation.resolution
                    } else {
                        format!("possible_{}", relation.resolution)
                    },
                    source_path: relation.source_path,
                    start_line: relation.start_line,
                    end_line: relation.end_line,
                    text: relation.evidence,
                });
            }
        }
        output.sort_by(|left, right| {
            right
                .confidence
                .cmp(&left.confidence)
                .then(left.source_path.cmp(&right.source_path))
                .then(left.start_line.cmp(&right.start_line))
        });
        output.truncate(25);
        Ok(output)
    }
}

fn collect_git_changes(repo: &Path, base_ref: &str, include_untracked: bool) -> Result<GitChanges> {
    let resolved = git_output(
        repo,
        [
            "rev-parse",
            "--verify",
            "--end-of-options",
            &format!("{base_ref}^{{commit}}"),
        ],
    )?;
    let resolved_base = String::from_utf8(resolved)?.trim().to_string();
    if resolved_base.is_empty() {
        bail!("Git base `{base_ref}` did not resolve to a commit");
    }
    let raw = git_output(
        repo,
        ["diff", "--name-status", "-z", "-M", &resolved_base, "--"],
    )?;
    let mut files = parse_name_status(&raw)?;
    if include_untracked {
        let raw = git_output(repo, ["ls-files", "--others", "--exclude-standard", "-z"])?;
        for path in nul_strings(&raw)? {
            files.push(ChangedFile {
                status: "untracked".to_string(),
                old_path: None,
                path,
            });
        }
    }
    files.retain(|file| {
        supported_path(&file.path) || file.old_path.as_deref().is_some_and(supported_path)
    });
    files.sort_by(|left, right| {
        left.path
            .cmp(&right.path)
            .then(left.status.cmp(&right.status))
    });
    files.dedup_by(|left, right| left.path == right.path && left.old_path == right.old_path);
    Ok(GitChanges {
        resolved_base,
        files,
    })
}

fn git_output<const N: usize>(repo: &Path, args: [&str; N]) -> Result<Vec<u8>> {
    let output = Command::new("git")
        .current_dir(repo)
        .args(args)
        .output()
        .context("running Git for change analysis")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        bail!("Git change analysis failed: {stderr}");
    }
    Ok(output.stdout)
}

fn parse_name_status(raw: &[u8]) -> Result<Vec<ChangedFile>> {
    let fields = nul_strings(raw)?;
    let mut cursor = 0;
    let mut output = Vec::new();
    while cursor < fields.len() {
        let status = fields[cursor].clone();
        cursor += 1;
        if status.starts_with('R') || status.starts_with('C') {
            let old_path = fields
                .get(cursor)
                .context("missing old rename path")?
                .clone();
            let path = fields
                .get(cursor + 1)
                .context("missing new rename path")?
                .clone();
            cursor += 2;
            output.push(ChangedFile {
                status: "renamed".to_string(),
                old_path: Some(old_path),
                path,
            });
        } else {
            let path = fields.get(cursor).context("missing changed path")?.clone();
            cursor += 1;
            let status = match status.as_str() {
                "A" => "added",
                "D" => "deleted",
                "M" => "modified",
                other if other.starts_with('T') => "modified",
                _ => "modified",
            };
            output.push(ChangedFile {
                status: status.to_string(),
                old_path: None,
                path,
            });
        }
    }
    Ok(output)
}

fn nul_strings(raw: &[u8]) -> Result<Vec<String>> {
    raw.split(|byte| *byte == 0)
        .filter(|field| !field.is_empty())
        .map(|field| String::from_utf8(field.to_vec()).context("Git returned a non-UTF-8 path"))
        .collect()
}

fn supported_path(path: &str) -> bool {
    matches!(
        Path::new(path).extension().and_then(|value| value.to_str()),
        Some("rs" | "ts" | "tsx" | "mts" | "cts")
    )
}

#[derive(Default)]
struct ParsedFileSymbols {
    source: String,
    symbols: Vec<IndexedSymbol>,
}

fn changed_symbols(repo: &Path, base: &str, files: &[ChangedFile]) -> Result<Vec<ChangedSymbol>> {
    let mut output = Vec::new();
    for file in files {
        let old_path = file.old_path.as_deref().unwrap_or(&file.path);
        let old = if matches!(file.status.as_str(), "added" | "untracked") {
            ParsedFileSymbols::default()
        } else {
            git_file_symbols(repo, base, old_path)?
        };
        let current = if file.status == "deleted" {
            ParsedFileSymbols::default()
        } else {
            working_file_symbols(repo, &file.path)?
        };
        output.extend(classify_symbols(old, current));
    }
    output.sort_by(|left, right| {
        symbol_path(left)
            .cmp(symbol_path(right))
            .then(symbol_line(left).cmp(&symbol_line(right)))
            .then(left.change.cmp(&right.change))
    });
    Ok(output)
}

fn git_file_symbols(repo: &Path, base: &str, path: &str) -> Result<ParsedFileSymbols> {
    let spec = format!("{base}:{path}");
    let output = Command::new("git")
        .current_dir(repo)
        .args(["show", "--no-ext-diff", &spec])
        .output()?;
    if !output.status.success() {
        return Ok(ParsedFileSymbols::default());
    }
    let source = String::from_utf8(output.stdout)?;
    let symbols = extract_symbols(
        &repo.display().to_string(),
        path,
        Path::new(path),
        &source,
        "change-analysis",
        "base",
    )?;
    Ok(ParsedFileSymbols { source, symbols })
}

fn working_file_symbols(repo: &Path, path: &str) -> Result<ParsedFileSymbols> {
    let absolute = repo.join(path);
    let source = std::fs::read_to_string(&absolute)
        .with_context(|| format!("reading changed file {}", absolute.display()))?;
    let symbols = extract_symbols(
        &repo.display().to_string(),
        path,
        Path::new(path),
        &source,
        "change-analysis",
        "working",
    )?;
    Ok(ParsedFileSymbols { source, symbols })
}

fn classify_symbols(old: ParsedFileSymbols, current: ParsedFileSymbols) -> Vec<ChangedSymbol> {
    let mut old_by_key = old
        .symbols
        .into_iter()
        .map(|symbol| (symbol.logical_key.clone(), symbol))
        .collect::<BTreeMap<_, _>>();
    let mut current_by_key = current
        .symbols
        .into_iter()
        .map(|symbol| (symbol.logical_key.clone(), symbol))
        .collect::<BTreeMap<_, _>>();
    let exact = old_by_key
        .keys()
        .filter(|key| current_by_key.contains_key(*key))
        .cloned()
        .collect::<Vec<_>>();
    let mut output = Vec::new();
    for key in exact {
        let old_symbol = old_by_key.remove(&key).expect("old exact symbol exists");
        let current_symbol = current_by_key
            .remove(&key)
            .expect("current exact symbol exists");
        if symbol_source(&old.source, &old_symbol)
            != symbol_source(&current.source, &current_symbol)
        {
            output.push(ChangedSymbol {
                change: "modified".to_string(),
                old: Some(old_symbol),
                current: Some(current_symbol),
            });
        }
    }

    let old_qualified = unique_by_qualified(&old_by_key);
    let current_qualified = unique_by_qualified(&current_by_key);
    let paired = old_qualified
        .keys()
        .filter(|name| current_qualified.contains_key(*name))
        .cloned()
        .collect::<Vec<_>>();
    for name in paired {
        let old_key = &old_qualified[&name];
        let current_key = &current_qualified[&name];
        let old = old_by_key
            .remove(old_key)
            .expect("old qualified symbol exists");
        let current = current_by_key
            .remove(current_key)
            .expect("current qualified symbol exists");
        output.push(ChangedSymbol {
            change: if old.signature != current.signature {
                "signature_changed".to_string()
            } else {
                "renamed".to_string()
            },
            old: Some(old),
            current: Some(current),
        });
    }

    let old_shape = unique_by_shape(&old_by_key);
    let current_shape = unique_by_shape(&current_by_key);
    let renamed = old_shape
        .keys()
        .filter(|shape| current_shape.contains_key(*shape))
        .cloned()
        .collect::<Vec<_>>();
    for shape in renamed {
        let old = old_by_key
            .remove(&old_shape[&shape])
            .expect("old shape symbol exists");
        let current = current_by_key
            .remove(&current_shape[&shape])
            .expect("current shape symbol exists");
        output.push(ChangedSymbol {
            change: "renamed".to_string(),
            old: Some(old),
            current: Some(current),
        });
    }
    output.extend(old_by_key.into_values().map(|symbol| ChangedSymbol {
        change: "deleted".to_string(),
        old: Some(symbol),
        current: None,
    }));
    output.extend(current_by_key.into_values().map(|symbol| ChangedSymbol {
        change: "added".to_string(),
        old: None,
        current: Some(symbol),
    }));
    output
}

fn symbol_source<'a>(source: &'a str, symbol: &IndexedSymbol) -> &'a str {
    let mut offsets = source
        .split_inclusive('\n')
        .scan(0usize, |offset, line| {
            let start = *offset;
            *offset += line.len();
            Some((start, *offset))
        })
        .collect::<Vec<_>>();
    if !source.ends_with('\n') {
        offsets.push((source.len(), source.len()));
    }
    let start_index = symbol.start_line.saturating_sub(1) as usize;
    let end_index = symbol.end_line.saturating_sub(1) as usize;
    let Some((start, _)) = offsets.get(start_index).copied() else {
        return "";
    };
    let end = offsets
        .get(end_index)
        .map(|(_, end)| *end)
        .unwrap_or(source.len());
    source.get(start..end).unwrap_or_default().trim()
}

fn unique_by_qualified(symbols: &BTreeMap<String, IndexedSymbol>) -> BTreeMap<String, String> {
    let mut counts = BTreeMap::<String, usize>::new();
    for symbol in symbols.values() {
        *counts.entry(symbol.qualified_name.clone()).or_default() += 1;
    }
    symbols
        .iter()
        .filter(|(_, symbol)| counts.get(&symbol.qualified_name) == Some(&1))
        .map(|(key, symbol)| (symbol.qualified_name.clone(), key.clone()))
        .collect()
}

fn unique_by_shape(symbols: &BTreeMap<String, IndexedSymbol>) -> BTreeMap<String, String> {
    let mut counts = BTreeMap::<String, usize>::new();
    for symbol in symbols.values() {
        let shape = format!(
            "{}:{}",
            symbol.kind,
            symbol.signature.replace(&symbol.name, "<name>")
        );
        *counts.entry(shape).or_default() += 1;
    }
    symbols
        .iter()
        .filter_map(|(key, symbol)| {
            let shape = format!(
                "{}:{}",
                symbol.kind,
                symbol.signature.replace(&symbol.name, "<name>")
            );
            (counts.get(&shape) == Some(&1)).then_some((shape, key.clone()))
        })
        .collect()
}

fn symbol_path(change: &ChangedSymbol) -> &str {
    change
        .current
        .as_ref()
        .or(change.old.as_ref())
        .map(|symbol| symbol.relative_path.as_str())
        .unwrap_or_default()
}

fn symbol_line(change: &ChangedSymbol) -> u64 {
    change
        .current
        .as_ref()
        .or(change.old.as_ref())
        .map(|symbol| symbol.start_line)
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_nul_name_status_with_renames() {
        let parsed =
            parse_name_status(b"M\0src/a.rs\0R100\0src/old.ts\0src/new.ts\0D\0src/gone.rs\0")
                .unwrap();
        assert_eq!(parsed.len(), 3);
        assert_eq!(parsed[1].status, "renamed");
        assert_eq!(parsed[1].old_path.as_deref(), Some("src/old.ts"));
        assert_eq!(parsed[1].path, "src/new.ts");
    }

    fn parsed(source: &str) -> ParsedFileSymbols {
        ParsedFileSymbols {
            source: source.to_string(),
            symbols: extract_symbols(
                "/repo",
                "src/lib.rs",
                Path::new("src/lib.rs"),
                source,
                "now",
                "hash",
            )
            .unwrap(),
        }
    }

    #[test]
    fn classifies_only_changed_declarations_and_distinguishes_signature_and_rename() {
        let body = classify_symbols(
            parsed("fn stable() {}\nfn changed() -> u64 { 1 }"),
            parsed("fn stable() {}\nfn changed() -> u64 { 2 }"),
        );
        assert_eq!(body.len(), 1);
        assert_eq!(body[0].change, "modified");
        assert_eq!(body[0].current.as_ref().unwrap().name, "changed");

        let signature = classify_symbols(
            parsed("fn changed(value: u64) {}"),
            parsed("fn changed(value: usize) {}"),
        );
        assert_eq!(signature.len(), 1);
        assert_eq!(signature[0].change, "signature_changed");

        let renamed = classify_symbols(parsed("fn old_name() {}"), parsed("fn new_name() {}"));
        assert_eq!(renamed.len(), 1);
        assert_eq!(renamed[0].change, "renamed");
    }
}
