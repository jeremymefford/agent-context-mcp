use anyhow::{Context, Result, bail};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fs;
use std::path::Path;

use super::relationships::{
    CONFIDENCE_LEXICAL, RelationKind, RepoRelationshipCoverage, ResolvedRelation,
};
use super::symbols::IndexedSymbol;
use super::{
    Engine, GRAPH_VERIFICATION_AUDIT_INTERVAL, ResolvedScope, build_root_hash, hash_text_like_file,
    scan_repo,
};

#[derive(Debug, Clone)]
pub struct ImpactRequest {
    pub repo: Option<String>,
    pub symbol_id: Option<String>,
    pub file: Option<String>,
    pub line: Option<u64>,
    pub max_depth: usize,
    pub max_nodes: usize,
    pub include_tests: bool,
    pub min_confidence: u64,
    pub include_possible: bool,
}

#[derive(Debug, Clone)]
pub struct TracePathRequest {
    pub repo: Option<String>,
    pub from_symbol_id: Option<String>,
    pub from_file: Option<String>,
    pub from_line: Option<u64>,
    pub to_symbol_id: Option<String>,
    pub to_file: Option<String>,
    pub to_line: Option<u64>,
    pub max_depth: usize,
    pub max_paths: usize,
    pub min_confidence: u64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ImpactResponse {
    pub scope: String,
    pub repo: String,
    pub root: ImpactNode,
    pub direct_dependents: Vec<ImpactNode>,
    pub transitive_dependents: Vec<ImpactNode>,
    pub affected_tests: Vec<ImpactNode>,
    pub possible_dependents: Vec<ImpactNode>,
    pub possible_tests: Vec<ImpactNode>,
    pub ambiguities: Vec<ImpactAmbiguity>,
    pub coverage: RepoRelationshipCoverage,
    pub diagnostic: Option<String>,
    pub truncated: bool,
    pub truncation_notices: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ImpactNode {
    pub logical_key: String,
    pub symbol_id: Option<String>,
    pub name: String,
    pub kind: String,
    pub relative_path: String,
    pub start_line: u64,
    pub end_line: u64,
    pub source_role: String,
    pub depth: usize,
    pub score: f64,
    pub possible: bool,
    pub evidence: Vec<ImpactEvidence>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ImpactEvidence {
    pub relation: RelationKind,
    pub confidence: u64,
    pub resolution: String,
    pub source_path: String,
    pub start_line: u64,
    pub end_line: u64,
    pub text: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ImpactAmbiguity {
    pub source_path: String,
    pub start_line: u64,
    pub target_name: String,
    pub resolution: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TracePathResponse {
    pub scope: String,
    pub repo: String,
    pub from: ImpactNode,
    pub to: ImpactNode,
    pub found: bool,
    pub paths: Vec<DependencyPath>,
    pub ambiguities: Vec<ImpactAmbiguity>,
    pub coverage: RepoRelationshipCoverage,
    pub diagnostic: Option<String>,
    pub truncated: bool,
    pub truncation_notices: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DependencyPath {
    pub score: f64,
    pub nodes: Vec<ImpactNode>,
}

#[derive(Clone)]
struct TraversalStep {
    key: String,
    depth: usize,
    score: f64,
    possible: bool,
    evidence: Vec<ImpactEvidence>,
}

#[derive(Clone)]
pub(super) struct GraphSource {
    pub(super) repo_key: String,
    pub(super) storage_repo: std::path::PathBuf,
    pub(super) suppressed_paths: BTreeSet<String>,
}

pub(super) struct GraphView {
    pub(super) sources: Vec<GraphSource>,
    pub(super) coverage: RepoRelationshipCoverage,
}

impl Engine {
    pub async fn analyze_impact(
        &self,
        scope: ResolvedScope,
        request: ImpactRequest,
    ) -> Result<ImpactResponse> {
        let repo = self
            .resolve_prepare_repo(&scope, request.repo.as_deref(), true)?
            .context("impact analysis requires one repository")?;
        let root = self.resolve_impact_root(&repo, &request).await?;
        self.verify_selected_symbols_current(&repo, std::slice::from_ref(&root))
            .await?;
        let view = self.verified_graph_view_for_repo(&repo, true).await?;
        self.analyze_impact_with_view(scope.id, &repo, &root, &view, &request)
            .await
    }

    pub(super) async fn analyze_impacts_for_symbols(
        &self,
        scope_id: String,
        repo: &Path,
        roots: &[IndexedSymbol],
        request: &ImpactRequest,
    ) -> Result<Vec<ImpactResponse>> {
        let view = self.verified_graph_view_for_repo(repo, true).await?;
        let mut output = Vec::with_capacity(roots.len());
        for root in roots {
            output.push(
                self.analyze_impact_with_view(scope_id.clone(), repo, root, &view, request)
                    .await?,
            );
        }
        Ok(output)
    }

    async fn analyze_impact_with_view(
        &self,
        scope_id: String,
        repo: &Path,
        root: &IndexedSymbol,
        view: &GraphView,
        request: &ImpactRequest,
    ) -> Result<ImpactResponse> {
        let coverage = view.coverage.clone();
        if !matches!(root.language.as_str(), "rust" | "typescript") {
            return Ok(ImpactResponse {
                scope: scope_id,
                repo: repo.display().to_string(),
                root: node_from_symbol(root, 0, 1.0, Vec::new()),
                direct_dependents: Vec::new(),
                transitive_dependents: Vec::new(),
                affected_tests: Vec::new(),
                possible_dependents: Vec::new(),
                possible_tests: Vec::new(),
                ambiguities: Vec::new(),
                coverage,
                diagnostic: Some(format!(
                    "unsupported language `{}`; structural impact is guaranteed only for Rust and TypeScript",
                    root.language
                )),
                truncated: false,
                truncation_notices: Vec::new(),
            });
        }
        let min_confidence = if request.include_possible {
            request.min_confidence.min(CONFIDENCE_LEXICAL)
        } else {
            request.min_confidence.max(650)
        };
        let max_depth = request.max_depth.clamp(1, 5);
        let max_nodes = request.max_nodes.clamp(1, 250);
        let mut queue = VecDeque::from([TraversalStep {
            key: root.logical_key.clone(),
            depth: 0,
            score: 1.0,
            possible: false,
            evidence: Vec::new(),
        }]);
        let mut visited = BTreeSet::from([root.logical_key.clone()]);
        let mut steps = BTreeMap::<String, TraversalStep>::new();
        let mut ambiguities = Vec::new();
        let mut truncated = false;

        while let Some(frontier_depth) = queue.front().map(|step| step.depth) {
            if frontier_depth >= max_depth {
                break;
            }
            let mut frontier = Vec::new();
            while queue
                .front()
                .is_some_and(|step| step.depth == frontier_depth)
            {
                frontier.push(queue.pop_front().expect("frontier entry must exist"));
            }
            let frontier_cap = max_nodes.min(100);
            let frontier_keys = frontier
                .iter()
                .map(|step| step.key.clone())
                .collect::<Vec<_>>();
            let relations = self
                .graph_relations_to(view, &frontier_keys, 0, frontier_cap + 1)
                .await?;
            let mut relations_by_target = BTreeMap::<String, Vec<ResolvedRelation>>::new();
            for relation in relations {
                if let Some(target_key) = relation.target_key.clone() {
                    relations_by_target
                        .entry(target_key)
                        .or_default()
                        .push(relation);
                }
            }
            for step in frontier {
                let relations = relations_by_target.remove(&step.key).unwrap_or_default();
                if relations.len() > frontier_cap {
                    truncated = true;
                }
                for relation in relations.into_iter().take(frontier_cap) {
                    if relation.confidence < 650 {
                        ambiguities.push(ambiguity_from_relation(&relation));
                    }
                    if relation.confidence < min_confidence {
                        continue;
                    }
                    let next_depth = step.depth + 1;
                    let next_score = step.score
                        * (relation.confidence as f64 / 1000.0)
                        * relation_weight(relation.kind)
                        * if next_depth > 1 { 0.7 } else { 1.0 };
                    let mut evidence = step.evidence.clone();
                    evidence.push(evidence_from_relation(&relation));
                    let candidate = TraversalStep {
                        key: relation.source_key.clone(),
                        depth: next_depth,
                        score: next_score,
                        possible: step.possible || relation.confidence < 650,
                        evidence,
                    };
                    steps
                        .entry(candidate.key.clone())
                        .and_modify(|existing| {
                            if candidate.score > existing.score {
                                *existing = candidate.clone();
                            }
                        })
                        .or_insert_with(|| candidate.clone());
                    if visited.insert(candidate.key.clone()) {
                        if visited.len() > max_nodes {
                            truncated = true;
                            break;
                        }
                        queue.push_back(candidate);
                    }
                }
                if truncated {
                    break;
                }
            }
            if truncated {
                break;
            }
        }

        let keys = steps.keys().cloned().collect::<Vec<_>>();
        let symbols = self.load_graph_symbols(view, &keys).await?;
        let by_key = symbols
            .into_iter()
            .map(|symbol| (symbol.logical_key.clone(), symbol))
            .collect::<BTreeMap<_, _>>();
        let mut nodes = steps
            .into_values()
            .map(|step| {
                let symbol = by_key.get(&step.key);
                node_from_step(step, symbol)
            })
            .collect::<Vec<_>>();
        nodes.sort_by(compare_nodes);
        let mut affected_tests = Vec::new();
        let mut possible_tests = Vec::new();
        let mut possible_dependents = Vec::new();
        let mut direct = Vec::new();
        let mut transitive = Vec::new();
        for node in nodes {
            if node.possible && node.source_role == "test" {
                if request.include_tests {
                    possible_tests.push(node);
                }
            } else if node.possible {
                possible_dependents.push(node);
            } else if node.source_role == "test" {
                if request.include_tests {
                    affected_tests.push(node);
                }
            } else if node.depth == 1 {
                direct.push(node);
            } else {
                transitive.push(node);
            }
        }
        ambiguities.sort_by(|left, right| {
            left.source_path
                .cmp(&right.source_path)
                .then(left.start_line.cmp(&right.start_line))
                .then(left.target_name.cmp(&right.target_name))
        });
        ambiguities.dedup_by(|left, right| {
            left.source_path == right.source_path
                && left.start_line == right.start_line
                && left.target_name == right.target_name
        });
        Ok(ImpactResponse {
            scope: scope_id,
            repo: repo.display().to_string(),
            root: node_from_symbol(root, 0, 1.0, Vec::new()),
            direct_dependents: direct,
            transitive_dependents: transitive,
            affected_tests,
            possible_dependents,
            possible_tests,
            ambiguities,
            coverage,
            diagnostic: None,
            truncated,
            truncation_notices: truncated
                .then(|| {
                    format!(
                        "Traversal stopped at maxNodes={} or the per-frontier cap={}",
                        max_nodes,
                        max_nodes.min(100)
                    )
                })
                .into_iter()
                .collect(),
        })
    }

    pub async fn trace_dependency_path(
        &self,
        scope: ResolvedScope,
        request: TracePathRequest,
    ) -> Result<TracePathResponse> {
        let repo = self
            .resolve_prepare_repo(&scope, request.repo.as_deref(), true)?
            .context("path tracing requires one repository")?;
        let from = self
            .resolve_graph_symbol_selector(
                &repo,
                request.from_symbol_id.as_deref(),
                request.from_file.as_deref(),
                request.from_line,
                "from",
            )
            .await?;
        let to = self
            .resolve_graph_symbol_selector(
                &repo,
                request.to_symbol_id.as_deref(),
                request.to_file.as_deref(),
                request.to_line,
                "to",
            )
            .await?;
        self.verify_selected_symbols_current(&repo, &[from.clone(), to.clone()])
            .await?;
        let view = self.verified_graph_view_for_repo(&repo, true).await?;
        let coverage = view.coverage.clone();
        if !matches!(from.language.as_str(), "rust" | "typescript")
            || !matches!(to.language.as_str(), "rust" | "typescript")
        {
            return Ok(TracePathResponse {
                scope: scope.id,
                repo: repo.display().to_string(),
                from: node_from_symbol(&from, 0, 1.0, Vec::new()),
                to: node_from_symbol(&to, 0, 1.0, Vec::new()),
                found: false,
                paths: Vec::new(),
                ambiguities: Vec::new(),
                coverage,
                diagnostic: Some(
                    "unsupported language; structural paths are guaranteed only for Rust and TypeScript"
                        .to_string(),
                ),
                truncated: false,
                truncation_notices: Vec::new(),
            });
        }
        let max_depth = request.max_depth.clamp(1, 10);
        let max_paths = request.max_paths.clamp(1, 10);
        let mut queue = VecDeque::from([(
            from.logical_key.clone(),
            Vec::<ResolvedRelation>::new(),
            1.0f64,
        )]);
        let mut paths = Vec::<(f64, Vec<ResolvedRelation>)>::new();
        let mut ambiguities = Vec::new();
        let mut best_scores = BTreeMap::<(String, usize), Vec<f64>>::new();
        let mut expanded = 0usize;
        let mut found_depth = None;
        let mut truncated = false;
        let mut stop_search = false;
        let node_cap = 1_000usize;
        let frontier_cap = 100usize;
        while let Some((key, path, path_score)) = queue.pop_front() {
            if found_depth.is_some_and(|depth| path.len() >= depth) {
                break;
            }
            if path.len() >= max_depth {
                continue;
            }
            expanded += 1;
            if expanded > node_cap {
                truncated = true;
                break;
            }
            let relations = self
                .graph_relations_from(&view, &[key], 0, frontier_cap + 1)
                .await?;
            if relations.len() > frontier_cap {
                truncated = true;
            }
            for relation in relations.into_iter().take(frontier_cap) {
                if relation.confidence < 650 {
                    ambiguities.push(ambiguity_from_relation(&relation));
                }
                if relation.confidence < request.min_confidence.max(650) {
                    continue;
                }
                let Some(target_key) = relation.target_key.clone() else {
                    continue;
                };
                if path.iter().any(|edge| {
                    edge.source_key == target_key
                        || edge.target_key.as_deref() == Some(target_key.as_str())
                }) {
                    continue;
                }
                let mut next = path.clone();
                let next_score = path_score
                    * (relation.confidence as f64 / 1000.0)
                    * relation_weight(relation.kind);
                next.push(relation);
                if target_key == to.logical_key {
                    found_depth.get_or_insert(next.len());
                    paths.push((next_score, next));
                    paths.sort_by(|left, right| right.0.total_cmp(&left.0));
                    if paths.len() > max_paths {
                        paths.truncate(max_paths);
                        truncated = true;
                    }
                } else if found_depth.is_none() {
                    let scores = best_scores
                        .entry((target_key.clone(), next.len()))
                        .or_default();
                    if scores.len() >= max_paths
                        && scores.last().is_some_and(|lowest| next_score <= *lowest)
                    {
                        continue;
                    }
                    scores.push(next_score);
                    scores.sort_by(|left, right| right.total_cmp(left));
                    scores.truncate(max_paths);
                    queue.push_back((target_key, next, next_score));
                    if queue.len() > node_cap {
                        truncated = true;
                        stop_search = true;
                        break;
                    }
                }
            }
            if stop_search {
                break;
            }
        }
        let mut rendered = Vec::new();
        for (_, path) in paths {
            rendered.push(self.render_dependency_path(&view, &from, &path).await?);
        }
        rendered.sort_by(|left, right| right.score.total_cmp(&left.score));
        ambiguities.sort_by(|left, right| {
            left.source_path
                .cmp(&right.source_path)
                .then(left.start_line.cmp(&right.start_line))
                .then(left.target_name.cmp(&right.target_name))
        });
        ambiguities.dedup_by(|left, right| {
            left.source_path == right.source_path
                && left.start_line == right.start_line
                && left.target_name == right.target_name
        });
        Ok(TracePathResponse {
            scope: scope.id,
            repo: repo.display().to_string(),
            from: node_from_symbol(&from, 0, 1.0, Vec::new()),
            to: node_from_symbol(&to, 0, 1.0, Vec::new()),
            found: !rendered.is_empty(),
            paths: rendered,
            ambiguities,
            coverage,
            diagnostic: None,
            truncated,
            truncation_notices: truncated
                .then(|| {
                    format!(
                        "Path search stopped at maxPaths={} or a traversal safety cap",
                        max_paths
                    )
                })
                .into_iter()
                .collect(),
        })
    }

    pub async fn relationship_coverage(
        &self,
        scope: ResolvedScope,
        repo_hint: Option<&str>,
    ) -> Result<Vec<RepoRelationshipCoverage>> {
        let repos = self.resolve_prepare_repos(&scope, repo_hint)?;
        let mut output = Vec::new();
        for repo in repos {
            output.push(
                self.verified_graph_view_for_repo(&repo, false)
                    .await?
                    .coverage,
            );
        }
        Ok(output)
    }

    pub async fn relationship_readiness(
        &self,
        scope: ResolvedScope,
        repo_hint: Option<&str>,
    ) -> Result<Vec<RepoRelationshipCoverage>> {
        let repos = self.resolve_prepare_repos(&scope, repo_hint)?;
        let mut output = Vec::new();
        for repo in repos {
            output.push(self.graph_view_for_repo(&repo, false).await?.coverage);
        }
        Ok(output)
    }

    async fn resolve_impact_root(
        &self,
        repo: &Path,
        request: &ImpactRequest,
    ) -> Result<IndexedSymbol> {
        if let Some(symbol_id) = request.symbol_id.as_deref() {
            return self
                .resolve_symbol_in_repo(repo, symbol_id)
                .await?
                .map(|resolved| resolved.symbol)
                .context("symbolId was not found");
        }
        let file = request
            .file
            .as_deref()
            .context("provide symbolId or file + line")?;
        let line = request.line.context("line is required with file")?;
        let symbols = self.load_file_outline_symbols_for_repo(repo, file).await?;
        symbols
            .into_iter()
            .filter(|symbol| symbol.start_line <= line && line <= symbol.end_line)
            .min_by_key(|symbol| symbol.end_line.saturating_sub(symbol.start_line))
            .context("no indexed symbol contains the requested line")
    }

    async fn resolve_graph_symbol_selector(
        &self,
        repo: &Path,
        symbol_id: Option<&str>,
        file: Option<&str>,
        line: Option<u64>,
        label: &str,
    ) -> Result<IndexedSymbol> {
        if let Some(symbol_id) = symbol_id.filter(|value| !value.trim().is_empty()) {
            return self
                .resolve_symbol_in_repo(repo, symbol_id)
                .await?
                .map(|resolved| resolved.symbol)
                .with_context(|| format!("{label}SymbolId was not found"));
        }
        let file =
            file.with_context(|| format!("provide {label}SymbolId or {label}File + {label}Line"))?;
        let line = line.with_context(|| format!("{label}Line is required with {label}File"))?;
        let symbols = self.load_file_outline_symbols_for_repo(repo, file).await?;
        symbols
            .into_iter()
            .filter(|symbol| symbol.start_line <= line && line <= symbol.end_line)
            .min_by_key(|symbol| symbol.end_line.saturating_sub(symbol.start_line))
            .with_context(|| format!("no indexed {label} symbol contains the requested line"))
    }

    async fn graph_view_for_repo(&self, repo: &Path, require_ready: bool) -> Result<GraphView> {
        let ctx = self.repo_context(repo)?;
        if !self.index_features_for_repo(&ctx.canonical_root).graph {
            if require_ready {
                bail!(
                    "relationship graph indexing is disabled for `{}`; enable the `graph` feature and refresh this repository",
                    ctx.canonical_root.display()
                );
            }
            return Ok(disabled_graph_view(repo));
        }
        let graph = self.inner.graph_store.clone();

        if let Some(overlay) = ctx.overlay.as_ref() {
            let state = self
                .load_overlay_state(overlay)
                .await?
                .context("worktree relationship overlay is not indexed")?;
            let overlay_key = overlay.repo_key.clone();
            let canonical_key = ctx.canonical_root.display().to_string();
            let overlay_graph = graph.clone();
            let canonical_graph = graph.clone();
            let overlay_key_for_query = overlay_key.clone();
            let canonical_key_for_query = canonical_key.clone();
            let (mut overlay_coverage, canonical_coverage) = tokio::try_join!(
                self.run_search_lexical_blocking("overlay_graph_state", move || {
                    overlay_graph.coverage_cached(&overlay_key_for_query)
                }),
                self.run_search_lexical_blocking("canonical_graph_state", move || {
                    canonical_graph.coverage_cached(&canonical_key_for_query)
                })
            )?;
            let ready = overlay_coverage.graph_status == "ready"
                && canonical_coverage.graph_status == "ready"
                && matches!(
                    state.overlay_status.as_deref(),
                    Some("completed") | Some("empty")
                );
            if !ready && require_ready {
                bail!(
                    "relationship graph is stale or incomplete; refresh this worktree index before graph analysis"
                );
            }
            if !ready {
                overlay_coverage.graph_status = "stale".to_string();
            }
            let suppressed_paths = super::overlay_suppressed_paths(&state);
            merge_coverage(&mut overlay_coverage, &canonical_coverage);
            overlay_coverage.repo = repo.display().to_string();
            return Ok(GraphView {
                sources: vec![
                    GraphSource {
                        repo_key: overlay_key,
                        storage_repo: overlay.storage_root.clone(),
                        suppressed_paths: BTreeSet::new(),
                    },
                    GraphSource {
                        repo_key: canonical_key,
                        storage_repo: ctx.canonical_root.clone(),
                        suppressed_paths,
                    },
                ],
                coverage: overlay_coverage,
            });
        }

        let repo_key = ctx.canonical_root.display().to_string();
        let coverage_graph = graph.clone();
        let key_for_query = repo_key.clone();
        let mut coverage = self
            .run_search_lexical_blocking("graph_state_check", move || {
                coverage_graph.coverage_cached(&key_for_query)
            })
            .await?;
        if coverage.graph_status != "ready" {
            if require_ready {
                bail!(
                    "relationship graph is stale or incomplete; refresh the index before graph analysis"
                );
            }
            coverage.graph_status = "stale".to_string();
        }
        Ok(GraphView {
            sources: vec![GraphSource {
                repo_key,
                storage_repo: ctx.canonical_root.clone(),
                suppressed_paths: BTreeSet::new(),
            }],
            coverage,
        })
    }

    pub(super) async fn verified_graph_view_for_repo(
        &self,
        repo: &Path,
        require_ready: bool,
    ) -> Result<GraphView> {
        let ctx = self.repo_context(repo)?;
        if !self.index_features_for_repo(&ctx.canonical_root).graph {
            if require_ready {
                bail!(
                    "relationship graph indexing is disabled for `{}`; enable the `graph` feature and refresh this repository",
                    ctx.canonical_root.display()
                );
            }
            return Ok(disabled_graph_view(repo));
        }
        let live_hashes = self.current_graph_hashes(repo).await?;
        let live_root = build_root_hash(&live_hashes);
        let graph = self.inner.graph_store.clone();

        if let Some(overlay) = ctx.overlay.as_ref() {
            let state = self
                .load_overlay_state(overlay)
                .await?
                .context("worktree relationship overlay is not indexed")?;
            let overlay_key = overlay.repo_key.clone();
            let canonical_key = ctx.canonical_root.display().to_string();
            let overlay_graph = graph.clone();
            let canonical_graph = graph.clone();
            let overlay_key_for_query = overlay_key.clone();
            let canonical_key_for_query = canonical_key.clone();
            let (mut overlay_coverage, canonical_coverage) = tokio::try_join!(
                self.run_search_lexical_blocking("overlay_graph_coverage", move || {
                    overlay_graph.coverage(&overlay_key_for_query)
                }),
                self.run_search_lexical_blocking("canonical_graph_coverage", move || {
                    canonical_graph.coverage(&canonical_key_for_query)
                })
            )?;
            let suppressed_paths = super::overlay_suppressed_paths(&state);
            let mut composed_hashes = self.graph_hashes(&canonical_key).await?;
            composed_hashes.retain(|path, _| !suppressed_paths.contains(path));
            composed_hashes.extend(self.graph_hashes(&overlay_key).await?);
            let overlay_consistent = self
                .graph_index_consistent(&overlay.storage_root, &overlay_key)
                .await?;
            let canonical_consistent = self
                .graph_index_consistent(&ctx.canonical_root, &canonical_key)
                .await?;
            let ready = overlay_coverage.graph_status == "ready"
                && canonical_coverage.graph_status == "ready"
                && overlay_consistent
                && canonical_consistent
                && overlay_coverage.root_hash.as_deref() == Some(live_root.as_str())
                && matches!(
                    state.overlay_status.as_deref(),
                    Some("completed") | Some("empty")
                );
            if !ready {
                overlay_coverage.graph_status = "stale".to_string();
                overlay_coverage.stale_files = stale_paths(&live_hashes, &composed_hashes);
                if require_ready {
                    bail!(
                        "relationship graph is stale or incomplete; refresh this worktree index before graph analysis"
                    );
                }
            }
            merge_coverage(&mut overlay_coverage, &canonical_coverage);
            overlay_coverage.repo = repo.display().to_string();
            return Ok(GraphView {
                sources: vec![
                    GraphSource {
                        repo_key: overlay_key,
                        storage_repo: overlay.storage_root.clone(),
                        suppressed_paths: BTreeSet::new(),
                    },
                    GraphSource {
                        repo_key: canonical_key,
                        storage_repo: ctx.canonical_root.clone(),
                        suppressed_paths,
                    },
                ],
                coverage: overlay_coverage,
            });
        }

        let repo_key = ctx.canonical_root.display().to_string();
        let coverage_graph = graph.clone();
        let key_for_query = repo_key.clone();
        let mut coverage = self
            .run_search_lexical_blocking("graph_ready_check", move || {
                coverage_graph.coverage(&key_for_query)
            })
            .await?;
        let graph_hashes = self.graph_hashes(&repo_key).await?;
        let index_consistent = self
            .graph_index_consistent(&ctx.canonical_root, &repo_key)
            .await?;
        let ready = coverage.graph_status == "ready"
            && index_consistent
            && coverage.root_hash.as_deref() == Some(live_root.as_str());
        if !ready {
            coverage.graph_status = "stale".to_string();
            coverage.stale_files = stale_paths(&live_hashes, &graph_hashes);
            if require_ready {
                bail!(
                    "relationship graph is stale or incomplete; refresh the index before graph analysis"
                );
            }
        }
        Ok(GraphView {
            sources: vec![GraphSource {
                repo_key,
                storage_repo: ctx.canonical_root.clone(),
                suppressed_paths: BTreeSet::new(),
            }],
            coverage,
        })
    }

    async fn current_graph_hashes(&self, repo: &Path) -> Result<BTreeMap<String, String>> {
        let cached = self
            .inner
            .graph_verification_cache
            .lock()
            .map_err(|_| anyhow::anyhow!("graph verification cache poisoned"))?
            .get(repo)
            .cloned();
        if let Some(cached) = cached {
            if cached.verified_at.elapsed() < GRAPH_VERIFICATION_AUDIT_INTERVAL {
                return Ok(cached.hashes);
            }
            let token_repo = repo.to_path_buf();
            let paths = cached.hashes.keys().cloned().collect::<Vec<_>>();
            let filesystem_token =
                tokio::task::spawn_blocking(move || graph_filesystem_token(&token_repo, &paths))
                    .await
                    .context("joining graph filesystem validation")??;
            if cached.filesystem_token == filesystem_token {
                if let Some(entry) = self
                    .inner
                    .graph_verification_cache
                    .lock()
                    .map_err(|_| anyhow::anyhow!("graph verification cache poisoned"))?
                    .get_mut(repo)
                {
                    entry.verified_at = std::time::Instant::now();
                }
                return Ok(cached.hashes);
            }
        }

        let root = repo.to_path_buf();
        let policy = self.inner.config.indexing.policy_for_repo(repo);
        let files = tokio::task::spawn_blocking(move || scan_repo(&root, &[], &[], &policy))
            .await
            .context("joining graph freshness scan")??;
        let hashes = files
            .into_iter()
            .map(|(path, file)| (path, file.hash))
            .collect::<BTreeMap<_, _>>();
        let token_repo = repo.to_path_buf();
        let paths = hashes.keys().cloned().collect::<Vec<_>>();
        let filesystem_token =
            tokio::task::spawn_blocking(move || graph_filesystem_token(&token_repo, &paths))
                .await
                .context("joining graph filesystem validation")??;
        self.inner
            .graph_verification_cache
            .lock()
            .map_err(|_| anyhow::anyhow!("graph verification cache poisoned"))?
            .insert(
                repo.to_path_buf(),
                super::GraphVerificationCacheEntry {
                    filesystem_token,
                    hashes: hashes.clone(),
                    verified_at: std::time::Instant::now(),
                },
            );
        Ok(hashes)
    }

    async fn verify_selected_symbols_current(
        &self,
        repo: &Path,
        symbols: &[IndexedSymbol],
    ) -> Result<()> {
        let repo = repo.to_path_buf();
        let files = symbols
            .iter()
            .map(|symbol| (symbol.relative_path.clone(), symbol.file_hash.clone()))
            .collect::<BTreeMap<_, _>>();
        tokio::task::spawn_blocking(move || {
            for (relative_path, expected_hash) in files {
                let current_hash = hash_text_like_file(&repo.join(&relative_path))?;
                if current_hash.as_deref() != Some(expected_hash.as_str()) {
                    bail!(
                        "relationship graph is stale for `{relative_path}`; refresh the index before graph analysis"
                    );
                }
            }
            Ok(())
        })
        .await
        .context("joining selected graph-file validation")?
    }

    async fn graph_hashes(&self, repo_key: &str) -> Result<BTreeMap<String, String>> {
        let graph = self.inner.graph_store.clone();
        let key = repo_key.to_string();
        self.run_search_lexical_blocking("graph_file_hashes", move || graph.file_hashes(&key))
            .await
    }

    async fn graph_index_consistent(&self, storage_repo: &Path, repo_key: &str) -> Result<bool> {
        let graph = self.inner.graph_store.clone();
        let index = self.inner.local_index.clone();
        let key = repo_key.to_string();
        let repo = storage_repo.to_path_buf();
        let (canonical_count, indexed_count) = tokio::try_join!(
            self.run_search_lexical_blocking("canonical_relationship_count", move || {
                graph.relation_document_count(&key)
            }),
            self.run_search_lexical_blocking("tantivy_relationship_count", move || {
                index.relation_document_count(&repo)
            })
        )?;
        Ok(canonical_count == indexed_count)
    }

    async fn graph_relations_to(
        &self,
        view: &GraphView,
        keys: &[String],
        min_confidence: u64,
        limit: usize,
    ) -> Result<Vec<ResolvedRelation>> {
        self.graph_relations(view, keys, min_confidence, limit, true)
            .await
    }

    async fn graph_relations_from(
        &self,
        view: &GraphView,
        keys: &[String],
        min_confidence: u64,
        limit: usize,
    ) -> Result<Vec<ResolvedRelation>> {
        self.graph_relations(view, keys, min_confidence, limit, false)
            .await
    }

    async fn graph_relations(
        &self,
        view: &GraphView,
        keys: &[String],
        min_confidence: u64,
        limit: usize,
        reverse: bool,
    ) -> Result<Vec<ResolvedRelation>> {
        let mut output = Vec::new();
        let mut seen = BTreeSet::new();
        for source in &view.sources {
            let index = self.inner.local_index.clone();
            let repo = source.storage_repo.clone();
            let keys = keys.to_vec();
            let relations = self
                .run_search_lexical_blocking("relationship_frontier", move || {
                    index.relation_frontier(&repo, &keys, min_confidence, limit, reverse)
                })
                .await?;
            for relation in relations {
                if source.suppressed_paths.contains(&relation.source_path) {
                    continue;
                }
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
        output.sort_by(compare_relations);
        Ok(output)
    }

    async fn load_graph_symbols(
        &self,
        view: &GraphView,
        keys: &[String],
    ) -> Result<Vec<IndexedSymbol>> {
        let mut output = Vec::new();
        let mut seen = BTreeSet::new();
        for source in &view.sources {
            let store = self.inner.symbol_store.clone();
            let repo = source.repo_key.clone();
            let keys = keys.to_vec();
            let symbols = self
                .run_search_lexical_blocking("load_graph_symbols", move || {
                    store.symbols_by_logical_keys(&repo, &keys)
                })
                .await?;
            for symbol in symbols {
                if !source.suppressed_paths.contains(&symbol.relative_path)
                    && seen.insert(symbol.logical_key.clone())
                {
                    output.push(symbol);
                }
            }
        }
        Ok(output)
    }

    async fn render_dependency_path(
        &self,
        view: &GraphView,
        root: &IndexedSymbol,
        relations: &[ResolvedRelation],
    ) -> Result<DependencyPath> {
        let keys = relations
            .iter()
            .filter_map(|relation| relation.target_key.clone())
            .collect::<Vec<_>>();
        let symbols = self.load_graph_symbols(view, &keys).await?;
        let by_key = symbols
            .into_iter()
            .map(|symbol| (symbol.logical_key.clone(), symbol))
            .collect::<BTreeMap<_, _>>();
        let mut nodes = vec![node_from_symbol(root, 0, 1.0, Vec::new())];
        let mut score = 1.0;
        for (depth, relation) in relations.iter().enumerate() {
            score *= relation.confidence as f64 / 1000.0 * relation_weight(relation.kind);
            let key = relation.target_key.as_deref().unwrap_or_default();
            let step = TraversalStep {
                key: key.to_string(),
                depth: depth + 1,
                score,
                possible: false,
                evidence: vec![evidence_from_relation(relation)],
            };
            nodes.push(node_from_step(step, by_key.get(key)));
        }
        Ok(DependencyPath { score, nodes })
    }
}

fn graph_filesystem_token(repo: &Path, paths: &[String]) -> Result<String> {
    let mut hasher = xxhash_rust::xxh3::Xxh3::new();
    let fingerprint = super::freshness::fingerprint_repo(repo)?;
    hasher.update(fingerprint.head.as_deref().unwrap_or_default().as_bytes());
    hasher.update(&fingerprint.index_mtime.unwrap_or_default().to_le_bytes());
    hasher.update(&fingerprint.root_mtime.unwrap_or_default().to_le_bytes());

    let mut directories = BTreeSet::from([repo.to_path_buf()]);
    for path in paths {
        let absolute = repo.join(path);
        hasher.update(path.as_bytes());
        hash_metadata(&mut hasher, &absolute);
        let mut parent = absolute.parent();
        while let Some(directory) = parent {
            if !directory.starts_with(repo) {
                break;
            }
            directories.insert(directory.to_path_buf());
            if directory == repo {
                break;
            }
            parent = directory.parent();
        }
    }
    for directory in directories {
        hasher.update(
            directory
                .strip_prefix(repo)
                .unwrap_or(&directory)
                .as_os_str()
                .as_encoded_bytes(),
        );
        hash_metadata(&mut hasher, &directory);
    }
    Ok(format!("{:032x}", hasher.digest128()))
}

fn hash_metadata(hasher: &mut xxhash_rust::xxh3::Xxh3, path: &Path) {
    match fs::symlink_metadata(path) {
        Ok(metadata) => {
            hasher.update(&metadata.len().to_le_bytes());
            if let Ok(modified) = metadata.modified()
                && let Ok(elapsed) = modified.duration_since(std::time::UNIX_EPOCH)
            {
                hasher.update(&elapsed.as_nanos().to_le_bytes());
            }
        }
        Err(_) => hasher.update(b"<missing>"),
    }
}

fn relation_weight(kind: RelationKind) -> f64 {
    match kind {
        RelationKind::Calls | RelationKind::Implements | RelationKind::Inherits => 1.0,
        RelationKind::TypeUses | RelationKind::ValueUses => 0.9,
        RelationKind::Reexports => 0.85,
        RelationKind::Imports => 0.75,
    }
}

fn compare_relations(left: &ResolvedRelation, right: &ResolvedRelation) -> std::cmp::Ordering {
    right
        .confidence
        .cmp(&left.confidence)
        .then(left.source_path.cmp(&right.source_path))
        .then(left.start_line.cmp(&right.start_line))
        .then(left.source_key.cmp(&right.source_key))
        .then(left.target_key.cmp(&right.target_key))
}

fn stale_paths(live: &BTreeMap<String, String>, indexed: &BTreeMap<String, String>) -> Vec<String> {
    live.keys()
        .chain(indexed.keys())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .filter(|path| live.get(*path) != indexed.get(*path))
        .cloned()
        .collect()
}

fn disabled_graph_view(repo: &Path) -> GraphView {
    GraphView {
        sources: Vec::new(),
        coverage: RepoRelationshipCoverage {
            repo: repo.display().to_string(),
            graph_status: "disabled".to_string(),
            ..RepoRelationshipCoverage::default()
        },
    }
}

fn merge_coverage(primary: &mut RepoRelationshipCoverage, canonical: &RepoRelationshipCoverage) {
    primary.supported_files += canonical.supported_files;
    primary.unsupported_files += canonical.unsupported_files;
    primary.definitions += canonical.definitions;
    primary.references += canonical.references;
    primary.definite += canonical.definite;
    primary.probable += canonical.probable;
    primary.possible += canonical.possible;
    primary.unresolved += canonical.unresolved;
    primary.unstable_identities += canonical.unstable_identities;
    primary
        .unsupported_paths
        .extend(canonical.unsupported_paths.clone());
    primary.unsupported_paths.sort();
    primary.unsupported_paths.dedup();
    let resolved = primary.definite + primary.probable;
    primary.resolution_percentage = if primary.references == 0 {
        100.0
    } else {
        resolved as f64 * 100.0 / primary.references as f64
    };
    for language in &canonical.by_language {
        if let Some(existing) = primary
            .by_language
            .iter_mut()
            .find(|existing| existing.language == language.language)
        {
            existing.files += language.files;
            existing.definitions += language.definitions;
            existing.references += language.references;
            existing.definite += language.definite;
            existing.probable += language.probable;
            existing.possible += language.possible;
            existing.unresolved += language.unresolved;
            existing.resolution_percentage = if existing.references == 0 {
                100.0
            } else {
                (existing.definite + existing.probable) as f64 * 100.0 / existing.references as f64
            };
        } else {
            primary.by_language.push(language.clone());
        }
    }
    primary
        .by_language
        .sort_by(|left, right| left.language.cmp(&right.language));
}

fn evidence_from_relation(relation: &ResolvedRelation) -> ImpactEvidence {
    ImpactEvidence {
        relation: relation.kind,
        confidence: relation.confidence,
        resolution: relation.resolution.clone(),
        source_path: relation.source_path.clone(),
        start_line: relation.start_line,
        end_line: relation.end_line,
        text: relation.evidence.clone(),
    }
}

fn ambiguity_from_relation(relation: &ResolvedRelation) -> ImpactAmbiguity {
    ImpactAmbiguity {
        source_path: relation.source_path.clone(),
        start_line: relation.start_line,
        target_name: relation.target_name.clone(),
        resolution: relation.resolution.clone(),
    }
}

fn node_from_symbol(
    symbol: &IndexedSymbol,
    depth: usize,
    score: f64,
    evidence: Vec<ImpactEvidence>,
) -> ImpactNode {
    ImpactNode {
        logical_key: symbol.logical_key.clone(),
        symbol_id: Some(symbol.symbol_id.clone()),
        name: symbol.name.clone(),
        kind: symbol.kind.clone(),
        relative_path: symbol.relative_path.clone(),
        start_line: symbol.start_line,
        end_line: symbol.end_line,
        source_role: symbol.source_role.clone(),
        depth,
        score,
        possible: false,
        evidence,
    }
}

fn node_from_step(step: TraversalStep, symbol: Option<&IndexedSymbol>) -> ImpactNode {
    if let Some(symbol) = symbol {
        let mut node = node_from_symbol(symbol, step.depth, step.score, step.evidence);
        node.possible = step.possible;
        node
    } else {
        ImpactNode {
            logical_key: step.key,
            symbol_id: None,
            name: "<file>".to_string(),
            kind: "file".to_string(),
            relative_path: step
                .evidence
                .first()
                .map(|value| value.source_path.clone())
                .unwrap_or_default(),
            start_line: step
                .evidence
                .first()
                .map(|value| value.start_line)
                .unwrap_or(0),
            end_line: step
                .evidence
                .first()
                .map(|value| value.end_line)
                .unwrap_or(0),
            source_role: "production".to_string(),
            depth: step.depth,
            score: step.score,
            possible: step.possible,
            evidence: step.evidence,
        }
    }
}

fn compare_nodes(left: &ImpactNode, right: &ImpactNode) -> std::cmp::Ordering {
    right
        .score
        .total_cmp(&left.score)
        .then(left.relative_path.cmp(&right.relative_path))
        .then(left.start_line.cmp(&right.start_line))
        .then(left.logical_key.cmp(&right.logical_key))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, ScopeKind};
    use crate::engine::changes::AnalyzeChangesRequest;
    use crate::engine::relationships::extract_relationships;
    use crate::engine::symbols::extract_symbols;
    use crate::engine::{build_root_hash, scan_repo};
    use std::fs;
    use std::path::PathBuf;
    use std::process::Command;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("agent-context-impact-{label}-{nonce}"))
    }

    fn git(repo: &Path, args: &[&str]) {
        let output = Command::new("git")
            .current_dir(repo)
            .args(args)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "git {:?} failed: {}",
            args,
            String::from_utf8_lossy(&output.stderr)
        );
    }

    #[test]
    fn graph_filesystem_token_tracks_edits_and_new_files() {
        let root = temp_dir("git-token");
        fs::create_dir_all(&root).unwrap();
        git(&root, &["init", "-q"]);
        fs::write(root.join("source.rs"), "fn first() {}\n").unwrap();
        let paths = vec!["source.rs".to_string()];
        let first = graph_filesystem_token(&root, &paths).unwrap();
        fs::write(root.join("source.rs"), "fn something_longer() {}\n").unwrap();
        let second = graph_filesystem_token(&root, &paths).unwrap();
        assert_ne!(first, second);
        fs::write(root.join("new.rs"), "fn added() {}\n").unwrap();
        let third = graph_filesystem_token(&root, &paths).unwrap();
        assert_ne!(second, third);
        let _ = fs::remove_dir_all(root);
    }

    async fn index_local_graph(engine: &Engine, config: &Config, repo: &Path) {
        let policy = config.indexing.policy_for_repo(repo);
        let files = scan_repo(repo, &[], &[], &policy).unwrap();
        let repo_key = repo.display().to_string();
        for (relative_path, file) in &files {
            let source = fs::read_to_string(&file.absolute_path).unwrap();
            let symbols = extract_symbols(
                &repo_key,
                relative_path,
                Path::new(relative_path),
                &source,
                "now",
                &file.hash,
            )
            .unwrap();
            engine
                .inner
                .symbol_store
                .replace_file_symbols(&repo_key, relative_path, &symbols)
                .unwrap();
            let extracted = extract_relationships(
                &repo_key,
                relative_path,
                Path::new(relative_path),
                &source,
                &symbols,
                &file.hash,
            )
            .unwrap();
            engine
                .inner
                .graph_store
                .replace_file(
                    &repo_key,
                    relative_path,
                    &extracted.references,
                    &extracted.coverage,
                    &file.hash,
                )
                .unwrap();
        }
        engine.resolve_graph(repo, &repo_key, &[]).await.unwrap();
        let hashes = files
            .into_iter()
            .map(|(path, file)| (path, file.hash))
            .collect::<BTreeMap<_, _>>();
        engine
            .set_graph_state(&repo_key, "ready", Some(&build_root_hash(&hashes)))
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn rust_fixture_runs_coverage_impact_path_and_change_analysis_end_to_end() {
        let root = temp_dir("end-to-end");
        let repo = root.join("repo");
        fs::create_dir_all(repo.join("src")).unwrap();
        let baseline = r#"
pub fn target(value: u64) -> u64 { value }
pub fn caller() -> u64 { target(1) }
pub fn transitive() -> u64 { caller() }
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn impacted_test() { let value = transitive(); assert_eq!(value, 1); }
}
"#;
        fs::write(repo.join("src/lib.rs"), baseline).unwrap();
        git(&repo, &["init", "-q"]);
        git(&repo, &["config", "user.email", "test@example.invalid"]);
        git(&repo, &["config", "user.name", "Impact Test"]);
        let signing_key = root.join("fixture-signing-key");
        let keygen = Command::new("ssh-keygen")
            .args(["-q", "-t", "ed25519", "-N", "", "-f"])
            .arg(&signing_key)
            .output()
            .unwrap();
        assert!(
            keygen.status.success(),
            "ssh-keygen failed: {}",
            String::from_utf8_lossy(&keygen.stderr)
        );
        git(&repo, &["config", "gpg.format", "ssh"]);
        git(&repo, &["config", "gpg.ssh.program", "ssh-keygen"]);
        git(
            &repo,
            &["config", "user.signingkey", signing_key.to_str().unwrap()],
        );
        git(&repo, &["config", "commit.gpgsign", "true"]);
        git(&repo, &["add", "src/lib.rs"]);
        git(&repo, &["commit", "-q", "-m", "baseline"]);

        let current = baseline.replace("value: u64", "value: u32");
        fs::write(repo.join("src/lib.rs"), &current).unwrap();
        let config_path = root.join("config.toml");
        fs::write(
            &config_path,
            format!(
                r#"
snapshot_path = "./snapshot.json"
index_root = "./index"

[embedding]
provider = "ollama"
model = "qwen3-embedding"

[embedding.ollama]
base_url = "http://127.0.0.1:11434"
dimensions = 8

[milvus]
address = "127.0.0.1:19530"

[worktrees]
mode = "full"

[[groups]]
id = "fixture"
repos = ["{}"]
"#,
                repo.display()
            ),
        )
        .unwrap();
        let config = Config::load_from_path(&config_path).unwrap();
        let engine = Engine::new(&config).await.unwrap();
        index_local_graph(&engine, &config, &repo).await;
        let scope = ResolvedScope {
            kind: ScopeKind::Repo,
            id: repo.display().to_string(),
            label: "fixture".to_string(),
            repos: vec![repo.clone()],
        };
        let symbols = engine
            .inner
            .symbol_store
            .file_symbols(&repo.display().to_string(), "src/lib.rs")
            .unwrap();
        let target = symbols
            .iter()
            .find(|symbol| symbol.name == "target")
            .unwrap();
        let transitive = symbols
            .iter()
            .find(|symbol| symbol.name == "transitive")
            .unwrap();

        let coverage = engine
            .relationship_coverage(scope.clone(), None)
            .await
            .unwrap();
        assert_eq!(coverage[0].graph_status, "ready");
        assert!(coverage[0].references > 0);

        let impact = engine
            .analyze_impact(
                scope.clone(),
                ImpactRequest {
                    repo: None,
                    symbol_id: Some(target.symbol_id.clone()),
                    file: None,
                    line: None,
                    max_depth: 3,
                    max_nodes: 50,
                    include_tests: true,
                    min_confidence: 650,
                    include_possible: false,
                },
            )
            .await
            .unwrap();
        assert!(
            impact
                .direct_dependents
                .iter()
                .any(|node| node.name == "caller")
        );
        assert!(
            impact
                .transitive_dependents
                .iter()
                .any(|node| node.name == "transitive")
        );
        assert!(
            impact
                .affected_tests
                .iter()
                .any(|node| node.name == "impacted_test"),
            "affected={:?} transitive={:?} possible_tests={:?}",
            impact.affected_tests,
            impact.transitive_dependents,
            impact.possible_tests,
        );

        let path = engine
            .trace_dependency_path(
                scope.clone(),
                TracePathRequest {
                    repo: None,
                    from_symbol_id: Some(transitive.symbol_id.clone()),
                    from_file: None,
                    from_line: None,
                    to_symbol_id: Some(target.symbol_id.clone()),
                    to_file: None,
                    to_line: None,
                    max_depth: 4,
                    max_paths: 3,
                    min_confidence: 650,
                },
            )
            .await
            .unwrap();
        assert!(path.found);
        assert!(path.paths[0].nodes.len() >= 3);

        let path_by_location = engine
            .trace_dependency_path(
                scope.clone(),
                TracePathRequest {
                    repo: None,
                    from_symbol_id: None,
                    from_file: Some(transitive.relative_path.clone()),
                    from_line: Some(transitive.start_line),
                    to_symbol_id: None,
                    to_file: Some(target.relative_path.clone()),
                    to_line: Some(target.start_line),
                    max_depth: 4,
                    max_paths: 3,
                    min_confidence: 650,
                },
            )
            .await
            .unwrap();
        assert!(path_by_location.found);
        assert_eq!(
            path_by_location.from.symbol_id,
            Some(transitive.symbol_id.clone())
        );
        assert_eq!(
            path_by_location.to.symbol_id,
            Some(target.symbol_id.clone())
        );

        let changes = engine
            .analyze_changes(
                scope.clone(),
                AnalyzeChangesRequest {
                    repo: None,
                    base_ref: "HEAD".to_string(),
                    include_untracked: false,
                    max_depth: 2,
                    max_nodes: 50,
                    include_tests: true,
                    min_confidence: 650,
                },
            )
            .await
            .unwrap();
        assert!(!changes.needs_index);
        assert!(!changes.invalid_base);
        assert!(changes.symbols.iter().any(|change| {
            change.change == "signature_changed"
                && change
                    .current
                    .as_ref()
                    .is_some_and(|symbol| symbol.name == "target")
        }));
        assert!(
            changes
                .impacts
                .iter()
                .any(|impact| impact.analysis.is_some())
        );

        let deleted = current
            .lines()
            .filter(|line| !line.trim_start().starts_with("pub fn target"))
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(repo.join("src/lib.rs"), &deleted).unwrap();
        index_local_graph(&engine, &config, &repo).await;
        let deletion = engine
            .analyze_changes(
                scope.clone(),
                AnalyzeChangesRequest {
                    repo: None,
                    base_ref: "HEAD".to_string(),
                    include_untracked: false,
                    max_depth: 2,
                    max_nodes: 50,
                    include_tests: true,
                    min_confidence: 650,
                },
            )
            .await
            .unwrap();
        let deleted_impact = deletion
            .impacts
            .iter()
            .find(|impact| impact.change == "deleted" && impact.qualified_name == "target")
            .expect("deleted target impact");
        assert!(deleted_impact.analysis.is_none());
        assert!(deleted_impact.possible_evidence.iter().any(|evidence| {
            evidence.confidence == CONFIDENCE_LEXICAL && evidence.text.contains("target(1)")
        }));

        let caller = engine
            .inner
            .symbol_store
            .file_symbols(&repo.display().to_string(), "src/lib.rs")
            .unwrap()
            .into_iter()
            .find(|symbol| symbol.name == "caller")
            .unwrap();
        fs::write(repo.join("src/lib.rs"), format!("{deleted}\n// stale\n")).unwrap();
        let stale_error = engine
            .analyze_impact(
                scope,
                ImpactRequest {
                    repo: None,
                    symbol_id: Some(caller.symbol_id),
                    file: None,
                    line: None,
                    max_depth: 2,
                    max_nodes: 50,
                    include_tests: true,
                    min_confidence: 650,
                    include_possible: false,
                },
            )
            .await
            .unwrap_err();
        assert!(stale_error.to_string().contains("stale"));
        let _ = fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn trace_prefers_the_highest_confidence_equal_depth_path() {
        let root = temp_dir("trace-ranking");
        let repo = root.join("repo");
        fs::create_dir_all(repo.join("src")).unwrap();
        fs::write(
            repo.join("src/paths.ts"),
            r#"
class Target {}
class D extends Target {}
class C extends D {}
function B(): D { return new D(); }
function A(value: C) {
  B();
}
"#,
        )
        .unwrap();
        let config_path = root.join("config.toml");
        fs::write(
            &config_path,
            format!(
                r#"
snapshot_path = "./snapshot.json"
index_root = "./index"

[embedding]
provider = "ollama"
model = "qwen3-embedding"

[embedding.ollama]
base_url = "http://127.0.0.1:11434"
dimensions = 8

[milvus]
address = "127.0.0.1:19530"

[worktrees]
mode = "full"

[[groups]]
id = "fixture"
repos = ["{}"]
"#,
                repo.display()
            ),
        )
        .unwrap();
        let config = Config::load_from_path(&config_path).unwrap();
        let engine = Engine::new(&config).await.unwrap();
        index_local_graph(&engine, &config, &repo).await;
        let symbols = engine
            .inner
            .symbol_store
            .file_symbols(&repo.display().to_string(), "src/paths.ts")
            .unwrap();
        let from = symbols.iter().find(|symbol| symbol.name == "A").unwrap();
        let to = symbols
            .iter()
            .find(|symbol| symbol.name == "Target")
            .unwrap();
        let scope = ResolvedScope {
            kind: ScopeKind::Repo,
            id: repo.display().to_string(),
            label: "fixture".to_string(),
            repos: vec![repo.clone()],
        };
        let result = engine
            .trace_dependency_path(
                scope,
                TracePathRequest {
                    repo: None,
                    from_symbol_id: Some(from.symbol_id.clone()),
                    from_file: None,
                    from_line: None,
                    to_symbol_id: Some(to.symbol_id.clone()),
                    to_file: None,
                    to_line: None,
                    max_depth: 4,
                    max_paths: 3,
                    min_confidence: 650,
                },
            )
            .await
            .unwrap();
        assert!(result.found);
        assert!(result.paths.len() >= 2);
        assert!(
            result.paths[0].nodes.iter().any(|node| node.name == "B"),
            "highest path was {:?}",
            result.paths[0]
                .nodes
                .iter()
                .map(|node| node.name.as_str())
                .collect::<Vec<_>>()
        );
        assert!(
            result
                .paths
                .iter()
                .any(|path| { path.nodes.iter().any(|node| node.name == "C") })
        );
        let _ = fs::remove_dir_all(root);
    }
}
