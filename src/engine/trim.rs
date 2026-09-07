use anyhow::{Context, Result};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::fs;
use std::path::Path;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};
use xxhash_rust::xxh3::xxh3_128;

use super::relationships::{
    AnalysisRelation, RelationKind, RepoRelationshipCoverage, compatible_symbol,
};
use super::symbols::{IndexedSymbol, source_role_for_path};
use super::{Engine, ResolvedScope};

const DEFAULT_MIN_CONFIDENCE: u64 = 650;
const MAX_GROUPS: usize = 2_000;
const MAX_RETURNED_ROOTS: usize = 200;
const MAX_VALIDATION_GROUPS: usize = 100;

#[derive(Debug, Clone)]
pub struct DeadCodeRequest {
    pub repo: Option<String>,
    pub min_confidence: u64,
    pub max_groups: usize,
    pub include_tests: bool,
    pub include_risk_candidates: bool,
}

impl Default for DeadCodeRequest {
    fn default() -> Self {
        Self {
            repo: None,
            min_confidence: DEFAULT_MIN_CONFIDENCE,
            max_groups: 250,
            include_tests: true,
            include_risk_candidates: true,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DeadCodeClassification {
    Unreachable,
    TestOnly,
    IntentionalTestSeam,
    ReplacedOrLegacy,
    DynamicOrExternalRisk,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DeadCodeRoot {
    pub logical_key: String,
    pub name: String,
    pub relative_path: String,
    pub start_line: u64,
    pub kind: String,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DeadCodeSymbol {
    pub logical_key: String,
    pub symbol_id: String,
    pub name: String,
    pub qualified_name: String,
    pub kind: String,
    pub relative_path: String,
    pub start_line: u64,
    pub end_line: u64,
    pub source_role: String,
    pub signature: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RemovalRange {
    pub relative_path: String,
    pub start_line: u64,
    pub end_line: u64,
    pub loc: u64,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RemovalBlocker {
    pub kind: String,
    pub detail: String,
    pub relative_path: Option<String>,
    pub line: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DeadCodeGroup {
    pub group_id: String,
    pub classification: DeadCodeClassification,
    pub confidence: u64,
    pub symbols: Vec<DeadCodeSymbol>,
    pub removal_ranges: Vec<RemovalRange>,
    pub source_loc: u64,
    pub compiler_validated_loc: u64,
    pub test_reference_count: u64,
    pub incoming_possible_count: u64,
    pub whole_files: Vec<String>,
    pub blockers: Vec<RemovalBlocker>,
    pub rationale: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TrimTotals {
    pub indexed_rust_symbols: u64,
    pub production_reachable_symbols: u64,
    pub candidate_symbols: u64,
    pub candidate_loc: u64,
    pub clean_candidate_loc: u64,
    pub compiler_validated_loc: u64,
    pub review_required_loc: u64,
    pub excluded_risk_loc: u64,
    pub unresolved_internal_documents: u64,
    pub unresolved_external_documents: u64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DeadCodeAnalysis {
    pub scope: String,
    pub repo: String,
    pub status: String,
    pub coverage: RepoRelationshipCoverage,
    pub root_count: u64,
    pub roots_truncated: bool,
    pub roots: Vec<DeadCodeRoot>,
    pub group_count: u64,
    pub groups: Vec<DeadCodeGroup>,
    pub totals: TrimTotals,
    pub analysis_blockers: Vec<RemovalBlocker>,
    pub truncated: bool,
    pub diagnostic: String,
}

#[derive(Debug, Clone)]
pub struct ExplainRemovalRequest {
    pub repo: Option<String>,
    pub group_id: String,
    pub min_confidence: u64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RemovalExplanation {
    pub scope: String,
    pub repo: String,
    pub status: String,
    pub group: Option<DeadCodeGroup>,
    pub coverage: RepoRelationshipCoverage,
    pub analysis_blockers: Vec<RemovalBlocker>,
    pub diagnostic: String,
}

#[derive(Debug, Clone)]
pub struct ValidateRemovalRequest {
    pub repo: Option<String>,
    pub group_ids: Vec<String>,
    pub min_confidence: u64,
    pub all_features: bool,
    pub include_tests: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RemovalValidation {
    pub scope: String,
    pub repo: String,
    pub status: String,
    pub group_ids: Vec<String>,
    pub validated_group_ids: Vec<String>,
    pub rejected_group_ids: Vec<String>,
    pub requested_loc: u64,
    pub compiler_validated_loc: u64,
    pub baseline_passed: bool,
    pub command: Vec<String>,
    pub exit_code: Option<i32>,
    pub diagnostics: String,
    pub source_checkout_untouched: bool,
    pub temporary_checkout_removed: bool,
}

#[derive(Debug, Clone)]
pub struct EstimateTrimRequest {
    pub repo: Option<String>,
    pub min_confidence: u64,
    pub max_groups: usize,
    pub include_tests: bool,
    pub include_risk_candidates: bool,
    pub validate: bool,
    pub validation_group_limit: usize,
    pub all_features: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TrimEstimate {
    pub analysis: DeadCodeAnalysis,
    pub validation: Option<RemovalValidation>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum RootClass {
    Hard,
    PublicApi,
    Dynamic,
}

#[derive(Default)]
struct SymbolSignals {
    public_api: bool,
    dynamic: bool,
    feature_gated: bool,
    intentional_test_seam: bool,
    legacy: bool,
    reasons: Vec<String>,
}

struct GraphSnapshot {
    symbols: Vec<IndexedSymbol>,
    relations: Vec<AnalysisRelation>,
    coverage: RepoRelationshipCoverage,
}

impl Engine {
    pub async fn analyze_dead_code(
        &self,
        scope: ResolvedScope,
        request: DeadCodeRequest,
    ) -> Result<DeadCodeAnalysis> {
        let repo = self
            .resolve_prepare_repo(&scope, request.repo.as_deref(), true)?
            .context("dead-code analysis requires one repository")?;
        let graph = self.verified_trim_graph(&repo).await?;
        tokio::task::spawn_blocking(move || analyze_snapshot(scope.id, &repo, graph, &request))
            .await
            .context("joining repository trim analysis")?
    }

    pub async fn explain_removal(
        &self,
        scope: ResolvedScope,
        request: ExplainRemovalRequest,
    ) -> Result<RemovalExplanation> {
        let analysis = self
            .analyze_dead_code(
                scope,
                DeadCodeRequest {
                    repo: request.repo,
                    min_confidence: request.min_confidence,
                    max_groups: MAX_GROUPS,
                    include_tests: true,
                    include_risk_candidates: true,
                },
            )
            .await?;
        let group = analysis
            .groups
            .iter()
            .find(|group| group.group_id == request.group_id)
            .cloned();
        Ok(RemovalExplanation {
            scope: analysis.scope.clone(),
            repo: analysis.repo.clone(),
            status: if group.is_some() {
                "found".to_string()
            } else {
                "not_found".to_string()
            },
            group,
            coverage: analysis.coverage.clone(),
            analysis_blockers: analysis.analysis_blockers.clone(),
            diagnostic: if analysis.truncated {
                "The requested group was resolved against the maximum bounded analysis set."
                    .to_string()
            } else {
                "Removal evidence is read-only and remains advisory until compiler validation passes."
                    .to_string()
            },
        })
    }

    pub async fn validate_removal(
        &self,
        scope: ResolvedScope,
        request: ValidateRemovalRequest,
    ) -> Result<RemovalValidation> {
        let repo = self
            .resolve_prepare_repo(&scope, request.repo.as_deref(), true)?
            .context("removal validation requires one repository")?;
        if request.group_ids.is_empty() || request.group_ids.len() > MAX_VALIDATION_GROUPS {
            return Ok(RemovalValidation {
                scope: scope.id,
                repo: repo.display().to_string(),
                status: "invalid_selection".to_string(),
                group_ids: request.group_ids,
                validated_group_ids: Vec::new(),
                rejected_group_ids: Vec::new(),
                requested_loc: 0,
                compiler_validated_loc: 0,
                baseline_passed: false,
                command: Vec::new(),
                exit_code: None,
                diagnostics: format!("select between 1 and {MAX_VALIDATION_GROUPS} removal groups"),
                source_checkout_untouched: true,
                temporary_checkout_removed: true,
            });
        }
        let analysis = self
            .analyze_dead_code(
                scope.clone(),
                DeadCodeRequest {
                    repo: request.repo,
                    min_confidence: request.min_confidence,
                    max_groups: MAX_GROUPS,
                    include_tests: true,
                    include_risk_candidates: true,
                },
            )
            .await?;
        let requested = request.group_ids.iter().cloned().collect::<BTreeSet<_>>();
        let groups = analysis
            .groups
            .iter()
            .filter(|group| requested.contains(&group.group_id))
            .cloned()
            .collect::<Vec<_>>();
        if groups.len() != requested.len() {
            let found = groups
                .iter()
                .map(|group| group.group_id.clone())
                .collect::<BTreeSet<_>>();
            let missing = requested.difference(&found).cloned().collect::<Vec<_>>();
            return Ok(RemovalValidation {
                scope: scope.id,
                repo: repo.display().to_string(),
                status: "invalid_selection".to_string(),
                group_ids: request.group_ids,
                validated_group_ids: Vec::new(),
                rejected_group_ids: missing.clone(),
                requested_loc: groups.iter().map(|group| group.source_loc).sum(),
                compiler_validated_loc: 0,
                baseline_passed: false,
                command: Vec::new(),
                exit_code: None,
                diagnostics: format!("unknown or truncated group ids: {}", missing.join(", ")),
                source_checkout_untouched: true,
                temporary_checkout_removed: true,
            });
        }
        let scope_id = scope.id;
        let repo_for_validation = repo.clone();
        let validation = tokio::task::spawn_blocking(move || {
            validate_groups_in_temporary_checkout(
                &scope_id,
                &repo_for_validation,
                &groups,
                request.all_features,
                request.include_tests,
            )
        })
        .await
        .context("joining disposable removal validation")??;
        Ok(validation)
    }

    pub async fn estimate_trim(
        &self,
        scope: ResolvedScope,
        request: EstimateTrimRequest,
    ) -> Result<TrimEstimate> {
        let analysis = self
            .analyze_dead_code(
                scope.clone(),
                DeadCodeRequest {
                    repo: request.repo.clone(),
                    min_confidence: request.min_confidence,
                    max_groups: request.max_groups,
                    include_tests: request.include_tests,
                    include_risk_candidates: request.include_risk_candidates,
                },
            )
            .await?;
        if !request.validate {
            return Ok(TrimEstimate {
                analysis,
                validation: None,
            });
        }
        let group_ids = analysis
            .groups
            .iter()
            .filter(|group| group_is_validation_eligible(group))
            .take(request.validation_group_limit.clamp(1, 100))
            .map(|group| group.group_id.clone())
            .collect::<Vec<_>>();
        if group_ids.is_empty() {
            return Ok(TrimEstimate {
                analysis,
                validation: None,
            });
        }
        let validation = self
            .validate_removal(
                scope,
                ValidateRemovalRequest {
                    repo: request.repo,
                    group_ids,
                    min_confidence: request.min_confidence,
                    all_features: request.all_features,
                    include_tests: request.include_tests,
                },
            )
            .await?;
        let mut analysis = analysis;
        analysis.totals.compiler_validated_loc = validation.compiler_validated_loc;
        Ok(TrimEstimate {
            analysis,
            validation: Some(validation),
        })
    }

    async fn verified_trim_graph(&self, repo: &Path) -> Result<GraphSnapshot> {
        let view = self.verified_graph_view_for_repo(repo, true).await?;
        let mut symbols_by_key = BTreeMap::<String, IndexedSymbol>::new();
        let mut relations = Vec::new();

        for source in &view.sources {
            let symbol_store = self.inner.symbol_store.clone();
            let graph_store = self.inner.graph_store.clone();
            let repo_key = source.repo_key.clone();
            let repo_key_for_relations = repo_key.clone();
            let (symbols, source_relations) = tokio::try_join!(
                self.run_search_lexical_blocking("trim_symbols", move || {
                    symbol_store.all_symbols(&repo_key)
                }),
                self.run_search_lexical_blocking("trim_relations", move || {
                    graph_store.all_analysis_relations(&repo_key_for_relations)
                })
            )?;
            for symbol in symbols {
                if source.suppressed_paths.contains(&symbol.relative_path) {
                    continue;
                }
                symbols_by_key
                    .entry(symbol.logical_key.clone())
                    .or_insert(symbol);
            }
            relations.extend(
                source_relations
                    .into_iter()
                    .filter(|relation| !source.suppressed_paths.contains(&relation.source_path)),
            );
        }

        Ok(GraphSnapshot {
            symbols: symbols_by_key.into_values().collect(),
            relations,
            coverage: view.coverage,
        })
    }
}

fn analyze_snapshot(
    scope: String,
    repo: &Path,
    graph: GraphSnapshot,
    request: &DeadCodeRequest,
) -> Result<DeadCodeAnalysis> {
    let min_confidence = request.min_confidence.clamp(DEFAULT_MIN_CONFIDENCE, 1_000);
    let symbols = graph
        .symbols
        .into_iter()
        .filter(|symbol| symbol.language == "rust")
        .collect::<Vec<_>>();
    let symbol_map = symbols
        .iter()
        .map(|symbol| (symbol.logical_key.clone(), symbol))
        .collect::<HashMap<_, _>>();
    let symbols_by_name = symbols.iter().fold(
        HashMap::<String, Vec<&IndexedSymbol>>::new(),
        |mut by_name, symbol| {
            by_name.entry(symbol.name.clone()).or_default().push(symbol);
            by_name
        },
    );
    let source_text = load_symbol_files(repo, &symbols);
    let source_lines = source_text
        .iter()
        .map(|(path, text)| (path.clone(), text.lines().collect::<Vec<_>>()))
        .collect::<HashMap<_, _>>();
    let signals = symbols
        .iter()
        .map(|symbol| {
            (
                symbol.logical_key.clone(),
                symbol_signals(
                    symbol,
                    source_lines
                        .get(&symbol.relative_path)
                        .map(|lines| lines.as_slice()),
                ),
            )
        })
        .collect::<HashMap<_, _>>();
    let dynamic_container_keys = signals
        .iter()
        .filter(|(_, signal)| signal.dynamic || signal.feature_gated)
        .map(|(key, _)| key.clone())
        .collect::<BTreeSet<_>>();

    let mut adjacency = HashMap::<String, Vec<String>>::new();
    let mut definite_incoming = HashMap::<String, Vec<&AnalysisRelation>>::new();
    let mut possible_incoming = HashMap::<String, Vec<&AnalysisRelation>>::new();
    let mut unresolved_by_name = HashMap::<String, Vec<&AnalysisRelation>>::new();
    let mut trait_implementation_keys = BTreeSet::new();
    for relation in &graph.relations {
        if relation.language != "rust" {
            continue;
        }
        if matches!(
            relation.kind,
            RelationKind::Implements | RelationKind::Inherits
        ) {
            trait_implementation_keys.insert(relation.source_key.clone());
        }
        let Some(target) = relation.target_key.as_ref() else {
            unresolved_by_name
                .entry(relation.target_name.clone())
                .or_default()
                .push(relation);
            continue;
        };
        if relation.confidence >= min_confidence {
            adjacency
                .entry(relation.source_key.clone())
                .or_default()
                .push(target.clone());
            definite_incoming
                .entry(target.clone())
                .or_default()
                .push(relation);
        } else {
            possible_incoming
                .entry(target.clone())
                .or_default()
                .push(relation);
        }
    }
    for targets in adjacency.values_mut() {
        targets.sort();
        targets.dedup();
    }
    let (unresolved_internal_documents, unresolved_external_documents) = graph
        .relations
        .iter()
        .filter(|relation| relation.language == "rust" && relation.target_key.is_none())
        .fold((0_u64, 0_u64), |(internal, external), relation| {
            if symbols_by_name
                .get(&relation.target_name)
                .is_some_and(|symbols| {
                    symbols
                        .iter()
                        .any(|symbol| compatible_symbol(relation.kind, symbol))
                })
            {
                (internal + 1, external)
            } else {
                (internal, external + 1)
            }
        });

    let mut hard_roots = BTreeSet::new();
    let mut public_roots = BTreeSet::new();
    let mut dynamic_roots = BTreeSet::new();
    let mut test_roots = BTreeSet::new();
    let mut root_rows = Vec::new();
    for symbol in &symbols {
        if symbol.source_role == "test" || source_role_for_path(&symbol.relative_path) == "test" {
            test_roots.insert(symbol.logical_key.clone());
            continue;
        }
        let signal = &signals[&symbol.logical_key];
        let mut classes = root_classes(symbol, signal);
        if symbol
            .parent_logical_key
            .as_ref()
            .is_some_and(|parent| dynamic_container_keys.contains(parent))
            && !classes
                .iter()
                .any(|(class, _)| *class == RootClass::Dynamic)
        {
            classes.push((
                RootClass::Dynamic,
                "member of a dynamically registered or feature-gated container".to_string(),
            ));
        }
        if trait_implementation_keys.contains(&symbol.logical_key)
            || symbol
                .parent_logical_key
                .as_ref()
                .is_some_and(|parent| trait_implementation_keys.contains(parent))
        {
            classes.push((
                RootClass::Dynamic,
                "trait implementation or inheritance may be reached through dynamic dispatch"
                    .to_string(),
            ));
        }
        for (class, reason) in classes {
            match class {
                RootClass::Hard => {
                    hard_roots.insert(symbol.logical_key.clone());
                }
                RootClass::PublicApi => {
                    public_roots.insert(symbol.logical_key.clone());
                }
                RootClass::Dynamic => {
                    dynamic_roots.insert(symbol.logical_key.clone());
                }
            }
            root_rows.push(DeadCodeRoot {
                logical_key: symbol.logical_key.clone(),
                name: symbol.name.clone(),
                relative_path: symbol.relative_path.clone(),
                start_line: symbol.start_line,
                kind: match class {
                    RootClass::Hard => "production".to_string(),
                    RootClass::PublicApi => "public_api".to_string(),
                    RootClass::Dynamic => "dynamic_registration".to_string(),
                },
                reason,
            });
        }
    }
    root_rows.sort_by(|left, right| {
        left.relative_path
            .cmp(&right.relative_path)
            .then(left.start_line.cmp(&right.start_line))
            .then(left.logical_key.cmp(&right.logical_key))
    });
    root_rows
        .dedup_by(|left, right| left.logical_key == right.logical_key && left.kind == right.kind);
    let root_count = root_rows.len();
    root_rows.truncate(MAX_RETURNED_ROOTS);

    let hard_reachable = reachable(&hard_roots, &adjacency);
    let public_reachable = reachable(&public_roots, &adjacency);
    let dynamic_reachable = reachable(&dynamic_roots, &adjacency);
    let production_reachable = hard_reachable
        .union(&public_reachable)
        .cloned()
        .collect::<BTreeSet<_>>()
        .union(&dynamic_reachable)
        .cloned()
        .collect::<BTreeSet<_>>();
    let test_reachable = if request.include_tests {
        reachable(&test_roots, &adjacency)
    } else {
        BTreeSet::new()
    };
    let candidate_keys = symbols
        .iter()
        .filter(|symbol| {
            symbol.source_role == "production"
                && source_role_for_path(&symbol.relative_path) == "production"
                && symbol.kind != "impl"
                && !hard_reachable.contains(&symbol.logical_key)
        })
        .map(|symbol| symbol.logical_key.clone())
        .collect::<BTreeSet<_>>();
    let components = strongly_connected_components(&candidate_keys, &adjacency);

    let file_production_keys = symbols
        .iter()
        .filter(|symbol| {
            symbol.source_role == "production"
                && source_role_for_path(&symbol.relative_path) == "production"
                && symbol.kind != "impl"
        })
        .fold(
            HashMap::<String, BTreeSet<String>>::new(),
            |mut by_file, symbol| {
                by_file
                    .entry(symbol.relative_path.clone())
                    .or_default()
                    .insert(symbol.logical_key.clone());
                by_file
            },
        );
    let mut groups = Vec::new();
    for component in components {
        let component_symbols = component
            .iter()
            .filter_map(|key| symbol_map.get(key).copied())
            .collect::<Vec<_>>();
        if component_symbols.is_empty() {
            continue;
        }
        let mut blockers = Vec::new();
        let mut rationale = Vec::new();
        let mut test_reference_count = 0_u64;
        let mut incoming_possible_count = 0_u64;
        let mut has_test_reach = false;
        let mut has_public_reach = false;
        let mut has_dynamic_reach = false;
        let mut intentional_test_seam = false;
        let mut legacy = false;

        for symbol in &component_symbols {
            let signal = &signals[&symbol.logical_key];
            has_test_reach |= test_reachable.contains(&symbol.logical_key);
            has_public_reach |= public_reachable.contains(&symbol.logical_key);
            has_dynamic_reach |= dynamic_reachable.contains(&symbol.logical_key);
            intentional_test_seam |= signal.intentional_test_seam;
            legacy |= signal.legacy;
            if !symbol.identity_stable {
                blockers.push(blocker(
                    "unstable_identity",
                    "declaration identity is unstable",
                    Some(symbol.relative_path.clone()),
                    Some(symbol.start_line),
                ));
            }
            if signal.public_api {
                blockers.push(blocker(
                    "public_api",
                    "public API may have consumers outside the indexed repository",
                    Some(symbol.relative_path.clone()),
                    Some(symbol.start_line),
                ));
            }
            if signal.dynamic || signal.feature_gated {
                blockers.push(blocker(
                    if signal.feature_gated {
                        "feature_gated"
                    } else {
                        "dynamic_registration"
                    },
                    signal.reasons.join("; "),
                    Some(symbol.relative_path.clone()),
                    Some(symbol.start_line),
                ));
            }
            if let Some(relations) = possible_incoming.get(&symbol.logical_key) {
                incoming_possible_count += relations.len() as u64;
                for relation in relations.iter().take(3) {
                    blockers.push(blocker(
                        "possible_incoming_reference",
                        format!(
                            "{} reference at confidence {} ({})",
                            relation.kind.as_str(),
                            relation.confidence,
                            relation.resolution
                        ),
                        Some(relation.source_path.clone()),
                        Some(relation.start_line),
                    ));
                }
            }
            if let Some(relations) = unresolved_by_name.get(&symbol.name) {
                for relation in relations
                    .iter()
                    .filter(|relation| compatible_symbol(relation.kind, symbol))
                    .take(3)
                {
                    blockers.push(blocker(
                        "unresolved_name_match",
                        format!(
                            "unresolved {} reference names `{}`",
                            relation.kind.as_str(),
                            symbol.name
                        ),
                        Some(relation.source_path.clone()),
                        Some(relation.start_line),
                    ));
                }
            }
            if let Some(relations) = definite_incoming.get(&symbol.logical_key) {
                test_reference_count += relations
                    .iter()
                    .filter(|relation| {
                        relation.source_role == "test"
                            || source_role_for_path(&relation.source_path) == "test"
                    })
                    .count() as u64;
            }
        }

        let classification = if intentional_test_seam && has_test_reach {
            DeadCodeClassification::IntentionalTestSeam
        } else if has_dynamic_reach || has_public_reach {
            DeadCodeClassification::DynamicOrExternalRisk
        } else if legacy {
            DeadCodeClassification::ReplacedOrLegacy
        } else if has_test_reach {
            DeadCodeClassification::TestOnly
        } else {
            DeadCodeClassification::Unreachable
        };
        if matches!(classification, DeadCodeClassification::Unreachable) {
            rationale.push("not reachable from any discovered production root".to_string());
        }
        if has_test_reach {
            rationale.push("reachable from one or more test roots".to_string());
        }
        if has_public_reach {
            rationale.push("reachable from an externally public API root".to_string());
        }
        if has_dynamic_reach {
            rationale.push("reachable from a dynamic registration root".to_string());
        }

        blockers.sort_by(|left, right| {
            left.kind
                .cmp(&right.kind)
                .then(left.relative_path.cmp(&right.relative_path))
                .then(left.line.cmp(&right.line))
                .then(left.detail.cmp(&right.detail))
        });
        blockers.dedup_by(|left, right| {
            left.kind == right.kind
                && left.relative_path == right.relative_path
                && left.line == right.line
                && left.detail == right.detail
        });
        let removal_ranges = removal_ranges(repo, &component_symbols, &source_lines);
        let source_loc = removal_ranges.iter().map(|range| range.loc).sum::<u64>();
        let component_set = component.iter().cloned().collect::<BTreeSet<_>>();
        let whole_files = component_symbols
            .iter()
            .map(|symbol| symbol.relative_path.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .filter(|path| file_production_keys.get(path) == Some(&component_set))
            .collect::<Vec<_>>();
        let confidence = group_confidence(&classification, &blockers, &graph.coverage);
        let id_material = component.join("\n");
        let group_id = format!("dead_{:032x}", xxh3_128(id_material.as_bytes()));
        groups.push(DeadCodeGroup {
            group_id,
            classification,
            confidence,
            symbols: component_symbols.into_iter().map(dead_symbol).collect(),
            removal_ranges,
            source_loc,
            compiler_validated_loc: 0,
            test_reference_count,
            incoming_possible_count,
            whole_files,
            blockers,
            rationale,
        });
    }

    groups = merge_removal_closure_groups(
        groups,
        &adjacency,
        &candidate_keys,
        &file_production_keys,
        &source_text,
        &graph.coverage,
    );
    append_dedicated_test_closures(repo, &mut groups, &symbols, &graph.relations, &source_lines);
    for group in &mut groups {
        normalize_group_ranges(group);
    }

    groups.sort_by(|left, right| {
        right
            .source_loc
            .cmp(&left.source_loc)
            .then(right.confidence.cmp(&left.confidence))
            .then(left.group_id.cmp(&right.group_id))
    });
    let all_group_count = groups.len();
    let clean_candidate_loc = groups
        .iter()
        .filter(|group| group_is_validation_eligible(group))
        .map(|group| group.source_loc)
        .sum::<u64>();
    let review_required_loc = groups
        .iter()
        .filter(|group| !group_is_validation_eligible(group) && !group_is_excluded_risk(group))
        .map(|group| group.source_loc)
        .sum::<u64>();
    let excluded_risk_loc = groups
        .iter()
        .filter(|group| group_is_excluded_risk(group))
        .map(|group| group.source_loc)
        .sum::<u64>();
    let candidate_loc = clean_candidate_loc.saturating_add(review_required_loc);
    let candidate_symbol_count = groups
        .iter()
        .filter(|group| !group_is_excluded_risk(group))
        .flat_map(|group| group.symbols.iter())
        .filter(|symbol| candidate_keys.contains(&symbol.logical_key))
        .map(|symbol| symbol.logical_key.clone())
        .collect::<BTreeSet<_>>()
        .len();
    if !request.include_risk_candidates {
        groups.retain(|group| !group_is_excluded_risk(group));
    }
    let visible_group_count = groups.len();
    groups.truncate(request.max_groups.clamp(1, MAX_GROUPS));
    let analysis_blockers = coverage_blockers(&graph.coverage);
    let status = if analysis_blockers.is_empty() {
        "ok"
    } else {
        "inconclusive"
    };
    Ok(DeadCodeAnalysis {
        scope,
        repo: repo.display().to_string(),
        status: status.to_string(),
        coverage: graph.coverage,
        root_count: root_count as u64,
        roots_truncated: root_count > MAX_RETURNED_ROOTS,
        roots: root_rows,
        group_count: all_group_count as u64,
        totals: TrimTotals {
            indexed_rust_symbols: symbols.len() as u64,
            production_reachable_symbols: production_reachable.len() as u64,
            candidate_symbols: candidate_symbol_count as u64,
            candidate_loc,
            clean_candidate_loc,
            compiler_validated_loc: 0,
            review_required_loc,
            excluded_risk_loc,
            unresolved_internal_documents,
            unresolved_external_documents,
        },
        groups,
        analysis_blockers,
        truncated: visible_group_count > request.max_groups.clamp(1, MAX_GROUPS),
        diagnostic: "Reachability is advisory until candidate groups pass disposable compiler validation; unresolved references, public APIs, macros, feature gates, and dynamic registration are reported as blockers.".to_string(),
    })
}

fn root_classes(symbol: &IndexedSymbol, signals: &SymbolSignals) -> Vec<(RootClass, String)> {
    let path = symbol.relative_path.as_str();
    let mut roots = Vec::new();
    let executable_source = path == "build.rs"
        || path.ends_with("/build.rs")
        || path == "src/main.rs"
        || path.ends_with("/src/main.rs")
        || path.starts_with("src/bin/")
        || path.contains("/src/bin/")
        || path.starts_with("examples/")
        || path.contains("/examples/");
    if symbol.name == "main" && executable_source {
        roots.push((
            RootClass::Hard,
            "executable or build entry point".to_string(),
        ));
    } else if symbol.name == "main" {
        roots.push((
            RootClass::Dynamic,
            "main function outside a conventional target path may be configured in Cargo.toml"
                .to_string(),
        ));
    }
    if signals.public_api {
        roots.push((
            RootClass::PublicApi,
            "externally visible Rust API".to_string(),
        ));
    }
    if signals.dynamic {
        roots.push((RootClass::Dynamic, signals.reasons.join("; ")));
    }
    if signals.feature_gated {
        roots.push((
            RootClass::Dynamic,
            "feature-gated declaration may be a production root in another build".to_string(),
        ));
    }
    roots
}

fn symbol_signals(symbol: &IndexedSymbol, lines: Option<&[&str]>) -> SymbolSignals {
    let signature = symbol.signature.trim_start();
    let public_api = signature.starts_with("pub ") || signature.starts_with("pub async ");
    let mut window = signature.to_string();
    if let Some(lines) = lines {
        let start = symbol.start_line.saturating_sub(8) as usize;
        let end = (symbol.start_line as usize + 2).min(lines.len());
        if start < end {
            window.push('\n');
            window.push_str(&lines[start..end].join("\n"));
        }
    }
    let dynamic_tokens = [
        "#[proc_macro",
        "#[no_mangle]",
        "#[unsafe(no_mangle)]",
        "#[export_name",
        "#[cfg(",
        "#[cfg_attr(",
        "#[target_feature",
        "#[used]",
        "#[link_section",
        "extern \"C\"",
        "extern \"system\"",
        "inventory::submit!",
        "linkme::distributed_slice",
        "#[graphql",
        "#[Object]",
        "#[ComplexObject]",
        "#[tauri::command]",
        "#[get(",
        "#[post(",
        "#[put(",
        "#[delete(",
        "#[patch(",
        "#[route(",
        "#[handler]",
        "#[job]",
        "#[worker]",
        "#[inject]",
        "#[provider]",
        "#[derive(Serialize",
        "#[derive(Deserialize",
        "serde::Serialize",
        "serde::Deserialize",
        "typetag::serde",
        "abi_stable",
        "plugin_registry",
    ];
    let dynamic_hits = dynamic_tokens
        .iter()
        .filter(|token| window.contains(**token))
        .map(|token| (*token).to_string())
        .collect::<Vec<_>>();
    let feature_gated = window.contains("#[cfg(feature") || window.contains("cfg_attr(feature");
    let lower_name = symbol.name.to_ascii_lowercase();
    let intentional_test_seam = lower_name.starts_with("test_")
        || lower_name.ends_with("_test")
        || lower_name.contains("_test_")
        || lower_name.contains("snapshot")
        || lower_name.starts_with("mock_")
        || lower_name.ends_with("_for_test");
    let legacy = lower_name.contains("legacy")
        || lower_name.contains("deprecated")
        || lower_name.contains("compat")
        || window.contains("#[deprecated");
    let mut reasons = dynamic_hits
        .into_iter()
        .map(|token| format!("dynamic marker `{token}`"))
        .collect::<Vec<_>>();
    if feature_gated {
        reasons.push("feature-gated declaration".to_string());
    }
    SymbolSignals {
        public_api,
        dynamic: reasons
            .iter()
            .any(|reason| reason.starts_with("dynamic marker")),
        feature_gated,
        intentional_test_seam,
        legacy,
        reasons,
    }
}

fn reachable(
    roots: &BTreeSet<String>,
    adjacency: &HashMap<String, Vec<String>>,
) -> BTreeSet<String> {
    let mut visited = roots.clone();
    let mut queue = roots.iter().cloned().collect::<VecDeque<_>>();
    while let Some(source) = queue.pop_front() {
        if let Some(targets) = adjacency.get(&source) {
            for target in targets {
                if visited.insert(target.clone()) {
                    queue.push_back(target.clone());
                }
            }
        }
    }
    visited
}

fn strongly_connected_components(
    candidates: &BTreeSet<String>,
    adjacency: &HashMap<String, Vec<String>>,
) -> Vec<Vec<String>> {
    let mut visited = BTreeSet::new();
    let mut finish_order = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        if visited.contains(candidate) {
            continue;
        }
        let mut stack = vec![(candidate.clone(), false)];
        while let Some((node, expanded)) = stack.pop() {
            if expanded {
                finish_order.push(node);
                continue;
            }
            if !visited.insert(node.clone()) {
                continue;
            }
            stack.push((node.clone(), true));
            if let Some(targets) = adjacency.get(&node) {
                for target in targets.iter().rev() {
                    if candidates.contains(target) && !visited.contains(target) {
                        stack.push((target.clone(), false));
                    }
                }
            }
        }
    }

    let mut reverse = HashMap::<String, Vec<String>>::new();
    for (source, targets) in adjacency {
        if !candidates.contains(source) {
            continue;
        }
        for target in targets {
            if candidates.contains(target) {
                reverse
                    .entry(target.clone())
                    .or_default()
                    .push(source.clone());
            }
        }
    }
    for sources in reverse.values_mut() {
        sources.sort();
        sources.dedup();
    }

    let mut assigned = BTreeSet::new();
    let mut components = Vec::new();
    for candidate in finish_order.into_iter().rev() {
        if !assigned.insert(candidate.clone()) {
            continue;
        }
        let mut component = Vec::new();
        let mut stack = vec![candidate];
        while let Some(node) = stack.pop() {
            component.push(node.clone());
            if let Some(sources) = reverse.get(&node) {
                for source in sources.iter().rev() {
                    if assigned.insert(source.clone()) {
                        stack.push(source.clone());
                    }
                }
            }
        }
        component.sort();
        components.push(component);
    }
    components
}

fn load_symbol_files(repo: &Path, symbols: &[IndexedSymbol]) -> HashMap<String, String> {
    symbols
        .iter()
        .map(|symbol| symbol.relative_path.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .filter_map(|relative_path| {
            fs::read_to_string(repo.join(&relative_path))
                .ok()
                .map(|text| (relative_path, text))
        })
        .collect()
}

fn removal_ranges(
    repo: &Path,
    symbols: &[&IndexedSymbol],
    source_lines: &HashMap<String, Vec<&str>>,
) -> Vec<RemovalRange> {
    let mut by_file = BTreeMap::<String, Vec<(u64, u64)>>::new();
    for symbol in symbols {
        let lines = source_lines.get(&symbol.relative_path);
        let start = lines
            .map(|lines| declaration_start_line_from_lines(lines, symbol.start_line))
            .unwrap_or(symbol.start_line);
        by_file
            .entry(symbol.relative_path.clone())
            .or_default()
            .push((start, symbol.end_line));
    }
    let mut output = Vec::new();
    for (path, mut ranges) in by_file {
        ranges.sort();
        let mut merged = Vec::<(u64, u64)>::new();
        for (start, end) in ranges {
            if let Some(last) = merged.last_mut()
                && start <= last.1.saturating_add(1)
            {
                last.1 = last.1.max(end);
                continue;
            }
            merged.push((start, end));
        }
        let file_exists = repo.join(&path).is_file();
        for (start, end) in merged {
            output.push(RemovalRange {
                relative_path: path.clone(),
                start_line: start,
                end_line: end,
                loc: end.saturating_sub(start).saturating_add(1),
                reason: if file_exists {
                    "declaration closure including contiguous attributes and documentation"
                        .to_string()
                } else {
                    "indexed declaration span; source file is unavailable".to_string()
                },
            });
        }
    }
    output
}

fn merge_removal_closure_groups(
    groups: Vec<DeadCodeGroup>,
    adjacency: &HashMap<String, Vec<String>>,
    candidate_keys: &BTreeSet<String>,
    file_production_keys: &HashMap<String, BTreeSet<String>>,
    source_text: &HashMap<String, String>,
    coverage: &RepoRelationshipCoverage,
) -> Vec<DeadCodeGroup> {
    let dead_files = file_production_keys
        .iter()
        .filter(|(_, keys)| !keys.is_empty() && keys.is_subset(candidate_keys))
        .map(|(path, _)| path.clone())
        .collect::<BTreeSet<_>>();
    let module_wiring_index = build_module_wiring_index(source_text);
    let group_by_key = groups
        .iter()
        .enumerate()
        .flat_map(|(index, group)| {
            group
                .symbols
                .iter()
                .map(move |symbol| (symbol.logical_key.clone(), index))
        })
        .collect::<HashMap<_, _>>();
    let mut parents = (0..groups.len()).collect::<Vec<_>>();
    for (source, targets) in adjacency {
        let Some(source_group) = group_by_key.get(source).copied() else {
            continue;
        };
        for target in targets {
            if let Some(target_group) = group_by_key.get(target).copied() {
                let source_risk = classification_risk(&groups[source_group].classification);
                let target_risk = classification_risk(&groups[target_group].classification);
                if source_risk >= target_risk {
                    union_group_indexes(&mut parents, source_group, target_group);
                }
            }
        }
    }
    for keys in dead_files
        .iter()
        .filter_map(|path| file_production_keys.get(path))
    {
        let indexes = keys
            .iter()
            .filter_map(|key| group_by_key.get(key).copied())
            .collect::<Vec<_>>();
        if let Some(first) = indexes.first().copied() {
            for index in indexes.into_iter().skip(1) {
                union_group_indexes(&mut parents, first, index);
            }
        }
    }
    let mut buckets = BTreeMap::<usize, Vec<usize>>::new();
    for index in 0..groups.len() {
        let root = find_group_root(&mut parents, index);
        buckets.entry(root).or_default().push(index);
    }

    let mut merged_groups = Vec::with_capacity(buckets.len());
    for indexes in buckets.into_values() {
        let mut symbols = indexes
            .iter()
            .flat_map(|index| groups[*index].symbols.clone())
            .collect::<Vec<_>>();
        symbols.sort_by(|left, right| {
            left.relative_path
                .cmp(&right.relative_path)
                .then(left.start_line.cmp(&right.start_line))
                .then(left.logical_key.cmp(&right.logical_key))
        });
        symbols.dedup_by(|left, right| left.logical_key == right.logical_key);
        let group_keys = symbols
            .iter()
            .map(|symbol| symbol.logical_key.clone())
            .collect::<BTreeSet<_>>();
        let whole_files = dead_files
            .iter()
            .filter(|path| {
                file_production_keys
                    .get(*path)
                    .is_some_and(|keys| keys.is_subset(&group_keys))
            })
            .cloned()
            .collect::<Vec<_>>();
        let mut ranges = indexes
            .iter()
            .flat_map(|index| groups[*index].removal_ranges.clone())
            .collect::<Vec<_>>();
        for path in &whole_files {
            if let Some(text) = source_text.get(path) {
                let line_count = text.lines().count() as u64;
                if line_count > 0 {
                    ranges.push(RemovalRange {
                        relative_path: path.clone(),
                        start_line: 1,
                        end_line: line_count,
                        loc: line_count,
                        reason: "whole file contains no production-reachable declarations"
                            .to_string(),
                    });
                }
            }
            ranges.extend(module_wiring_ranges(path, &module_wiring_index));
        }
        let mut blockers = indexes
            .iter()
            .flat_map(|index| groups[*index].blockers.clone())
            .collect::<Vec<_>>();
        blockers.sort_by(|left, right| {
            left.kind
                .cmp(&right.kind)
                .then(left.relative_path.cmp(&right.relative_path))
                .then(left.line.cmp(&right.line))
                .then(left.detail.cmp(&right.detail))
        });
        blockers.dedup_by(|left, right| {
            left.kind == right.kind
                && left.relative_path == right.relative_path
                && left.line == right.line
                && left.detail == right.detail
        });
        let classification = indexes
            .iter()
            .map(|index| &groups[*index].classification)
            .max_by_key(|classification| classification_risk(classification))
            .cloned()
            .unwrap_or(DeadCodeClassification::Unreachable);
        let mut rationale = indexes
            .iter()
            .flat_map(|index| groups[*index].rationale.clone())
            .collect::<BTreeSet<_>>();
        if indexes.len() > 1 {
            rationale.insert(format!(
                "complete removal closure joins {} strongly connected components linked by dead-code dependencies or a wholly dead file",
                indexes.len()
            ));
        }
        if !whole_files.is_empty() {
            rationale.insert(
                "the removal closure includes whole dead files and their simple module wiring"
                    .to_string(),
            );
        }
        let key_material = symbols
            .iter()
            .map(|symbol| symbol.logical_key.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        let mut group = DeadCodeGroup {
            group_id: format!("dead_{:032x}", xxh3_128(key_material.as_bytes())),
            classification: classification.clone(),
            confidence: group_confidence(&classification, &blockers, coverage),
            symbols,
            removal_ranges: ranges,
            source_loc: 0,
            compiler_validated_loc: 0,
            test_reference_count: indexes
                .iter()
                .map(|index| groups[*index].test_reference_count)
                .sum(),
            incoming_possible_count: indexes
                .iter()
                .map(|index| groups[*index].incoming_possible_count)
                .sum(),
            whole_files,
            blockers,
            rationale: rationale.into_iter().collect(),
        };
        normalize_group_ranges(&mut group);
        merged_groups.push(group);
    }
    merged_groups
}

fn find_group_root(parents: &mut [usize], index: usize) -> usize {
    let mut root = index;
    while parents[root] != root {
        root = parents[root];
    }
    let mut current = index;
    while parents[current] != current {
        let next = parents[current];
        parents[current] = root;
        current = next;
    }
    root
}

fn union_group_indexes(parents: &mut [usize], left: usize, right: usize) {
    let left_root = find_group_root(parents, left);
    let right_root = find_group_root(parents, right);
    if left_root != right_root {
        let (root, child) = if left_root < right_root {
            (left_root, right_root)
        } else {
            (right_root, left_root)
        };
        parents[child] = root;
    }
}

fn module_wiring_ranges(
    dead_path: &str,
    wiring_index: &HashMap<String, Vec<RemovalRange>>,
) -> Vec<RemovalRange> {
    let file_name = Path::new(dead_path)
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or_default();
    let module_name = if file_name == "mod.rs" {
        Path::new(dead_path)
            .parent()
            .and_then(Path::file_name)
            .and_then(|value| value.to_str())
            .unwrap_or_default()
    } else {
        Path::new(file_name)
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or_default()
    };
    if module_name.is_empty() {
        return Vec::new();
    }
    let mut ranges = wiring_index
        .get(module_name)
        .into_iter()
        .flatten()
        .chain(
            wiring_index
                .get(&format!("@file:{file_name}"))
                .into_iter()
                .flatten(),
        )
        .filter(|range| range.relative_path != dead_path)
        .cloned()
        .collect::<Vec<_>>();
    ranges.sort_by(|left, right| {
        left.relative_path
            .cmp(&right.relative_path)
            .then(left.start_line.cmp(&right.start_line))
    });
    ranges.dedup_by(|left, right| {
        left.relative_path == right.relative_path && left.start_line == right.start_line
    });
    ranges
}

fn build_module_wiring_index(
    source_text: &HashMap<String, String>,
) -> HashMap<String, Vec<RemovalRange>> {
    let mut index = HashMap::<String, Vec<RemovalRange>>::new();
    for (path, text) in source_text {
        for (line_index, line) in text.lines().enumerate() {
            let trimmed = line.trim();
            let mut keys = BTreeSet::new();
            for prefix in ["mod ", "pub mod ", "pub(crate) mod "] {
                if let Some(module) = trimmed
                    .strip_prefix(prefix)
                    .and_then(|value| value.strip_suffix(';'))
                    .filter(|value| {
                        !value.is_empty()
                            && value
                                .chars()
                                .all(|character| character == '_' || character.is_alphanumeric())
                    })
                {
                    keys.insert(module.to_string());
                }
            }
            if let Some(file_name) = trimmed
                .strip_prefix("include!(\"")
                .and_then(|value| value.strip_suffix("\");"))
            {
                keys.insert(format!("@file:{file_name}"));
            }
            if let Some(path_expr) = trimmed
                .strip_prefix("pub use ")
                .and_then(|value| value.strip_suffix(';'))
                .filter(|value| !value.contains('{'))
            {
                for segment in path_expr.split("::") {
                    if let Some(identifier) = segment.split_whitespace().next()
                        && !identifier.is_empty()
                    {
                        keys.insert(identifier.to_string());
                    }
                }
            }
            for key in keys {
                index.entry(key).or_default().push(RemovalRange {
                    relative_path: path.clone(),
                    start_line: line_index as u64 + 1,
                    end_line: line_index as u64 + 1,
                    loc: 1,
                    reason: "module declaration, include, or simple reexport for a dead file"
                        .to_string(),
                });
            }
        }
    }
    index
}

fn append_dedicated_test_closures(
    repo: &Path,
    groups: &mut [DeadCodeGroup],
    symbols: &[IndexedSymbol],
    relations: &[AnalysisRelation],
    source_lines: &HashMap<String, Vec<&str>>,
) {
    let test_symbols = symbols
        .iter()
        .filter(|symbol| {
            symbol.source_role == "test" || source_role_for_path(&symbol.relative_path) == "test"
        })
        .map(|symbol| (symbol.logical_key.clone(), symbol))
        .collect::<HashMap<_, _>>();
    let production_keys = symbols
        .iter()
        .filter(|symbol| {
            symbol.source_role == "production"
                && source_role_for_path(&symbol.relative_path) == "production"
        })
        .map(|symbol| symbol.logical_key.clone())
        .collect::<BTreeSet<_>>();
    let group_by_key = groups
        .iter()
        .enumerate()
        .flat_map(|(index, group)| {
            group
                .symbols
                .iter()
                .map(move |symbol| (symbol.logical_key.clone(), index))
        })
        .collect::<HashMap<_, _>>();
    let mut outgoing_production = HashMap::<String, BTreeSet<String>>::new();
    for relation in relations {
        if relation.language != "rust" {
            continue;
        }
        let Some(target) = relation.target_key.as_ref() else {
            continue;
        };
        if (relation.source_role == "test" || source_role_for_path(&relation.source_path) == "test")
            && production_keys.contains(target)
        {
            outgoing_production
                .entry(relation.source_key.clone())
                .or_default()
                .insert(target.clone());
        }
    }
    let mut dedicated_by_group = HashMap::<usize, Vec<&IndexedSymbol>>::new();
    for (source, targets) in outgoing_production {
        let mut target_group = None;
        let mut dedicated = !targets.is_empty();
        for target in targets {
            let Some(index) = group_by_key.get(&target).copied() else {
                dedicated = false;
                break;
            };
            if target_group.is_some_and(|existing| existing != index) {
                dedicated = false;
                break;
            }
            target_group = Some(index);
        }
        if dedicated
            && let (Some(index), Some(symbol)) = (target_group, test_symbols.get(&source).copied())
        {
            dedicated_by_group.entry(index).or_default().push(symbol);
        }
    }
    for (index, dedicated) in dedicated_by_group {
        for mut range in removal_ranges(repo, &dedicated, source_lines) {
            range.reason =
                "test declaration dedicated exclusively to this removal group".to_string();
            groups[index].removal_ranges.push(range);
        }
    }
}

fn normalize_group_ranges(group: &mut DeadCodeGroup) {
    let mut by_file = BTreeMap::<String, Vec<(u64, u64, String)>>::new();
    for range in group.removal_ranges.drain(..) {
        by_file.entry(range.relative_path).or_default().push((
            range.start_line,
            range.end_line,
            range.reason,
        ));
    }
    let mut normalized = Vec::new();
    for (path, mut ranges) in by_file {
        ranges.sort_by_key(|(start, end, _)| (*start, *end));
        let mut merged = Vec::<(u64, u64, BTreeSet<String>)>::new();
        for (start, end, reason) in ranges {
            if let Some(last) = merged.last_mut()
                && start <= last.1.saturating_add(1)
            {
                last.1 = last.1.max(end);
                last.2.insert(reason);
            } else {
                merged.push((start, end, BTreeSet::from([reason])));
            }
        }
        for (start, end, reasons) in merged {
            normalized.push(RemovalRange {
                relative_path: path.clone(),
                start_line: start,
                end_line: end,
                loc: end.saturating_sub(start).saturating_add(1),
                reason: reasons.into_iter().collect::<Vec<_>>().join("; "),
            });
        }
    }
    group.source_loc = normalized.iter().map(|range| range.loc).sum();
    group.removal_ranges = normalized;
}

fn classification_risk(classification: &DeadCodeClassification) -> u8 {
    match classification {
        DeadCodeClassification::Unreachable => 0,
        DeadCodeClassification::ReplacedOrLegacy => 1,
        DeadCodeClassification::TestOnly => 2,
        DeadCodeClassification::IntentionalTestSeam => 3,
        DeadCodeClassification::DynamicOrExternalRisk => 4,
    }
}

#[cfg(test)]
fn declaration_start_line(text: &str, declaration_line: u64) -> u64 {
    let lines = text.lines().collect::<Vec<_>>();
    declaration_start_line_from_lines(&lines, declaration_line)
}

fn declaration_start_line_from_lines(lines: &[&str], declaration_line: u64) -> u64 {
    if declaration_line <= 1 || declaration_line as usize > lines.len() {
        return declaration_line;
    }
    let mut start = declaration_line as usize - 1;
    while start > 0 {
        let previous = lines[start - 1].trim();
        if previous.starts_with("///")
            || previous.starts_with("//!")
            || previous.starts_with("#[")
            || previous.starts_with("#[cfg_attr")
            || (previous.is_empty()
                && start >= 2
                && (lines[start - 2].trim().starts_with("///")
                    || lines[start - 2].trim().starts_with("#[")))
        {
            start -= 1;
        } else {
            break;
        }
    }
    start as u64 + 1
}

fn group_confidence(
    classification: &DeadCodeClassification,
    blockers: &[RemovalBlocker],
    coverage: &RepoRelationshipCoverage,
) -> u64 {
    let mut confidence: u64 = match classification {
        DeadCodeClassification::Unreachable => 850,
        DeadCodeClassification::TestOnly => 800,
        DeadCodeClassification::ReplacedOrLegacy => 775,
        DeadCodeClassification::IntentionalTestSeam => 500,
        DeadCodeClassification::DynamicOrExternalRisk => 350,
    };
    confidence = confidence.saturating_sub((blockers.len() as u64).saturating_mul(40));
    if coverage.graph_status != "ready" || !coverage.stale_files.is_empty() {
        confidence = confidence.saturating_sub(150);
    }
    confidence
}

fn group_is_validation_eligible(group: &DeadCodeGroup) -> bool {
    group.blockers.is_empty() && !group_is_excluded_risk(group)
}

fn group_is_excluded_risk(group: &DeadCodeGroup) -> bool {
    matches!(
        group.classification,
        DeadCodeClassification::DynamicOrExternalRisk | DeadCodeClassification::IntentionalTestSeam
    )
}

fn coverage_blockers(coverage: &RepoRelationshipCoverage) -> Vec<RemovalBlocker> {
    let mut blockers = Vec::new();
    if coverage.graph_status != "ready" || !coverage.stale_files.is_empty() {
        blockers.push(blocker(
            "stale_graph",
            "relationship graph is not current",
            None,
            None,
        ));
    }
    blockers
}

fn validate_groups_in_temporary_checkout(
    scope: &str,
    repo: &Path,
    groups: &[DeadCodeGroup],
    all_features: bool,
    include_tests: bool,
) -> Result<RemovalValidation> {
    if !repo.join("Cargo.toml").is_file() {
        return Ok(RemovalValidation {
            scope: scope.to_string(),
            repo: repo.display().to_string(),
            status: "unsupported".to_string(),
            group_ids: groups.iter().map(|group| group.group_id.clone()).collect(),
            validated_group_ids: Vec::new(),
            rejected_group_ids: Vec::new(),
            requested_loc: groups.iter().map(|group| group.source_loc).sum(),
            compiler_validated_loc: 0,
            baseline_passed: false,
            command: Vec::new(),
            exit_code: None,
            diagnostics: "compiler validation currently supports Rust Cargo repositories"
                .to_string(),
            source_checkout_untouched: true,
            temporary_checkout_removed: true,
        });
    }

    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let temp_root = std::env::temp_dir().join(format!(
        "agent-context-removal-validation-{}-{nonce}",
        std::process::id()
    ));
    let checkout = temp_root.join("repo");
    fs::create_dir_all(&checkout)
        .with_context(|| format!("creating validation checkout {}", checkout.display()))?;

    let group_ids = groups
        .iter()
        .map(|group| group.group_id.clone())
        .collect::<Vec<_>>();
    let validation_result = (|| -> Result<ValidationSelection> {
        copy_repository_tree(repo, &checkout)?;
        let mut command = vec![
            "cargo".to_string(),
            "check".to_string(),
            "--workspace".to_string(),
            "--locked".to_string(),
            "--offline".to_string(),
        ];
        if all_features {
            command.push("--all-features".to_string());
        }
        if include_tests {
            command.push("--all-targets".to_string());
        }
        let target_dir = temp_root.join("target");
        let baseline = run_validation_compiler(&checkout, &target_dir, &command)?;
        if !baseline.passed {
            return Ok(ValidationSelection {
                command,
                baseline_passed: false,
                requested_loc: groups.iter().map(|group| group.source_loc).sum(),
                validated_groups: Vec::new(),
                rejected_groups: Vec::new(),
                validated_loc: 0,
                final_run: baseline,
                initial_failure: None,
            });
        }

        let all_paths = groups
            .iter()
            .flat_map(|group| &group.removal_ranges)
            .map(|range| range.relative_path.clone())
            .collect::<BTreeSet<_>>();
        let requested_loc = write_validation_state(repo, &checkout, &all_paths, groups)?;
        let initial = run_validation_compiler(&checkout, &target_dir, &command)?;
        if initial.passed {
            return Ok(ValidationSelection {
                command,
                baseline_passed: true,
                requested_loc,
                validated_groups: groups.to_vec(),
                rejected_groups: Vec::new(),
                validated_loc: requested_loc,
                final_run: initial,
                initial_failure: None,
            });
        }

        let initial_failure = initial.diagnostics.clone();
        let mut validated_groups = Vec::new();
        let mut rejected_groups = Vec::new();
        let mut last_failure = initial_failure.clone();
        if groups.len() == 1 {
            rejected_groups.push(groups[0].group_id.clone());
        } else {
            let midpoint = groups.len() / 2;
            isolate_validation_groups(
                repo,
                &checkout,
                &all_paths,
                &command,
                &target_dir,
                &mut validated_groups,
                &mut rejected_groups,
                &groups[..midpoint],
                &mut last_failure,
            )?;
            isolate_validation_groups(
                repo,
                &checkout,
                &all_paths,
                &command,
                &target_dir,
                &mut validated_groups,
                &mut rejected_groups,
                &groups[midpoint..],
                &mut last_failure,
            )?;
        }
        let validated_loc = write_validation_state(repo, &checkout, &all_paths, &validated_groups)?;
        let final_run = if validated_groups.is_empty() {
            baseline
        } else {
            run_validation_compiler(&checkout, &target_dir, &command)?
        };
        if !final_run.passed {
            return Ok(ValidationSelection {
                command,
                baseline_passed: true,
                requested_loc,
                validated_groups: Vec::new(),
                rejected_groups: group_ids.clone(),
                validated_loc: 0,
                final_run,
                initial_failure: Some(last_failure),
            });
        }
        Ok(ValidationSelection {
            command,
            baseline_passed: true,
            requested_loc,
            validated_groups,
            rejected_groups,
            validated_loc,
            final_run,
            initial_failure: Some(last_failure),
        })
    })();

    let cleanup = fs::remove_dir_all(&temp_root).is_ok();
    let selection = validation_result?;
    let validated_group_ids = selection
        .validated_groups
        .iter()
        .map(|group| group.group_id.clone())
        .collect::<Vec<_>>();
    let status = if !selection.baseline_passed {
        "baseline_failed"
    } else if selection.rejected_groups.is_empty() {
        "passed"
    } else if validated_group_ids.is_empty() {
        "failed"
    } else {
        "partial"
    };
    let diagnostics = if status == "partial" {
        format!(
            "{} of {} groups passed adaptive compiler isolation; rejected groups: {}\n{}",
            validated_group_ids.len(),
            group_ids.len(),
            selection.rejected_groups.join(", "),
            selection.initial_failure.unwrap_or_default(),
        )
    } else {
        selection.final_run.diagnostics.clone()
    };
    Ok(RemovalValidation {
        scope: scope.to_string(),
        repo: repo.display().to_string(),
        status: status.to_string(),
        group_ids,
        validated_group_ids,
        rejected_group_ids: selection.rejected_groups,
        requested_loc: selection.requested_loc,
        compiler_validated_loc: selection.validated_loc,
        baseline_passed: selection.baseline_passed,
        command: selection.command,
        exit_code: selection.final_run.exit_code,
        diagnostics: tail_chars(&diagnostics, 24_000),
        source_checkout_untouched: true,
        temporary_checkout_removed: cleanup,
    })
}

struct CompilerRun {
    passed: bool,
    exit_code: Option<i32>,
    diagnostics: String,
}

struct ValidationSelection {
    command: Vec<String>,
    baseline_passed: bool,
    requested_loc: u64,
    validated_groups: Vec<DeadCodeGroup>,
    rejected_groups: Vec<String>,
    validated_loc: u64,
    final_run: CompilerRun,
    initial_failure: Option<String>,
}

#[allow(clippy::too_many_arguments)]
fn isolate_validation_groups(
    source: &Path,
    checkout: &Path,
    all_paths: &BTreeSet<String>,
    command: &[String],
    target_dir: &Path,
    validated: &mut Vec<DeadCodeGroup>,
    rejected: &mut Vec<String>,
    batch: &[DeadCodeGroup],
    last_failure: &mut String,
) -> Result<()> {
    let mut proposed = validated.clone();
    proposed.extend_from_slice(batch);
    write_validation_state(source, checkout, all_paths, &proposed)?;
    let run = run_validation_compiler(checkout, target_dir, command)?;
    if run.passed {
        validated.extend_from_slice(batch);
    } else if batch.len() == 1 {
        rejected.push(batch[0].group_id.clone());
        *last_failure = run.diagnostics;
    } else {
        let midpoint = batch.len() / 2;
        isolate_validation_groups(
            source,
            checkout,
            all_paths,
            command,
            target_dir,
            validated,
            rejected,
            &batch[..midpoint],
            last_failure,
        )?;
        isolate_validation_groups(
            source,
            checkout,
            all_paths,
            command,
            target_dir,
            validated,
            rejected,
            &batch[midpoint..],
            last_failure,
        )?;
    }
    Ok(())
}

fn write_validation_state(
    source: &Path,
    checkout: &Path,
    all_paths: &BTreeSet<String>,
    groups: &[DeadCodeGroup],
) -> Result<u64> {
    for relative_path in all_paths {
        fs::copy(source.join(relative_path), checkout.join(relative_path)).with_context(|| {
            format!("restoring validation input {relative_path} from the source checkout")
        })?;
    }
    apply_validation_ranges(checkout, &merged_validation_ranges(groups))
}

fn run_validation_compiler(
    checkout: &Path,
    target_dir: &Path,
    command: &[String],
) -> Result<CompilerRun> {
    let output = Command::new(&command[0])
        .args(&command[1..])
        .current_dir(checkout)
        .env("CARGO_TARGET_DIR", target_dir)
        .output()
        .context("running cargo check in disposable checkout")?;
    let mut diagnostics = String::from_utf8_lossy(&output.stderr).to_string();
    if diagnostics.trim().is_empty() {
        diagnostics = String::from_utf8_lossy(&output.stdout).to_string();
    }
    Ok(CompilerRun {
        passed: output.status.success(),
        exit_code: output.status.code(),
        diagnostics: tail_chars(&diagnostics, 24_000),
    })
}

fn merged_validation_ranges(groups: &[DeadCodeGroup]) -> BTreeMap<String, Vec<(u64, u64)>> {
    let mut by_file = BTreeMap::<String, Vec<(u64, u64)>>::new();
    for range in groups.iter().flat_map(|group| &group.removal_ranges) {
        by_file
            .entry(range.relative_path.clone())
            .or_default()
            .push((range.start_line, range.end_line));
    }
    for ranges in by_file.values_mut() {
        ranges.sort();
        let mut merged = Vec::<(u64, u64)>::new();
        for (start, end) in ranges.drain(..) {
            if let Some(last) = merged.last_mut()
                && start <= last.1.saturating_add(1)
            {
                last.1 = last.1.max(end);
            } else {
                merged.push((start, end));
            }
        }
        *ranges = merged;
    }
    by_file
}

fn apply_validation_ranges(
    checkout: &Path,
    ranges: &BTreeMap<String, Vec<(u64, u64)>>,
) -> Result<u64> {
    let mut removed = 0_u64;
    for (relative_path, file_ranges) in ranges {
        let path = checkout.join(relative_path);
        let text = fs::read_to_string(&path)
            .with_context(|| format!("reading validation target {}", path.display()))?;
        let had_trailing_newline = text.ends_with('\n');
        let lines = text.lines().collect::<Vec<_>>();
        let mut retained = Vec::with_capacity(lines.len());
        for (index, line) in lines.iter().enumerate() {
            let line_number = index as u64 + 1;
            if file_ranges
                .iter()
                .any(|(start, end)| *start <= line_number && line_number <= *end)
            {
                removed += 1;
            } else {
                retained.push(*line);
            }
        }
        let mut updated = retained.join("\n");
        if had_trailing_newline {
            updated.push('\n');
        }
        fs::write(&path, updated)
            .with_context(|| format!("writing validation target {}", path.display()))?;
    }
    Ok(removed)
}

fn copy_repository_tree(source: &Path, destination: &Path) -> Result<()> {
    for entry in fs::read_dir(source)
        .with_context(|| format!("reading repository directory {}", source.display()))?
    {
        let entry = entry?;
        let name = entry.file_name();
        if matches!(
            name.to_str(),
            Some(
                ".git"
                    | "target"
                    | "node_modules"
                    | ".next"
                    | ".turbo"
                    | ".cache"
                    | ".venv"
                    | "dist"
                    | "coverage"
            )
        ) {
            continue;
        }
        let source_path = entry.path();
        let destination_path = destination.join(&name);
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            fs::create_dir_all(&destination_path)?;
            copy_repository_tree(&source_path, &destination_path)?;
        } else if file_type.is_symlink() {
            copy_symlink(&source_path, &destination_path)?;
        } else if file_type.is_file() {
            fs::copy(&source_path, &destination_path).with_context(|| {
                format!(
                    "copying validation input {} to {}",
                    source_path.display(),
                    destination_path.display()
                )
            })?;
        }
    }
    Ok(())
}

#[cfg(unix)]
fn copy_symlink(source: &Path, destination: &Path) -> Result<()> {
    std::os::unix::fs::symlink(fs::read_link(source)?, destination)
        .with_context(|| format!("copying symlink {}", source.display()))
}

#[cfg(windows)]
fn copy_symlink(source: &Path, destination: &Path) -> Result<()> {
    let target = fs::read_link(source)?;
    if source.is_dir() {
        std::os::windows::fs::symlink_dir(target, destination)?;
    } else {
        std::os::windows::fs::symlink_file(target, destination)?;
    }
    Ok(())
}

fn tail_chars(value: &str, limit: usize) -> String {
    let count = value.chars().count();
    if count <= limit {
        return value.to_string();
    }
    value.chars().skip(count - limit).collect()
}

fn dead_symbol(symbol: &IndexedSymbol) -> DeadCodeSymbol {
    DeadCodeSymbol {
        logical_key: symbol.logical_key.clone(),
        symbol_id: symbol.symbol_id.clone(),
        name: symbol.name.clone(),
        qualified_name: symbol.qualified_name.clone(),
        kind: symbol.kind.clone(),
        relative_path: symbol.relative_path.clone(),
        start_line: symbol.start_line,
        end_line: symbol.end_line,
        source_role: symbol.source_role.clone(),
        signature: symbol.signature.clone(),
    }
}

fn blocker(
    kind: &str,
    detail: impl Into<String>,
    relative_path: Option<String>,
    line: Option<u64>,
) -> RemovalBlocker {
    RemovalBlocker {
        kind: kind.to_string(),
        detail: detail.into(),
        relative_path,
        line,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DeadCodeClassification, DeadCodeGroup, DeadCodeRequest, GraphSnapshot, RemovalRange,
        analyze_snapshot, declaration_start_line, group_is_validation_eligible, reachable,
        strongly_connected_components, symbol_signals, validate_groups_in_temporary_checkout,
    };
    use crate::config::Config;
    use crate::engine::relationships::{
        AnalysisRelation, GraphStore, RelationKind, RepoRelationshipCoverage,
    };
    use crate::engine::symbols::{IndexedSymbol, SymbolStore};
    use std::collections::{BTreeSet, HashMap};
    use std::fs;
    use std::path::PathBuf;

    #[test]
    fn reachability_follows_directed_edges() {
        let roots = BTreeSet::from(["main".to_string()]);
        let adjacency = HashMap::from([
            ("main".to_string(), vec!["a".to_string()]),
            ("a".to_string(), vec!["b".to_string()]),
        ]);
        assert_eq!(
            reachable(&roots, &adjacency),
            BTreeSet::from(["main".to_string(), "a".to_string(), "b".to_string()])
        );
    }

    #[test]
    fn mutually_referencing_candidates_form_one_component() {
        let candidates = BTreeSet::from(["a".to_string(), "b".to_string(), "leaf".to_string()]);
        let adjacency = HashMap::from([
            ("a".to_string(), vec!["b".to_string()]),
            ("b".to_string(), vec!["a".to_string()]),
        ]);
        let mut components = strongly_connected_components(&candidates, &adjacency);
        components.sort();
        assert!(components.contains(&vec!["a".to_string(), "b".to_string()]));
        assert!(components.contains(&vec!["leaf".to_string()]));
    }

    #[test]
    fn one_way_dead_dependencies_form_one_complete_removal_closure() {
        let repo = temp_repo("dependency-closure");
        fs::create_dir_all(repo.join("src")).unwrap();
        fs::write(repo.join("src/entry.rs"), "fn dead_entry() { helper(); }\n").unwrap();
        fs::write(repo.join("src/helper.rs"), "fn helper() {}\n").unwrap();
        let analysis = analyze_snapshot(
            "fixture".to_string(),
            &repo,
            GraphSnapshot {
                symbols: vec![
                    symbol("dead_entry", "src/entry.rs", 1, "production"),
                    symbol("helper", "src/helper.rs", 1, "production"),
                ],
                relations: vec![relation(
                    "dead_entry",
                    "helper",
                    "src/entry.rs",
                    "production",
                )],
                coverage: RepoRelationshipCoverage {
                    repo: repo.display().to_string(),
                    graph_status: "ready".to_string(),
                    resolution_percentage: 100.0,
                    ..RepoRelationshipCoverage::default()
                },
            },
            &DeadCodeRequest::default(),
        )
        .unwrap();

        assert_eq!(analysis.group_count, 1);
        assert_eq!(analysis.groups[0].symbols.len(), 2);
        assert_eq!(analysis.groups[0].source_loc, 2);
        assert!(
            analysis.groups[0]
                .rationale
                .iter()
                .any(|reason| reason.contains("complete removal closure"))
        );

        fs::remove_dir_all(repo).unwrap();
    }

    #[test]
    fn clean_dead_caller_does_not_inherit_external_risk_from_dependency() {
        let repo = temp_repo("risk-boundary");
        fs::create_dir_all(repo.join("src")).unwrap();
        fs::write(
            repo.join("src/caller.rs"),
            "fn dead_caller() { public_api(); }\n",
        )
        .unwrap();
        fs::write(repo.join("src/api.rs"), "pub fn public_api() {}\n").unwrap();
        let mut public_api = symbol("public_api", "src/api.rs", 1, "production");
        public_api.signature = "pub fn public_api()".to_string();
        let analysis = analyze_snapshot(
            "fixture".to_string(),
            &repo,
            GraphSnapshot {
                symbols: vec![
                    symbol("dead_caller", "src/caller.rs", 1, "production"),
                    public_api,
                ],
                relations: vec![relation(
                    "dead_caller",
                    "public_api",
                    "src/caller.rs",
                    "production",
                )],
                coverage: RepoRelationshipCoverage {
                    repo: repo.display().to_string(),
                    graph_status: "ready".to_string(),
                    resolution_percentage: 100.0,
                    ..RepoRelationshipCoverage::default()
                },
            },
            &DeadCodeRequest {
                include_risk_candidates: true,
                ..DeadCodeRequest::default()
            },
        )
        .unwrap();

        assert_eq!(analysis.group_count, 2);
        assert!(analysis.groups.iter().any(|group| {
            matches!(group.classification, DeadCodeClassification::Unreachable)
                && group
                    .symbols
                    .iter()
                    .any(|symbol| symbol.name == "dead_caller")
        }));
        assert!(analysis.groups.iter().any(|group| {
            matches!(
                group.classification,
                DeadCodeClassification::DynamicOrExternalRisk
            ) && group
                .symbols
                .iter()
                .any(|symbol| symbol.name == "public_api")
        }));

        fs::remove_dir_all(repo).unwrap();
    }

    #[test]
    fn impl_containers_are_not_independent_removal_candidates() {
        let repo = temp_repo("impl-container");
        fs::create_dir_all(repo.join("src")).unwrap();
        fs::write(
            repo.join("src/main.rs"),
            "fn main() {}\nimpl App {\n    fn dead_method() {}\n}\n",
        )
        .unwrap();
        let main = symbol("main", "src/main.rs", 1, "production");
        let mut implementation = symbol("App", "src/main.rs", 2, "production");
        implementation.kind = "impl".to_string();
        implementation.end_line = 4;
        let mut method = symbol("dead_method", "src/main.rs", 3, "production");
        method.parent_logical_key = Some(implementation.logical_key.clone());
        let analysis = analyze_snapshot(
            "fixture".to_string(),
            &repo,
            GraphSnapshot {
                symbols: vec![main, implementation, method],
                relations: Vec::new(),
                coverage: RepoRelationshipCoverage {
                    repo: repo.display().to_string(),
                    graph_status: "ready".to_string(),
                    resolution_percentage: 100.0,
                    ..RepoRelationshipCoverage::default()
                },
            },
            &DeadCodeRequest::default(),
        )
        .unwrap();

        assert_eq!(analysis.totals.candidate_symbols, 1);
        assert_eq!(analysis.totals.candidate_loc, 1);
        assert_eq!(analysis.groups[0].symbols[0].name, "dead_method");

        fs::remove_dir_all(repo).unwrap();
    }

    #[test]
    fn removal_range_includes_contiguous_attributes_and_docs() {
        let text = "fn live() {}\n\n/// docs\n#[cfg(feature = \"x\")]\npub fn candidate() {}\n";
        assert_eq!(declaration_start_line(text, 5), 3);
    }

    #[test]
    fn only_main_is_a_hard_root_in_an_executable_source_file() {
        let repo = temp_repo("main-roots");
        fs::create_dir_all(repo.join("src")).unwrap();
        fs::write(
            repo.join("src/main.rs"),
            "fn main() { live(); }\nfn live() {}\nfn dead_helper() {}\n",
        )
        .unwrap();
        let analysis = analyze_snapshot(
            "fixture".to_string(),
            &repo,
            GraphSnapshot {
                symbols: vec![
                    symbol("main", "src/main.rs", 1, "production"),
                    symbol("live", "src/main.rs", 2, "production"),
                    symbol("dead_helper", "src/main.rs", 3, "production"),
                ],
                relations: vec![relation("main", "live", "src/main.rs", "production")],
                coverage: RepoRelationshipCoverage {
                    repo: repo.display().to_string(),
                    graph_status: "ready".to_string(),
                    resolution_percentage: 100.0,
                    ..RepoRelationshipCoverage::default()
                },
            },
            &DeadCodeRequest::default(),
        )
        .unwrap();

        assert_eq!(analysis.root_count, 1);
        assert_eq!(analysis.totals.production_reachable_symbols, 2);
        assert!(analysis.groups.iter().any(|group| {
            group
                .symbols
                .iter()
                .any(|symbol| symbol.name == "dead_helper")
        }));

        fs::remove_dir_all(repo).unwrap();
    }

    #[test]
    fn dynamic_markers_do_not_taint_unrelated_file_declarations() {
        let mut text = "inventory::submit! { Registration }\n".to_string();
        text.push_str(&"\n".repeat(20));
        text.push_str("fn ordinary_helper() {}\n");
        let helper = symbol("ordinary_helper", "src/registry.rs", 22, "production");
        let lines = text.lines().collect::<Vec<_>>();

        assert!(!symbol_signals(&helper, Some(&lines)).dynamic);
    }

    #[test]
    fn intentional_test_seam_detection_uses_name_boundaries() {
        let latest = symbol("latest", "src/lib.rs", 1, "production");
        let helper = symbol("helper_for_test", "src/lib.rs", 2, "production");

        assert!(!symbol_signals(&latest, None).intentional_test_seam);
        assert!(symbol_signals(&helper, None).intentional_test_seam);
    }

    #[test]
    fn analysis_groups_test_only_cycle_with_file_wiring_and_dedicated_test() {
        let repo = temp_repo("closure");
        fs::create_dir_all(repo.join("src")).unwrap();
        fs::write(
            repo.join("src/main.rs"),
            "mod dead;\nfn main() { live(); }\nfn live() {}\n",
        )
        .unwrap();
        fs::write(
            repo.join("src/dead.rs"),
            "fn dead_a() { dead_b(); }\nfn dead_b() { dead_a(); }\n",
        )
        .unwrap();
        fs::write(repo.join("src/tests.rs"), "fn dead_test() { dead_a(); }\n").unwrap();

        let symbols = vec![
            symbol("main", "src/main.rs", 2, "production"),
            symbol("live", "src/main.rs", 3, "production"),
            symbol("dead_a", "src/dead.rs", 1, "production"),
            symbol("dead_b", "src/dead.rs", 2, "production"),
            symbol("dead_test", "src/tests.rs", 1, "test"),
        ];
        let relations = vec![
            relation("main", "live", "src/main.rs", "production"),
            relation("dead_a", "dead_b", "src/dead.rs", "production"),
            relation("dead_b", "dead_a", "src/dead.rs", "production"),
            relation("dead_test", "dead_a", "src/tests.rs", "test"),
        ];
        let analysis = analyze_snapshot(
            "fixture".to_string(),
            &repo,
            GraphSnapshot {
                symbols,
                relations,
                coverage: RepoRelationshipCoverage {
                    repo: repo.display().to_string(),
                    graph_status: "ready".to_string(),
                    resolution_percentage: 100.0,
                    ..RepoRelationshipCoverage::default()
                },
            },
            &DeadCodeRequest::default(),
        )
        .unwrap();

        let group = analysis
            .groups
            .iter()
            .find(|group| group.whole_files == vec!["src/dead.rs".to_string()])
            .unwrap();
        assert!(matches!(
            group.classification,
            DeadCodeClassification::TestOnly
        ));
        assert_eq!(group.symbols.len(), 2);
        assert_eq!(group.source_loc, 4);
        assert!(group.blockers.is_empty());
        assert!(group.removal_ranges.iter().any(|range| {
            range.relative_path == "src/main.rs" && range.reason.contains("module")
        }));
        assert!(group.removal_ranges.iter().any(|range| {
            range.relative_path == "src/tests.rs" && range.reason.contains("test declaration")
        }));

        fs::remove_dir_all(repo).unwrap();
    }

    #[test]
    fn compiler_validation_uses_disposable_checkout_and_preserves_source() {
        let repo = temp_repo("compiler");
        fs::create_dir_all(repo.join("src")).unwrap();
        fs::write(
            repo.join("Cargo.toml"),
            "[package]\nname = \"trim-fixture\"\nversion = \"0.1.0\"\nedition = \"2024\"\n",
        )
        .unwrap();
        fs::write(
            repo.join("Cargo.lock"),
            "# This file is automatically @generated by Cargo.\nversion = 4\n\n[[package]]\nname = \"trim-fixture\"\nversion = \"0.1.0\"\n",
        )
        .unwrap();
        fs::write(repo.join("src/main.rs"), "fn dead() {}\nfn main() {\n}\n").unwrap();
        let group = DeadCodeGroup {
            group_id: "dead-fixture".to_string(),
            classification: DeadCodeClassification::Unreachable,
            confidence: 850,
            symbols: Vec::new(),
            removal_ranges: vec![RemovalRange {
                relative_path: "src/main.rs".to_string(),
                start_line: 1,
                end_line: 1,
                loc: 1,
                reason: "fixture".to_string(),
            }],
            source_loc: 1,
            compiler_validated_loc: 0,
            test_reference_count: 0,
            incoming_possible_count: 0,
            whole_files: Vec::new(),
            blockers: Vec::new(),
            rationale: Vec::new(),
        };

        let validation =
            validate_groups_in_temporary_checkout("fixture", &repo, &[group], false, false)
                .unwrap();
        assert_eq!(validation.status, "passed", "{}", validation.diagnostics);
        assert_eq!(validation.compiler_validated_loc, 1);
        assert!(validation.source_checkout_untouched);
        assert!(validation.temporary_checkout_removed);
        assert_eq!(
            fs::read_to_string(repo.join("src/main.rs")).unwrap(),
            "fn dead() {}\nfn main() {\n}\n"
        );

        fs::remove_dir_all(repo).unwrap();
    }

    #[test]
    fn failed_compiler_validation_never_promotes_candidate_loc() {
        let repo = temp_repo("compiler-failure");
        fs::create_dir_all(repo.join("src")).unwrap();
        fs::write(
            repo.join("Cargo.toml"),
            "[package]\nname = \"trim-failure-fixture\"\nversion = \"0.1.0\"\nedition = \"2024\"\n",
        )
        .unwrap();
        fs::write(
            repo.join("Cargo.lock"),
            "# This file is automatically @generated by Cargo.\nversion = 4\n\n[[package]]\nname = \"trim-failure-fixture\"\nversion = \"0.1.0\"\n",
        )
        .unwrap();
        fs::write(repo.join("src/main.rs"), "fn dead() {}\nfn main() {}\n").unwrap();
        let group = DeadCodeGroup {
            group_id: "main-fixture".to_string(),
            classification: DeadCodeClassification::Unreachable,
            confidence: 850,
            symbols: Vec::new(),
            removal_ranges: vec![RemovalRange {
                relative_path: "src/main.rs".to_string(),
                start_line: 2,
                end_line: 2,
                loc: 1,
                reason: "fixture".to_string(),
            }],
            source_loc: 1,
            compiler_validated_loc: 0,
            test_reference_count: 0,
            incoming_possible_count: 0,
            whole_files: Vec::new(),
            blockers: Vec::new(),
            rationale: Vec::new(),
        };

        let validation =
            validate_groups_in_temporary_checkout("fixture", &repo, &[group], false, false)
                .unwrap();
        assert_eq!(validation.status, "failed");
        assert_eq!(validation.compiler_validated_loc, 0);
        assert_eq!(validation.requested_loc, 1);
        assert!(validation.source_checkout_untouched);
        assert_eq!(
            fs::read_to_string(repo.join("src/main.rs")).unwrap(),
            "fn dead() {}\nfn main() {}\n"
        );

        fs::remove_dir_all(repo).unwrap();
    }

    #[test]
    fn compiler_validation_isolates_passing_groups() {
        let repo = temp_repo("compiler-partial");
        fs::create_dir_all(repo.join("src")).unwrap();
        fs::write(
            repo.join("Cargo.toml"),
            "[package]\nname = \"trim-partial-fixture\"\nversion = \"0.1.0\"\nedition = \"2024\"\n",
        )
        .unwrap();
        fs::write(
            repo.join("Cargo.lock"),
            "# This file is automatically @generated by Cargo.\nversion = 4\n\n[[package]]\nname = \"trim-partial-fixture\"\nversion = \"0.1.0\"\n",
        )
        .unwrap();
        fs::write(repo.join("src/main.rs"), "fn dead() {}\nfn main() {}\n").unwrap();
        let group = |id: &str, line: u64| DeadCodeGroup {
            group_id: id.to_string(),
            classification: DeadCodeClassification::Unreachable,
            confidence: 850,
            symbols: Vec::new(),
            removal_ranges: vec![RemovalRange {
                relative_path: "src/main.rs".to_string(),
                start_line: line,
                end_line: line,
                loc: 1,
                reason: "fixture".to_string(),
            }],
            source_loc: 1,
            compiler_validated_loc: 0,
            test_reference_count: 0,
            incoming_possible_count: 0,
            whole_files: Vec::new(),
            blockers: Vec::new(),
            rationale: Vec::new(),
        };
        let groups = vec![group("dead-fixture", 1), group("main-fixture", 2)];

        let validation =
            validate_groups_in_temporary_checkout("fixture", &repo, &groups, false, false).unwrap();
        assert_eq!(validation.status, "partial", "{}", validation.diagnostics);
        assert!(validation.baseline_passed);
        assert_eq!(validation.validated_group_ids, vec!["dead-fixture"]);
        assert_eq!(validation.rejected_group_ids, vec!["main-fixture"]);
        assert_eq!(validation.compiler_validated_loc, 1);
        assert_eq!(validation.requested_loc, 2);
        assert_eq!(
            fs::read_to_string(repo.join("src/main.rs")).unwrap(),
            "fn dead() {}\nfn main() {}\n"
        );

        fs::remove_dir_all(repo).unwrap();
    }

    #[test]
    fn bounded_group_details_preserve_full_census_totals() {
        let repo = temp_repo("bounded-totals");
        fs::create_dir_all(repo.join("src")).unwrap();
        fs::write(repo.join("src/first.rs"), "fn first() {}\n").unwrap();
        fs::write(repo.join("src/second.rs"), "fn second() {}\n").unwrap();
        fs::write(repo.join("src/third.rs"), "fn third() {}\n").unwrap();
        let analysis = analyze_snapshot(
            "fixture".to_string(),
            &repo,
            GraphSnapshot {
                symbols: vec![
                    symbol("first", "src/first.rs", 1, "production"),
                    symbol("second", "src/second.rs", 1, "production"),
                    symbol("third", "src/third.rs", 1, "production"),
                ],
                relations: Vec::new(),
                coverage: RepoRelationshipCoverage {
                    repo: repo.display().to_string(),
                    graph_status: "ready".to_string(),
                    resolution_percentage: 100.0,
                    ..RepoRelationshipCoverage::default()
                },
            },
            &DeadCodeRequest {
                max_groups: 1,
                ..DeadCodeRequest::default()
            },
        )
        .unwrap();

        assert_eq!(analysis.group_count, 3);
        assert_eq!(analysis.groups.len(), 1);
        assert_eq!(analysis.totals.candidate_loc, 3);
        assert!(analysis.truncated);

        fs::remove_dir_all(repo).unwrap();
    }

    #[test]
    fn hidden_risk_details_remain_accounted_in_aggregate_totals() {
        let repo = temp_repo("risk-totals");
        fs::create_dir_all(repo.join("src")).unwrap();
        fs::write(repo.join("src/private.rs"), "fn private_dead() {}\n").unwrap();
        fs::write(repo.join("src/public.rs"), "pub fn public_api() {}\n").unwrap();
        let mut public = symbol("public_api", "src/public.rs", 1, "production");
        public.signature = "pub fn public_api()".to_string();
        let analysis = analyze_snapshot(
            "fixture".to_string(),
            &repo,
            GraphSnapshot {
                symbols: vec![
                    symbol("private_dead", "src/private.rs", 1, "production"),
                    public,
                ],
                relations: Vec::new(),
                coverage: RepoRelationshipCoverage {
                    repo: repo.display().to_string(),
                    graph_status: "ready".to_string(),
                    resolution_percentage: 100.0,
                    ..RepoRelationshipCoverage::default()
                },
            },
            &DeadCodeRequest {
                include_risk_candidates: false,
                ..DeadCodeRequest::default()
            },
        )
        .unwrap();

        assert_eq!(analysis.group_count, 2);
        assert_eq!(analysis.groups.len(), 1);
        assert_eq!(analysis.totals.candidate_loc, 1);
        assert_eq!(analysis.totals.clean_candidate_loc, 1);
        assert_eq!(analysis.totals.review_required_loc, 0);
        assert_eq!(analysis.totals.excluded_risk_loc, 1);

        fs::remove_dir_all(repo).unwrap();
    }

    #[test]
    #[ignore = "manual read-only trim benchmark"]
    fn bench_dead_code_analysis_manual() {
        let config_path = PathBuf::from(
            std::env::var("CC_TRIM_BENCH_CONFIG")
                .expect("set CC_TRIM_BENCH_CONFIG to an agent-context config path"),
        );
        let repo = std::env::var("CC_TRIM_BENCH_REPO")
            .expect("set CC_TRIM_BENCH_REPO to a configured repository root");
        let config = Config::load_from_path(&config_path).expect("load benchmark config");
        let repo_path = PathBuf::from(&repo);
        let symbols = SymbolStore::new(config.symbol_db_path());
        let graph = GraphStore::new(config.symbol_db_path());
        let started = std::time::Instant::now();
        let phase = std::time::Instant::now();
        let loaded_symbols = symbols.all_symbols(&repo).expect("load benchmark symbols");
        eprintln!(
            "trim_benchmark_phase symbols={} elapsed_ms={}",
            loaded_symbols.len(),
            phase.elapsed().as_millis()
        );
        let phase = std::time::Instant::now();
        let loaded_relations = graph
            .all_analysis_relations(&repo)
            .expect("load benchmark relations");
        eprintln!(
            "trim_benchmark_phase relations={} elapsed_ms={}",
            loaded_relations.len(),
            phase.elapsed().as_millis()
        );
        let phase = std::time::Instant::now();
        let coverage = graph.coverage(&repo).expect("load benchmark coverage");
        eprintln!(
            "trim_benchmark_phase coverage elapsed_ms={}",
            phase.elapsed().as_millis()
        );
        let snapshot = GraphSnapshot {
            symbols: loaded_symbols,
            relations: loaded_relations,
            coverage,
        };
        let graph_status = snapshot.coverage.graph_status.clone();
        let phase = std::time::Instant::now();
        let analysis = analyze_snapshot(
            repo.clone(),
            &repo_path,
            snapshot,
            &DeadCodeRequest {
                repo: Some(repo),
                max_groups: 2_000,
                ..DeadCodeRequest::default()
            },
        )
        .expect("run unverified read-only dead-code benchmark");
        eprintln!(
            "trim_benchmark_phase analysis elapsed_ms={}",
            phase.elapsed().as_millis()
        );

        eprintln!(
            "trim_benchmark verified=false graph_status={} elapsed_ms={} symbols={} reachable={} roots={} groups={} candidate_loc={} clean_candidate_loc={} review_required_loc={} excluded_risk_loc={} unresolved_internal_docs={} unresolved_external_docs={} compiler_validated_loc={} status={}",
            graph_status,
            started.elapsed().as_millis(),
            analysis.totals.indexed_rust_symbols,
            analysis.totals.production_reachable_symbols,
            analysis.root_count,
            analysis.group_count,
            analysis.totals.candidate_loc,
            analysis.totals.clean_candidate_loc,
            analysis.totals.review_required_loc,
            analysis.totals.excluded_risk_loc,
            analysis.totals.unresolved_internal_documents,
            analysis.totals.unresolved_external_documents,
            analysis.totals.compiler_validated_loc,
            analysis.status,
        );
        for group in analysis
            .groups
            .iter()
            .filter(|group| group_is_validation_eligible(group))
            .take(30)
        {
            let symbols = group
                .symbols
                .iter()
                .take(6)
                .map(|symbol| {
                    format!(
                        "{}:{}:{}",
                        symbol.relative_path, symbol.start_line, symbol.name
                    )
                })
                .collect::<Vec<_>>()
                .join(",");
            let blocker_kinds = group
                .blockers
                .iter()
                .map(|blocker| blocker.kind.as_str())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>()
                .join(",");
            eprintln!(
                "trim_benchmark_clean_group id={} class={:?} confidence={} loc={} symbols={} whole_files={} blockers={} examples={}",
                group.group_id,
                group.classification,
                group.confidence,
                group.source_loc,
                group.symbols.len(),
                group.whole_files.len(),
                blocker_kinds,
                symbols,
            );
        }
    }

    fn symbol(name: &str, path: &str, line: u64, role: &str) -> IndexedSymbol {
        IndexedSymbol {
            symbol_id: format!("sym-{name}"),
            logical_key: name.to_string(),
            repo: "fixture".to_string(),
            relative_path: path.to_string(),
            name: name.to_string(),
            kind: "function".to_string(),
            container: None,
            language: "rust".to_string(),
            start_line: line,
            end_line: line,
            indexed_at: "now".to_string(),
            file_hash: "hash".to_string(),
            parent_symbol_id: None,
            parent_logical_key: None,
            qualified_name: name.to_string(),
            signature: format!("fn {name}()"),
            source_role: role.to_string(),
            identity_stable: true,
        }
    }

    fn relation(source: &str, target: &str, path: &str, role: &str) -> AnalysisRelation {
        AnalysisRelation {
            source_key: source.to_string(),
            source_path: path.to_string(),
            target_key: Some(target.to_string()),
            target_name: target.to_string(),
            kind: RelationKind::Calls,
            confidence: 900,
            resolution: "same_module_unique".to_string(),
            start_line: 1,
            source_role: role.to_string(),
            language: "rust".to_string(),
        }
    }

    fn temp_repo(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "agent-context-trim-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }
}
