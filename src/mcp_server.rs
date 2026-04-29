use crate::config::{Config, ResolvedScope};
use crate::engine::splitter::SplitterKind;
use crate::engine::symbols::OutlineNode;
use crate::engine::{
    AnchorQuality, EditResolutionType, EditTargetAnchor, EditTargetReasonCode, EditTargetStatus,
    Engine, FileOutlineResponse, PrepareEditTargetRequest, PrepareEditTargetResponse,
    RepoSearchError, SearchMode, SearchPlanSummary, SearchRequest, SearchResponse,
    SymbolSearchResponse, SymbolSearchScopeRequest, TextSearchResponse, TextSearchScopeRequest,
    render_clear_text, render_search_explanation_text, render_status_text,
};
use anyhow::{Context, Result, bail};
use axum::{Router, routing::get};
use rmcp::{
    ErrorData as McpError, RoleServer, ServerHandler,
    model::{
        CallToolRequestParams, CallToolResult, Content, ListToolsResult, PaginatedRequestParams,
        ServerCapabilities, ServerInfo, Tool, ToolAnnotations,
    },
    service::RequestContext,
    transport::streamable_http_server::{
        StreamableHttpServerConfig, session::local::LocalSessionManager,
        tower::StreamableHttpService,
    },
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::borrow::Cow;
use std::collections::HashSet;
use std::future::Future;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::signal;
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

#[derive(Clone)]
struct NativeServer {
    engine: Engine,
    active_repos: Arc<Mutex<HashSet<String>>>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
struct SearchCodeArgs {
    #[serde(default)]
    scope: Option<String>,
    #[serde(default)]
    path: Option<String>,
    query: String,
    #[serde(default = "default_limit")]
    limit: usize,
    #[serde(default)]
    mode: Option<String>,
    #[serde(default)]
    extension_filter: Vec<String>,
    #[serde(default)]
    path_prefix: Option<String>,
    #[serde(default)]
    language: Option<String>,
    #[serde(default)]
    file: Option<String>,
    #[serde(default = "default_dedupe_by_file")]
    dedupe_by_file: bool,
    #[serde(default)]
    include_diagnostics: bool,
    #[serde(default = "default_include_content")]
    include_content: bool,
    #[serde(default = "default_snippet_chars")]
    snippet_chars: usize,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
struct IndexCodebaseArgs {
    #[serde(default)]
    scope: Option<String>,
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    force: bool,
    #[serde(default = "default_splitter")]
    splitter: String,
    #[serde(default)]
    custom_extensions: Vec<String>,
    #[serde(default)]
    ignore_patterns: Vec<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
struct ScopeArgs {
    #[serde(default)]
    scope: Option<String>,
    #[serde(default)]
    path: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
struct SearchSymbolsArgs {
    #[serde(default)]
    scope: Option<String>,
    #[serde(default)]
    path: Option<String>,
    query: String,
    #[serde(default = "default_limit")]
    limit: usize,
    #[serde(default)]
    path_prefix: Option<String>,
    #[serde(default)]
    language: Option<String>,
    #[serde(default)]
    kind: Option<String>,
    #[serde(default)]
    container: Option<String>,
    #[serde(default)]
    include_diagnostics: bool,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
struct SearchTextArgs {
    #[serde(default)]
    scope: Option<String>,
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    repo: Option<String>,
    query: String,
    #[serde(default = "default_text_search_limit")]
    limit: usize,
    #[serde(default)]
    path_prefix: Option<String>,
    #[serde(default)]
    language: Option<String>,
    #[serde(default)]
    file: Option<String>,
    #[serde(default)]
    extension_filter: Vec<String>,
    #[serde(default = "default_case_sensitive")]
    case_sensitive: bool,
    #[serde(default)]
    whole_word: bool,
    #[serde(default = "default_text_search_context_lines")]
    context_lines: usize,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
struct PrepareEditTargetArgs {
    #[serde(default)]
    scope: Option<String>,
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    repo: Option<String>,
    #[serde(default)]
    file: Option<String>,
    #[serde(default)]
    symbol_id: Option<String>,
    #[serde(default)]
    line_hint: Option<u64>,
    #[serde(default)]
    query: Option<String>,
    #[serde(default)]
    occurrence: Option<usize>,
    #[serde(default = "default_before_lines")]
    before_lines: usize,
    #[serde(default = "default_after_lines")]
    after_lines: usize,
    #[serde(default = "default_max_lines")]
    max_lines: usize,
    #[serde(default = "default_anchor_count")]
    anchor_count: usize,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
struct FileOutlineArgs {
    #[serde(default)]
    scope: Option<String>,
    #[serde(default)]
    path: Option<String>,
    file: String,
    #[serde(default)]
    detail: Option<String>,
    #[serde(default = "default_outline_depth")]
    max_depth: usize,
    #[serde(default = "default_outline_max_nodes")]
    max_nodes: usize,
    #[serde(default = "default_outline_top_level_limit")]
    top_level_limit: usize,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct IndexLaunchResult {
    scope: String,
    label: String,
    started: bool,
    force: bool,
    started_repos: Vec<String>,
    already_indexing: Vec<String>,
}

#[derive(Debug, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "camelCase")]
struct ListScopesArgs {
    #[serde(default)]
    include_repos: bool,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ScopeSummary {
    id: String,
    label: String,
    repo_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    repos: Option<Vec<String>>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ListScopesResult {
    default_scope: String,
    groups: Vec<ScopeSummary>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct CompactSearchResponse {
    scope: String,
    label: String,
    #[serde(skip_serializing_if = "is_false")]
    partial: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    repo_errors: Vec<RepoSearchError>,
    #[serde(skip_serializing_if = "Option::is_none")]
    plan: Option<SearchPlanSummary>,
    hits: Vec<CompactSearchHit>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct CompactSearchHit {
    repo_label: String,
    relative_path: String,
    start_line: u64,
    end_line: u64,
    language: String,
    match_type: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    content: String,
    #[serde(skip_serializing_if = "is_false")]
    stale: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    repo: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    score: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dense_score: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    lexical_score: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    symbol_score: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    indexed_at: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct CompactSymbolSearchResponse {
    scope: String,
    label: String,
    #[serde(skip_serializing_if = "is_false")]
    partial: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    repo_errors: Vec<RepoSearchError>,
    hits: Vec<CompactSymbolHit>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct CompactSymbolHit {
    symbol_id: String,
    repo_label: String,
    relative_path: String,
    name: String,
    kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    container: Option<String>,
    start_line: u64,
    end_line: u64,
    language: String,
    #[serde(skip_serializing_if = "is_false")]
    stale: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    repo: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    score: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    lexical_score: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    semantic_score: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    indexed_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    file_hash: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct CompactTextSearchResponse {
    scope: String,
    label: String,
    #[serde(skip_serializing_if = "is_false")]
    partial: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    repo_errors: Vec<RepoSearchError>,
    hits: Vec<CompactTextSearchHit>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct CompactTextSearchHit {
    repo_label: String,
    relative_path: String,
    start_line: u64,
    end_line: u64,
    preview: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct CompactPrepareEditTargetResponse {
    status: EditTargetStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    repo_label: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    relative_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    start_line: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    end_line: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    anchors: Vec<EditTargetAnchor>,
    #[serde(skip_serializing_if = "Option::is_none")]
    anchor_quality: Option<AnchorQuality>,
    #[serde(skip_serializing_if = "Option::is_none")]
    resolution_type: Option<EditResolutionType>,
    #[serde(skip_serializing_if = "is_false")]
    stale: bool,
    #[serde(skip_serializing_if = "is_false")]
    unindexed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    symbol_id: Option<String>,
    #[serde(skip_serializing_if = "is_false")]
    truncated: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    candidates: Vec<CompactEditTargetCandidate>,
    #[serde(skip_serializing_if = "Option::is_none")]
    symbol_start_line: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    symbol_end_line: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reason_code: Option<EditTargetReasonCode>,
    #[serde(skip_serializing_if = "Option::is_none")]
    suggested_next_tool: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct CompactEditTargetCandidate {
    repo_label: String,
    relative_path: String,
    start_line: u64,
    end_line: u64,
    preview: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutlineDetail {
    Summary,
    Compact,
    Full,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct CompactFileOutlineResponse {
    scope: String,
    label: String,
    file: String,
    matches: Vec<CompactFileOutlineMatch>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct CompactFileOutlineMatch {
    #[serde(skip_serializing_if = "Option::is_none")]
    repo: Option<String>,
    repo_label: String,
    relative_path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    language: Option<String>,
    #[serde(skip_serializing_if = "is_false")]
    stale: bool,
    #[serde(skip_serializing_if = "is_false")]
    truncated: bool,
    returned_node_count: usize,
    total_node_count: usize,
    #[serde(skip_serializing_if = "is_zero")]
    remaining_node_count: usize,
    symbols: Vec<CompactOutlineNode>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct CompactOutlineNode {
    symbol_id: String,
    name: String,
    kind: String,
    start_line: u64,
    end_line: u64,
    #[serde(skip_serializing_if = "is_false")]
    has_children: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    children: Vec<CompactOutlineNode>,
}

pub async fn serve(
    config: &Config,
    listen: &str,
    allow_remote_unauthenticated: bool,
) -> Result<()> {
    enforce_loopback_bind(listen, allow_remote_unauthenticated)?;
    let engine = Engine::new(config).await?;
    engine
        .mark_interrupted_indexing_failed("agent-context restarted while indexing was in progress")
        .await?;
    let server = NativeServer {
        engine,
        active_repos: Arc::new(Mutex::new(HashSet::new())),
    };
    let cancellation = CancellationToken::new();

    if let Some(interval_secs) = config.freshness.audit_interval_secs {
        let audit_engine = server.engine.clone();
        let audit_token = cancellation.child_token();
        tokio::spawn(async move {
            let duration = std::time::Duration::from_secs(interval_secs);
            loop {
                tokio::select! {
                    _ = audit_token.cancelled() => break,
                    _ = tokio::time::sleep(duration) => {
                        let _ = audit_engine.cheap_audit_once().await;
                    }
                }
            }
        });
    }

    let service = StreamableHttpService::<NativeServer, LocalSessionManager>::new(
        {
            let server = server.clone();
            move || Ok(server.clone())
        },
        Default::default(),
        StreamableHttpServerConfig::default()
            .with_sse_keep_alive(None)
            .with_cancellation_token(cancellation.child_token()),
    );

    let router = Router::new()
        .route("/health", get(health))
        .nest_service("/mcp", service);

    let listener = TcpListener::bind(listen)
        .await
        .with_context(|| format!("binding MCP listener on {listen}"))?;

    let shutdown_token = cancellation.clone();
    axum::serve(listener, router)
        .with_graceful_shutdown(async move {
            wait_for_signal().await;
            shutdown_token.cancel();
        })
        .await
        .context("serving native HTTP MCP endpoint")
}

const SERVER_INSTRUCTIONS: &str = "Use list_scopes first. Use search_symbols for exact definitions. Use search_code for broader discovery; its snippets are discovery hints, not authoritative reads. Use search_text for exact literals, identifiers, test names, and log lines in a known repo, file, or subtree instead of shell rg. Use get_file_outline for compact file structure once the file is known. Use prepare_edit_target only when the exact patch location is already known; it is the final pre-patch step, not a general reader or overview tool. If prepare_edit_target returns needsNarrowing or ambiguous, go back to search_text or get_file_outline. Fall back to shell rg/sed/bat only when regex is required or MCP exact inspection is unavailable. scope defaults to the configured default group.";

pub fn tool_list() -> Vec<Tool> {
    vec![
        build_tool(
            "index_codebase",
            "Start background indexing for a configured scope. Defaults to the configured default group.",
            false,
            index_codebase_schema(),
        ),
        build_tool(
            "search_symbols",
            "Exact symbol and definition lookup. Use this when you know or suspect the definition name.",
            true,
            search_symbols_schema(),
        ),
        build_tool(
            "search_code",
            "Broader code discovery across indexed repos. Snippets are discovery hints; use search_text for exact follow-up once a file or pathPrefix is known.",
            true,
            search_code_schema(),
        ),
        build_tool(
            "search_text",
            "Exact literal confirmation inside a known repo, file, or pathPrefix. Use this instead of shell rg for identifiers, test names, log lines, and exact strings once discovery is done.",
            true,
            search_text_schema(),
        ),
        build_tool(
            "get_file_outline",
            "Return a compact indexed symbol outline summary for a known repo-relative file. Use this for structure instead of broad file reads.",
            true,
            get_file_outline_schema(),
        ),
        build_tool(
            "prepare_edit_target",
            "Final pre-patch step only. Use immediately before editing instead of sed/bat/cat after the exact patch location is known. Not for overview or broad inspection.",
            true,
            prepare_edit_target_schema(),
        ),
        build_tool(
            "explain_search",
            "Explain how search_code will classify and weight a query.",
            true,
            explain_search_schema(),
        ),
        build_tool(
            "clear_index",
            "Drop the index for a configured scope.",
            false,
            scope_args_schema(
                "Configured group id or repo root. Defaults to the configured default group.",
            ),
        ),
        build_tool(
            "get_indexing_status",
            "Report indexing status for a configured scope.",
            true,
            scope_args_schema(
                "Configured group id or repo root. Defaults to the configured default group.",
            ),
        ),
        build_tool(
            "list_scopes",
            "Preferred first call. Lists configured scopes.",
            true,
            list_scopes_schema(),
        ),
    ]
}

async fn health() -> &'static str {
    "ok"
}

fn enforce_loopback_bind(listen: &str, allow_remote_unauthenticated: bool) -> Result<()> {
    if allow_remote_unauthenticated || listen_is_loopback(listen) {
        return Ok(());
    }

    bail!(
        "refusing to bind unauthenticated HTTP MCP server to non-loopback address `{listen}`; use --allow-remote-unauthenticated to override"
    );
}

fn listen_is_loopback(listen: &str) -> bool {
    if let Ok(addr) = listen.parse::<SocketAddr>() {
        return addr.ip().is_loopback();
    }

    if let Some(rest) = listen.strip_prefix("localhost:") {
        return !rest.is_empty();
    }

    false
}

async fn wait_for_signal() {
    let ctrl_c = async {
        let _ = signal::ctrl_c().await;
    };

    #[cfg(unix)]
    let terminate = async {
        use tokio::signal::unix::{SignalKind, signal};
        if let Ok(mut stream) = signal(SignalKind::terminate()) {
            let _ = stream.recv().await;
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {}
        _ = terminate => {}
    }
}

impl ServerHandler for NativeServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_instructions(SERVER_INSTRUCTIONS)
    }

    #[allow(clippy::manual_async_fn)]
    fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> impl Future<Output = Result<ListToolsResult, McpError>> + Send + '_ {
        async move {
            Ok(ListToolsResult {
                tools: tool_list(),
                next_cursor: None,
                meta: None,
            })
        }
    }

    #[allow(clippy::manual_async_fn)]
    fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> impl Future<Output = Result<CallToolResult, McpError>> + Send + '_ {
        async move {
            let args = request.arguments.unwrap_or_default();
            match request.name.as_ref() {
                "index_codebase" => {
                    let args: IndexCodebaseArgs = parse_args(args)?;
                    let splitter = parse_splitter_kind(&args.splitter).map_err(invalid_params)?;
                    let scope = self
                        .engine
                        .config()
                        .resolve_mcp_scope(args.scope.as_deref(), args.path.as_deref())
                        .map_err(invalid_params)?;
                    let result = self
                        .start_background_indexing(
                            scope,
                            args.force,
                            splitter,
                            args.custom_extensions,
                            args.ignore_patterns,
                        )
                        .await
                        .map_err(internal_error)?;
                    Ok(tool_success(
                        render_index_launch_text(&result),
                        serde_json::to_value(result).ok(),
                    ))
                }
                "search_code" => {
                    let args: SearchCodeArgs = parse_args(args)?;
                    let scope = self
                        .engine
                        .config()
                        .resolve_mcp_scope(args.scope.as_deref(), args.path.as_deref())
                        .map_err(invalid_params)?;
                    let extension_filter = normalize_extension_filter(&args.extension_filter)
                        .map_err(invalid_params)?;
                    let mode = parse_search_mode(args.mode.as_deref()).map_err(invalid_params)?;
                    let result = self
                        .engine
                        .search_scope(
                            scope,
                            SearchRequest {
                                query: args.query,
                                limit: args.limit.clamp(1, 50),
                                mode,
                                extension_filter,
                                path_prefix: normalize_optional_path(&args.path_prefix),
                                language: normalize_optional_language(&args.language),
                                file: normalize_optional_path(&args.file),
                                dedupe_by_file: args.dedupe_by_file,
                                snippet_chars: if args.include_content {
                                    args.snippet_chars.min(1200)
                                } else {
                                    0
                                },
                            },
                        )
                        .await
                        .map_err(internal_error)?;
                    Ok(tool_success(
                        render_search_summary_text(&result),
                        serde_json::to_value(compact_search_response(
                            &result,
                            args.include_diagnostics,
                        ))
                        .ok(),
                    ))
                }
                "search_symbols" => {
                    let args: SearchSymbolsArgs = parse_args(args)?;
                    let scope = self
                        .engine
                        .config()
                        .resolve_mcp_scope(args.scope.as_deref(), args.path.as_deref())
                        .map_err(invalid_params)?;
                    let result = self
                        .engine
                        .search_symbols(
                            scope,
                            SymbolSearchScopeRequest {
                                query: args.query,
                                limit: args.limit.clamp(1, 50),
                                path_prefix: normalize_optional_path(&args.path_prefix),
                                language: normalize_optional_language(&args.language),
                                kind: normalize_optional_kind(&args.kind),
                                container: normalize_optional_string(&args.container),
                            },
                        )
                        .await
                        .map_err(internal_error)?;
                    Ok(tool_success(
                        render_symbol_search_summary_text(&result),
                        serde_json::to_value(compact_symbol_search_response(
                            &result,
                            args.include_diagnostics,
                        ))
                        .ok(),
                    ))
                }
                "search_text" => {
                    let args: SearchTextArgs = parse_args(args)?;
                    let scope = self
                        .engine
                        .config()
                        .resolve_mcp_scope(args.scope.as_deref(), args.path.as_deref())
                        .map_err(invalid_params)?;
                    let extension_filter = normalize_extension_filter(&args.extension_filter)
                        .map_err(invalid_params)?;
                    let result = self
                        .engine
                        .search_text_scope(
                            scope,
                            TextSearchScopeRequest {
                                repo: normalize_optional_string(&args.repo),
                                query: args.query,
                                limit: args.limit.clamp(1, 50),
                                path_prefix: normalize_optional_path(&args.path_prefix),
                                language: normalize_optional_language(&args.language),
                                file: normalize_optional_path(&args.file),
                                extension_filter,
                                case_sensitive: args.case_sensitive,
                                whole_word: args.whole_word,
                                context_lines: args.context_lines.min(12),
                            },
                        )
                        .await
                        .map_err(internal_error)?;
                    Ok(tool_success(
                        render_text_search_summary_text(&result),
                        serde_json::to_value(compact_text_search_response(&result)).ok(),
                    ))
                }
                "prepare_edit_target" => {
                    let args: PrepareEditTargetArgs = parse_args(args)?;
                    let scope = self
                        .engine
                        .config()
                        .resolve_mcp_scope(args.scope.as_deref(), args.path.as_deref())
                        .map_err(invalid_params)?;
                    let result = self
                        .engine
                        .prepare_edit_target(
                            scope,
                            PrepareEditTargetRequest {
                                repo: normalize_optional_string(&args.repo),
                                file: normalize_optional_path(&args.file),
                                symbol_id: normalize_optional_string(&args.symbol_id),
                                line_hint: args.line_hint,
                                query: args.query.filter(|value| !value.is_empty()),
                                occurrence: args.occurrence,
                                before_lines: args.before_lines.min(4),
                                after_lines: args.after_lines.min(8),
                                max_lines: args.max_lines.clamp(1, 32),
                                anchor_count: args.anchor_count.clamp(1, 3),
                            },
                        )
                        .await
                        .map_err(internal_error)?;
                    Ok(tool_success(
                        render_prepare_edit_target_summary_text(&result),
                        serde_json::to_value(compact_prepare_edit_target_response(&result)).ok(),
                    ))
                }
                "get_file_outline" => {
                    let args: FileOutlineArgs = parse_args(args)?;
                    let scope = self
                        .engine
                        .config()
                        .resolve_mcp_scope(args.scope.as_deref(), args.path.as_deref())
                        .map_err(invalid_params)?;
                    let detail =
                        parse_outline_detail(args.detail.as_deref()).map_err(invalid_params)?;
                    let result = self
                        .engine
                        .get_file_outline(scope, &args.file)
                        .await
                        .map_err(internal_error)?;
                    let result = compact_outline_response(
                        result,
                        OutlineCompactionOptions {
                            detail,
                            max_depth: args.max_depth.clamp(1, 16),
                            max_nodes: args.max_nodes.clamp(1, 512),
                            top_level_limit: args.top_level_limit.clamp(1, 256),
                        },
                    );
                    Ok(tool_success(
                        render_outline_summary_text(&result),
                        serde_json::to_value(result).ok(),
                    ))
                }
                "explain_search" => {
                    let args: SearchCodeArgs = parse_args(args)?;
                    let scope = self
                        .engine
                        .config()
                        .resolve_mcp_scope(args.scope.as_deref(), args.path.as_deref())
                        .map_err(invalid_params)?;
                    let extension_filter = normalize_extension_filter(&args.extension_filter)
                        .map_err(invalid_params)?;
                    let mode = parse_search_mode(args.mode.as_deref()).map_err(invalid_params)?;
                    let result = self
                        .engine
                        .explain_search(
                            scope,
                            &SearchRequest {
                                query: args.query,
                                limit: args.limit.clamp(1, 50),
                                mode,
                                extension_filter,
                                path_prefix: normalize_optional_path(&args.path_prefix),
                                language: normalize_optional_language(&args.language),
                                file: normalize_optional_path(&args.file),
                                dedupe_by_file: args.dedupe_by_file,
                                snippet_chars: 0,
                            },
                        )
                        .await
                        .map_err(internal_error)?;
                    Ok(tool_success(
                        render_search_explanation_text(&result),
                        serde_json::to_value(result).ok(),
                    ))
                }
                "clear_index" => {
                    let args: ScopeArgs = parse_args(args)?;
                    let scope = self
                        .engine
                        .config()
                        .resolve_mcp_scope(args.scope.as_deref(), args.path.as_deref())
                        .map_err(invalid_params)?;
                    let result = self
                        .engine
                        .clear_scope(scope)
                        .await
                        .map_err(internal_error)?;
                    Ok(tool_success(
                        render_clear_text(&result),
                        serde_json::to_value(result).ok(),
                    ))
                }
                "get_indexing_status" => {
                    let args: ScopeArgs = parse_args(args)?;
                    let scope = self
                        .engine
                        .config()
                        .resolve_mcp_scope(args.scope.as_deref(), args.path.as_deref())
                        .map_err(invalid_params)?;
                    let result = self
                        .engine
                        .status_scope(scope)
                        .await
                        .map_err(internal_error)?;
                    Ok(tool_success(
                        render_status_text(&result),
                        serde_json::to_value(result).ok(),
                    ))
                }
                "list_scopes" => {
                    let args: ListScopesArgs = parse_args(args)?;
                    let result = self.list_scopes(args.include_repos);
                    Ok(tool_success(
                        render_list_scopes_text(&result),
                        serde_json::to_value(result).ok(),
                    ))
                }
                _ => Err(McpError::method_not_found::<
                    rmcp::model::CallToolRequestMethod,
                >()),
            }
        }
    }
}

impl NativeServer {
    fn list_scopes(&self, include_repos: bool) -> ListScopesResult {
        let config = self.engine.config();
        ListScopesResult {
            default_scope: config.default_group.clone(),
            groups: config
                .groups
                .iter()
                .map(|group| ScopeSummary {
                    id: group.id.clone(),
                    label: group.label.clone().unwrap_or_else(|| group.id.clone()),
                    repo_count: group.repos.len(),
                    repos: include_repos.then(|| group.repos.clone()),
                })
                .collect(),
        }
    }

    async fn start_background_indexing(
        &self,
        scope: ResolvedScope,
        force: bool,
        splitter: SplitterKind,
        custom_extensions: Vec<String>,
        ignore_patterns: Vec<String>,
    ) -> Result<IndexLaunchResult> {
        let mut active_repos = self.active_repos.lock().await;
        let mut started_repos = Vec::new();
        let mut already_indexing = Vec::new();
        let mut runnable_repos = Vec::new();

        for repo in scope.repos {
            let repo_key = repo.display().to_string();
            if active_repos.contains(&repo_key) {
                already_indexing.push(repo_key);
            } else {
                active_repos.insert(repo_key.clone());
                started_repos.push(repo_key);
                runnable_repos.push(repo);
            }
        }
        drop(active_repos);

        if runnable_repos.is_empty() {
            return Ok(IndexLaunchResult {
                scope: scope.id,
                label: scope.label,
                started: false,
                force,
                started_repos,
                already_indexing,
            });
        }

        let runnable_scope = ResolvedScope {
            kind: scope.kind,
            id: scope.id.clone(),
            label: scope.label.clone(),
            repos: runnable_repos,
        };
        if let Err(error) = self.engine.mark_scope_indexing(&runnable_scope).await {
            let mut active_repos = self.active_repos.lock().await;
            for repo_key in &started_repos {
                active_repos.remove(repo_key);
            }
            return Err(error);
        }

        let engine = self.engine.clone();
        let active_repos = self.active_repos.clone();
        let completion_repos = started_repos.clone();
        tokio::spawn(async move {
            let _ = engine
                .index_scope(
                    runnable_scope,
                    force,
                    splitter,
                    &custom_extensions,
                    &ignore_patterns,
                )
                .await;

            let mut active_repos = active_repos.lock().await;
            for repo_key in completion_repos {
                active_repos.remove(&repo_key);
            }
        });

        Ok(IndexLaunchResult {
            scope: scope.id,
            label: scope.label,
            started: true,
            force,
            started_repos,
            already_indexing,
        })
    }
}

fn build_tool(
    name: &'static str,
    description: &'static str,
    read_only: bool,
    input_schema: Map<String, Value>,
) -> Tool {
    let mut annotations = ToolAnnotations::default();
    annotations.read_only_hint = Some(read_only);
    annotations.destructive_hint = Some(!read_only);
    annotations.idempotent_hint = Some(read_only);
    annotations.open_world_hint = Some(false);

    let mut tool = Tool::default();
    tool.name = Cow::Borrowed(name);
    tool.description = Some(Cow::Borrowed(description));
    tool.input_schema = Arc::new(input_schema);
    tool.annotations = Some(annotations);
    tool
}

fn parse_args<T>(arguments: Map<String, Value>) -> Result<T, McpError>
where
    T: for<'de> Deserialize<'de>,
{
    serde_json::from_value(Value::Object(arguments))
        .map_err(|error| invalid_params(anyhow::anyhow!(error)))
}

fn tool_success(text: String, structured: Option<Value>) -> CallToolResult {
    let mut result = CallToolResult::default();
    result.content = vec![Content::text(text)];
    result.structured_content = structured;
    result.is_error = Some(false);
    result
}

fn render_index_launch_text(result: &IndexLaunchResult) -> String {
    let mut lines = vec![format!("Scope: {}", result.label)];
    if result.started {
        lines.push(format!(
            "Started background indexing for {} repo(s). force={}",
            result.started_repos.len(),
            result.force
        ));
    } else {
        lines.push("No new background indexing jobs started.".to_string());
    }

    if !result.started_repos.is_empty() {
        lines.push(format!("Started: {}", result.started_repos.join(", ")));
    }
    if !result.already_indexing.is_empty() {
        lines.push(format!(
            "Already indexing: {}",
            result.already_indexing.join(", ")
        ));
    }

    lines.join("\n")
}

fn render_list_scopes_text(result: &ListScopesResult) -> String {
    let mut lines = vec![format!("Default scope: {}", result.default_scope)];
    if result.groups.is_empty() {
        lines.push("No configured group scopes.".to_string());
        return lines.join("\n");
    }

    for group in &result.groups {
        lines.push(format!(
            "{} ({}) repos={}",
            group.label, group.id, group.repo_count
        ));
    }
    lines.join("\n")
}

fn render_search_summary_text(result: &SearchResponse) -> String {
    let mut lines = vec![format!(
        "Scope: {} hits={} partial={}",
        result.label,
        result.hits.len(),
        result.partial
    )];
    for (index, hit) in result.hits.iter().enumerate() {
        lines.push(format!(
            "{}. {} :: {}:{}-{} {}",
            index + 1,
            hit.repo_label,
            hit.relative_path,
            hit.start_line,
            hit.end_line,
            hit.match_type
        ));
    }
    for error in &result.repo_errors {
        lines.push(format!("ERR {}: {}", error.repo, error.error));
    }
    lines.join("\n")
}

fn render_symbol_search_summary_text(result: &SymbolSearchResponse) -> String {
    let mut lines = vec![format!(
        "Scope: {} symbols={} partial={}",
        result.label,
        result.hits.len(),
        result.partial
    )];
    for (index, hit) in result.hits.iter().enumerate() {
        lines.push(format!(
            "{}. {} :: {}:{}-{} {} {}",
            index + 1,
            hit.repo_label,
            hit.relative_path,
            hit.start_line,
            hit.end_line,
            hit.kind,
            hit.name
        ));
    }
    for error in &result.repo_errors {
        lines.push(format!("ERR {}: {}", error.repo, error.error));
    }
    lines.join("\n")
}

fn render_text_search_summary_text(result: &TextSearchResponse) -> String {
    let mut lines = vec![format!(
        "Scope: {} hits={} partial={}",
        result.label,
        result.hits.len(),
        result.partial
    )];
    if result.hits.is_empty() {
        lines.push("No exact literal hits.".to_string());
    }
    for error in &result.repo_errors {
        lines.push(format!("ERR {}: {}", error.repo, error.error));
    }
    lines.join("\n")
}

fn render_prepare_edit_target_summary_text(
    result: &crate::engine::PrepareEditTargetResponse,
) -> String {
    match result.status {
        EditTargetStatus::Ready => format!(
            "Ready {} :: {}:{}-{} anchors={}",
            result.repo_label.as_deref().unwrap_or("unknown"),
            result.relative_path.as_deref().unwrap_or("unknown"),
            result.start_line.unwrap_or(0),
            result.end_line.unwrap_or(0),
            result.anchors.len()
        ),
        EditTargetStatus::Ambiguous => format!(
            "Ambiguous {} candidates={} next={}",
            result.relative_path.as_deref().unwrap_or("target"),
            result.candidates.len(),
            result
                .suggested_next_tool
                .as_deref()
                .unwrap_or("prepare_edit_target")
        ),
        EditTargetStatus::NeedsNarrowing => format!(
            "Needs narrowing {} reason={} next={}",
            result.relative_path.as_deref().unwrap_or("target"),
            result
                .reason_code
                .map(reason_code_label)
                .unwrap_or("unknown"),
            result
                .suggested_next_tool
                .as_deref()
                .unwrap_or("search_text")
        ),
        EditTargetStatus::NotFound => format!(
            "Not found {}",
            result.relative_path.as_deref().unwrap_or("target")
        ),
    }
}

fn reason_code_label(reason: EditTargetReasonCode) -> &'static str {
    match reason {
        EditTargetReasonCode::WindowTooBroad => "window_too_broad",
        EditTargetReasonCode::LargeSymbol => "large_symbol",
        EditTargetReasonCode::MultipleMatches => "multiple_matches",
        EditTargetReasonCode::WeakAnchors => "weak_anchors",
    }
}

fn render_outline_summary_text(result: &CompactFileOutlineResponse) -> String {
    let mut lines = vec![format!("Scope: {} file={}", result.label, result.file)];
    if result.matches.is_empty() {
        lines.push("No indexed outline.".to_string());
        return lines.join("\n");
    }
    for entry in &result.matches {
        lines.push(format!(
            "{} :: {} nodes={}/{} truncated={}",
            entry.repo_label,
            entry.relative_path,
            entry.returned_node_count,
            entry.total_node_count,
            entry.truncated
        ));
    }
    lines.join("\n")
}

fn compact_search_response(
    result: &SearchResponse,
    include_diagnostics: bool,
) -> CompactSearchResponse {
    CompactSearchResponse {
        scope: result.scope.clone(),
        label: result.label.clone(),
        partial: result.partial,
        repo_errors: result.repo_errors.clone(),
        plan: include_diagnostics.then(|| result.plan.clone()),
        hits: result
            .hits
            .iter()
            .map(|hit| CompactSearchHit {
                repo_label: hit.repo_label.clone(),
                relative_path: hit.relative_path.clone(),
                start_line: hit.start_line,
                end_line: hit.end_line,
                language: hit.language.clone(),
                match_type: hit.match_type.clone(),
                content: hit.content.clone(),
                stale: hit.stale,
                repo: include_diagnostics.then(|| hit.repo.clone()),
                score: include_diagnostics.then_some(round_score(hit.score)),
                dense_score: include_diagnostics
                    .then(|| hit.dense_score.map(round_score))
                    .flatten(),
                lexical_score: include_diagnostics
                    .then(|| hit.lexical_score.map(round_score))
                    .flatten(),
                symbol_score: include_diagnostics
                    .then(|| hit.symbol_score.map(round_score))
                    .flatten(),
                indexed_at: include_diagnostics
                    .then(|| hit.indexed_at.clone())
                    .flatten(),
            })
            .collect(),
    }
}

fn compact_symbol_search_response(
    result: &SymbolSearchResponse,
    include_diagnostics: bool,
) -> CompactSymbolSearchResponse {
    CompactSymbolSearchResponse {
        scope: result.scope.clone(),
        label: result.label.clone(),
        partial: result.partial,
        repo_errors: result.repo_errors.clone(),
        hits: result
            .hits
            .iter()
            .map(|hit| CompactSymbolHit {
                symbol_id: hit.symbol_id.clone(),
                repo_label: hit.repo_label.clone(),
                relative_path: hit.relative_path.clone(),
                name: hit.name.clone(),
                kind: hit.kind.clone(),
                container: hit.container.clone(),
                start_line: hit.start_line,
                end_line: hit.end_line,
                language: hit.language.clone(),
                stale: hit.stale,
                repo: include_diagnostics.then(|| hit.repo.clone()),
                score: include_diagnostics.then_some(round_score(hit.score)),
                lexical_score: include_diagnostics
                    .then(|| hit.lexical_score.map(round_score))
                    .flatten(),
                semantic_score: include_diagnostics
                    .then(|| hit.semantic_score.map(round_score))
                    .flatten(),
                indexed_at: include_diagnostics.then(|| hit.indexed_at.clone()),
                file_hash: include_diagnostics.then(|| hit.file_hash.clone()),
            })
            .collect(),
    }
}

fn compact_text_search_response(result: &TextSearchResponse) -> CompactTextSearchResponse {
    CompactTextSearchResponse {
        scope: result.scope.clone(),
        label: result.label.clone(),
        partial: result.partial,
        repo_errors: result.repo_errors.clone(),
        hits: result
            .hits
            .iter()
            .map(|hit| CompactTextSearchHit {
                repo_label: hit.repo_label.clone(),
                relative_path: hit.relative_path.clone(),
                start_line: hit.start_line,
                end_line: hit.end_line,
                preview: hit.preview.clone(),
            })
            .collect(),
    }
}

fn compact_prepare_edit_target_response(
    result: &PrepareEditTargetResponse,
) -> CompactPrepareEditTargetResponse {
    CompactPrepareEditTargetResponse {
        status: result.status,
        repo_label: result.repo_label.clone(),
        relative_path: result.relative_path.clone(),
        start_line: result.start_line,
        end_line: result.end_line,
        content: result.content.clone(),
        anchors: result.anchors.clone(),
        anchor_quality: result.anchor_quality,
        resolution_type: result.resolution_type,
        stale: result.stale.unwrap_or(false),
        unindexed: matches!(result.indexed, Some(false)),
        symbol_id: result.symbol_id.clone(),
        truncated: result.truncated.unwrap_or(false),
        candidates: result
            .candidates
            .iter()
            .map(|candidate| CompactEditTargetCandidate {
                repo_label: candidate.repo_label.clone(),
                relative_path: candidate.relative_path.clone(),
                start_line: candidate.start_line,
                end_line: candidate.end_line,
                preview: candidate.preview.clone(),
            })
            .collect(),
        symbol_start_line: result.symbol_start_line,
        symbol_end_line: result.symbol_end_line,
        reason_code: result.reason_code,
        suggested_next_tool: result.suggested_next_tool.clone(),
    }
}

#[derive(Debug, Clone, Copy)]
struct OutlineCompactionOptions {
    detail: OutlineDetail,
    max_depth: usize,
    max_nodes: usize,
    top_level_limit: usize,
}

#[derive(Debug)]
struct OutlineBudget {
    remaining_nodes: usize,
    truncated: bool,
}

fn compact_outline_response(
    result: FileOutlineResponse,
    options: OutlineCompactionOptions,
) -> CompactFileOutlineResponse {
    let include_repo = result.matches.len() > 1;
    CompactFileOutlineResponse {
        scope: result.scope,
        label: result.label,
        file: result.file,
        matches: result
            .matches
            .into_iter()
            .map(|entry| {
                let total_node_count = count_outline_nodes(&entry.symbols);
                let mut budget = OutlineBudget {
                    remaining_nodes: options.max_nodes.max(1),
                    truncated: false,
                };
                let symbols = compact_outline_nodes(&entry.symbols, options, 1, true, &mut budget);
                let returned_node_count = count_compact_outline_nodes(&symbols);
                let remaining_node_count = total_node_count.saturating_sub(returned_node_count);
                CompactFileOutlineMatch {
                    repo: include_repo.then_some(entry.repo),
                    repo_label: entry.repo_label,
                    relative_path: entry.relative_path,
                    language: entry.language,
                    stale: entry.stale,
                    truncated: budget.truncated || remaining_node_count > 0,
                    returned_node_count,
                    total_node_count,
                    remaining_node_count,
                    symbols,
                }
            })
            .collect(),
    }
}

fn compact_outline_nodes(
    nodes: &[OutlineNode],
    options: OutlineCompactionOptions,
    depth: usize,
    is_top_level: bool,
    budget: &mut OutlineBudget,
) -> Vec<CompactOutlineNode> {
    let limit = if is_top_level {
        options.top_level_limit.max(1)
    } else {
        usize::MAX
    };
    if nodes.len() > limit {
        budget.truncated = true;
    }

    let mut compact = Vec::new();
    for (index, node) in nodes.iter().enumerate() {
        if index >= limit {
            break;
        }
        if budget.remaining_nodes == 0 {
            budget.truncated = true;
            break;
        }
        budget.remaining_nodes -= 1;

        let has_children = !node.children.is_empty();
        let children = if should_descend_outline(options, depth) && has_children {
            compact_outline_nodes(&node.children, options, depth + 1, false, budget)
        } else {
            Vec::new()
        };
        compact.push(CompactOutlineNode {
            symbol_id: node.symbol_id.clone(),
            name: node.name.clone(),
            kind: node.kind.clone(),
            start_line: node.start_line,
            end_line: node.end_line,
            has_children,
            children,
        });
    }
    compact
}

fn should_descend_outline(options: OutlineCompactionOptions, depth: usize) -> bool {
    match options.detail {
        OutlineDetail::Summary => false,
        OutlineDetail::Compact | OutlineDetail::Full => depth < options.max_depth,
    }
}

fn count_compact_outline_nodes(nodes: &[CompactOutlineNode]) -> usize {
    nodes
        .iter()
        .map(|node| 1 + count_compact_outline_nodes(&node.children))
        .sum()
}

fn count_outline_nodes(nodes: &[OutlineNode]) -> usize {
    nodes
        .iter()
        .map(|node| 1 + count_outline_nodes(&node.children))
        .sum()
}

fn round_score(value: f64) -> f64 {
    (value * 1_000_000.0).round() / 1_000_000.0
}

fn is_false(value: &bool) -> bool {
    !*value
}

fn is_zero(value: &usize) -> bool {
    *value == 0
}

fn invalid_params(error: anyhow::Error) -> McpError {
    McpError::invalid_params(error.to_string(), None)
}

fn internal_error(error: anyhow::Error) -> McpError {
    McpError::internal_error(error.to_string(), None)
}

fn default_limit() -> usize {
    5
}

fn default_text_search_limit() -> usize {
    10
}

fn default_text_search_context_lines() -> usize {
    1
}

fn default_dedupe_by_file() -> bool {
    true
}

fn default_case_sensitive() -> bool {
    true
}

fn default_include_content() -> bool {
    true
}

fn default_snippet_chars() -> usize {
    360
}

fn default_outline_depth() -> usize {
    2
}

fn default_outline_max_nodes() -> usize {
    64
}

fn default_outline_top_level_limit() -> usize {
    32
}

fn default_before_lines() -> usize {
    2
}

fn default_after_lines() -> usize {
    6
}

fn default_max_lines() -> usize {
    32
}

fn default_anchor_count() -> usize {
    2
}

fn default_splitter() -> String {
    "ast".to_string()
}

fn normalize_extension_filter(extension_filter: &[String]) -> Result<Vec<String>> {
    let mut normalized = extension_filter
        .iter()
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
        .collect::<Vec<_>>();

    for extension in &normalized {
        if !(extension.starts_with('.')
            && extension.len() > 1
            && !extension.chars().any(char::is_whitespace))
        {
            anyhow::bail!(
                "invalid extensionFilter entry `{extension}`; use dotted extensions like `.ts` or `.py`"
            );
        }
    }

    normalized.sort();
    normalized.dedup();
    Ok(normalized)
}

fn parse_splitter_kind(value: &str) -> Result<SplitterKind> {
    match value {
        "ast" => Ok(SplitterKind::Ast),
        "langchain" => Ok(SplitterKind::LangChain),
        other => anyhow::bail!("invalid splitter `{other}`; must be `ast` or `langchain`"),
    }
}

fn parse_search_mode(value: Option<&str>) -> Result<SearchMode> {
    match value.unwrap_or("auto") {
        "auto" => Ok(SearchMode::Auto),
        "semantic" => Ok(SearchMode::Semantic),
        "hybrid" => Ok(SearchMode::Hybrid),
        "identifier" => Ok(SearchMode::Identifier),
        "path" => Ok(SearchMode::Path),
        other => anyhow::bail!(
            "invalid mode `{other}`; must be one of `auto`, `semantic`, `hybrid`, `identifier`, or `path`"
        ),
    }
}

fn parse_outline_detail(detail: Option<&str>) -> Result<OutlineDetail> {
    match detail
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("summary")
    {
        "summary" => Ok(OutlineDetail::Summary),
        "compact" => Ok(OutlineDetail::Compact),
        "full" => Ok(OutlineDetail::Full),
        other => anyhow::bail!(
            "invalid detail `{other}`; must be one of `summary`, `compact`, or `full`"
        ),
    }
}

fn normalize_optional_string(value: &Option<String>) -> Option<String> {
    value
        .as_ref()
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
}

fn normalize_optional_path(value: &Option<String>) -> Option<String> {
    normalize_optional_string(value).map(|value| value.replace('\\', "/"))
}

fn normalize_optional_language(value: &Option<String>) -> Option<String> {
    normalize_optional_string(value).map(|value| value.to_lowercase())
}

fn normalize_optional_kind(value: &Option<String>) -> Option<String> {
    normalize_optional_string(value).map(|value| value.to_lowercase())
}

fn nullable_string_schema(description: &str) -> Value {
    json!({
        "description": description,
        "type": ["string", "null"]
    })
}

fn index_codebase_schema() -> Map<String, Value> {
    json!({
        "type": "object",
        "description": "Start background indexing for a configured scope.",
        "properties": {
            "scope": nullable_string_schema("Configured group id or repo root. Defaults to the configured default group."),
            "force": {
                "type": "boolean",
                "default": false,
                "description": "Fully rebuild before indexing."
            },
            "splitter": {
                "type": "string",
                "enum": ["ast", "langchain"],
                "default": "ast",
                "description": "Chunking strategy."
            },
            "customExtensions": {
                "type": "array",
                "default": [],
                "description": "Additional dotted file extensions.",
                "items": {
                    "type": "string"
                }
            },
            "ignorePatterns": {
                "type": "array",
                "default": [],
                "description": "Additional ignore patterns.",
                "items": {
                    "type": "string"
                }
            }
        }
    })
    .as_object()
    .cloned()
    .unwrap_or_default()
}

fn search_code_schema() -> Map<String, Value> {
    json!({
        "type": "object",
        "description": "Broader code discovery. Use search_symbols first for exact definitions. Treat snippets as discovery hints, then use search_text or get_file_outline once the file is known. Use prepare_edit_target only when the exact patch location is already known.",
        "properties": {
            "scope": nullable_string_schema("Configured group id or repo root. Defaults to the configured default group."),
            "query": {
                "type": "string",
                "description": "Natural-language, identifier, or path-like query."
            },
            "limit": {
                "type": "integer",
                "format": "uint",
                "minimum": 1,
                "maximum": 50,
                "default": 5,
                "description": "Maximum hits."
            },
            "mode": {
                "type": ["string", "null"],
                "enum": ["auto", "semantic", "hybrid", "identifier", "path", null],
                "default": "auto",
                "description": "Search mode."
            },
            "extensionFilter": {
                "type": "array",
                "default": [],
                "description": "Dotted file extensions to include.",
                "items": {
                    "type": "string"
                }
            },
            "pathPrefix": nullable_string_schema("Repo-relative path prefix filter."),
            "language": nullable_string_schema("Normalized language filter."),
            "file": nullable_string_schema("Repo-relative file path filter."),
            "dedupeByFile": {
                "type": "boolean",
                "default": true,
                "description": "Return one best hit per file."
            },
            "includeContent": {
                "type": "boolean",
                "default": true,
                "description": "Include compact discovery snippets. Use search_text or prepare_edit_target for authoritative follow-up."
            },
            "snippetChars": {
                "type": "integer",
                "format": "uint",
                "minimum": 0,
                "maximum": 1200,
                "default": 360,
                "description": "Approximate max discovery snippet characters."
            },
            "includeDiagnostics": {
                "type": "boolean",
                "default": false,
                "description": "Include scores, plan, timestamps, and absolute repo paths."
            }
        },
        "required": ["query"]
    })
    .as_object()
    .cloned()
    .unwrap_or_default()
}

fn search_symbols_schema() -> Map<String, Value> {
    json!({
        "type": "object",
        "description": "Exact symbol and definition lookup. Prefer this before raw file reads when the definition name is known or suspected.",
        "properties": {
            "scope": nullable_string_schema("Configured group id or repo root. Defaults to the configured default group."),
            "query": {
                "type": "string",
                "description": "Definition name, container, or path-like symbol query."
            },
            "limit": {
                "type": "integer",
                "format": "uint",
                "minimum": 1,
                "maximum": 50,
                "default": 5,
                "description": "Maximum hits."
            },
            "pathPrefix": nullable_string_schema("Repo-relative path prefix filter."),
            "language": nullable_string_schema("Normalized language filter."),
            "kind": nullable_string_schema("Symbol kind filter."),
            "container": nullable_string_schema("Container/module/class filter."),
            "includeDiagnostics": {
                "type": "boolean",
                "default": false,
                "description": "Include scores, timestamps, file hashes, and absolute repo paths."
            }
        },
        "required": ["query"]
    })
    .as_object()
    .cloned()
    .unwrap_or_default()
}

fn search_text_schema() -> Map<String, Value> {
    json!({
        "type": "object",
        "description": "Exact literal confirmation after discovery. Use when the repo, file, or pathPrefix is already known. This is the MCP replacement for narrow shell rg when you need exact strings, identifiers, test names, or log lines.",
        "properties": {
            "scope": nullable_string_schema("Configured group id or repo root. Defaults to the configured default group."),
            "repo": nullable_string_schema("Absolute configured repo root to narrow a group scope to one repo."),
            "query": {
                "type": "string",
                "description": "Exact literal text to match. Regex is not supported."
            },
            "limit": {
                "type": "integer",
                "format": "uint",
                "minimum": 1,
                "maximum": 50,
                "default": 10,
                "description": "Maximum hits."
            },
            "file": nullable_string_schema("Repo-relative file path to search."),
            "pathPrefix": nullable_string_schema("Repo-relative path prefix to bound subtree search."),
            "language": nullable_string_schema("Normalized language filter."),
            "extensionFilter": {
                "type": "array",
                "default": [],
                "description": "Dotted file extensions to include.",
                "items": {
                    "type": "string"
                }
            },
            "caseSensitive": {
                "type": "boolean",
                "default": true,
                "description": "Match case exactly."
            },
            "wholeWord": {
                "type": "boolean",
                "default": false,
                "description": "Require whole-word matches."
            },
            "contextLines": {
                "type": "integer",
                "format": "uint",
                "minimum": 0,
                "maximum": 12,
                "default": 1,
                "description": "Extra surrounding lines in each preview."
            }
        },
        "required": ["query"]
    })
    .as_object()
    .cloned()
    .unwrap_or_default()
}

fn prepare_edit_target_schema() -> Map<String, Value> {
    json!({
        "type": "object",
        "description": "Final pre-patch step only. Use after search_symbols, search_code, search_text, or get_file_outline has already identified the exact patch location. Not for overview or broad inspection. If it returns ambiguous or needsNarrowing, go back to search_text or get_file_outline.",
        "properties": {
            "scope": nullable_string_schema("Configured group id or repo root. Defaults to the configured default group."),
            "repo": nullable_string_schema("Repo root or repo basename when disambiguating within a multi-repo scope."),
            "file": nullable_string_schema("Repo-relative file path. Required if symbolId is not provided."),
            "symbolId": nullable_string_schema("Indexed symbol id to prepare for editing."),
            "lineHint": {
                "type": ["integer", "null"],
                "format": "uint64",
                "description": "1-based line hint used to narrow a known edit location."
            },
            "query": nullable_string_schema("Exact literal text used to narrow a known edit target."),
            "occurrence": {
                "type": ["integer", "null"],
                "format": "uint",
                "minimum": 1,
                "description": "1-based occurrence selector for repeated literal matches when the exact occurrence is already known."
            }
        }
    })
    .as_object()
    .cloned()
    .unwrap_or_default()
}

fn explain_search_schema() -> Map<String, Value> {
    json!({
        "type": "object",
        "description": "Explain search_code query planning.",
        "properties": {
            "scope": nullable_string_schema("Configured group id or repo root. Defaults to the configured default group."),
            "query": {
                "type": "string",
                "description": "Query to explain."
            },
            "mode": {
                "type": ["string", "null"],
                "enum": ["auto", "semantic", "hybrid", "identifier", "path", null],
                "default": "auto",
                "description": "Search mode."
            },
            "extensionFilter": {
                "type": "array",
                "default": [],
                "description": "Dotted file extensions to include.",
                "items": {
                    "type": "string"
                }
            },
            "pathPrefix": nullable_string_schema("Repo-relative path prefix filter."),
            "language": nullable_string_schema("Normalized language filter."),
            "file": nullable_string_schema("Repo-relative file path filter.")
        },
        "required": ["query"]
    })
    .as_object()
    .cloned()
    .unwrap_or_default()
}

fn get_file_outline_schema() -> Map<String, Value> {
    json!({
        "type": "object",
        "description": "Return a compact indexed symbol outline summary for a known file. Defaults to a top-level summary and is not a full symbol dump.",
        "properties": {
            "scope": nullable_string_schema("Configured group id or repo root. Defaults to the configured default group."),
            "file": {
                "type": "string",
                "description": "Repo-relative file path."
            },
            "detail": {
                "type": "string",
                "enum": ["summary", "compact", "full"],
                "default": "summary",
                "description": "Outline detail level. `summary` is top-level only, `compact` expands within a small budget, and `full` is opt-in but still capped."
            },
            "maxDepth": {
                "type": "integer",
                "format": "uint",
                "minimum": 1,
                "maximum": 16,
                "default": 2,
                "description": "Maximum outline tree depth for `compact` or `full` detail."
            },
            "maxNodes": {
                "type": "integer",
                "format": "uint",
                "minimum": 1,
                "maximum": 512,
                "default": 64,
                "description": "Maximum outline nodes returned across the file."
            },
            "topLevelLimit": {
                "type": "integer",
                "format": "uint",
                "minimum": 1,
                "maximum": 256,
                "default": 32,
                "description": "Maximum top-level declarations returned before truncation."
            }
        },
        "required": ["file"]
    })
    .as_object()
    .cloned()
    .unwrap_or_default()
}

fn scope_args_schema(scope_description: &str) -> Map<String, Value> {
    json!({
        "type": "object",
        "properties": {
            "scope": nullable_string_schema(scope_description)
        }
    })
    .as_object()
    .cloned()
    .unwrap_or_default()
}

fn list_scopes_schema() -> Map<String, Value> {
    json!({
        "type": "object",
        "description": "Lists configured scopes.",
        "properties": {
            "includeRepos": {
                "type": "boolean",
                "default": false,
                "description": "Include repo paths for each scope."
            }
        }
    })
    .as_object()
    .cloned()
    .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::{
        OutlineCompactionOptions, OutlineDetail, SERVER_INSTRUCTIONS, SearchMode,
        compact_outline_response, compact_prepare_edit_target_response, compact_search_response,
        compact_text_search_response, default_limit, enforce_loopback_bind,
        get_file_outline_schema, list_scopes_schema, listen_is_loopback,
        normalize_extension_filter, parse_search_mode, parse_splitter_kind,
        prepare_edit_target_schema, search_symbols_schema, search_text_schema, tool_list,
    };
    use crate::engine::splitter::SplitterKind;
    use crate::engine::symbols::OutlineNode;
    use crate::engine::{
        AnchorQuality, EditResolutionType, EditTargetAnchor, EditTargetReasonCode,
        EditTargetStatus, FileOutlineMatch, FileOutlineResponse, PrepareEditTargetResponse,
        SearchHit, SearchPlanSummary, SearchResponse, TextSearchHit, TextSearchResponse,
    };
    use serde_json::{Value, to_value};

    #[test]
    fn search_limit_default_matches_node_contract() {
        assert_eq!(default_limit(), 5);
    }

    #[test]
    fn loopback_listeners_are_allowed_by_default() {
        assert!(listen_is_loopback("127.0.0.1:8765"));
        assert!(listen_is_loopback("[::1]:8765"));
        assert!(listen_is_loopback("localhost:8765"));
        assert!(enforce_loopback_bind("127.0.0.1:8765", false).is_ok());
    }

    #[test]
    fn remote_listener_requires_explicit_override() {
        let error = enforce_loopback_bind("0.0.0.0:8765", false).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("refusing to bind unauthenticated")
        );
        assert!(enforce_loopback_bind("0.0.0.0:8765", true).is_ok());
    }

    #[test]
    fn extension_filter_normalizes_and_deduplicates() {
        let normalized = normalize_extension_filter(&[
            ".rs".to_string(),
            " .py ".to_string(),
            ".rs".to_string(),
        ])
        .unwrap();

        assert_eq!(normalized, vec![".py".to_string(), ".rs".to_string()]);
    }

    #[test]
    fn extension_filter_rejects_invalid_values() {
        let error = normalize_extension_filter(&["rs".to_string()]).unwrap_err();
        assert!(error.to_string().contains("invalid extensionFilter entry"));
    }

    #[test]
    fn splitter_parser_rejects_unknown_values() {
        let error = parse_splitter_kind("bogus").unwrap_err();
        assert!(error.to_string().contains("invalid splitter"));
    }

    #[test]
    fn splitter_parser_accepts_supported_values() {
        assert!(matches!(
            parse_splitter_kind("ast").unwrap(),
            SplitterKind::Ast
        ));
        assert!(matches!(
            parse_splitter_kind("langchain").unwrap(),
            SplitterKind::LangChain
        ));
    }

    #[test]
    fn search_mode_parser_defaults_to_auto() {
        assert!(matches!(parse_search_mode(None).unwrap(), SearchMode::Auto));
        assert!(matches!(
            parse_search_mode(Some("identifier")).unwrap(),
            SearchMode::Identifier
        ));
    }

    #[test]
    fn tool_list_includes_scope_discovery() {
        let tools = tool_list();
        assert!(tools.iter().any(|tool| tool.name == "list_scopes"));
        assert!(tools.iter().any(|tool| tool.name == "search_symbols"));
        assert!(tools.iter().any(|tool| tool.name == "search_text"));
        assert!(tools.iter().any(|tool| tool.name == "prepare_edit_target"));
        assert!(tools.iter().any(|tool| tool.name == "get_file_outline"));
        assert!(tools.iter().any(|tool| tool.name == "explain_search"));
    }

    #[test]
    fn tool_descriptions_route_agents_toward_symbol_search() {
        let tools = tool_list();
        let search_code = tools
            .iter()
            .find(|tool| tool.name == "search_code")
            .and_then(|tool| tool.description.as_deref())
            .unwrap_or_default();
        let search_symbols = tools
            .iter()
            .find(|tool| tool.name == "search_symbols")
            .and_then(|tool| tool.description.as_deref())
            .unwrap_or_default();
        let search_text = tools
            .iter()
            .find(|tool| tool.name == "search_text")
            .and_then(|tool| tool.description.as_deref())
            .unwrap_or_default();
        let prepare_edit_target = tools
            .iter()
            .find(|tool| tool.name == "prepare_edit_target")
            .and_then(|tool| tool.description.as_deref())
            .unwrap_or_default();
        let list_scopes = tools
            .iter()
            .find(|tool| tool.name == "list_scopes")
            .and_then(|tool| tool.description.as_deref())
            .unwrap_or_default();

        assert!(search_code.contains("Snippets are discovery hints"));
        assert!(search_symbols.contains("definition name"));
        assert!(search_text.contains("instead of shell rg"));
        assert!(prepare_edit_target.contains("Final pre-patch step only"));
        assert!(list_scopes.contains("Preferred first call"));
    }

    #[test]
    fn tool_list_orders_navigation_before_prepare_edit() {
        let tools = tool_list();
        let names = tools
            .iter()
            .map(|tool| tool.name.as_ref())
            .collect::<Vec<_>>();
        let search_symbols = names
            .iter()
            .position(|name| *name == "search_symbols")
            .expect("search_symbols missing");
        let search_code = names
            .iter()
            .position(|name| *name == "search_code")
            .expect("search_code missing");
        let search_text = names
            .iter()
            .position(|name| *name == "search_text")
            .expect("search_text missing");
        let outline = names
            .iter()
            .position(|name| *name == "get_file_outline")
            .expect("get_file_outline missing");
        let prepare = names
            .iter()
            .position(|name| *name == "prepare_edit_target")
            .expect("prepare_edit_target missing");

        assert!(search_symbols < search_code);
        assert!(search_code < search_text);
        assert!(search_text < outline);
        assert!(outline < prepare);
    }

    #[test]
    fn search_tool_schema_advertises_limit_and_extension_filter_metadata() {
        let tools = tool_list();
        let search_tool = tools
            .iter()
            .find(|tool| tool.name == "search_code")
            .expect("search_code tool missing");
        let schema = Value::Object((*search_tool.input_schema).clone());

        assert_eq!(schema["properties"]["limit"]["default"].as_u64(), Some(5));
        assert_eq!(schema["properties"]["limit"]["maximum"].as_u64(), Some(50));
        assert_eq!(
            schema["properties"]["mode"]["default"].as_str(),
            Some("auto")
        );
        assert!(
            schema["properties"]["extensionFilter"]["description"]
                .as_str()
                .unwrap_or_default()
                .contains("Dotted file extensions")
        );
        assert!(
            schema["description"]
                .as_str()
                .unwrap_or_default()
                .contains("Treat snippets as discovery hints")
        );
        assert!(schema["properties"].get("path").is_none());
        assert_eq!(
            schema["properties"]["snippetChars"]["default"].as_u64(),
            Some(360)
        );
        assert!(
            schema["properties"]["includeContent"]["description"]
                .as_str()
                .unwrap_or_default()
                .contains("authoritative follow-up")
        );
    }

    #[test]
    fn index_tool_schema_advertises_splitter_enum() {
        let tools = tool_list();
        let index_tool = tools
            .iter()
            .find(|tool| tool.name == "index_codebase")
            .expect("index_codebase tool missing");
        let schema = Value::Object((*index_tool.input_schema).clone());
        let enum_values = schema["properties"]["splitter"]["enum"]
            .as_array()
            .expect("splitter enum missing")
            .iter()
            .filter_map(Value::as_str)
            .collect::<Vec<_>>();

        assert_eq!(enum_values, vec!["ast", "langchain"]);
    }

    #[test]
    fn symbol_and_scope_schemas_advertise_preferred_usage() {
        let search_symbols = Value::Object(search_symbols_schema());
        let search_text = Value::Object(search_text_schema());
        let prepare = Value::Object(prepare_edit_target_schema());
        let list_scopes = Value::Object(list_scopes_schema());

        assert!(
            search_symbols["description"]
                .as_str()
                .unwrap_or_default()
                .contains("Exact symbol and definition lookup")
        );
        assert!(
            search_symbols["properties"]["query"]["description"]
                .as_str()
                .unwrap_or_default()
                .contains("Definition name")
        );
        assert!(
            search_text["description"]
                .as_str()
                .unwrap_or_default()
                .contains("replacement for narrow shell rg")
        );
        assert!(
            search_text["description"]
                .as_str()
                .unwrap_or_default()
                .contains("repo, file, or pathPrefix")
        );
        assert!(
            search_text["properties"]["repo"]["description"]
                .as_str()
                .unwrap_or_default()
                .contains("narrow a group scope")
        );
        assert_eq!(
            search_text["properties"]["contextLines"]["default"].as_u64(),
            Some(1)
        );
        assert!(
            prepare["description"]
                .as_str()
                .unwrap_or_default()
                .contains("Final pre-patch step only")
        );
        assert!(
            search_text["properties"]["repo"]["description"]
                .as_str()
                .unwrap_or_default()
                .contains("narrow a group scope")
        );
        assert!(prepare["properties"].get("beforeLines").is_none());
        assert!(prepare["properties"].get("afterLines").is_none());
        assert!(prepare["properties"].get("maxLines").is_none());
        assert!(prepare["properties"].get("anchorCount").is_none());
        assert!(
            list_scopes["description"]
                .as_str()
                .unwrap_or_default()
                .contains("Lists configured scopes")
        );
        let outline = Value::Object(get_file_outline_schema());
        assert_eq!(
            outline["properties"]["detail"]["default"].as_str(),
            Some("summary")
        );
        assert_eq!(
            outline["properties"]["maxNodes"]["default"].as_u64(),
            Some(64)
        );
        assert_eq!(
            outline["properties"]["topLevelLimit"]["default"].as_u64(),
            Some(32)
        );
        assert!(SERVER_INSTRUCTIONS.contains("search_symbols for exact definitions"));
        assert!(SERVER_INSTRUCTIONS.contains("search_text for exact literals"));
        assert!(SERVER_INSTRUCTIONS.contains("instead of shell rg"));
        assert!(SERVER_INSTRUCTIONS.contains("final pre-patch step"));
    }

    #[test]
    fn compact_text_search_response_stays_small() {
        let response = TextSearchResponse {
            scope: "workspace".to_string(),
            label: "Workspace".to_string(),
            partial: false,
            repo_errors: Vec::new(),
            hits: vec![TextSearchHit {
                repo: "/tmp/repo".to_string(),
                repo_label: "repo".to_string(),
                relative_path: "src/lib.rs".to_string(),
                start_line: 10,
                end_line: 10,
                preview: "fn build() {}".to_string(),
                stale: false,
            }],
        };

        let json = to_value(compact_text_search_response(&response)).unwrap();
        assert!(json["hits"][0].get("repo").is_none());
        assert_eq!(json["hits"][0]["preview"], "fn build() {}");
    }

    #[test]
    fn compact_prepare_edit_target_response_omits_heavy_diagnostics() {
        let response = PrepareEditTargetResponse {
            status: EditTargetStatus::Ready,
            repo: Some("/tmp/repo".to_string()),
            repo_label: Some("repo".to_string()),
            relative_path: Some("src/lib.rs".to_string()),
            start_line: Some(10),
            end_line: Some(14),
            content: Some("fn build() {}\n".to_string()),
            anchors: vec![EditTargetAnchor {
                line: 10,
                text: "fn build() {".to_string(),
                unique_in_file: true,
            }],
            anchor_quality: Some(AnchorQuality::Strong),
            resolution_type: Some(EditResolutionType::Literal),
            file_hash: Some("abc".to_string()),
            indexed: Some(true),
            stale: Some(false),
            indexed_at: Some("2026-01-01T00:00:00Z".to_string()),
            indexed_file_hash: Some("abc".to_string()),
            symbol_id: Some("sym_123".to_string()),
            truncated: Some(false),
            candidates: Vec::new(),
            symbol_start_line: None,
            symbol_end_line: None,
            reason_code: None,
            suggested_next_tool: None,
        };

        let json = to_value(compact_prepare_edit_target_response(&response)).unwrap();
        assert_eq!(json["repoLabel"], "repo");
        assert!(json.get("repo").is_none());
        assert!(json.get("fileHash").is_none());
        assert!(json.get("indexedAt").is_none());
        assert!(json.get("indexedFileHash").is_none());
        assert!(json.get("stale").is_none());
        assert!(json.get("unindexed").is_none());
        assert!(json.get("truncated").is_none());
        assert_eq!(json["anchors"][0]["text"], "fn build() {");
    }

    #[test]
    fn compact_prepare_edit_target_response_keeps_guidance_for_non_ready_states() {
        let response = PrepareEditTargetResponse {
            status: EditTargetStatus::NeedsNarrowing,
            repo: Some("/tmp/repo".to_string()),
            repo_label: Some("repo".to_string()),
            relative_path: Some("src/lib.rs".to_string()),
            start_line: None,
            end_line: None,
            content: None,
            anchors: Vec::new(),
            anchor_quality: None,
            resolution_type: Some(EditResolutionType::Symbol),
            file_hash: Some("abc".to_string()),
            indexed: Some(true),
            stale: Some(false),
            indexed_at: Some("2026-01-01T00:00:00Z".to_string()),
            indexed_file_hash: Some("abc".to_string()),
            symbol_id: Some("sym_123".to_string()),
            truncated: Some(true),
            candidates: Vec::new(),
            symbol_start_line: Some(10),
            symbol_end_line: Some(80),
            reason_code: Some(EditTargetReasonCode::LargeSymbol),
            suggested_next_tool: Some("get_file_outline".to_string()),
        };

        let json = to_value(compact_prepare_edit_target_response(&response)).unwrap();
        assert_eq!(json["reasonCode"], "large_symbol");
        assert_eq!(json["suggestedNextTool"], "get_file_outline");
        assert!(json.get("content").is_none());
    }

    #[test]
    fn compact_search_response_omits_diagnostics_by_default() {
        let response = SearchResponse {
            scope: "workspace".to_string(),
            label: "Workspace".to_string(),
            partial: false,
            repo_errors: Vec::new(),
            plan: SearchPlanSummary {
                requested_mode: "auto".to_string(),
                effective_mode: "hybrid".to_string(),
                query_kind: "mixed".to_string(),
                dense_weight: 1.0,
                lexical_weight: 1.0,
                symbol_weight: 1.0,
                symbol_lexical_share: 0.5,
                symbol_semantic_share: 0.5,
                dedupe_by_file: true,
            },
            hits: vec![SearchHit {
                repo: "/tmp/repo".to_string(),
                repo_label: "repo".to_string(),
                relative_path: "src/lib.rs".to_string(),
                start_line: 10,
                end_line: 12,
                language: "rust".to_string(),
                score: 0.42,
                match_type: "hybrid".to_string(),
                dense_score: Some(0.1),
                lexical_score: Some(0.2),
                symbol_score: Some(0.3),
                indexed_at: Some("2026-01-01T00:00:00Z".to_string()),
                stale: false,
                content: "fn example() {}".to_string(),
            }],
        };

        let value = to_value(compact_search_response(&response, false)).unwrap();
        let hit = &value["hits"][0];

        assert!(value.get("plan").is_none());
        assert!(value.get("repoErrors").is_none());
        assert!(hit.get("repo").is_none());
        assert!(hit.get("score").is_none());
        assert!(hit.get("denseScore").is_none());
        assert!(hit.get("indexedAt").is_none());
        assert!(hit.get("stale").is_none());
        assert_eq!(hit["relativePath"].as_str(), Some("src/lib.rs"));
        assert_eq!(hit["content"].as_str(), Some("fn example() {}"));
    }

    #[test]
    fn compact_outline_response_prunes_to_requested_depth() {
        let result = FileOutlineResponse {
            scope: "workspace".to_string(),
            label: "Workspace".to_string(),
            file: "src/lib.rs".to_string(),
            matches: vec![FileOutlineMatch {
                repo: "/tmp/repo".to_string(),
                repo_label: "repo".to_string(),
                relative_path: "src/lib.rs".to_string(),
                language: Some("rust".to_string()),
                indexed_at: Some("2026-01-01T00:00:00Z".to_string()),
                stale: false,
                symbols: vec![OutlineNode {
                    symbol_id: "root".to_string(),
                    name: "Root".to_string(),
                    kind: "struct".to_string(),
                    container: None,
                    language: "rust".to_string(),
                    start_line: 1,
                    end_line: 20,
                    children: vec![OutlineNode {
                        symbol_id: "child".to_string(),
                        name: "child".to_string(),
                        kind: "method".to_string(),
                        container: Some("Root".to_string()),
                        language: "rust".to_string(),
                        start_line: 5,
                        end_line: 8,
                        children: vec![OutlineNode {
                            symbol_id: "grandchild".to_string(),
                            name: "grandchild".to_string(),
                            kind: "function".to_string(),
                            container: Some("child".to_string()),
                            language: "rust".to_string(),
                            start_line: 6,
                            end_line: 7,
                            children: Vec::new(),
                        }],
                    }],
                }],
            }],
        };

        let compact = compact_outline_response(
            result,
            OutlineCompactionOptions {
                detail: OutlineDetail::Compact,
                max_depth: 2,
                max_nodes: 64,
                top_level_limit: 32,
            },
        );

        assert_eq!(compact.matches[0].symbols[0].children.len(), 1);
        assert!(
            compact.matches[0].symbols[0].children[0]
                .children
                .is_empty()
        );
        assert_eq!(compact.matches[0].returned_node_count, 2);
        assert_eq!(compact.matches[0].total_node_count, 3);
        assert_eq!(compact.matches[0].remaining_node_count, 1);
        assert!(compact.matches[0].truncated);
    }

    #[test]
    fn compact_outline_response_defaults_to_summary_budget() {
        let result = FileOutlineResponse {
            scope: "workspace".to_string(),
            label: "Workspace".to_string(),
            file: "src/lib.rs".to_string(),
            matches: vec![FileOutlineMatch {
                repo: "/tmp/repo".to_string(),
                repo_label: "repo".to_string(),
                relative_path: "src/lib.rs".to_string(),
                language: Some("rust".to_string()),
                indexed_at: Some("2026-01-01T00:00:00Z".to_string()),
                stale: false,
                symbols: vec![
                    OutlineNode {
                        symbol_id: "root".to_string(),
                        name: "Root".to_string(),
                        kind: "struct".to_string(),
                        container: None,
                        language: "rust".to_string(),
                        start_line: 1,
                        end_line: 20,
                        children: vec![OutlineNode {
                            symbol_id: "child".to_string(),
                            name: "child".to_string(),
                            kind: "method".to_string(),
                            container: Some("Root".to_string()),
                            language: "rust".to_string(),
                            start_line: 5,
                            end_line: 8,
                            children: Vec::new(),
                        }],
                    },
                    OutlineNode {
                        symbol_id: "other".to_string(),
                        name: "Other".to_string(),
                        kind: "enum".to_string(),
                        container: None,
                        language: "rust".to_string(),
                        start_line: 30,
                        end_line: 40,
                        children: Vec::new(),
                    },
                ],
            }],
        };

        let compact = compact_outline_response(
            result,
            OutlineCompactionOptions {
                detail: OutlineDetail::Summary,
                max_depth: 2,
                max_nodes: 64,
                top_level_limit: 32,
            },
        );

        let match_entry = &compact.matches[0];
        assert!(match_entry.repo.is_none());
        assert_eq!(match_entry.returned_node_count, 2);
        assert_eq!(match_entry.total_node_count, 3);
        assert_eq!(match_entry.remaining_node_count, 1);
        assert!(match_entry.truncated);
        assert!(match_entry.symbols[0].children.is_empty());
        assert!(match_entry.symbols[0].has_children);
    }

    #[test]
    fn compact_outline_response_caps_top_level_nodes() {
        let symbols = (0..4)
            .map(|index| OutlineNode {
                symbol_id: format!("sym_{index}"),
                name: format!("Item{index}"),
                kind: "function".to_string(),
                container: None,
                language: "rust".to_string(),
                start_line: index + 1,
                end_line: index + 1,
                children: Vec::new(),
            })
            .collect();
        let result = FileOutlineResponse {
            scope: "workspace".to_string(),
            label: "Workspace".to_string(),
            file: "src/lib.rs".to_string(),
            matches: vec![FileOutlineMatch {
                repo: "/tmp/repo".to_string(),
                repo_label: "repo".to_string(),
                relative_path: "src/lib.rs".to_string(),
                language: Some("rust".to_string()),
                indexed_at: None,
                stale: false,
                symbols,
            }],
        };

        let compact = compact_outline_response(
            result,
            OutlineCompactionOptions {
                detail: OutlineDetail::Summary,
                max_depth: 2,
                max_nodes: 64,
                top_level_limit: 2,
            },
        );

        assert_eq!(compact.matches[0].symbols.len(), 2);
        assert_eq!(compact.matches[0].returned_node_count, 2);
        assert_eq!(compact.matches[0].total_node_count, 4);
        assert_eq!(compact.matches[0].remaining_node_count, 2);
        assert!(compact.matches[0].truncated);
    }

    #[test]
    fn compact_outline_response_includes_repo_when_multiple_matches_exist() {
        let match_entry = FileOutlineMatch {
            repo: "/tmp/repo-a".to_string(),
            repo_label: "repo-a".to_string(),
            relative_path: "src/lib.rs".to_string(),
            language: Some("rust".to_string()),
            indexed_at: None,
            stale: false,
            symbols: vec![OutlineNode {
                symbol_id: "root".to_string(),
                name: "Root".to_string(),
                kind: "struct".to_string(),
                container: None,
                language: "rust".to_string(),
                start_line: 1,
                end_line: 10,
                children: Vec::new(),
            }],
        };
        let result = FileOutlineResponse {
            scope: "workspace".to_string(),
            label: "Workspace".to_string(),
            file: "src/lib.rs".to_string(),
            matches: vec![
                match_entry.clone(),
                FileOutlineMatch {
                    repo: "/tmp/repo-b".to_string(),
                    repo_label: "repo-b".to_string(),
                    ..match_entry
                },
            ],
        };

        let compact = compact_outline_response(
            result,
            OutlineCompactionOptions {
                detail: OutlineDetail::Summary,
                max_depth: 2,
                max_nodes: 64,
                top_level_limit: 32,
            },
        );

        assert_eq!(compact.matches[0].repo.as_deref(), Some("/tmp/repo-a"));
        assert_eq!(compact.matches[1].repo.as_deref(), Some("/tmp/repo-b"));
    }
}
