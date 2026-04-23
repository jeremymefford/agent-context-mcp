use crate::config::{Config, ResolvedScope};
use crate::engine::splitter::SplitterKind;
use crate::engine::symbols::OutlineNode;
use crate::engine::{
    Engine, FileOutlineResponse, RepoSearchError, SearchMode, SearchPlanSummary, SearchRequest,
    SearchResponse, SymbolSearchResponse, SymbolSearchScopeRequest, render_clear_text,
    render_search_explanation_text, render_status_text,
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
struct FileOutlineArgs {
    #[serde(default)]
    scope: Option<String>,
    #[serde(default)]
    path: Option<String>,
    file: String,
    #[serde(default = "default_outline_depth")]
    max_depth: usize,
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

const SERVER_INSTRUCTIONS: &str = "Use list_scopes first. Use search_symbols for exact definitions. Use get_file_outline when the file is known. Use search_code for broader code search. scope defaults to the configured default group.";

pub fn tool_list() -> Vec<Tool> {
    vec![
        build_tool(
            "index_codebase",
            "Start background indexing for a configured scope. Defaults to the configured default group.",
            false,
            index_codebase_schema(),
        ),
        build_tool(
            "search_code",
            "Broader code search over a configured scope. Use search_symbols first for exact definitions.",
            true,
            search_code_schema(),
        ),
        build_tool(
            "search_symbols",
            "Preferred tool for exact symbol and definition lookup.",
            true,
            search_symbols_schema(),
        ),
        build_tool(
            "get_file_outline",
            "Return the indexed symbol outline for a known repo-relative file.",
            true,
            get_file_outline_schema(),
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
                "get_file_outline" => {
                    let args: FileOutlineArgs = parse_args(args)?;
                    let scope = self
                        .engine
                        .config()
                        .resolve_mcp_scope(args.scope.as_deref(), args.path.as_deref())
                        .map_err(invalid_params)?;
                    let result = self
                        .engine
                        .get_file_outline(scope, &args.file)
                        .await
                        .map_err(internal_error)?;
                    let result = compact_outline_response(result, args.max_depth.clamp(1, 16));
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

fn render_outline_summary_text(result: &FileOutlineResponse) -> String {
    let mut lines = vec![format!("Scope: {} file={}", result.label, result.file)];
    if result.matches.is_empty() {
        lines.push("No indexed outline.".to_string());
        return lines.join("\n");
    }
    for entry in &result.matches {
        lines.push(format!(
            "{} :: {} symbols={}",
            entry.repo_label,
            entry.relative_path,
            count_outline_nodes(&entry.symbols)
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

fn compact_outline_response(
    mut result: FileOutlineResponse,
    max_depth: usize,
) -> FileOutlineResponse {
    for entry in &mut result.matches {
        entry.symbols = prune_outline_nodes(&entry.symbols, max_depth);
    }
    result
}

fn prune_outline_nodes(nodes: &[OutlineNode], max_depth: usize) -> Vec<OutlineNode> {
    nodes
        .iter()
        .map(|node| OutlineNode {
            symbol_id: node.symbol_id.clone(),
            name: node.name.clone(),
            kind: node.kind.clone(),
            container: node.container.clone(),
            language: node.language.clone(),
            start_line: node.start_line,
            end_line: node.end_line,
            children: if max_depth <= 1 {
                Vec::new()
            } else {
                prune_outline_nodes(&node.children, max_depth - 1)
            },
        })
        .collect()
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

fn invalid_params(error: anyhow::Error) -> McpError {
    McpError::invalid_params(error.to_string(), None)
}

fn internal_error(error: anyhow::Error) -> McpError {
    McpError::internal_error(error.to_string(), None)
}

fn default_limit() -> usize {
    5
}

fn default_dedupe_by_file() -> bool {
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
        "description": "Broader code search. Use search_symbols first for exact definitions.",
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
                "description": "Include compact snippets."
            },
            "snippetChars": {
                "type": "integer",
                "format": "uint",
                "minimum": 0,
                "maximum": 1200,
                "default": 360,
                "description": "Approximate max snippet characters."
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
        "description": "Preferred exact symbol and definition lookup.",
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
        "description": "Return an indexed symbol outline for a known file.",
        "properties": {
            "scope": nullable_string_schema("Configured group id or repo root. Defaults to the configured default group."),
            "file": {
                "type": "string",
                "description": "Repo-relative file path."
            },
            "maxDepth": {
                "type": "integer",
                "format": "uint",
                "minimum": 1,
                "maximum": 16,
                "default": 2,
                "description": "Maximum outline tree depth."
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
        SERVER_INSTRUCTIONS, SearchMode, compact_outline_response, compact_search_response,
        default_limit, enforce_loopback_bind, list_scopes_schema, listen_is_loopback,
        normalize_extension_filter, parse_search_mode, parse_splitter_kind, search_symbols_schema,
        tool_list,
    };
    use crate::engine::splitter::SplitterKind;
    use crate::engine::symbols::OutlineNode;
    use crate::engine::{
        FileOutlineMatch, FileOutlineResponse, SearchHit, SearchPlanSummary, SearchResponse,
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
        let list_scopes = tools
            .iter()
            .find(|tool| tool.name == "list_scopes")
            .and_then(|tool| tool.description.as_deref())
            .unwrap_or_default();

        assert!(search_code.contains("Use search_symbols first"));
        assert!(search_symbols.contains("Preferred tool for exact symbol"));
        assert!(list_scopes.contains("Preferred first call"));
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
                .contains("Use search_symbols first")
        );
        assert!(schema["properties"].get("path").is_none());
        assert_eq!(
            schema["properties"]["snippetChars"]["default"].as_u64(),
            Some(360)
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
        let list_scopes = Value::Object(list_scopes_schema());

        assert!(
            search_symbols["description"]
                .as_str()
                .unwrap_or_default()
                .contains("Preferred exact symbol")
        );
        assert!(
            search_symbols["properties"]["query"]["description"]
                .as_str()
                .unwrap_or_default()
                .contains("Definition name")
        );
        assert!(
            list_scopes["description"]
                .as_str()
                .unwrap_or_default()
                .contains("Lists configured scopes")
        );
        assert!(SERVER_INSTRUCTIONS.contains("search_symbols for exact definitions"));
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

        let compact = compact_outline_response(result, 2);

        assert_eq!(compact.matches[0].symbols[0].children.len(), 1);
        assert!(
            compact.matches[0].symbols[0].children[0]
                .children
                .is_empty()
        );
    }
}
