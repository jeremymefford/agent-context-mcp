use crate::config::{Config, ResolvedScope};
use crate::engine::splitter::SplitterKind;
use crate::engine::{
    Engine, SearchMode, SearchRequest, SymbolSearchScopeRequest, render_clear_text,
    render_outline_text, render_search_explanation_text, render_search_text, render_status_text,
    render_symbol_search_text,
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
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
struct FileOutlineArgs {
    #[serde(default)]
    scope: Option<String>,
    #[serde(default)]
    path: Option<String>,
    file: String,
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
struct ListScopesArgs {}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ScopeSummary {
    id: String,
    label: String,
    repos: Vec<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ListScopesResult {
    default_scope: String,
    groups: Vec<ScopeSummary>,
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

const SERVER_INSTRUCTIONS: &str = "Rust-native agent-context MCP server. One HTTP MCP server talks directly to the configured embedding provider and Milvus, supports named repo groups, and avoids spawned Node MCP workers. Preferred routing for agents: call `list_scopes` before guessing configured group ids; use `search_symbols` first for exact symbol and definition lookup by name; use `get_file_outline` after a symbol or file is known and you need indexed structure; use `search_code` for broader semantic or hybrid discovery when you do not already know the exact definition name; use `explain_search` when you need to understand how `search_code` will classify and weight a query. `scope` accepts a configured group id; `path` accepts a configured absolute repo root as a compatibility alias; if both are omitted, the configured default group is used.";

pub fn tool_list() -> Vec<Tool> {
    vec![
        build_tool(
            "index_codebase",
            "Start background indexing for one configured repo or every repo in a configured group. If neither `scope` nor `path` is provided, the configured default group is indexed.",
            false,
            index_codebase_schema(),
        ),
        build_tool(
            "search_code",
            "Search indexed code in one configured repo or across a configured group using query-planned dense, lexical, and symbol-aware ranking. Use this for broader semantic or hybrid code discovery. For exact definition or symbol lookup by name, prefer `search_symbols` first. Prefer `scope` for configured group ids and configured absolute repo roots; if omitted, the configured default group is searched.",
            true,
            search_code_schema(),
        ),
        build_tool(
            "search_symbols",
            "Preferred tool for exact symbol and definition lookup by name across one configured repo or a configured group. Use this before `search_code` when the user asks for a function, class, struct, trait, module, method, or other named definition. Prefer `scope` for configured groups and configured absolute repo roots; if omitted, the configured default group is searched.",
            true,
            search_symbols_schema(),
        ),
        build_tool(
            "get_file_outline",
            "Return the indexed symbol outline for one repo-relative file across one repo or a configured group. Use this after `search_symbols` or `search_code` when you already know the file and want indexed structure. If a group contains multiple repos with the same file path, all matching outlines are returned.",
            true,
            get_file_outline_schema(),
        ),
        build_tool(
            "explain_search",
            "Explain how `search_code` will classify a query, which indexes it will use, and how it will weight dense, lexical, and symbol signals. Use this when tuning retrieval or when an agent is unsure whether `search_code` or `search_symbols` is the better fit.",
            true,
            search_code_schema(),
        ),
        build_tool(
            "clear_index",
            "Drop the Rust-managed index for one repo or every repo in a configured group. If neither `scope` nor `path` is provided, the configured default group is cleared.",
            false,
            scope_args_schema(
                "Optional configured group id or configured absolute repo root. Prefer calling `list_scopes` first for configured groups. If omitted, the configured default group is used.",
            ),
        ),
        build_tool(
            "get_indexing_status",
            "Report indexing status, counts, and progress for one repo or a configured group. If neither `scope` nor `path` is provided, the configured default group is used.",
            true,
            scope_args_schema(
                "Optional configured group id or configured absolute repo root. Prefer calling `list_scopes` first for configured groups. If omitted, the configured default group is used.",
            ),
        ),
        build_tool(
            "list_scopes",
            "Preferred first call for unfamiliar agents. Lists configured group scopes and the default scope so searches and indexing can target the right workspace without guessing group ids.",
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
                            },
                        )
                        .await
                        .map_err(internal_error)?;
                    Ok(tool_success(
                        render_search_text(&result),
                        serde_json::to_value(result).ok(),
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
                        render_symbol_search_text(&result),
                        serde_json::to_value(result).ok(),
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
                    Ok(tool_success(
                        render_outline_text(&result),
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
                    let _: ListScopesArgs = parse_args(args)?;
                    let result = self.list_scopes();
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
    fn list_scopes(&self) -> ListScopesResult {
        let config = self.engine.config();
        ListScopesResult {
            default_scope: config.default_group.clone(),
            groups: config
                .groups
                .iter()
                .map(|group| ScopeSummary {
                    id: group.id.clone(),
                    label: group.label.clone().unwrap_or_else(|| group.id.clone()),
                    repos: group.repos.clone(),
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
            group.label,
            group.id,
            group.repos.len()
        ));
    }
    lines.join("\n")
}

fn invalid_params(error: anyhow::Error) -> McpError {
    McpError::invalid_params(error.to_string(), None)
}

fn internal_error(error: anyhow::Error) -> McpError {
    McpError::internal_error(error.to_string(), None)
}

fn default_limit() -> usize {
    10
}

fn default_dedupe_by_file() -> bool {
    true
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
        "default": Value::Null,
        "type": ["string", "null"]
    })
}

fn index_codebase_schema() -> Map<String, Value> {
    json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "IndexCodebaseArgs",
        "type": "object",
        "description": "Start background indexing. Prefer `scope` for configured groups and configured absolute repo roots. `path` is accepted as a compatibility alias. If neither is provided, the configured default group is indexed.",
        "properties": {
            "scope": nullable_string_schema("Configured group id or configured absolute repo root. Preferred over `path`. If omitted together with `path`, the configured default group is indexed."),
            "path": nullable_string_schema("Compatibility alias for `scope`. Accepts a configured absolute repo root or configured group id. Prefer `scope` in new clients."),
            "force": {
                "type": "boolean",
                "default": false,
                "description": "When true, drop and fully rebuild the Rust-managed collection before indexing."
            },
            "splitter": {
                "type": "string",
                "enum": ["ast", "langchain"],
                "default": "ast",
                "description": "`ast` uses syntax-aware splitting with fallback; `langchain` uses the text-splitter path directly."
            },
            "customExtensions": {
                "type": "array",
                "default": [],
                "description": "Optional additional file extensions to include, such as ['.vue', '.svelte']",
                "items": {
                    "type": "string"
                }
            },
            "ignorePatterns": {
                "type": "array",
                "default": [],
                "description": "Optional additional ignore patterns beyond built-in and file-based ignores, such as ['dist/**', '*.tmp']",
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
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "SearchCodeArgs",
        "type": "object",
        "description": "Search indexed code using dense semantic search, a local lexical index, and symbol-aware boosts. Use this for broader semantic or hybrid code discovery. For exact definition or symbol lookup by name, prefer `search_symbols` first. Prefer `scope` for configured groups and configured absolute repo roots. `path` is accepted as a compatibility alias. If neither is provided, the configured default group is searched.",
        "properties": {
            "scope": nullable_string_schema("Configured group id or configured absolute repo root. Preferred over `path`. If omitted together with `path`, the configured default group is searched."),
            "path": nullable_string_schema("Compatibility alias for `scope`. Accepts a configured absolute repo root or configured group id. Prefer `scope` in new clients."),
            "query": {
                "type": "string",
                "description": "Natural-language, identifier-like, or path/module-like query to search for. Prefer `search_symbols` instead when the intent is exact symbol or definition lookup by name."
            },
            "limit": {
                "type": "integer",
                "format": "uint",
                "minimum": 1,
                "maximum": 50,
                "default": 10,
                "description": "Maximum number of final merged results to return. Values above 50 are capped."
            },
            "mode": {
                "type": ["string", "null"],
                "enum": ["auto", "semantic", "hybrid", "identifier", "path", null],
                "default": "auto",
                "description": "`auto` classifies the query and chooses dense/lexical/symbol weighting automatically. Override with a fixed search mode when needed."
            },
            "extensionFilter": {
                "type": "array",
                "default": [],
                "description": "Optional list of dotted file extensions to filter search results, such as ['.rs', '.py'].",
                "items": {
                    "type": "string"
                }
            },
            "pathPrefix": nullable_string_schema("Optional repo-relative path prefix filter, such as `src/graphql/` or `server/crates/`."),
            "language": nullable_string_schema("Optional normalized language filter, such as `rust`, `typescript`, or `python`."),
            "file": nullable_string_schema("Optional repo-relative file path to constrain search to one indexed file."),
            "dedupeByFile": {
                "type": "boolean",
                "default": true,
                "description": "When true, only the best hit per file is returned. Set false to allow multiple top hits from the same file."
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
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "SearchSymbolsArgs",
        "type": "object",
        "description": "Preferred tool for exact symbol and definition lookup by name. Search indexed definitions using the local symbol index before falling back to broader `search_code` queries.",
        "properties": {
            "scope": nullable_string_schema("Configured group id or configured absolute repo root. Preferred over `path`. If omitted together with `path`, the configured default group is searched."),
            "path": nullable_string_schema("Compatibility alias for `scope`. Accepts a configured absolute repo root or configured group id. Prefer `scope` in new clients."),
            "query": {
                "type": "string",
                "description": "Exact or near-exact definition name, container, or path-oriented symbol query. Use this when the user asks for a function, class, struct, trait, module, method, or named definition."
            },
            "limit": {
                "type": "integer",
                "format": "uint",
                "minimum": 1,
                "maximum": 50,
                "default": 10,
                "description": "Maximum number of symbol hits to return."
            },
            "pathPrefix": nullable_string_schema("Optional repo-relative path prefix filter."),
            "language": nullable_string_schema("Optional normalized language filter."),
            "kind": nullable_string_schema("Optional normalized symbol kind filter such as `function`, `class`, `struct`, or `module`."),
            "container": nullable_string_schema("Optional container/module/class filter.")
        },
        "required": ["query"]
    })
    .as_object()
    .cloned()
    .unwrap_or_default()
}

fn get_file_outline_schema() -> Map<String, Value> {
    json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "FileOutlineArgs",
        "type": "object",
        "description": "Return the indexed symbol outline for one repo-relative file. Use this after `search_symbols` or `search_code` when you already know the file and want indexed structure. If the requested group contains multiple repos with the same file path, each matching outline is returned.",
        "properties": {
            "scope": nullable_string_schema("Configured group id or configured absolute repo root. Preferred over `path`. If omitted together with `path`, the configured default group is searched."),
            "path": nullable_string_schema("Compatibility alias for `scope`. Accepts a configured absolute repo root or configured group id. Prefer `scope` in new clients."),
            "file": {
                "type": "string",
                "description": "Repo-relative file path, such as `src/lib.rs` or `server/crates/api/src/schema.rs`."
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
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "ScopeArgs",
        "type": "object",
        "properties": {
            "scope": nullable_string_schema(scope_description),
            "path": nullable_string_schema("Compatibility alias for `scope`. Accepts a configured absolute repo root or configured group id. Prefer `scope` in new clients.")
        }
    })
    .as_object()
    .cloned()
    .unwrap_or_default()
}

fn list_scopes_schema() -> Map<String, Value> {
    json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "ListScopesArgs",
        "type": "object",
        "description": "No arguments. Preferred first call for unfamiliar agents. Returns the configured default scope and all configured group scopes so search tools can be targeted without guessing group ids.",
        "properties": {}
    })
    .as_object()
    .cloned()
    .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::{
        SERVER_INSTRUCTIONS, SearchMode, default_limit, enforce_loopback_bind, list_scopes_schema,
        listen_is_loopback, normalize_extension_filter, parse_search_mode, parse_splitter_kind,
        search_symbols_schema, tool_list,
    };
    use crate::engine::splitter::SplitterKind;
    use serde_json::Value;

    #[test]
    fn search_limit_default_matches_node_contract() {
        assert_eq!(default_limit(), 10);
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

        assert!(search_code.contains("prefer `search_symbols` first"));
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

        assert_eq!(schema["properties"]["limit"]["default"].as_u64(), Some(10));
        assert_eq!(schema["properties"]["limit"]["maximum"].as_u64(), Some(50));
        assert_eq!(
            schema["properties"]["mode"]["default"].as_str(),
            Some("auto")
        );
        assert!(
            schema["properties"]["extensionFilter"]["description"]
                .as_str()
                .unwrap_or_default()
                .contains("dotted file extensions")
        );
        assert!(
            schema["description"]
                .as_str()
                .unwrap_or_default()
                .contains("prefer `search_symbols` first")
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
                .contains("Preferred tool for exact symbol")
        );
        assert!(
            search_symbols["properties"]["query"]["description"]
                .as_str()
                .unwrap_or_default()
                .contains("Use this when the user asks for a function")
        );
        assert!(
            list_scopes["description"]
                .as_str()
                .unwrap_or_default()
                .contains("Preferred first call")
        );
        assert!(SERVER_INSTRUCTIONS.contains("use `search_symbols` first"));
    }
}
