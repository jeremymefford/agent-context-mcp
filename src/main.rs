mod commands;
mod config;
mod engine;
mod mcp_server;
mod snapshot;

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "agent-context",
    version,
    about = "Rust-native MCP code search server for Milvus-backed local code intelligence"
)]
struct Cli {
    #[arg(long, global = true)]
    config: Option<PathBuf>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Write a starter config for a first repo or workspace
    Init {
        /// Embedding provider: voyage, openai, or ollama
        #[arg(long)]
        provider: Option<String>,
        /// Embedding model name
        #[arg(long)]
        model: Option<String>,
        /// Group id for the starter workspace
        #[arg(long, default_value = "workspace")]
        group_id: String,
        /// One or more starter repos
        #[arg(long = "repo")]
        repo: Vec<String>,
        /// Overwrite an existing config file
        #[arg(long)]
        force: bool,
    },
    /// Validate config, credentials, backend reachability, and launchd state
    Doctor {
        /// launchd label to inspect
        #[arg(long)]
        label: Option<String>,
        /// Health endpoint bind address to probe
        #[arg(long)]
        listen: Option<String>,
    },
    /// Install the managed post-commit hook into one git repo
    InstallHook { repo: PathBuf },
    /// Install or refresh the macOS launchd service for the MCP server
    InstallLaunchd {
        #[arg(long)]
        label: Option<String>,
        #[arg(long, default_value = "127.0.0.1:8765")]
        listen: String,
        #[arg(long)]
        workdir: Option<PathBuf>,
    },
    /// Remove the macOS launchd service for the MCP server
    UninstallLaunchd {
        #[arg(long)]
        label: Option<String>,
    },
    /// Print client-ready MCP config for a supported editor or agent
    PrintMcpConfig {
        #[arg(long)]
        client: String,
        #[arg(long, default_value = "http://127.0.0.1:8765/mcp")]
        url: String,
    },
    /// Incrementally refresh a single codebase or group
    RefreshOne {
        /// Absolute path to a repo or configured group id
        path: String,
    },
    /// Incrementally refresh every configured codebase
    RefreshAll,
    /// Clear and fully reindex every configured codebase
    ReindexAll,
    /// Ad-hoc search in a repo or configured group
    Search {
        /// Absolute path to a repo or configured group id
        path: String,
        /// Natural-language query
        query: String,
        /// Max results
        #[arg(long, default_value_t = 5)]
        limit: usize,
    },
    /// Print the native MCP tool list
    ListTools,
    /// Serve a shared local HTTP MCP endpoint
    Serve {
        /// Bind address for the local MCP endpoint
        #[arg(long, default_value = "127.0.0.1:8765")]
        listen: String,
    },
}

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Init {
            provider,
            model,
            group_id,
            repo,
            force,
        } => {
            commands::init::run(
                cli.config.as_deref(),
                provider.as_deref(),
                model.as_deref(),
                &group_id,
                &repo,
                force,
            )
            .await
        }
        Command::Doctor { label, listen } => {
            let cfg = load_config(cli.config.as_deref())?;
            commands::doctor::run(&cfg, label.as_deref(), listen.as_deref()).await
        }
        Command::InstallHook { repo } => commands::install_hook::run(&repo).await,
        Command::InstallLaunchd {
            label,
            listen,
            workdir,
        } => {
            commands::install_launchd::run(
                label.as_deref(),
                &listen,
                workdir.as_deref(),
                cli.config.as_deref(),
            )
            .await
        }
        Command::UninstallLaunchd { label } => {
            commands::uninstall_launchd::run(label.as_deref()).await
        }
        Command::PrintMcpConfig { client, url } => {
            commands::print_mcp_config::run(&client, &url).await
        }
        Command::RefreshOne { path } => {
            let cfg = load_config(cli.config.as_deref())?;
            commands::refresh_one::run(&cfg, &path).await
        }
        Command::RefreshAll => {
            let cfg = load_config(cli.config.as_deref())?;
            commands::refresh_all::run(&cfg).await
        }
        Command::ReindexAll => {
            let cfg = load_config(cli.config.as_deref())?;
            commands::reindex_all::run(&cfg).await
        }
        Command::Search { path, query, limit } => {
            let cfg = load_config(cli.config.as_deref())?;
            commands::search::run(&cfg, &path, &query, limit).await
        }
        Command::ListTools => commands::list_tools::run().await,
        Command::Serve { listen } => {
            let cfg = load_config(cli.config.as_deref())?;
            commands::serve::run(&cfg, &listen).await
        }
    }
}

fn load_config(path: Option<&std::path::Path>) -> Result<config::Config> {
    match path {
        Some(path) => config::Config::load_from_path(path),
        None => config::Config::load(),
    }
}
