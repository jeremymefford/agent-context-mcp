use crate::commands::support::DEFAULT_LISTEN;
use crate::config::{Config, ResolvedScope, ScopeKind};
use crate::engine::{Engine, RepoStatus, StatusReport};
use anyhow::{Context, Result, bail};
use reqwest::StatusCode;
use std::path::Path;
use std::process::Command;
use std::time::Duration;

pub async fn run(config: &Config, listen: Option<&str>) -> Result<()> {
    let mut failures = 0usize;
    let listen = listen.unwrap_or(DEFAULT_LISTEN);

    print_check("config", true, config.config_path.display().to_string());

    match config.embedding.api_key() {
        Ok(Some(_)) => print_check(
            "embedding credentials",
            true,
            format!(
                "provider={} model={}",
                config.embedding.provider_name(),
                config.embedding.model
            ),
        ),
        Ok(None) => print_check(
            "embedding credentials",
            true,
            format!(
                "provider={} model={}",
                config.embedding.provider_name(),
                config.embedding.model
            ),
        ),
        Err(error) => {
            failures += 1;
            print_check("embedding credentials", false, error.to_string());
        }
    }

    for path in [
        config.index_root.clone(),
        config.merkle_dir(),
        config
            .snapshot_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf(),
    ] {
        match std::fs::create_dir_all(&path) {
            Ok(()) => print_check("filesystem", true, path.display().to_string()),
            Err(error) => {
                failures += 1;
                print_check("filesystem", false, format!("{}: {error}", path.display()));
            }
        }
    }

    print_check(
        "search budgets",
        true,
        format!(
            "requests={} repoSearches={} lexical={} dense={} warmRepos={}",
            config.search.max_concurrent_requests,
            config.search.max_concurrent_repo_searches,
            config.search.max_concurrent_lexical_tasks,
            config.search.max_concurrent_dense_tasks,
            config.search.max_warm_repos
        ),
    );

    let engine = match Engine::new(config).await {
        Ok(engine) => engine,
        Err(error) => {
            failures += 1;
            print_check("engine construction", false, error.to_string());
            bail!("doctor failed with {failures} issue(s)");
        }
    };

    match engine.healthcheck().await {
        Ok(()) => print_check("backend health", true, config.milvus.address.clone()),
        Err(error) => {
            failures += 1;
            print_check("backend health", false, error.to_string());
        }
    }

    match engine.index_identity_status().await {
        Ok(status) if status.compatible => print_check(
            "index identity",
            true,
            format!(
                "format={} searchRoot={} embedding={}",
                status.index_format_version,
                status.search_root_version,
                status.configured_embedding_fingerprint
            ),
        ),
        Ok(status) => {
            failures += 1;
            print_check(
                "index identity",
                false,
                status
                    .reason
                    .unwrap_or_else(|| "index identity mismatch".to_string()),
            );
        }
        Err(error) => {
            failures += 1;
            print_check("index identity", false, error.to_string());
        }
    }

    match index_status_report(config, &engine).await {
        Ok(report) => {
            let failed_repos = failed_index_repos(&report);
            let indexing_repos = indexing_repos(&report);
            let not_indexed_repos = not_indexed_repos(&report);
            if failed_repos.is_empty() {
                print_check(
                    "index coverage",
                    true,
                    format!(
                        "status={} indexedRepos={} files={} chunks={}",
                        report.overall_status,
                        report
                            .repos
                            .iter()
                            .filter(|repo| repo.status == "indexed")
                            .count(),
                        report.indexed_files,
                        report.total_chunks
                    ),
                );
            } else {
                failures += 1;
                print_check(
                    "index coverage",
                    false,
                    format_failed_index_repos(&failed_repos),
                );
            }

            if !indexing_repos.is_empty() {
                print_warning(
                    "indexing in progress",
                    format!(
                        "{} repo(s) currently indexing: {}",
                        indexing_repos.len(),
                        format_repo_status_list(&indexing_repos)
                    ),
                );
            }

            if !not_indexed_repos.is_empty() {
                print_warning(
                    "not indexed",
                    format!(
                        "{} configured repo(s) have no completed index yet: {}",
                        not_indexed_repos.len(),
                        format_repo_status_list(&not_indexed_repos)
                    ),
                );
            }
        }
        Err(error) => {
            failures += 1;
            print_check("index coverage", false, error.to_string());
        }
    }

    match engine.vector_hygiene_report().await {
        Ok(report) => {
            print_check(
                "vector collections",
                true,
                format!(
                    "configured={} present={} loaded={}/{}",
                    report.expected_collections,
                    report.agent_context_collections.len(),
                    report.loaded_collections.len(),
                    report.recommended_loaded_collection_limit
                ),
            );
            if !report.stale_collections.is_empty() {
                print_warning(
                    "stale vector collections",
                    format!(
                        "{} unconfigured collection(s): {}",
                        report.stale_collections.len(),
                        format_collection_list(&report.stale_collections)
                    ),
                );
            }
            if report.loaded_collections.len() > report.recommended_loaded_collection_limit {
                print_warning(
                    "loaded vector collections",
                    format!(
                        "{} loaded collection(s) exceeds recommended limit {}; refresh/reindex now releases indexing loads, but stale or broad semantic searches can still keep Milvus warm: {}",
                        report.loaded_collections.len(),
                        report.recommended_loaded_collection_limit,
                        format_collection_list(&report.loaded_collections)
                    ),
                );
                print_warning(
                    "vector release",
                    "run `agent-context release-vector-collections` to unload warm Milvus collections without deleting indexes".to_string(),
                );
            }
        }
        Err(error) => print_warning(
            "vector collections",
            format!("unable to inspect Milvus vector hygiene: {error}"),
        ),
    }

    match homebrew_service_status()? {
        Some(status) if status.status == "started" => print_check(
            "brew service",
            true,
            format!("{} ({})", status.name, status.status),
        ),
        Some(status) if status.status == "none" => print_warning(
            "brew service",
            format!(
                "{} is not running; start it with `brew services start agent-context`",
                status.name
            ),
        ),
        Some(status) => print_warning(
            "brew service",
            format!(
                "{} status={} file={}",
                status.name, status.status, status.file
            ),
        ),
        None => print_warning(
            "brew service",
            "agent-context is not registered with Homebrew services".to_string(),
        ),
    }

    if matches!(homebrew_service_status()?, Some(status) if status.status == "started") {
        match health_endpoint(listen).await {
            Ok(()) => print_check("health endpoint", true, format!("http://{listen}/health")),
            Err(error) => {
                failures += 1;
                print_check("health endpoint", false, error.to_string());
            }
        }
    }

    if failures > 0 {
        bail!("doctor failed with {failures} issue(s)");
    }

    println!("doctor completed with no blocking issues");
    Ok(())
}

#[derive(serde::Deserialize)]
struct BrewServiceEntry {
    name: String,
    status: String,
    file: String,
}

fn homebrew_service_status() -> Result<Option<BrewServiceEntry>> {
    let output = Command::new("brew")
        .args(["services", "list", "--json"])
        .output()
        .context("running `brew services list --json`")?;
    if !output.status.success() {
        bail!("`brew services list --json` failed");
    }
    let services: Vec<BrewServiceEntry> =
        serde_json::from_slice(&output.stdout).context("parsing brew services JSON")?;
    Ok(services
        .into_iter()
        .find(|entry| entry.name == "agent-context"))
}

async fn health_endpoint(listen: &str) -> Result<()> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .context("building doctor HTTP client")?;
    let url = format!("http://{listen}/health");
    let response = client
        .get(&url)
        .send()
        .await
        .with_context(|| format!("GET {url}"))?;
    if response.status() != StatusCode::OK {
        bail!("health endpoint returned {}", response.status());
    }
    Ok(())
}

fn print_check(name: &str, ok: bool, detail: String) {
    let status = if ok { "ok" } else { "fail" };
    println!("[{status}] {name}: {detail}");
}

fn print_warning(name: &str, detail: String) {
    println!("[warn] {name}: {detail}");
}

async fn index_status_report(config: &Config, engine: &Engine) -> Result<StatusReport> {
    let repos = config.all_repos()?;
    engine
        .status_scope(ResolvedScope {
            kind: ScopeKind::Group,
            id: "__all_configured_repos__".to_string(),
            label: "configured repositories".to_string(),
            repos,
        })
        .await
}

fn failed_index_repos(report: &StatusReport) -> Vec<&RepoStatus> {
    report
        .repos
        .iter()
        .filter(|repo| {
            repo.status == "indexfailed"
                || matches!(
                    repo.index_status.as_deref(),
                    Some("failed" | "incomplete" | "limit_reached")
                )
        })
        .collect()
}

fn indexing_repos(report: &StatusReport) -> Vec<&RepoStatus> {
    report
        .repos
        .iter()
        .filter(|repo| repo.status == "indexing")
        .collect()
}

fn not_indexed_repos(report: &StatusReport) -> Vec<&RepoStatus> {
    report
        .repos
        .iter()
        .filter(|repo| repo.status == "not_indexed")
        .collect()
}

fn format_failed_index_repos(repos: &[&RepoStatus]) -> String {
    format!(
        "{} repo(s) have failed indexes: {}; recover with `agent-context refresh-one --force <repo>`",
        repos.len(),
        format_repo_status_list(repos)
    )
}

fn format_repo_status_list(repos: &[&RepoStatus]) -> String {
    const MAX_REPOS: usize = 5;
    let mut names = repos
        .iter()
        .take(MAX_REPOS)
        .map(|repo| {
            let mut detail = format!(
                "{} status={} index_status={}",
                repo.repo_label,
                repo.status,
                repo.index_status.as_deref().unwrap_or("unknown")
            );
            if let Some(progress) = repo.indexing_percentage.or(repo.last_attempted_percentage) {
                detail.push_str(&format!(" progress={progress:.1}%"));
            }
            if let Some(error) = &repo.error_message {
                detail.push_str(&format!(" error={error}"));
            }
            detail.push_str(&format!(" repo={}", repo.repo));
            detail
        })
        .collect::<Vec<_>>();
    if repos.len() > MAX_REPOS {
        names.push(format!("... and {} more", repos.len() - MAX_REPOS));
    }
    names.join("; ")
}

fn format_collection_list(collections: &[String]) -> String {
    const MAX_NAMES: usize = 5;
    let mut names = collections
        .iter()
        .take(MAX_NAMES)
        .cloned()
        .collect::<Vec<_>>();
    if collections.len() > MAX_NAMES {
        names.push(format!("... and {} more", collections.len() - MAX_NAMES));
    }
    names.join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn repo_status(status: &str, index_status: Option<&str>) -> RepoStatus {
        RepoStatus {
            repo: format!("/repo/{status}"),
            repo_label: status.to_string(),
            collection_name: format!("collection_{status}"),
            status: status.to_string(),
            indexed_files: None,
            total_chunks: None,
            index_status: index_status.map(str::to_string),
            indexing_percentage: None,
            last_attempted_percentage: None,
            error_message: None,
        }
    }

    fn status_report(repos: Vec<RepoStatus>) -> StatusReport {
        StatusReport {
            scope: "workspace".to_string(),
            label: "Workspace".to_string(),
            kind: "group".to_string(),
            overall_status: "indexfailed".to_string(),
            indexed_files: 0,
            total_chunks: 0,
            repos,
            identity_error: None,
        }
    }

    #[test]
    fn failed_index_repos_include_failed_status_and_failed_index_status() {
        let failed = repo_status("indexfailed", Some("failed"));
        let indexed_but_failed = repo_status("indexed", Some("failed"));
        let indexed_but_incomplete = repo_status("indexed", Some("incomplete"));
        let indexed_but_limited = repo_status("indexed", Some("limit_reached"));
        let indexed = repo_status("indexed", Some("completed"));
        let report = status_report(vec![
            failed,
            indexed_but_failed,
            indexed_but_incomplete,
            indexed_but_limited,
            indexed,
        ]);

        let repos = failed_index_repos(&report);

        assert_eq!(repos.len(), 4);
        assert_eq!(repos[0].repo_label, "indexfailed");
        assert_eq!(repos[1].repo_label, "indexed");
    }

    #[test]
    fn failed_index_format_points_to_force_refresh() {
        let mut failed = repo_status("indexfailed", Some("failed"));
        failed.last_attempted_percentage = Some(20.93);
        failed.error_message =
            Some("agent-context restarted while indexing was in progress".into());
        let repos = vec![&failed];

        let message = format_failed_index_repos(&repos);

        assert!(message.contains("agent-context refresh-one --force <repo>"));
        assert!(message.contains("progress=20.9%"));
        assert!(message.contains("restarted while indexing"));
    }
}
