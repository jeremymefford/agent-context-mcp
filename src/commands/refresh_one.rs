use crate::commands::support::DEFAULT_LISTEN;
use crate::config::{ResolvedScope, ScopeKind};
use anyhow::{Context, Result, bail};
use reqwest::StatusCode;
use serde::Deserialize;
use serde_json::json;
use std::time::Duration;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct EnqueueResponse {
    #[serde(rename = "scope")]
    _scope: String,
    label: String,
    started: bool,
    force: bool,
    queued_repos: Vec<String>,
    merged_repos: Vec<String>,
    already_running: Vec<String>,
}

pub async fn run(
    path: &str,
    force: bool,
    listen: Option<&str>,
    scope: Option<&ResolvedScope>,
) -> Result<()> {
    let listen = listen.unwrap_or(DEFAULT_LISTEN);
    let url = format!("http://{listen}/enqueue-refresh");
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .context("building refresh-one HTTP client")?;
    let response = client
        .post(&url)
        .json(&build_enqueue_request(path, force, scope))
        .send()
        .await
        .with_context(|| format!("POST {url}"))?;

    let status = response.status();
    if status != StatusCode::OK {
        let body = response.text().await.unwrap_or_default();
        bail!(
            "refresh-one requires the local agent-context service at http://{listen}; enqueue failed with {}{}",
            status,
            if body.is_empty() {
                String::new()
            } else {
                format!(": {body}")
            }
        );
    }

    let result: EnqueueResponse = response
        .json()
        .await
        .context("decoding enqueue-refresh response")?;

    println!("Scope: {}", result.label);
    println!("Worker started: {}", result.started);
    println!("Force: {}", result.force);
    if !result.queued_repos.is_empty() {
        println!("Queued: {}", result.queued_repos.join(", "));
    }
    if !result.merged_repos.is_empty() {
        println!("Merged: {}", result.merged_repos.join(", "));
    }
    if !result.already_running.is_empty() {
        println!("Already running: {}", result.already_running.join(", "));
    }
    if result.queued_repos.is_empty()
        && result.merged_repos.is_empty()
        && result.already_running.is_empty()
    {
        println!("No repos were queued.");
    }
    Ok(())
}

pub fn build_enqueue_request(
    path: &str,
    force: bool,
    scope: Option<&ResolvedScope>,
) -> serde_json::Value {
    let mut payload = json!({
        "path": path,
        "force": force,
    });
    if let Some(scope) = scope {
        payload["scope"] = json!(scope.id);
        payload["label"] = json!(scope.label);
        payload["kind"] = json!(match scope.kind {
            ScopeKind::Repo => "repo",
            ScopeKind::Group => "group",
        });
        payload["repos"] = json!(
            scope
                .repos
                .iter()
                .map(|repo| repo.display().to_string())
                .collect::<Vec<_>>()
        );
    }
    payload
}
