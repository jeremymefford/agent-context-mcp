use crate::commands::support::{self, DEFAULT_LAUNCHD_LABEL, DEFAULT_LISTEN};
use crate::config::Config;
use crate::engine::Engine;
use anyhow::{Context, Result, bail};
use reqwest::StatusCode;
use std::path::Path;
use std::process::Command;
use std::time::Duration;

pub async fn run(config: &Config, label: Option<&str>, listen: Option<&str>) -> Result<()> {
    let mut failures = 0usize;
    let listen = listen.unwrap_or(DEFAULT_LISTEN);
    let label = label
        .map(ToOwned::to_owned)
        .or_else(find_installed_label)
        .unwrap_or_else(|| DEFAULT_LAUNCHD_LABEL.to_string());

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

    let launchd_state = launchd_state(&label)?;
    match launchd_state {
        Some("running") => print_check("launchd", true, label.clone()),
        Some("installed") => print_check("launchd", true, format!("{label} (installed)")),
        _ => print_warning(
            "launchd",
            format!(
                "{} is not installed; run `agent-context install-launchd`",
                label
            ),
        ),
    }

    if launchd_state.is_some() {
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

fn find_installed_label() -> Option<String> {
    for label in support::existing_launchd_labels() {
        if launchd_state(label).ok().flatten().is_some() {
            return Some(label.to_string());
        }
    }
    None
}

fn launchd_state(label: &str) -> Result<Option<&'static str>> {
    let target = format!("gui/{}/{}", unsafe { libc::geteuid() }, label);
    let output = Command::new("launchctl")
        .args(["print", &target])
        .output()
        .context("running launchctl print")?;
    if output.status.success() {
        return Ok(Some("running"));
    }

    let plist_path = support::default_plist_path(label)?;
    if plist_path.exists() {
        return Ok(Some("installed"));
    }
    Ok(None)
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
