use crate::config::{Config, normalize_path};
use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};

pub async fn run(
    config_override: Option<&Path>,
    provider: Option<&str>,
    model: Option<&str>,
    group_id: &str,
    repos: &[String],
    force: bool,
) -> Result<()> {
    let config_path = match config_override {
        Some(path) => path.to_path_buf(),
        None => Config::default_config_path()?,
    };
    if config_path.exists() && !force {
        bail!(
            "config already exists at {}; pass --force to overwrite it",
            config_path.display()
        );
    }

    let provider = provider.unwrap_or("voyage").trim().to_ascii_lowercase();
    let default_model = match provider.as_str() {
        "voyage" => "voyage-code-3",
        "openai" => "text-embedding-3-small",
        "ollama" => "embeddinggemma",
        other => bail!("unsupported provider `{other}`; expected voyage, openai, or ollama"),
    };

    let repo_list = if repos.is_empty() {
        vec![
            std::env::current_dir()
                .context("reading current directory")?
                .display()
                .to_string(),
        ]
    } else {
        repos.to_vec()
    };
    let normalized_repos = repo_list
        .iter()
        .map(|repo| normalize_path(repo))
        .collect::<Result<Vec<_>>>()?;

    if let Some(parent) = config_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }

    let index_root = config_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("index-v1");
    let snapshot_path = config_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("state/snapshot.json");

    let body = render_config(
        &provider,
        model.unwrap_or(default_model),
        group_id,
        &normalized_repos,
        &snapshot_path,
        &index_root,
    );
    std::fs::write(&config_path, body)
        .with_context(|| format!("writing {}", config_path.display()))?;

    println!("Wrote {}", config_path.display());
    println!("Next steps:");
    println!("1. Start Milvus from `docker compose -f docker/milvus-compose.yml up -d`");
    println!("2. Export the provider API key env var if needed");
    println!(
        "3. Run `agent-context doctor{}`",
        render_config_flag(&config_path)
    );
    println!(
        "4. Preferred: `brew services start agent-context`; fallback: `agent-context serve --listen 127.0.0.1:8765{}`",
        render_config_flag(&config_path)
    );
    Ok(())
}

fn render_config(
    provider: &str,
    model: &str,
    group_id: &str,
    repos: &[PathBuf],
    snapshot_path: &Path,
    index_root: &Path,
) -> String {
    let embedding_block = match provider {
        "voyage" => r#"[embedding.voyage]
api_key_env = "VOYAGE_API_KEY"
# key_file = "~/Library/Application Support/agent-context/voyage_key"
"#
        .to_string(),
        "openai" => r#"[embedding.openai]
api_key_env = "OPENAI_API_KEY"
base_url = "https://api.openai.com/v1"
"#
        .to_string(),
        "ollama" => r#"[embedding.ollama]
base_url = "http://127.0.0.1:11434"
"#
        .to_string(),
        _ => String::new(),
    };

    let repos_block = repos
        .iter()
        .map(|repo| format!("  \"{}\",", repo.display()))
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        r#"# agent-context config
snapshot_path = "{}"
index_root = "{}"
default_group = "{}"

[embedding]
provider = "{}"
model = "{}"

{}
[milvus]
address = "127.0.0.1:19530"
# token_env = "MILVUS_TOKEN"

[freshness]
# audit_interval_secs = 600

[search]
max_concurrent_requests = 2
max_concurrent_repo_searches = 4
max_concurrent_lexical_tasks = 2
max_concurrent_dense_tasks = 2
max_warm_repos = 4

[[groups]]
id = "{}"
label = "Workspace"
repos = [
{}
]
"#,
        snapshot_path.display(),
        index_root.display(),
        group_id,
        provider,
        model,
        embedding_block,
        group_id,
        repos_block
    )
}

fn render_config_flag(config_path: &Path) -> String {
    format!(" --config \"{}\"", config_path.display())
}

#[cfg(test)]
mod tests {
    use super::render_config;
    use std::path::Path;

    #[test]
    fn starter_config_uses_requested_provider_block() {
        let config = render_config(
            "openai",
            "text-embedding-3-small",
            "workspace",
            &[Path::new("/tmp/repo").to_path_buf()],
            Path::new("/tmp/snapshot.json"),
            Path::new("/tmp/index-v1"),
        );
        assert!(config.contains("provider = \"openai\""));
        assert!(config.contains("[embedding.openai]"));
        assert!(config.contains("/tmp/repo"));
    }
}
