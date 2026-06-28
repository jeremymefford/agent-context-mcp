use crate::config::Config;
use crate::engine::Engine;
use anyhow::{Context, Result};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

pub async fn run(config: &Config, apply: bool) -> Result<()> {
    let paths = reset_paths(config);
    let vector_collections = match Engine::new(config).await {
        Ok(engine) => match engine.vector_hygiene_report().await {
            Ok(report) => report.agent_context_collections,
            Err(error) => {
                eprintln!("[warn] unable to inspect agent-context vector collections: {error}");
                Vec::new()
            }
        },
        Err(error) => {
            eprintln!("[warn] unable to initialize engine for vector cleanup: {error}");
            Vec::new()
        }
    };

    println!("Local state paths:");
    for path in &paths {
        println!("- {}", path.display());
    }

    if vector_collections.is_empty() {
        println!("Agent-context vector collections: none found");
    } else {
        println!(
            "Agent-context vector collections ({}):",
            vector_collections.len()
        );
        for collection in &vector_collections {
            println!("- {collection}");
        }
    }

    if !apply {
        println!("dry run only; rerun with --apply to remove this local state");
        return Ok(());
    }

    for path in &paths {
        remove_path_if_present(path)?;
    }

    if !vector_collections.is_empty() {
        match Engine::new(config).await {
            Ok(engine) => {
                let dropped = engine
                    .drop_vector_collections(&vector_collections)
                    .await
                    .context("dropping agent-context vector collections")?;
                println!(
                    "Dropped {} agent-context vector collection(s)",
                    dropped.len()
                );
            }
            Err(error) => {
                eprintln!(
                    "[warn] local state was removed, but vector collections were not dropped: {error}"
                );
            }
        }
    }

    println!(
        "Removed local agent-context state for {}",
        config.config_path.display()
    );
    Ok(())
}

fn reset_paths(config: &Config) -> Vec<PathBuf> {
    let ordered = vec![
        config.snapshot_path.clone(),
        config.index_root.clone(),
        config.merkle_dir(),
    ];

    let mut seen = BTreeSet::new();
    ordered
        .into_iter()
        .filter(|path| seen.insert(path.clone()))
        .collect()
}

fn remove_path_if_present(path: &Path) -> Result<()> {
    if !path.exists() {
        return Ok(());
    }

    let metadata = std::fs::symlink_metadata(path)
        .with_context(|| format!("reading metadata for {}", path.display()))?;
    if metadata.file_type().is_dir() {
        std::fs::remove_dir_all(path)
            .with_context(|| format!("removing directory {}", path.display()))?;
    } else {
        std::fs::remove_file(path).with_context(|| format!("removing file {}", path.display()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::reset_paths;
    use crate::config::{
        Config, EmbeddingConfig, EmbeddingProfileConfig, EmbeddingProvider, FreshnessConfig,
        GroupConfig, MilvusConfig, OllamaProviderConfig, OpenAiProviderConfig, SearchConfig,
        VoyageProviderConfig, WorktreeConfig,
    };
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    fn sample_config(index_root: PathBuf, merkle_dir: PathBuf) -> Config {
        Config {
            config_path: PathBuf::from("/tmp/config.toml"),
            embedding: EmbeddingConfig {
                default_profile: "default".to_string(),
                profiles: BTreeMap::from([(
                    "default".to_string(),
                    EmbeddingProfileConfig {
                        provider: EmbeddingProvider::Ollama,
                        model: "embeddinggemma".to_string(),
                        voyage: VoyageProviderConfig::default(),
                        openai: OpenAiProviderConfig::default(),
                        ollama: OllamaProviderConfig {
                            base_url: "http://127.0.0.1:11434".to_string(),
                        },
                    },
                )]),
                assignments: BTreeMap::new(),
            },
            milvus: MilvusConfig {
                address: "127.0.0.1:19530".to_string(),
                token_env: None,
                token: None,
                username: None,
                password: None,
                database: String::new(),
            },
            snapshot_path: PathBuf::from("/tmp/state/snapshot.json"),
            index_root,
            merkle_dir,
            default_group: "workspace".to_string(),
            groups: vec![GroupConfig {
                id: "workspace".to_string(),
                label: Some("Workspace".to_string()),
                repos: vec!["/tmp/repo".to_string()],
            }],
            freshness: FreshnessConfig::default(),
            search: SearchConfig::default(),
            worktrees: WorktreeConfig::default(),
            worktree_canonicals: BTreeMap::new(),
        }
    }

    #[test]
    fn reset_paths_dedupes_identical_entries() {
        let config = sample_config(
            PathBuf::from("/tmp/index-v1"),
            PathBuf::from("/tmp/index-v1"),
        );

        let paths = reset_paths(&config);

        assert_eq!(paths.len(), 2);
        assert_eq!(paths[0], PathBuf::from("/tmp/state/snapshot.json"));
        assert_eq!(paths[1], PathBuf::from("/tmp/index-v1"));
    }
}
