use crate::config::Config;
use crate::engine::Engine;
use anyhow::Result;

pub async fn run(config: &Config, apply: bool) -> Result<()> {
    let engine = Engine::new(config).await?;
    let report = engine.vector_hygiene_report().await?;

    if report.stale_collections.is_empty() {
        println!("no stale agent-context vector collections found");
        return Ok(());
    }

    println!(
        "found {} stale agent-context vector collection(s):",
        report.stale_collections.len()
    );
    for collection in &report.stale_collections {
        println!("- {collection}");
    }

    if !apply {
        println!("dry run only; rerun with --apply to drop these stale Milvus collections");
        return Ok(());
    }

    let dropped = engine.prune_stale_vector_collections().await?;
    println!("dropped {} stale vector collection(s)", dropped.len());
    Ok(())
}
