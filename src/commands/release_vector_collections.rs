use crate::config::Config;
use crate::engine::Engine;
use anyhow::Result;

pub async fn run(config: &Config) -> Result<()> {
    let engine = Engine::new(config).await?;
    let released = engine.release_loaded_vector_collections().await?;

    if released.is_empty() {
        println!("no loaded agent-context vector collections found");
        return Ok(());
    }

    println!(
        "released {} loaded agent-context vector collection(s)",
        released.len()
    );
    for collection in released {
        println!("- {collection}");
    }
    Ok(())
}
