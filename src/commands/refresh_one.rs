use crate::{
    config::Config,
    engine::splitter::SplitterKind,
    engine::{Engine, render_index_text},
};
use anyhow::Result;

pub async fn run(config: &Config, path: &str, force: bool) -> Result<()> {
    let engine = Engine::new(config).await?;
    let scope = config.resolve_scope(None, Some(path))?;
    let result = engine
        .index_scope(scope, force, SplitterKind::Ast, &[], &[])
        .await?;
    println!("{}", render_index_text(&result));
    if result.has_errors {
        std::process::exit(1);
    }
    Ok(())
}
