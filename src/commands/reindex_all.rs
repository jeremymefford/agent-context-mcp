use crate::{
    config::Config,
    engine::splitter::SplitterKind,
    engine::{Engine, render_index_text},
};
use anyhow::Result;

pub async fn run(config: &Config) -> Result<()> {
    let engine = Engine::new(config).await?;
    let result = engine
        .index_scope(engine.all_scope()?, true, SplitterKind::Ast, &[], &[])
        .await?;
    println!("{}", render_index_text(&result));
    if result.has_errors {
        std::process::exit(1);
    }
    Ok(())
}
