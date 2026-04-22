use crate::{
    config::Config,
    engine::{Engine, SearchMode, SearchRequest, render_search_text},
};
use anyhow::Result;

pub async fn run(config: &Config, path: &str, query: &str, limit: usize) -> Result<()> {
    let engine = Engine::new(config).await?;
    let scope = config.resolve_scope(None, Some(path))?;
    let result = engine
        .search_scope(
            scope,
            SearchRequest {
                query: query.to_string(),
                limit,
                mode: SearchMode::Auto,
                extension_filter: Vec::new(),
                path_prefix: None,
                language: None,
                file: None,
                dedupe_by_file: true,
            },
        )
        .await?;
    println!("{}", render_search_text(&result));
    Ok(())
}
