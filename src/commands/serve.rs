use crate::{config::Config, mcp_server};
use anyhow::Result;

pub async fn run(config: &Config, listen: &str) -> Result<()> {
    mcp_server::serve(config, listen).await
}
