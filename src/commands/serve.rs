use crate::{config::Config, mcp_server};
use anyhow::Result;

pub async fn run(config: &Config, listen: &str, allow_remote_unauthenticated: bool) -> Result<()> {
    mcp_server::serve(config, listen, allow_remote_unauthenticated).await
}
