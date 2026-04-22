use crate::mcp_server::tool_list;
use anyhow::Result;
use rmcp::model::ListToolsResult;

pub async fn run() -> Result<()> {
    let tools = ListToolsResult {
        tools: tool_list(),
        next_cursor: None,
        meta: None,
    };
    println!("{}", serde_json::to_string_pretty(&tools)?);
    Ok(())
}
