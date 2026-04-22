use anyhow::{Result, bail};

pub async fn run(client: &str, url: &str) -> Result<()> {
    let snippet = match client {
        "codex" => format!("[mcp_servers.agent-context]\nurl = \"{url}\"\n"),
        "claude" => format!(
            "{{\n  \"mcpServers\": {{\n    \"agent-context\": {{\n      \"type\": \"http\",\n      \"url\": \"{url}\"\n    }}\n  }}\n}}\n"
        ),
        "copilot" => format!(
            "{{\n  \"servers\": {{\n    \"agent-context\": {{\n      \"type\": \"http\",\n      \"url\": \"{url}\"\n    }}\n  }}\n}}\n"
        ),
        other => bail!("unsupported client `{other}`; expected codex, claude, or copilot"),
    };
    println!("{snippet}");
    Ok(())
}
