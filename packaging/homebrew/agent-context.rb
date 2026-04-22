class AgentContext < Formula
  desc "Rust-native MCP code search server for Milvus-backed local code intelligence"
  homepage "https://github.com/NZBMan/claude-context-rust-http-mcp"
  url "https://github.com/NZBMan/claude-context-rust-http-mcp/releases/download/v0.1.0/agent-context-darwin-arm64.tar.gz"
  sha256 "CHANGE_ME"
  license "Apache-2.0"

  def install
    bin.install "agent-context"
  end

  def caveats
    <<~EOS
      agent-context expects:

        1. a running Milvus instance
        2. a config file, usually created with `agent-context init`
        3. an embedding provider configured via env vars or Ollama

      See the project README for the macOS quickstart.
    EOS
  end
end
