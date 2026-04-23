class AgentContext < Formula
  desc "Rust-native MCP code search server for Milvus-backed local code intelligence"
  homepage "https://github.com/jeremymefford/agent-context-mcp"
  url "__DARWIN_ARM64_URL__"
  sha256 "__DARWIN_ARM64_SHA256__"
  license "GPL-3.0-only"

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
