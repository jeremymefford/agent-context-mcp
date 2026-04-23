class AgentContext < Formula
  desc "Rust-native MCP code search server for Milvus-backed local code intelligence"
  homepage "https://github.com/jeremymefford/agent-context-mcp"
  url "__DARWIN_ARM64_URL__"
  sha256 "__DARWIN_ARM64_SHA256__"
  license "GPL-3.0-only"
  depends_on arch: :arm64

  def install
    bin.install "agent-context"
  end

  service do
    run [opt_bin/"agent-context", "serve", "--listen", "127.0.0.1:8765"]
    keep_alive true
    working_dir var/"agent-context"
    log_path var/"log/agent-context.log"
    error_log_path var/"log/agent-context.err.log"
  end

  def caveats
    <<~EOS
      agent-context expects:

        1. a running Milvus instance
        2. a config file, usually created with `agent-context init`
        3. an embedding provider configured via env vars or Ollama

      Preferred local setup:

        brew services start agent-context

      Health check:

        curl http://127.0.0.1:8765/health

      See the project README for the macOS quickstart.
    EOS
  end
end
