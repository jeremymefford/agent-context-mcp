#!/bin/sh

# >>> agent-context managed block >>>
AGENT_CONTEXT_BIN="${AGENT_CONTEXT_BIN:-$(command -v agent-context 2>/dev/null || printf '%s' "$HOME/.cargo/bin/agent-context")}"
if [ -x "$AGENT_CONTEXT_BIN" ]; then
  nohup "$AGENT_CONTEXT_BIN" refresh-one "$(git rev-parse --show-toplevel)" >> "$HOME/Library/Logs/agent-context-hooks.log" 2>&1 &
fi
# <<< agent-context managed block <<<
