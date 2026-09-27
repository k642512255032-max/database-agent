#!/usr/bin/env bash
# Start `ollama serve` tuned for the pipeline on a Colab GPU, unless it is already running.
# Same server settings as deploy/windows/ollama_env.ps1 and all_in_one.py:
#   FLASH_ATTENTION + q8_0 KV cache: faster prompt reading and half the cache memory, so two 7B models fit a T4;
#   NUM_PARALLEL=4: prompt-cache slots per model (the agents send different system prompts in turn);
#   KEEP_ALIVE=-1: models are never unloaded between questions.
set -euo pipefail
LOG_DIR="${LOG_DIR:-/tmp/agent-logs}"
mkdir -p "$LOG_DIR"
export PATH="/usr/local/bin:$PATH"
if curl -fs http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  exit 0
fi
OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 OLLAMA_NUM_PARALLEL=4 OLLAMA_KEEP_ALIVE=-1   OLLAMA_MAX_LOADED_MODELS=3 setsid nohup ollama serve > "$LOG_DIR/ollama.log" 2>&1 < /dev/null &
for i in $(seq 1 30); do curl -fs http://127.0.0.1:11434/api/tags >/dev/null 2>&1 && break; sleep 1; done
