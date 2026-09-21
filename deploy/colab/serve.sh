#!/usr/bin/env bash
# Start the Streamlit app and open a public Cloudflare quick tunnel to it; print the shareable URL.
# The URL is random (https://<words>.trycloudflare.com), needs no account, and lives as long as this VM does.
set -euo pipefail

APP_DIR="${APP_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
PORT="${PORT:-8501}"
LOG_DIR="${LOG_DIR:-/tmp/agent-logs}"
mkdir -p "$LOG_DIR"

# Ollama must be up (bootstrap starts it; a notebook restart may have killed it)
if ! curl -fs http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  nohup ollama serve > "$LOG_DIR/ollama.log" 2>&1 &
  for i in $(seq 1 30); do curl -fs http://127.0.0.1:11434/api/tags >/dev/null 2>&1 && break; sleep 1; done
fi
mysqladmin ping --silent 2>/dev/null || service mysql start >/dev/null 2>&1 || true

pkill -f "streamlit run app.py" 2>/dev/null || true
pkill -f "cloudflared tunnel" 2>/dev/null || true

cd "$APP_DIR"
nohup python -m streamlit run app.py --server.headless true --server.port "$PORT" --server.address 0.0.0.0 \
  --browser.gatherUsageStats false > "$LOG_DIR/streamlit.log" 2>&1 &
for i in $(seq 1 60); do curl -fs "http://127.0.0.1:$PORT/_stcore/health" >/dev/null 2>&1 && break; sleep 1; done

nohup cloudflared tunnel --url "http://127.0.0.1:$PORT" --no-autoupdate > "$LOG_DIR/cloudflared.log" 2>&1 &
URL=""
for i in $(seq 1 60); do
  URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOG_DIR/cloudflared.log" | head -1 || true)
  [ -n "$URL" ] && break
  sleep 1
done

echo
echo "======================================================================"
if [ -n "$URL" ]; then
  echo "  Share this link:  $URL"
  echo "  (first answer is slow while the models load into the GPU; logs in $LOG_DIR)"
else
  echo "  Tunnel URL not found yet - check $LOG_DIR/cloudflared.log"
fi
echo "======================================================================"
