#!/usr/bin/env bash
# One-shot setup of the whole stack on a free GPU notebook VM (Google Colab, Kaggle):
#   MySQL 8 + the "employees" sample database + feature view/table + trained models,
#   Ollama on the GPU with the SQL model and the thinking model,
#   the app's Python dependencies and .env.
# Idempotent: re-running skips what is already done. Takes ~8-12 minutes the first time
# (most of it downloading ~10 GB of models).
set -euo pipefail

APP_DIR="${APP_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
MYSQL_ROOT_PW="${MYSQL_ROOT_PW:-rootpw}"
RO_USER="${RO_USER:-agent_ro}"
RO_PW="${RO_PW:-agent_ro_pw}"
SQL_MODEL="${SQL_MODEL:-qwen2.5-coder:7b}"
THINK_MODEL="${THINK_MODEL:-qwen3:8b}"
TEST_DB_DIR="${TEST_DB_DIR:-/tmp/test_db}"
LOG_DIR="${LOG_DIR:-/tmp/agent-logs}"
mkdir -p "$LOG_DIR"

say() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }

# ------------------------------------------------------------------ 1. MySQL
say "MySQL server"
if ! command -v mysqld >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq >/dev/null
  apt-get install -y -qq mysql-server >/dev/null
fi
if ! mysqladmin ping --silent 2>/dev/null; then
  service mysql start >/dev/null 2>&1 || (mysqld_safe > "$LOG_DIR/mysqld.log" 2>&1 &)
  for i in $(seq 1 30); do mysqladmin ping --silent 2>/dev/null && break; sleep 1; done
fi
# Ubuntu's root uses auth_socket; switch to a password so SQLAlchemy can connect over TCP
mysql -uroot -e "ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY '$MYSQL_ROOT_PW';" 2>/dev/null \
  || mysql -uroot -p"$MYSQL_ROOT_PW" -e "SELECT 1" >/dev/null
MYSQL="mysql -uroot -p$MYSQL_ROOT_PW"

# ------------------------------------------------------------------ 2. employees sample database
if $MYSQL -e "USE employees" 2>/dev/null && [ "$($MYSQL -N -e 'SELECT COUNT(*) FROM employees.employees')" -gt 0 ]; then
  say "employees database already loaded"
else
  say "Loading the employees sample database (datacharmer/test_db, ~1-2 min)"
  [ -d "$TEST_DB_DIR" ] || git clone -q --depth 1 https://github.com/datacharmer/test_db "$TEST_DB_DIR"
  (cd "$TEST_DB_DIR" && $MYSQL < employees.sql > "$LOG_DIR/load_employees.log" 2>&1)
fi
$MYSQL -e "CREATE USER IF NOT EXISTS '$RO_USER'@'%' IDENTIFIED BY '$RO_PW';
           GRANT SELECT, SHOW VIEW ON employees.* TO '$RO_USER'@'%'; FLUSH PRIVILEGES;"

say "Feature view + materialised table (employee_features)"
if [ -z "$($MYSQL -N -e "SELECT 1 FROM information_schema.tables WHERE table_schema='employees' AND table_name='employee_features'")" ]; then
  $MYSQL employees < "$APP_DIR/sample_data/employees_features.sql"
else
  echo "employee_features already exists - skipping view creation"
fi

# ------------------------------------------------------------------ 3. Python deps + .env
say "Python dependencies"
pip install -q -r "$APP_DIR/requirements.txt" 2>&1 | tail -1 || true
cat > "$APP_DIR/.env" <<EOF
DATABASE_URL=mysql+pymysql://$RO_USER:$RO_PW@127.0.0.1:3306/employees
OLLAMA_HOST=http://127.0.0.1:11434
OLLAMA_MODEL=$SQL_MODEL
OLLAMA_ANSWER_MODEL=$THINK_MODEL
OLLAMA_EXPERT_MODEL=$THINK_MODEL
LLM_NUM_CTX=8192
LLM_TEMPERATURE=0
MAX_ROWS=1000
EOF
(cd "$APP_DIR" && python sample_data/materialise_employee_features.py \
   --admin-url "mysql+pymysql://root:$MYSQL_ROOT_PW@127.0.0.1:3306/employees" > "$LOG_DIR/materialise.log" 2>&1) \
  && echo "employee_features materialised" || echo "materialise skipped (see $LOG_DIR/materialise.log); the view still works"

# ------------------------------------------------------------------ 4. Ollama on the GPU
say "Ollama"
command -v ollama >/dev/null 2>&1 || curl -fsSL https://ollama.com/install.sh | sh > "$LOG_DIR/ollama-install.log" 2>&1
if ! curl -fs http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  setsid nohup ollama serve > "$LOG_DIR/ollama.log" 2>&1 < /dev/null &
  for i in $(seq 1 30); do curl -fs http://127.0.0.1:11434/api/tags >/dev/null 2>&1 && break; sleep 1; done
fi
for m in "$SQL_MODEL" "$THINK_MODEL"; do
  ollama list 2>/dev/null | grep -q "^$m" || { say "Pulling $m"; ollama pull "$m"; }
done
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "(no NVIDIA GPU visible - Ollama will run on CPU, slowly)"

# ------------------------------------------------------------------ 5. trained models
say "Training the ML models (employees_training_config.yaml)"
if [ -f "$APP_DIR/models/manifest.json" ] && [ "$(ls "$APP_DIR"/models/*.joblib 2>/dev/null | wc -l)" -gt 0 ]; then
  echo "models already trained"
else
  (cd "$APP_DIR" && python train_models.py --config employees_training_config.yaml \
     --database-url "mysql+pymysql://root:$MYSQL_ROOT_PW@127.0.0.1:3306/employees" > "$LOG_DIR/train.log" 2>&1) \
    && tail -3 "$LOG_DIR/train.log" || echo "training failed (see $LOG_DIR/train.log); data queries still work"
fi

# ------------------------------------------------------------------ 6. tunnel binary
say "cloudflared (public tunnel, no account needed)"
if ! command -v cloudflared >/dev/null 2>&1; then
  curl -fsSL -o /tmp/cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
  dpkg -i /tmp/cloudflared.deb > /dev/null
fi

say "Ready. Start the app with:  bash $APP_DIR/deploy/colab/serve.sh"
