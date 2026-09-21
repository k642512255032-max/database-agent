#@title Local Data Agent on Colab - ALL IN ONE (install + verify + run + public link). Runtime type must be T4 GPU.
"""Everything runs on THIS Colab VM: MySQL + employees DB, Ollama on the GPU, the app, a public tunnel.
Every step prints OK / FAIL; the Ollama check at the end is the same check the app's sidebar does.
Re-running is safe (installed parts are skipped). Paste into one cell, or:  !curl -sL <raw url> | python
"""
import json, os, re, subprocess, sys, time, urllib.request

APP = "/content/database-agent"
LOG = "/tmp/agent-logs"
SQL_MODEL, THINK_MODEL = "qwen2.5-coder:7b", "qwen3:8b"
ROOT_PW, RO_USER, RO_PW = "rootpw", "agent_ro", "agent_ro_pw"
os.makedirs(LOG, exist_ok=True)
os.environ.pop("OLLAMA_HOST", None)                       # never inherit a stray host setting


def sh(cmd, check=False, quiet=True, **kw):
    r = subprocess.run(cmd, shell=True, text=True, capture_output=quiet, **kw)
    if check and r.returncode != 0:
        raise RuntimeError(f"command failed: {cmd}\n{(r.stderr or r.stdout or '')[-1500:]}")
    return (r.stdout or "").strip()


def daemon(cmd, log):
    """Start a background service that survives the end of the cell."""
    return subprocess.Popen(cmd, shell=True, stdout=open(log, "ab"), stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True)


def wait(url, seconds=60):
    for _ in range(seconds):
        try:
            urllib.request.urlopen(url, timeout=3)
            return True
        except Exception:
            time.sleep(1)
    return False


def step(title):
    print(f"\n\033[1;34m==> {title}\033[0m", flush=True)


ok = lambda m: print("  OK  ", m, flush=True)
bad = lambda m: print("  FAIL", m, flush=True)

# ------------------------------------------------------------------ 0. VM + GPU
step("Where am I")
print("  host:", sh("hostname"), "| user:", sh("whoami"), "| this is the Colab VM, not your PC")
gpu = sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
ok(f"GPU {gpu}") if gpu else bad("no GPU - Runtime > Change runtime type > T4 GPU, then run again")

# ------------------------------------------------------------------ 1. code
step("Code")
if not os.path.isdir(APP):
    sh(f"git clone -q https://github.com/k642512255032-max/database-agent.git {APP}", check=True)
sh(f"cd {APP} && git pull -q")
ok(f"{APP} @ {sh(f'cd {APP} && git log --oneline -1')}")

# ------------------------------------------------------------------ 2. MySQL + employees
step("MySQL")
if not sh("which mysqld"):
    sh("DEBIAN_FRONTEND=noninteractive apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq mysql-server", check=True)
if sh("mysqladmin ping --silent && echo up") != "up":
    sh("service mysql start")
    for _ in range(30):
        if sh("mysqladmin ping --silent && echo up") == "up":
            break
        time.sleep(1)
sh(f"mysql -uroot -e \"ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY '{ROOT_PW}';\"")
MY = f"mysql -uroot -p{ROOT_PW}"
ok("MySQL running") if sh(f"{MY} -N -e 'SELECT 1'") == "1" else bad("MySQL not answering")

step("employees sample database")
n = sh(f"{MY} -N -e 'SELECT COUNT(*) FROM employees.employees' 2>/dev/null")
if n.isdigit() and int(n) > 0:
    ok(f"already loaded ({int(n):,} employees)")
else:
    if not os.path.isdir("/tmp/test_db"):
        sh("git clone -q --depth 1 https://github.com/datacharmer/test_db /tmp/test_db", check=True)
    print("  loading (1-2 min)...", flush=True)
    sh(f"cd /tmp/test_db && {MY} < employees.sql", check=True)
    n = sh(f"{MY} -N -e 'SELECT COUNT(*) FROM employees.employees'")
    ok(f"loaded {int(n):,} employees")
sh(f"{MY} -e \"CREATE USER IF NOT EXISTS '{RO_USER}'@'%' IDENTIFIED BY '{RO_PW}'; "
   f"GRANT SELECT, SHOW VIEW ON employees.* TO '{RO_USER}'@'%'; FLUSH PRIVILEGES;\"", check=True)
kind = sh(f"{MY} -N -e \"SELECT TABLE_TYPE FROM information_schema.tables WHERE table_schema='employees' AND table_name='employee_features'\"")
if kind:
    ok(f"read-only user; employee_features already exists ({kind})")
else:
    sh(f"{MY} employees < {APP}/sample_data/employees_features.sql", check=True)
    ok("read-only user + employee_features view")

# ------------------------------------------------------------------ 3. python deps + .env
step("Python dependencies + .env")
sh(f"pip install -q -r {APP}/requirements.txt")
with open(f"{APP}/.env", "w") as f:
    f.write(f"""DATABASE_URL=mysql+pymysql://{RO_USER}:{RO_PW}@127.0.0.1:3306/employees
OLLAMA_HOST=http://127.0.0.1:11434
OLLAMA_MODEL={SQL_MODEL}
OLLAMA_ANSWER_MODEL={THINK_MODEL}
OLLAMA_EXPERT_MODEL={THINK_MODEL}
LLM_NUM_CTX=8192
LLM_TEMPERATURE=0
MAX_ROWS=1000
""")
ok(".env written (DB + models point at this VM)")
if kind == "BASE TABLE":
    ok("employee_features already materialised")
else:
    r = subprocess.run(f"cd {APP} && python sample_data/materialise_employee_features.py --admin-url mysql+pymysql://root:{ROOT_PW}@127.0.0.1:3306/employees",
                       shell=True, capture_output=True, text=True)
    ok("employee_features materialised") if r.returncode == 0 else print("  note: materialise skipped, the view still works")

# ------------------------------------------------------------------ 4. Ollama
step("Ollama on the GPU")
OLLAMA_TAR = "https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64.tar.zst"


def install_ollama():
    """Manual install from the GitHub release asset (Ollama ships .tar.zst now; install.sh is flaky on Colab)."""
    sh("DEBIAN_FRONTEND=noninteractive apt-get install -y -qq zstd >/dev/null 2>&1 || true")
    for attempt in range(1, 4):
        r = subprocess.run(f"curl -fL --retry 3 -o /tmp/ollama.tar.zst {OLLAMA_TAR} && rm -rf /usr/local/lib/ollama "
                           "&& tar --use-compress-program=unzstd -C /usr/local -xf /tmp/ollama.tar.zst && chmod +x /usr/local/bin/ollama",
                           shell=True, capture_output=True, text=True)
        if r.returncode == 0 and os.path.exists("/usr/local/bin/ollama"):
            return True
        print(f"  attempt {attempt} failed: {(r.stderr or r.stdout)[-400:].strip()}", flush=True)
        print("  disk:", sh("df -h /usr/local | tail -1"), flush=True)
        time.sleep(3)
    return False


if not (os.path.exists("/usr/local/bin/ollama") or sh("which ollama")):
    print("  installing Ollama (~160 MB download)...", flush=True)
    if not install_ollama():
        bad("could not install Ollama - paste the lines above")
        sys.exit(1)
os.environ["PATH"] = "/usr/local/bin:" + os.environ["PATH"]
ok(f"binary {sh('which ollama')} ({sh('ollama --version 2>&1 | tail -1')})")
if not wait("http://127.0.0.1:11434/api/tags", 2):
    daemon("ollama serve", f"{LOG}/ollama.log")
    if not wait("http://127.0.0.1:11434/api/tags", 40):
        bad("ollama serve did not come up; log:\n" + sh(f"tail -20 {LOG}/ollama.log"))
        sys.exit(1)
ok("ollama serve is running (pid " + sh("pgrep -f 'ollama serve' | head -1") + ")")
for m in (SQL_MODEL, THINK_MODEL):
    have = json.load(urllib.request.urlopen("http://127.0.0.1:11434/api/tags"))["models"]
    if any(x["name"] == m for x in have):
        ok(f"model {m} present")
    else:
        print(f"  pulling {m} (a few minutes)...", flush=True)
        subprocess.run(f"ollama pull {m}", shell=True, check=True)
        ok(f"model {m} pulled")
print("  test generation with the SQL model...", flush=True)
t0 = time.time()
resp = json.load(urllib.request.urlopen(urllib.request.Request(
    "http://127.0.0.1:11434/api/chat", data=json.dumps({"model": SQL_MODEL, "stream": False,
    "messages": [{"role": "user", "content": "Reply with the single word OK"}]}).encode(),
    headers={"Content-Type": "application/json"}), timeout=600))
ok(f"Ollama answered in {time.time() - t0:.1f}s: {resp['message']['content'][:40]!r}")
print("  GPU memory now:", sh("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader"))

# ------------------------------------------------------------------ 5. the app's own health check
step("Health check exactly as the app's sidebar does it")
probe = sh(f"cd {APP} && python -c \"from agent.llm import OllamaLLM; from agent.config import settings; "
           f"print(OllamaLLM(settings.model).health()); print(OllamaLLM(settings.answer_model).health()); "
           f"print(OllamaLLM(settings.expert_model).health())\"")
print("  " + probe.replace("\n", "\n  "))
if "OK" not in probe or "not pulled" in probe or "Cannot reach" in probe:
    bad("the app would show Ollama offline - stop here and paste this output")
    sys.exit(1)
ok("the app will see Ollama")

# ------------------------------------------------------------------ 6. trained models (non-fatal)
step("ML models")
if sh(f"ls {APP}/models/*.joblib 2>/dev/null"):
    ok("already trained")
else:
    r = subprocess.run(f"cd {APP} && python train_models.py --config employees_training_config.yaml "
                       f"--database-url mysql+pymysql://root:{ROOT_PW}@127.0.0.1:3306/employees",
                       shell=True, capture_output=True, text=True)
    ok("trained") if r.returncode == 0 else print("  note: training failed (data queries still work):", (r.stderr or "")[-300:])

# ------------------------------------------------------------------ 7. app
step("Streamlit app")
sh("pkill -f 'streamlit run app.py'")
daemon(f"cd {APP} && python -m streamlit run app.py --server.headless true --server.port 8501 "
       f"--server.address 0.0.0.0 --browser.gatherUsageStats false", f"{LOG}/streamlit.log")
if wait("http://127.0.0.1:8501/_stcore/health", 90):
    ok("app is serving on the VM, port 8501")
else:
    bad("app did not start; log:\n" + sh(f"tail -30 {LOG}/streamlit.log"))
    sys.exit(1)

# ------------------------------------------------------------------ 8. public link
step("Public link (Cloudflare quick tunnel, no account)")
if not sh("which cloudflared"):
    sh("curl -fsSL -o /tmp/cf.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb && dpkg -i /tmp/cf.deb", check=True)
url = ""
for attempt in range(1, 4):
    sh("pkill -f 'cloudflared tunnel'")
    open(f"{LOG}/cloudflared.log", "w").close()
    daemon("cloudflared tunnel --url http://127.0.0.1:8501 --no-autoupdate --protocol http2", f"{LOG}/cloudflared.log")
    for _ in range(45):
        m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", open(f"{LOG}/cloudflared.log").read())
        if m:
            url = m.group(0)
            break
        time.sleep(1)
    if url:
        break
    print(f"  attempt {attempt} gave no URL, retrying...", flush=True)
if not url:
    bad("no tunnel URL; log:\n" + sh(f"tail -20 {LOG}/cloudflared.log"))
    print("  fallback in a new cell:  !ssh -o StrictHostKeyChecking=no -p 443 -R0:localhost:8501 a.pinggy.io")
    sys.exit(1)

print("\n" + "=" * 70)
print(f"  SHARE THIS LINK:  {url}")
print("  Everything runs on this Colab VM. Keep the notebook open while people use it.")
print("  Sidebar should show: Database / SQL · qwen2.5-coder:7b / Answer · qwen3:8b / Expert · qwen3:8b in green.")
print("=" * 70)
