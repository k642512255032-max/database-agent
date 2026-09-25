"""Central configuration, read from environment variables / .env file."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"


def _load_env_file(path: Path) -> None:
    """Load KEY=VALUE lines from the project's .env file.

    The .env file always wins over variables already set in the shell, so editing
    .env is enough. Works even if python-dotenv is not installed.
    """
    if not path.exists():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(path, override=True)
        return
    except ImportError:
        pass
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip().strip('"').strip("'")
_load_env_file(ENV_FILE)


def _on(name: str, default: str = "1") -> bool:
    """True unless the env var is set to 0 / false / no / empty."""
    return os.getenv(name, default) not in ("0", "false", "no", "")


@dataclass
class Settings:
    # --- Database -----------------------------------------------------------
    # SQLAlchemy URL, e.g. mysql+pymysql://user:pass@localhost:3306/shop
    database_url: str = field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL", "mysql+pymysql://root:helloworld@127.0.0.1:3305/employees"
        )
    )
    max_rows: int = int(os.getenv("MAX_ROWS", "1000"))       # LIMIT for plain data queries (shown as a table)
    analysis_max_rows: int = int(os.getenv("ANALYSIS_MAX_ROWS", "100000"))  # LIMIT when rows feed stats / ML
    query_timeout_s: int = int(os.getenv("QUERY_TIMEOUT_S", "30"))

    # --- LLM (Ollama) -------------------------------------------------------
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    model: str = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:3b")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0"))
    num_ctx: int = int(os.getenv("LLM_NUM_CTX", "8192"))
    llm_timeout_s: int = int(os.getenv("LLM_TIMEOUT_S", "180"))
    # Thinking (reasoning) models: Ollama runs their reasoning first ("think": true), so they need more time.
    # Model names matching this regex get thinking switched on and the longer timeout.
    thinking_models: str = os.getenv("OLLAMA_THINKING_MODELS", r"qwen3|deepseek-r1|gpt-oss|magistral|phi4-reasoning")
    llm_think_timeout_s: int = int(os.getenv("LLM_THINK_TIMEOUT_S", "900"))
    # How long Ollama keeps a model (and its prompt cache) loaded after a call; "-1" = forever. Sent with every request.
    llm_keep_alive: str = os.getenv("LLM_KEEP_ALIVE", "30m")
    # Separate model for the answer agent (explanations + chart planning). Empty = same as OLLAMA_MODEL.
    # A general instruct model (e.g. qwen2.5:7b-instruct) writes far better prose than a coder model.
    answer_model: str = os.getenv("OLLAMA_ANSWER_MODEL", "")
    max_charts: int = int(os.getenv("MAX_CHARTS", "2"))
    # Also write a short plain-English summary for plain data queries (one extra answer-model call per query).
    summarise_data_queries: bool = os.getenv("SUMMARISE_DATA_QUERIES", "1") not in ("0", "false", "no", "")

    # --- Agent --------------------------------------------------------------
    max_sql_retries: int = int(os.getenv("MAX_SQL_RETRIES", "3"))
    max_tables_in_prompt: int = int(os.getenv("MAX_TABLES_IN_PROMPT", "6"))
    sample_rows_per_table: int = int(os.getenv("SAMPLE_ROWS_PER_TABLE", "3"))
    schema_probe_ms: int = int(os.getenv("SCHEMA_PROBE_MS", "4000"))   # max time per row-count / sample probe

    # --- ML -----------------------------------------------------------------
    models_dir: Path = Path(os.getenv("MODELS_DIR", str(ROOT / "models")))

    # --- Power BI export ---------------------------------------------------
    # "live": the .pbip queries MySQL with the generated SQL (needs MySQL Connector/NET on the Power BI machine);
    # "inline": the result rows are embedded in the semantic model, nothing to install, no refresh.
    powerbi_source: str = os.getenv("POWERBI_SOURCE", "live")
    powerbi_inline_max_rows: int = int(os.getenv("POWERBI_INLINE_MAX_ROWS", "5000"))

    # --- Dashboards --------------------------------------------------------
    netlify_token: str = os.getenv("NETLIFY_AUTH_TOKEN", "")          # personal access token; empty = publishing disabled
    dashboard_max_widgets: int = int(os.getenv("DASHBOARD_MAX_WIDGETS", "8"))
    dashboard_rows_per_widget: int = int(os.getenv("DASHBOARD_ROWS_PER_WIDGET", "500"))   # LIMIT for widget SQL and table rows
    dashboards_dir: Path = Path(os.getenv("DASHBOARDS_DIR", str(ROOT / "dashboards")))

    # --- Expert AI ---------------------------------------------------------
    # A specialist agent beside the answer agent: judges data quality, gives insights and advice.
    # Empty model = same as OLLAMA_ANSWER_MODEL (then OLLAMA_MODEL). A 14B instruct model gives noticeably better advice.
    expert_model: str = os.getenv("OLLAMA_EXPERT_MODEL", "")
    expert_persona: str = os.getenv("EXPERT_PERSONA", "")          # "Domain & goals" text, editable in the sidebar
    # Each agent can be switched on/off (sidebar "Agents" panel); these are the defaults. 1 = on.
    standardise_requests: bool = _on("STANDARDISE_REQUESTS", "1")   # step 0: request standardiser (SQL model)
    expert_reviews: bool = _on("EXPERT_REVIEWS", "1")               # both expert steps unless overridden below
    expert_plan: Optional[bool] = _on("EXPERT_PLAN") if os.getenv("EXPERT_PLAN") is not None else None      # step 2
    expert_assess: Optional[bool] = _on("EXPERT_ASSESS") if os.getenv("EXPERT_ASSESS") is not None else None  # step 7
    draw_charts: bool = _on("DRAW_CHARTS", "1")                     # step 9: chart planner (answer model)
    remember_conversation: bool = _on("REMEMBER_CONVERSATION", "1") # step 10: context builder (answer model)
    # --- Speed (see README "Speed on a laptop") ------------------------------
    # Charts, memory and the data-query summary only fill a JSON form: 0 = run them with thinking off (much faster
    # on a thinking model), 1 = let the model reason first as well.
    think_light_steps: bool = _on("THINK_LIGHT_STEPS", "0")
    # step 1: trust the standardiser's task for plain data queries and skip the router model call
    router_shortcut: bool = _on("ROUTER_SHORTCUT", "1")
    # step 10: update the conversation memory in the background after the answer is shown
    defer_memory_update: bool = _on("DEFER_MEMORY_UPDATE", "1")
    expert_audit_timeout_ms: int = int(os.getenv("EXPERT_AUDIT_TIMEOUT_MS", "20000"))   # per audit query (MySQL hint)
    expert_audit_sample_rows: int = int(os.getenv("EXPERT_AUDIT_SAMPLE_ROWS", "500"))    # rows fetched for sample checks

    # --- Fine-tune agents (pages/3_Fine_tune_agents.py, agent/knowledge.py) ---
    knowledge_dir: Path = Path(os.getenv("KNOWLEDGE_DIR", str(ROOT / "knowledge")))   # uploads + indexes per agent
    knowledge_top_k: int = int(os.getenv("KNOWLEDGE_TOP_K", "3"))              # passages added to an agent's prompt
    knowledge_max_chars: int = int(os.getenv("KNOWLEDGE_MAX_CHARS", "2400"))   # cap on the added passage text
    knowledge_min_score: float = float(os.getenv("KNOWLEDGE_MIN_SCORE", "0.35"))  # keep passages scoring >= this share of the best
    knowledge_query_chars: int = int(os.getenv("KNOWLEDGE_QUERY_CHARS", "3000"))  # tail of the prompt used as the search query


settings = Settings()
