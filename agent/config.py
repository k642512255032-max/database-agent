"""Central configuration, read from environment variables / .env file."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

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


settings = Settings()
