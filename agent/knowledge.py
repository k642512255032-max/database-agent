"""Per-agent knowledge: documents uploaded on the "Fine-tune agents" page, and the tuning each agent gets from them.

Two ways an agent learns from its documents:

1. Knowledge learning (runs here, no GPU). "Start fine-tuning" splits the documents into passages and builds a
   BM25 index per agent. On every call the agent's LLM is wrapped (KnowledgeLLM): the passages most relevant to
   the prompt are appended to the system prompt. The effect is immediate and the trace shows which passages
   were used.
2. LoRA training (GPU, Colab). `generate_examples` has the LLM turn each passage into input -> output examples
   in the agent's own format (uploaded .jsonl example files are used as they are). The dataset is exported
   as chat JSONL and trained with deploy/colab/finetune_agent.py, which registers a new Ollama model. Entering
   that model name for the agent makes the wrapper send the agent's calls to it.

Layout on disk (settings.knowledge_dir, one folder per agent):
    <agent>/sources/<name>.txt    extracted text of each uploaded document
    <agent>/examples.jsonl        hand-written examples uploaded as .jsonl
    <agent>/index.json            passages + BM25 statistics, written by build()
    <agent>/train.jsonl           generated training dataset (chat format)
    <agent>/model.txt             fine-tuned Ollama model used for this agent (optional)
"""
from __future__ import annotations
import io
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional
from .config import settings

# key -> (label, what the agent does, which documents help it)
AGENTS: dict[str, tuple[str, str, str]] = {
    "understanding": ("Understanding", "Request standardiser: rewrites each message into one explicit question.",
                      "Glossaries, abbreviations, business terms, how users phrase questions, examples of "
                      "questions and what they really mean."),
    "router": ("Router", "Decides whether a request is a plain data query, statistics or machine learning.",
               "Which kinds of questions need a statistical test or a model, and which are plain look-ups."),
    "sql": ("SQL writer", "Code agent: writes and repairs the SQL query.",
            "Data dictionary, table and column meanings, join rules, business definitions (e.g. 'current "
            "employee = to_date 9999-01-01'), example queries."),
    "answer": ("Answer & charts", "Explains the result in plain English and plans the charts.",
               "Reporting style guides, how figures should be interpreted and presented, domain background."),
    "expert": ("Expert AI", "Plans the data and reviews the answer as a domain expert.",
               "Domain knowledge, policies, regulations, KPIs and their targets, known data-quality issues."),
    "memory": ("Memory", "Folds each turn into the conversation memory.",
               "What should be remembered between questions (standing filters, preferences, entities)."),
}

# what each agent's training examples look like (input -> output), used when generating the dataset
EXAMPLE_FORMATS = {
    "understanding": "input: a short, informal user message about the data; output: one clear, complete English "
                     "question that states the measure, filters and grouping explicitly",
    "router": "input: a user question; output: one of data_query | statistics | machine_learning, then ' - ' and "
              "a one-sentence reason",
    "sql": "input: a question about the data; output: one MySQL SELECT statement that answers it, using only "
           "tables and columns named in the passage",
    "answer": "input: a question and a short result table in text; output: a plain-English answer of 2-4 sentences",
    "expert": "input: a question or a finding about the data; output: an expert assessment with one insight and "
              "one recommendation",
    "memory": "input: a question and its answer; output: the facts, entities and filters to remember, as a short list",
}

TEXT_TYPES = ("txt", "md", "sql", "csv", "json", "yaml", "yml", "log")
UPLOAD_TYPES = TEXT_TYPES + ("pdf", "docx", "jsonl")
CHUNK_CHARS = 900
WORD = re.compile(r"[a-z0-9_]+")
STOP = frozenset("a an and are as at be by for from has have how i in is it of on or that the this to was what "
                 "when where which who why will with you your do does did can".split())


def tokens(text: str) -> list[str]:
    return [w for w in WORD.findall(text.lower()) if w not in STOP and len(w) > 1]


# ------------------------------------------------------------------ documents
def extract_text(name: str, data: bytes) -> str:
    """Plain text of an uploaded file (.txt/.md/.sql/.csv/.json/..., .pdf, .docx)."""
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext == "pdf":
        from pypdf import PdfReader
        return "\n\n".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(data)).pages)
    if ext == "docx":
        import docx
        doc = docx.Document(io.BytesIO(data))
        return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
    if ext in TEXT_TYPES or ext == "jsonl":
        return data.decode("utf-8-sig", errors="replace")
    raise ValueError(f"Unsupported file type '.{ext}'. Use one of: {', '.join(UPLOAD_TYPES)}.")


def chunk(text: str, size: int = CHUNK_CHARS) -> list[str]:
    """Split on blank lines and pack paragraphs into passages of about `size` characters."""
    paras = [" ".join(p.split()) for p in re.split(r"\n\s*\n", text) if p.strip()]
    out, cur = [], ""
    for p in paras:
        while len(p) > size:                      # a very long paragraph: cut at a sentence end when possible
            cut = p.rfind(". ", 0, size)
            cut = cut + 1 if cut > size // 2 else size
            if cur:
                out.append(cur)
                cur = ""
            out.append(p[:cut].strip())
            p = p[cut:].strip()
        if cur and len(cur) + len(p) + 1 > size:
            out.append(cur)
            cur = p
        else:
            cur = f"{cur}\n{p}" if cur else p
    if cur:
        out.append(cur)
    return [c for c in out if c.strip()]


def parse_examples(text: str) -> list[dict]:
    """Examples from a .jsonl file: {"input", "output"} lines or chat {"messages": [...]} lines."""
    out = []
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {n} is not valid JSON: {exc}") from exc
        if "messages" in row:
            msgs = {m.get("role"): m.get("content", "") for m in row["messages"]}
            row = {"input": msgs.get("user", ""), "output": msgs.get("assistant", "")}
        if not str(row.get("input", "")).strip() or not str(row.get("output", "")).strip():
            raise ValueError(f"line {n} needs non-empty 'input' and 'output' (or chat 'messages').")
        out.append({"input": str(row["input"]), "output": str(row["output"])})
    return out


# ------------------------------------------------------------------ the store
@dataclass
class Passage:
    source: str
    text: str
    score: float = 0.0


class KnowledgeBase:
    def __init__(self, root: Path | str | None = None):
        self.root = Path(root or settings.knowledge_dir)
        self._cache: dict[str, tuple[float, dict]] = {}     # agent -> (index mtime, index)

    def _dir(self, agent: str) -> Path:
        if agent not in AGENTS:
            raise KeyError(f"unknown agent '{agent}'")
        return self.root / agent

    # ---------------------------------------------------------- documents
    def add_document(self, agent: str, name: str, data: bytes) -> str:
        """Store an upload; returns what was stored. .jsonl files are examples, everything else a document."""
        text = extract_text(name, data)
        d = self._dir(agent)
        if name.lower().endswith(".jsonl"):
            examples = parse_examples(text)
            d.mkdir(parents=True, exist_ok=True)
            with open(d / "examples.jsonl", "a", encoding="utf-8") as f:
                for ex in examples:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            return f"{len(examples)} examples"
        if not text.strip():
            raise ValueError(f"No text could be read from {name} (a scanned PDF needs OCR first).")
        (d / "sources").mkdir(parents=True, exist_ok=True)
        (d / "sources" / f"{Path(name).stem}.txt").write_text(text, encoding="utf-8")
        return f"{len(text):,} characters"

    def documents(self, agent: str) -> list[str]:
        src = self._dir(agent) / "sources"
        return sorted(p.stem for p in src.glob("*.txt")) if src.exists() else []

    def remove_document(self, agent: str, name: str) -> None:
        (self._dir(agent) / "sources" / f"{name}.txt").unlink(missing_ok=True)

    def examples(self, agent: str) -> list[dict]:
        path = self._dir(agent) / "examples.jsonl"
        return parse_examples(path.read_text(encoding="utf-8")) if path.exists() else []

    def clear_examples(self, agent: str) -> None:
        (self._dir(agent) / "examples.jsonl").unlink(missing_ok=True)

    # ---------------------------------------------------------- index
    def build(self, agent: str) -> dict:
        """'Start fine-tuning' for knowledge learning: (re)index every document of the agent."""
        d = self._dir(agent)
        passages = [{"source": name, "text": c}
                    for name in self.documents(agent)
                    for c in chunk((d / "sources" / f"{name}.txt").read_text(encoding="utf-8"))]
        df: Counter = Counter()
        for p in passages:
            toks = tokens(p["text"])
            p["len"] = len(toks)
            df.update(set(toks))
        index = {"agent": agent, "built_at": datetime.now().isoformat(timespec="seconds"),
                 "documents": self.documents(agent), "passages": passages, "df": dict(df),
                 "avg_len": (sum(p["len"] for p in passages) / len(passages)) if passages else 0.0}
        d.mkdir(parents=True, exist_ok=True)
        (d / "index.json").write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
        self._cache.pop(agent, None)
        return index

    def index(self, agent: str) -> Optional[dict]:
        """The built index, re-read only when the file changed (cheap to call on every LLM call)."""
        path = self._dir(agent) / "index.json"
        try:
            mtime = path.stat().st_mtime
        except FileNotFoundError:
            return None
        cached = self._cache.get(agent)
        if cached and cached[0] == mtime:
            return cached[1]
        index = json.loads(path.read_text(encoding="utf-8"))
        self._cache[agent] = (mtime, index)
        return index

    def reset(self, agent: str) -> None:
        """Forget everything the agent learned (documents, examples, index, dataset, model)."""
        d = self._dir(agent)
        for p in sorted(d.rglob("*"), reverse=True) if d.exists() else []:
            p.unlink() if p.is_file() else p.rmdir()
        self._cache.pop(agent, None)

    def search(self, agent: str, query: str, k: int | None = None) -> list[Passage]:
        """BM25 over the agent's passages. Only passages sharing terms with the query are returned, and of those
        only the ones scoring at least KNOWLEDGE_MIN_SCORE of the best hit (drops weak tails)."""
        index = self.index(agent)
        if not index or not index["passages"]:
            return []
        k = k or settings.knowledge_top_k
        n, avg, df = len(index["passages"]), index["avg_len"] or 1.0, index["df"]
        q = set(tokens(query))
        scored = []
        for p in index["passages"]:
            tf = Counter(t for t in tokens(p["text"]) if t in q)
            if not tf:
                continue
            # IDF floored above 0: with one or two short documents every term is in every passage and plain
            # BM25 would score them all ~0, so a single uploaded glossary would never be used
            s = sum(max(math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)), 0.1)
                    * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * p["len"] / avg)) for t, f in tf.items())
            scored.append(Passage(p["source"], p["text"], round(s, 2)))
        if not scored:
            return []
        scored.sort(key=lambda p: -p.score)
        floor = scored[0].score * settings.knowledge_min_score
        return [p for p in scored[:k] if p.score >= floor]

    # ---------------------------------------------------------- fine-tuned model
    def model_for(self, agent: str) -> str:
        path = self._dir(agent) / "model.txt"
        return path.read_text(encoding="utf-8").strip() if path.exists() else ""

    def set_model(self, agent: str, model: str) -> None:
        path = self._dir(agent) / "model.txt"
        if model.strip():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(model.strip(), encoding="utf-8")
        else:
            path.unlink(missing_ok=True)

    # ---------------------------------------------------------- LoRA dataset
    def dataset_path(self, agent: str) -> Path:
        return self._dir(agent) / "train.jsonl"

    def generate_examples(self, agent: str, llm: Any, per_passage: int = 3,
                          on_progress: Callable[[int, int], None] | None = None) -> int:
        """Write <agent>/train.jsonl: uploaded examples + LLM-written examples from every passage.
        Returns the number of examples. Each line is chat JSONL (system / user / assistant)."""
        index = self.index(agent) or self.build(agent)
        system = TRAIN_SYSTEM[agent]
        examples = list(self.examples(agent))
        total = len(index["passages"])
        for i, p in enumerate(index["passages"], 1):
            try:
                out = llm.chat_json(GEN_SYSTEM.format(n=per_passage, fmt=EXAMPLE_FORMATS[agent]),
                                    f"Passage (from {p['source']}):\n{p['text']}\n\nJSON:", GEN_SCHEMA)
                examples += [e for e in out.get("examples", [])[:per_passage]
                             if str(e.get("input", "")).strip() and str(e.get("output", "")).strip()]
            except Exception:
                pass                                 # one bad passage must not stop the dataset
            if on_progress:
                on_progress(i, total)
        path = self.dataset_path(agent)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for e in examples:
                f.write(json.dumps({"messages": [{"role": "system", "content": system},
                                                 {"role": "user", "content": str(e["input"])},
                                                 {"role": "assistant", "content": str(e["output"])}]},
                                   ensure_ascii=False) + "\n")
        return len(examples)

    def status(self, agent: str) -> dict:
        index = self.index(agent)
        ds = self.dataset_path(agent)
        return {"documents": len(self.documents(agent)), "examples": len(self.examples(agent)),
                "passages": len(index["passages"]) if index else 0,
                "built_at": index["built_at"] if index else None,
                "stale": bool(index) and index.get("documents") != self.documents(agent),
                "dataset": sum(1 for _ in open(ds, encoding="utf-8")) if ds.exists() else 0,
                "model": self.model_for(agent)}


GEN_SYSTEM = """You write training examples for one agent of a database assistant, based on a passage of its
reference documents. Write {n} varied, realistic examples that teach what the passage says.
Format of each example: {fmt}.
Use only facts, names, tables and columns that appear in the passage. Never invent IDs or values."""

GEN_SCHEMA = {
    "type": "object",
    "properties": {"examples": {"type": "array", "items": {
        "type": "object", "properties": {"input": {"type": "string"}, "output": {"type": "string"}},
        "required": ["input", "output"]}}},
    "required": ["examples"],
}

# system prompt stored with each training example (short: the fine-tuned model learns the behaviour itself)
TRAIN_SYSTEM = {a: f"You are the {label} agent of a database assistant. {what}" for a, (label, what, _) in AGENTS.items()}


# ------------------------------------------------------------------ LLM wrapper
KNOWLEDGE_HEADER = ("\n\nReference knowledge for this task (from documents uploaded for this agent). Follow it when "
                    "it is relevant; ignore it when it is not. It never overrides the output format above.\n")


class KnowledgeLLM:
    """Wraps an agent's LLM: appends relevant passages to the system prompt, and sends the call to the agent's
    fine-tuned model when one is set. Without documents or a model it passes calls through unchanged."""

    def __init__(self, inner: Any, kb: KnowledgeBase, agent: str, light: bool = False):
        self.inner, self.kb, self.agent, self._light = inner, kb, agent, light
        self.last_knowledge: list[str] = []
        self._target_llm: Any = inner

    def _target(self) -> Any:
        model = self.kb.model_for(self.agent)
        if not model or not hasattr(self.inner, "host"):      # test fakes have no host: never replaced
            self._target_llm = self.inner
            return self.inner
        if getattr(self._target_llm, "model", None) != model:
            from .llm import OllamaLLM
            self._target_llm = OllamaLLM(model, self.inner.host, think=False if self._light else None)
        return self._target_llm

    def _augment(self, system: str, user: str) -> str:
        passages = self.kb.search(self.agent, user[-settings.knowledge_query_chars:])
        self.last_knowledge = [f"{p.source} (score {p.score}): {p.text[:160]}" for p in passages]
        if not passages:
            return system
        body, used = [], 0
        for i, p in enumerate(passages, 1):
            if used + len(p.text) > settings.knowledge_max_chars:
                break
            body.append(f"[{i}] ({p.source}) {p.text}")
            used += len(p.text)
        return system + KNOWLEDGE_HEADER + "\n".join(body)

    def chat_json(self, system: str, user: str, schema: dict) -> dict:
        return self._target().chat_json(self._augment(system, user), user, schema)

    def chat(self, system: str, user: str, schema: Optional[dict] = None) -> str:
        return self._target().chat(self._augment(system, user), user, schema)

    def light(self) -> "KnowledgeLLM":
        if not hasattr(self.inner, "light"):
            return self
        return KnowledgeLLM(self.inner.light(), self.kb, self.agent, light=True)

    @property
    def model(self) -> str:
        return getattr(self._target(), "model", "?")

    def __getattr__(self, name: str) -> Any:            # last_thinking, last_stats, health, host, ...
        return getattr(self.__dict__.get("_target_llm") or self.__dict__["inner"], name)
