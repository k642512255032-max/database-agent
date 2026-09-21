# Local Data Agent: English → SQL → Statistics / ML, fully local

Ask your **MySQL** database questions in plain English. The agent writes the SQL, runs it
safely, and when needed runs **statistical tests** or **already-trained ML models**
(Decision Tree, Random Forest, KNN, K-Means, PCA, Isolation Forest, DBSCAN) with **automatic
feature engineering**. **Every step is shown**: what it did, why, the model's reasoning, and the
inputs and outputs.

The LLM is a lightweight **Qwen2.5-Coder** model served by **Ollama**. Nothing leaves your machine.

```
 "Predict churn for premium customers in Hanoi"
        │
 0 Standardise request ─ LLM ─▶ standalone question + filters (plan = premium, city = Hanoi), metrics, sort, limit;
                                references like "he" / "that city" resolved from the conversation memory
 1 Understand request ── LLM ─▶ intent = machine_learning, model = churn_random_forest
2 Expert data plan ── Expert ▶ (optional) the expert, briefed on the database, decides which tables / columns / filters
                                answer the request and writes the order for the SQL writer
 2b Select tables ──────────▶ from the expert's plan (lexical schema linking as fallback, scores shown)
 3 Generate SQL ──────── LLM ─▶ SELECT * FROM customer_features WHERE plan='premium' AND city='Hanoi'
                                (the "code agent": its prompt carries the expert's data order)
 4 Validate SQL ─────────────▶ sqlglot: single read-only SELECT, real tables/columns, LIMIT
 5 Execute SQL ──────────────▶ read-only MySQL session; on error the LLM repairs it (≤3 tries)
 6 Load model → Feature engineering → Inference → Explanations
 7 Expert assessment ─ Expert ▶ (optional) data-quality verdict, the expert's own answer, insights, advice
 8 Answer agent ── 2nd LLM ─▶ answer / key findings / interpretation / caveats, built from the computed facts
                                AND the expert's assessment
 9 Chart planner ─ 2nd LLM ─▶ chart spec (bar, line, scatter, histogram, box) validated against the real columns
 10 Context builder ─ 2nd LLM ▶ updates the conversation memory (entities, filters, preferences, facts found)
```

Two kinds of model do the work: a **coder** model writes SQL (`OLLAMA_MODEL`), and a **thinking** model explains the
results, plans the charts and acts as the Expert AI (`OLLAMA_ANSWER_MODEL` / `OLLAMA_EXPERT_MODEL`, default
`qwen3:8b`). Thinking models reason before they answer - Ollama returns that reasoning separately and the trace
shows it under *Model's thinking* - which helps with abstract or judgement questions; `qwen2.5:7b-instruct` still
works as a faster non-thinking alternative. Plain data queries skip the
prose and show the SQL and the result table directly.

Results can be downloaded as **CSV** or **Excel** (statistics tables included as extra sheets), every chart
as a **PNG** — rendered locally with `vl-convert`, so nothing leaves the machine — and the whole result as a
**Power BI project** (table + the planned charts + the SQL) that opens in Power BI Desktop.

A second page, **Dashboards**, turns a plain-English description into a hosted web dashboard: the agent designs the
widgets, writes and runs the SQL, generates the HTML / CSS / JavaScript, previews it, and on your approval publishes
it to Netlify.

## 1. Install

Requirements: Python 3.10+, [Ollama](https://ollama.com), and MySQL 8 (or MariaDB).

```bash
# LLM
ollama pull qwen2.5-coder:3b          # ~2 GB. Use :1.5b for weak laptops, :7b for better SQL
ollama pull qwen3:8b                  # ~5 GB thinking model for answers + the Expert AI (qwen3:4b on weak laptops)

# Python
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # then edit if needed
```

No MySQL yet? `docker compose up -d` starts MySQL 8.4 with root password `rootpw`.

## 2. Try it with the demo database

```bash
python sample_data/seed_mysql.py --admin-url mysql+pymysql://root:rootpw@127.0.0.1:3306
python train_models.py            # trains the 8 models in training_config.yaml (~10 s)
streamlit run app.py              # opens http://localhost:8501
```

The seed script creates `shop` (customers, products, orders, order_items), two feature
views (`customer_features`, `order_features`), and a **SELECT-only** user `agent_ro`.

Example questions:

| Type | Question |
|---|---|
| SQL | *Top 5 cities by total revenue from completed orders in 2025* |
| SQL | *Monthly number of orders in 2026* |
| Statistics | *Is there a significant difference in support tickets between churned and retained customers?* |
| Statistics | *Which factors explain total spent? run a linear regression* |
| ML, supervised | *Predict churn for premium customers in Hanoi using the random forest* |
| ML, clustering | *Segment customers in Da Nang into clusters* |
| ML, anomaly | *Find suspicious orders paid by bank transfer* |

In the sidebar, **Mode** lets you force *Data query / Statistics / Machine learning* and pick a
model yourself if the small LLM routes a question wrongly.

## 3. Use your own database

1. Set `DATABASE_URL` in `.env`. Use a read-only user:
   ```sql
   CREATE USER 'agent_ro'@'%' IDENTIFIED BY 'strong-password';
   GRANT SELECT, SHOW VIEW ON your_db.* TO 'agent_ro'@'%';
   ```
2. Add **table and column comments** in MySQL. They are sent to the LLM and improve SQL accuracy a lot.
3. For ML, create a **feature view** with one row per entity, for example `customer_features`. With a view,
   the agent only has to write `SELECT * FROM view WHERE ...` at inference time, which a 3B model handles reliably.
   If the view is heavy (aggregates over millions of rows), materialise it as a table with the same name so
   schema probes and inference queries stay fast. For the MySQL `employees` sample database this is done by
   `sample_data/employees_features.sql` (the view) + `python sample_data/materialise_employee_features.py` (the table).
4. Describe your models in `training_config.yaml`, then run `python train_models.py`.
5. Restart the app, or click *Refresh schema* in the sidebar.

## 4. What is included

### Explainability (see the 🪜 Step-by-step tab)
Each step records its **name, why it runs, the LLM's reasoning, inputs and outputs, status and duration**.
This includes the exact schema and facts that were sent to the model. Failed SQL attempts and their
repairs stay visible. You can download the whole trace as text.

### Request standardiser and conversation memory (`agent/request_agent.py`, `agent/context.py`)
Two small agents sit around the pipeline:

* **Request standardiser** (step 0, SQL model) turns every message into one explicit, standalone request:
  typos fixed, abbreviations expanded (*dept* → department, *avg* → average), the measure named, and references
  resolved from the memory, e.g. *"Who is the oldest employee?"* → *"How old is he?"* becomes
  *"How old is employee emp_no 10001 (Georgi Facello)?"*. It also extracts **filters, metrics, grouping, time
  range, sort, limit** and the **assumptions** it made (shown under the answer as *Assumed: …*). The router, table
  linking and SQL generation all work from this standardised request, and the SQL prompt gets the extracted
  details as a checklist.
* **Context builder** (last step, answer model) runs after every answer and rewrites a compact memory:
  a summary, the **entities in play** (with IDs), **standing filters**, **metrics of interest**, **user
  preferences** (*"always show the top 10"*), the **tables used** and up to 8 **facts found** (with the concrete
  numbers). It is shown in the sidebar (*Conversation memory*) and in the trace, and is what the next
  standardiser call and SQL prompt see instead of a raw dump of earlier turns. If either LLM call fails the turn
  still completes: the message is used as typed, and the memory is updated deterministically.

Turn *Remember conversation* off to skip both the memory lookup and the update; *Clear chat* resets it.

### Answer agent and charts (`agent/answer_agent.py`)
For statistics and ML answers a second model receives a **fact sheet** (row count, numeric summary, top categories,
the statistics tables or the model summary, first rows) and returns a structured answer: **Answer → Key findings →
Interpretation → Caveats**. It cannot compute anything, only restate the facts. The same model then proposes chart
specs (type, x, y, colour, aggregate); every spec is validated against the actual columns and drawn with Altair
(tooltips included). If the model proposes nothing usable, a deterministic heuristic picks the chart. Toggle
*Draw charts* in the sidebar; `MAX_CHARTS` caps the number per answer.

### Power BI export (`agent/powerbi.py`)
*Download Power BI* next to the CSV/Excel buttons writes a **Power BI Project** (`<question>-powerbi.zip`, a
`.pbip` folder). Unzip it and open the `.pbip` in Power BI Desktop: page *Overview* has the question, the answer,
a row-count card and one Power BI visual per chart the answer agent planned (bar → column/bar chart, line → line
chart, scatter → scatter chart, histogram → column chart over a DAX bin column, box → mean/min/max columns);
page *Data* has the SQL and the full table. Statistics tables become extra tables in the model.

`.pbix` is a binary format only Power BI Desktop can write, so the agent generates the documented text formats
instead: a **TMDL** semantic model and a **PBIR** report, straight from the result and the chart specs.
On older Desktop versions enable *Options → Preview features → Power BI Project (.pbip)* and *Store reports using
enhanced metadata format (PBIR)*.

The sidebar choice **Power BI data source** decides where the rows come from:

| Mode | What the file contains | Refresh in Power BI |
|---|---|---|
| **Live MySQL query** (default, `POWERBI_SOURCE=live`) | The generated SQL as the table's M query: `MySQL.Database(host:port, db, [Query=...])`. Host, port and database name only — **the password is never written**. | Yes. Power BI asks for credentials on first refresh: use the read-only `agent_ro` user. Needs [MySQL Connector/NET](https://dev.mysql.com/downloads/connector/net/) on the Power BI machine. |
| **Embedded rows** (`POWERBI_SOURCE=inline`) | The rows as a DAX `DATATABLE`, capped at `POWERBI_INLINE_MAX_ROWS` (default 5 000). Nothing to install, works offline. | No: a snapshot. |

ML answers are always embedded, because the predictions, clusters and anomaly scores exist only in the agent, not
in MySQL.

### Dashboards: describe → preview → publish (`dashboard/`, `pages/1_Dashboards.py`)
Open **Dashboards** in the sidebar navigation and describe what you want, e.g. *"Sales overview: total revenue KPI,
revenue by month (line), top 10 cities by revenue (bar), orders by status (pie), latest 20 orders (table)"*. Then:

1. **Design** - the answer model turns the description into a validated *dashboard spec*: 3-8 widgets, each with a
   kind (`kpi`, `bar`, `line`, `pie`, `table`), a grid width and a **standalone data question**.
2. **Build** - every widget's question goes through the normal pipeline (`DataAgent.ask`, data-query mode): schema
   linking, SQL generation, validation, execution. Each step is shown live; a widget whose SQL fails becomes an error
   card, the rest of the dashboard still builds.
3. **Preview** - the generated `index.html`, `style.css`, `app.js` and `data.js` (plus a vendored Chart.js) are
   rendered in the page. You can read the files, see every widget's SQL and row count, download the zip, or type a
   change request (*"make revenue by month a bar chart"*): only widgets whose question changed are re-queried.
4. **Publish** - after an explicit consent checkbox, the bundle is zip-deployed to **Netlify** and you get a public
   URL. *My dashboards* lists what you published: **Refresh** re-runs the stored SQL (no LLM) and republishes to the
   same URL; **Unpublish** deletes the site.

The LLM never writes HTML/CSS/JS - it produces the spec, Python renders the files from templates, so a 3B-7B local
model cannot produce a broken page. The published site is a **data snapshot**: it never connects to your database
and nothing on it is live. Everything on it is public to anyone with the link, so publish only what you are happy
to share.

Setup: create a Netlify personal access token (*User settings → Applications*) and put it in `.env` as
`NETLIFY_AUTH_TOKEN`. Without it the page still designs, builds and previews; only *Publish* is disabled.
Caps: `DASHBOARD_MAX_WIDGETS` (8) and `DASHBOARD_ROWS_PER_WIDGET` (500). Records live in `dashboards/*.json`.

### Expert AI: a briefed specialist in the middle of the pipeline (`agent/expert_agent.py`, `agent/briefing.py`, `agent/data_quality.py`, `pages/2_Expert_audit.py`)
The answer agent explains *what* the numbers say. The **Expert AI** decides *which data is needed*, judges *whether
it can be trusted* and says *what an expert would conclude*. It is built like the answer agent (own Ollama model, JSON
output, trace steps, never fatal) and works in two places of every turn:

1. **Expert data plan** (before any SQL) - briefed on the database, the expert reads the standardised request and
   returns a plan: tables, columns, filters, grouping, one-row-per, sort/limit, an explicit **order for the SQL
   writer** and the pitfalls to avoid. Table selection follows the plan (validated against the real schema; lexical
   linking is the fallback) and the order is placed at the top of the SQL prompt.
2. **Expert assessment** (after the data, before the answer) - the expert judges the rows against its own plan and the
   tool-computed quality report, and writes its **own answer**, data issues, insights, advice and a confidence. The
   answer agent receives this assessment inside its fact sheet and must build on it.

**Briefings** - the expert learns the database from `briefings/<database>.md`: a hand-written briefing for the MySQL
`employees` sample database and one for the `shop` demo are bundled (tables, grain, keys, joins and the rules of thumb
an expert applies, e.g. `to_date = '9999-01-01'` = still current, the data snapshot ends in 2002). Any other database
gets a deterministic briefing generated from its schema. The sidebar shows which briefing is loaded (*Database
briefing*), you can edit the file, and *Reload briefing* picks up changes. With *Expert AI* switched off the pipeline
is exactly the classic one.

The assessment uses **specialised prompts per task**:

| Task | When | Focus |
|---|---|---|
| Data-query review | plain SQL answers | completeness, suspicious values (negatives, outliers, future / sentinel dates), whether the rows answer the question or hide detail, which breakdown to add |
| Statistics review | t-test, ANOVA, regression... | test fit, sample size per group, effect size vs significance, assumptions, what to run next |
| ML review | predictions, clusters, anomalies | metrics vs trust, class balance, leakage risk, uncertain predictions, plausibility of explanations, how to validate |
| Table audit | *Expert audit* page | a whole table or view: structure, per-column statistics, findings with severity, report with impact / fix / recommendations |
| Data plan | every question (before SQL) | tables, columns, filters, one-row-per, order for the SQL writer, pitfalls from the briefing |

Every number the expert may cite is computed by tools first (`agent/data_quality.py`): missing values, duplicates,
constant columns, IQR outliers, negatives in amount-like columns, future dates, blank strings, case variants; for whole
tables also distinct counts, ranges, orphan foreign keys, duplicate rows and open-ended sentinel dates, with one
bounded query (`MAX_EXECUTION_TIME`) per group of columns. The model only judges, prioritises and advises - which is
what makes a small local model usable here. The deterministic findings are always shown next to the report.

* **In chat**: turn on *Expert AI* in the sidebar; an *Expert review* card (verdict, expert answer, quality score 1-5,
  data issues, insights, advice, confidence) appears under each answer, and the trace shows the briefing and schema
  the expert read, its plan, the order inside the SQL prompt, and the assessment the answer agent built on.
* **Expert audit page**: pick a table, click *Audit table*, read the findings and the report, download it as Markdown.
* **Persona**: the sidebar box *Domain & goals* (e.g. *"HR analytics; we care about pay equity and retention"*) is
  injected into every expert prompt so the advice is business-specific. `EXPERT_PERSONA` sets the default.
* **Model**: `OLLAMA_EXPERT_MODEL` (empty = the answer model). Use a **thinking** model - `qwen3:8b` by default,
  `qwen3:14b` if you have the memory; its reasoning is recorded in the trace. Nothing is fine-tuned.

### Statistical models (`agent/stats_tools.py`)
`describe`, `correlation` (Pearson and Spearman with p-values), `group_summary`, `ttest` (Welch + Cohen's d),
`anova`, `chi_square` (+ Cramér's V), `normality` (Shapiro-Wilk), `linear_regression` (OLS),
`logistic_regression` (odds ratios). The LLM only **chooses** the method and the columns. The numbers and the
interpretation are computed by scipy/statsmodels with fixed rules, so the LLM cannot make up statistics.

### Automatic feature engineering (`ml/features.py`)
| Detected column | Action |
|---|---|
| ID-like, constant, free text | dropped (reason recorded) |
| numeric (incl. DECIMAL) | median impute + missing flag, log1p if skewed, z-score |
| date / datetime | year, month, day-of-week, days-before-reference |
| categorical ≤15 values | one-hot + `__other__` for unseen values |
| categorical >15 values | frequency encoding |

The feature engineering is fitted at training time and saved inside the model bundle. At inference it replays
the same transformations, and its plan is shown in the UI.

### ML inference and explanations (`ml/inference.py`)
| Algorithm | Output | Explanation |
|---|---|---|
| Decision Tree | class/value + confidence | exact decision path as rules **in raw units** (e.g. `days_since_last_order > 102.5`) |
| Random Forest | class/value + probability | global importance, share of trees agreeing, local attribution |
| KNN | class/value | the k nearest training rows with their labels and distances |
| K-Means | cluster, distance to centroid | cluster sizes and what characterises each centroid |
| PCA | PC scores | explained variance, top loadings, PC1/PC2 scatter |
| Isolation Forest | anomaly flag + score | the most extreme features (z-scores) of each anomaly |
| DBSCAN | cluster / noise | nearest core sample within eps (DBSCAN has no native `predict`) |

*Local attribution* sets one column at a time to its typical training value and measures how much the
prediction changes. It works the same way for every supervised model.

## 4b. Share it temporarily for free (Google Colab GPU)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/k642512255032-max/database-agent/blob/main/deploy/colab/Host_on_Colab.ipynb)

`deploy/colab/Host_on_Colab.ipynb` runs the whole stack on a free Colab T4 GPU and opens a public link:
MySQL with the `employees` sample database, Ollama with `qwen2.5-coder:7b` + `qwen3:8b`, the trained models and the
app, then a Cloudflare quick tunnel (no account). Open the notebook, pick *Runtime → Change runtime type → T4 GPU*,
*Run all*, wait ~10 minutes, share the `https://….trycloudflare.com` link printed by cell 3 and leave cell 4 running.
The link lives while the notebook runs (Colab ends sessions after ~90 min idle / ~12 h) and changes on every run.
Anyone with the link can query the sample data and use the models; Netlify publishing stays off unless you add a
token on the VM. The same scripts (`deploy/colab/bootstrap.sh`, `serve.sh`) work on Kaggle or any Ubuntu GPU box.

## 5. Safety
* sqlglot allows a **single SELECT/WITH/UNION** only. INSERT/UPDATE/DELETE/DDL/`INTO OUTFILE`/multiple statements are rejected.
* Unknown tables and columns are rejected **before** execution, with a helpful message for the repair step.
* A `LIMIT` is always enforced (`MAX_ROWS`), and MySQL sessions are `SET SESSION TRANSACTION READ ONLY`.
* Model files are joblib pickles. **Only load models you trained yourself.**

## 6. Configuration (`.env`)
| Variable | Default |
|---|---|
| `DATABASE_URL` | `mysql+pymysql://agent_ro:agent_ro_pw@127.0.0.1:3306/shop` |
| `OLLAMA_HOST` / `OLLAMA_MODEL` | `http://127.0.0.1:11434` / `qwen2.5-coder:3b` |
| `OLLAMA_ANSWER_MODEL` / `MAX_CHARTS` | *(same as OLLAMA_MODEL)* / 2 — a thinking model such as `qwen3:8b` is recommended |
| `OLLAMA_THINKING_MODELS`, `LLM_THINK_TIMEOUT_S` | `qwen3\|deepseek-r1\|gpt-oss\|magistral\|phi4-reasoning`, 900 — names matching the regex run with thinking on and the longer timeout |
| `SCHEMA_PROBE_MS` | 4000 — time limit per row-count / sample probe at schema load (heavy views are skipped) |
| `MAX_ROWS`, `MAX_SQL_RETRIES`, `MAX_TABLES_IN_PROMPT` | 1000, 3, 6 |
| `LLM_NUM_CTX`, `LLM_TEMPERATURE` | 8192, 0 |
| `MODELS_DIR` | `./models` |
| `POWERBI_SOURCE`, `POWERBI_INLINE_MAX_ROWS` | `live`, 5000 |
| `NETLIFY_AUTH_TOKEN` | *(empty = dashboard publishing off)* |
| `DASHBOARD_MAX_WIDGETS`, `DASHBOARD_ROWS_PER_WIDGET`, `DASHBOARDS_DIR` | 8, 500, `./dashboards` |
| `OLLAMA_EXPERT_MODEL`, `EXPERT_PERSONA`, `EXPERT_REVIEWS` | *(answer model)*, *(empty)*, 1 |
| `EXPERT_AUDIT_TIMEOUT_MS`, `EXPERT_AUDIT_SAMPLE_ROWS` | 20000, 500 |

## 7. Tests
```bash
pytest -q tests/     # uses a scripted fake LLM against the seeded MySQL, so Ollama is not needed
```
Tests cover the SQL guard, the read-only session, SQL self-repair, t-test and regression, and all 7 ML algorithms.

## Project layout
```
app.py                  Streamlit UI (chat)     ui_shared.py  cached agent shared by the pages
pages/1_Dashboards.py   Dashboards page: describe -> build -> preview -> publish
pages/2_Expert_audit.py Expert audit page: profile a table -> findings -> expert report
briefings/              database briefings for the expert (employees.md, shop.md; add <database>.md for yours)
deploy/colab/           free temporary hosting: bootstrap.sh + serve.sh + Host_on_Colab.ipynb
dashboard/  spec.py  prompts.py  builder.py (design + fetch + render)  render.py (HTML/CSS/JS bundle)
            netlify.py (zip deploy)  store.py (dashboards/*.json)  assets/chart.umd.js (vendored Chart.js)
train_models.py         one-off training → models/*.joblib + manifest.json
training_config.yaml    which models to train, on which SQL
agent/  config.py  db.py (schema, linking, SQL guard)  llm.py (Ollama JSON-schema output)
        prompts.py  orchestrator.py (the pipeline)  answer_agent.py (explanations + chart plans)
        request_agent.py (request standardiser)  context.py (conversation memory + context builder)
        stats_tools.py  trace.py (explainability)
        export.py (CSV/Excel/PNG)  powerbi.py (.pbip project)
        expert_agent.py (Expert AI: data plan, assessment, table audit)  briefing.py (database briefings)
        data_quality.py (quality toolkit + audit SQL)
ml/     features.py (auto FE)  registry.py (bundles)  inference.py (predict + explain)
sample_data/seed_mysql.py   demo database
tests/  test_agent.py  test_export.py  test_powerbi.py  test_dashboard_*.py  test_data_quality.py  test_expert_*.py  test_briefing.py
```

## Tips for small models
* `qwen2.5-coder:3b` is a good default. If joins across many tables fail often, try `:7b`.
* Keep `MAX_TABLES_IN_PROMPT` low and use clear table/column names or comments.
* Views are the easiest way to hide complex joins from the model.
