"""Materialise the `employee_features` view of the MySQL "employees" sample database as a real table.

The view (sample_data/employees_features.sql) aggregates 2.8M salary rows and two window
functions for every one of the 300k employees, so every query against it - the agent's schema
probes and every ML inference query - recomputes all of that and takes tens of seconds.
The data is a fixed historical snapshot (it ends in 2002), so a static table loses nothing.

Run ONCE with an admin connection (needs CREATE/DROP on the database):

    python sample_data/materialise_employee_features.py
    python sample_data/materialise_employee_features.py --admin-url mysql+pymysql://root:PASSWORD@127.0.0.1:3305/employees

Without --admin-url the DATABASE_URL from .env / the environment is used. Re-running rebuilds the table.
Trained models keep working: the table has the same name and columns as the view.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.config import settings  # noqa: E402  (loads .env, same as the app)

TABLE = "employee_features"
TMP_VIEW = f"{TABLE}_v"
VIEW_SQL = Path(__file__).with_name("employees_features.sql")
IDENT = re.compile(r"^[A-Za-z0-9_]+$")


def view_definition() -> str:
    """The CREATE VIEW statement from employees_features.sql, renamed to the temporary view."""
    sql = VIEW_SQL.read_text(encoding="utf-8")
    sql = "\n".join(line for line in sql.splitlines() if not line.lstrip().startswith("--"))
    m = re.search(r"CREATE\s+VIEW\s+`?employee_features`?\s+AS(.*?);", sql, flags=re.S | re.I)
    if not m:
        sys.exit(f"could not find 'CREATE VIEW employee_features AS ...;' in {VIEW_SQL}")
    return f"CREATE OR REPLACE VIEW `{TMP_VIEW}` AS{m.group(1)}"


def object_type(conn, database: str, name: str) -> str | None:
    row = conn.execute(text("SELECT TABLE_TYPE FROM information_schema.TABLES "
                            "WHERE TABLE_SCHEMA = :db AND TABLE_NAME = :name"),
                       {"db": database, "name": name}).fetchone()
    return None if row is None else row[0]          # 'BASE TABLE' | 'VIEW'


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--admin-url", default=settings.database_url,
                    help="SQLAlchemy URL with CREATE/DROP rights on the employees database (default: DATABASE_URL)")
    args = ap.parse_args()
    if not args.admin_url:
        sys.exit("set DATABASE_URL (in .env or the environment) or pass --admin-url")

    engine = create_engine(args.admin_url)
    database = engine.url.database
    if not database or not IDENT.match(database):
        sys.exit(f"the URL must name a database (got {database!r})")

    t0 = time.time()
    with engine.begin() as c:
        c.execute(text(view_definition()))
        kind = object_type(c, database, TABLE)
        if kind == "VIEW":
            c.execute(text(f"DROP VIEW `{TABLE}`"))
        elif kind == "BASE TABLE":
            c.execute(text(f"DROP TABLE `{TABLE}`"))
        print(f"building `{TABLE}` from the view (this aggregates the whole salaries table) ...", flush=True)
        c.execute(text(f"CREATE TABLE `{TABLE}` AS SELECT * FROM `{TMP_VIEW}`"))
        c.execute(text(f"ALTER TABLE `{TABLE}` ADD PRIMARY KEY (emp_no), "
                       f"ADD INDEX idx_department (department), ADD INDEX idx_left (left_company), "
                       f"ADD INDEX idx_title (current_title)"))
        c.execute(text(f"DROP VIEW `{TMP_VIEW}`"))
        n = c.execute(text(f"SELECT COUNT(*) FROM `{TABLE}`")).scalar()
    print(f"done: `{database}`.`{TABLE}` is now a table with {n:,} rows ({time.time() - t0:.0f}s). "
          f"Restart the app so it re-reads the schema.")


if __name__ == "__main__":
    main()
