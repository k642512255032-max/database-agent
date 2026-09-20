"""Create a demo MySQL database `shop` with realistic synthetic e-commerce data,
feature VIEWs for the ML models, and a read-only user for the agent.

Usage:
    python sample_data/seed_mysql.py --admin-url mysql+pymysql://root:rootpw@127.0.0.1:3306

Tables : customers, products, orders, order_items
Views  : customer_features (1 row per customer, used by churn / segmentation models)
         order_features    (1 row per order, used by anomaly detection)
User   : agent_ro / agent_ro_pw  (SELECT only)
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

SNAPSHOT = datetime(2026, 6, 30)
CITIES = ["Hanoi", "Ho Chi Minh City", "Da Nang", "Hai Phong", "Can Tho", "Hue", "Nha Trang"]
CATEGORIES = {"Electronics": (80, 900), "Books": (5, 40), "Fashion": (15, 150),
              "Home": (20, 300), "Beauty": (8, 80), "Sports": (15, 250)}

DDL = """
DROP VIEW IF EXISTS order_features;
DROP VIEW IF EXISTS customer_features;
DROP TABLE IF EXISTS order_items;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS customers;

CREATE TABLE customers (
  customer_id INT PRIMARY KEY,
  full_name VARCHAR(100) NOT NULL,
  gender VARCHAR(10),
  age INT,
  city VARCHAR(50),
  signup_date DATE,
  plan VARCHAR(20) COMMENT 'basic | standard | premium membership',
  monthly_fee DECIMAL(8,2),
  support_tickets INT COMMENT 'tickets opened in the last 12 months',
  churned TINYINT(1) COMMENT '1 = customer cancelled membership'
) COMMENT='Registered customers';

CREATE TABLE products (
  product_id INT PRIMARY KEY,
  product_name VARCHAR(100),
  category VARCHAR(30),
  price DECIMAL(10,2)
) COMMENT='Product catalogue';

CREATE TABLE orders (
  order_id INT PRIMARY KEY,
  customer_id INT NOT NULL,
  order_date DATETIME,
  status VARCHAR(20) COMMENT 'completed | cancelled | returned',
  payment_method VARCHAR(20),
  total_amount DECIMAL(12,2),
  FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
) COMMENT='Customer orders';

CREATE TABLE order_items (
  order_item_id INT PRIMARY KEY,
  order_id INT NOT NULL,
  product_id INT NOT NULL,
  quantity INT,
  unit_price DECIMAL(10,2),
  FOREIGN KEY (order_id) REFERENCES orders(order_id),
  FOREIGN KEY (product_id) REFERENCES products(product_id)
) COMMENT='Line items of each order';

CREATE VIEW customer_features AS
SELECT c.customer_id, c.age, c.gender, c.city, c.plan, c.monthly_fee, c.signup_date,
       c.support_tickets,
       TIMESTAMPDIFF(MONTH, c.signup_date, '2026-06-30') AS tenure_months,
       COUNT(o.order_id) AS total_orders,
       COALESCE(SUM(o.total_amount), 0) AS total_spent,
       COALESCE(AVG(o.total_amount), 0) AS avg_order_value,
       COALESCE(SUM(o.status = 'returned'), 0) AS returned_orders,
       DATEDIFF('2026-06-30', MAX(o.order_date)) AS days_since_last_order,
       c.churned
FROM customers c
LEFT JOIN orders o ON o.customer_id = c.customer_id
GROUP BY c.customer_id, c.age, c.gender, c.city, c.plan, c.monthly_fee, c.signup_date,
         c.support_tickets, c.churned;

CREATE VIEW order_features AS
SELECT o.order_id, o.customer_id, o.order_date, HOUR(o.order_date) AS order_hour,
       o.payment_method, o.status, o.total_amount,
       COUNT(oi.order_item_id) AS n_lines, COALESCE(SUM(oi.quantity), 0) AS n_units,
       COALESCE(MAX(oi.unit_price), 0) AS max_unit_price
FROM orders o
LEFT JOIN order_items oi ON oi.order_id = o.order_id
GROUP BY o.order_id, o.customer_id, o.order_date, o.payment_method, o.status, o.total_amount;
"""


def generate(n_customers: int = 1500, seed: int = 42) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    first = ["An", "Binh", "Chi", "Dung", "Giang", "Hoa", "Khanh", "Linh", "Minh", "Nam", "Phuong", "Quang",
             "Thao", "Trang", "Tuan", "Vy", "Hung", "Lan", "Son", "Yen"]
    last = ["Nguyen", "Tran", "Le", "Pham", "Hoang", "Vu", "Dang", "Bui", "Do", "Ngo"]

    # ---------------- customers
    plan = rng.choice(["basic", "standard", "premium"], n_customers, p=[0.5, 0.35, 0.15])
    fee = np.select([plan == "basic", plan == "standard"], [4.99, 9.99], 19.99) + rng.normal(0, 0.3, n_customers).round(2)
    signup = [SNAPSHOT - timedelta(days=int(d)) for d in rng.integers(30, 1400, n_customers)]
    tenure = np.array([(SNAPSHOT - s).days / 30.4 for s in signup])
    tickets = rng.poisson(np.where(plan == "basic", 2.2, 1.2))
    age = rng.normal(34, 10, n_customers).clip(18, 75).round().astype(int)
    logit = (-1.2 + 0.45 * tickets - 0.035 * tenure + np.where(plan == "basic", 0.7, np.where(plan == "premium", -0.8, 0))
             - 0.01 * (age - 34) + rng.normal(0, 0.6, n_customers))
    churned = (rng.random(n_customers) < 1 / (1 + np.exp(-logit))).astype(int)
    customers = pd.DataFrame({
        "customer_id": np.arange(1, n_customers + 1),
        "full_name": [f"{rng.choice(last)} {rng.choice(first)}" for _ in range(n_customers)],
        "gender": rng.choice(["female", "male"], n_customers),
        "age": age,
        "city": rng.choice(CITIES, n_customers, p=[0.28, 0.32, 0.12, 0.09, 0.07, 0.06, 0.06]),
        "signup_date": [s.date() for s in signup],
        "plan": plan, "monthly_fee": fee.round(2), "support_tickets": tickets, "churned": churned,
    })

    # ---------------- products
    rows, pid = [], 1
    for cat, (lo, hi) in CATEGORIES.items():
        for i in range(12):
            rows.append({"product_id": pid, "product_name": f"{cat} item {i + 1}", "category": cat,
                         "price": round(float(rng.uniform(lo, hi)), 2)})
            pid += 1
    products = pd.DataFrame(rows)

    # ---------------- orders + items
    orders, items, oid, iid = [], [], 1, 1
    for c in customers.itertuples():
        months = max((SNAPSHOT.date() - c.signup_date).days / 30.4, 1)
        rate = {"basic": 0.6, "standard": 1.0, "premium": 1.6}[c.plan] * (0.45 if c.churned else 1.0)
        n = rng.poisson(rate * months / 3)
        active_until = SNAPSHOT - timedelta(days=int(rng.integers(90, 400))) if c.churned else SNAPSHOT
        start = datetime.combine(c.signup_date, datetime.min.time())
        span = max((active_until - start).days, 1)
        for _ in range(n):
            when = start + timedelta(days=int(rng.integers(0, span)), hours=int(rng.choice(range(7, 24))),
                                     minutes=int(rng.integers(0, 60)))
            lines = rng.integers(1, 4)
            total = 0.0
            for _ in range(lines):
                p = products.iloc[int(rng.integers(0, len(products)))]
                q = int(rng.integers(1, 3))
                items.append({"order_item_id": iid, "order_id": oid, "product_id": int(p.product_id),
                              "quantity": q, "unit_price": float(p.price)})
                total += q * float(p.price)
                iid += 1
            orders.append({"order_id": oid, "customer_id": c.customer_id, "order_date": when,
                           "status": rng.choice(["completed", "cancelled", "returned"], p=[0.88, 0.05, 0.07]),
                           "payment_method": rng.choice(["card", "e-wallet", "cod", "bank_transfer"], p=[0.4, 0.3, 0.2, 0.1]),
                           "total_amount": round(total, 2)})
            oid += 1

    # inject ~25 suspicious orders (night-time, huge quantities, very high value)
    for _ in range(25):
        c = customers.iloc[int(rng.integers(0, n_customers))]
        when = SNAPSHOT - timedelta(days=int(rng.integers(1, 200)), hours=int(rng.integers(0, 4)))
        when = when.replace(hour=int(rng.integers(1, 5)))
        p = products[products.category == "Electronics"].sample(1, random_state=int(rng.integers(1e6))).iloc[0]
        q = int(rng.integers(15, 40))
        items.append({"order_item_id": iid, "order_id": oid, "product_id": int(p.product_id), "quantity": q,
                      "unit_price": float(p.price)})
        orders.append({"order_id": oid, "customer_id": int(c.customer_id), "order_date": when, "status": "completed",
                       "payment_method": "bank_transfer", "total_amount": round(q * float(p.price), 2)})
        oid += 1
        iid += 1
    return {"customers": customers, "products": products, "orders": pd.DataFrame(orders),
            "order_items": pd.DataFrame(items)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--admin-url", default="mysql+pymysql://root:rootpw@127.0.0.1:3306")
    ap.add_argument("--database", default="shop")
    ap.add_argument("--customers", type=int, default=1500)
    ap.add_argument("--ro-user", default="agent_ro")
    ap.add_argument("--ro-password", default="agent_ro_pw")
    args = ap.parse_args()

    admin = create_engine(args.admin_url)
    with admin.begin() as c:
        c.execute(text(f"CREATE DATABASE IF NOT EXISTS `{args.database}` CHARACTER SET utf8mb4"))
    eng = create_engine(f"{args.admin_url.rstrip('/')}/{args.database}")
    with eng.begin() as c:
        for stmt in [s.strip() for s in DDL.split(";") if s.strip()]:
            c.execute(text(stmt))
    data = generate(args.customers)
    for name in ["customers", "products", "orders", "order_items"]:
        data[name].to_sql(name, eng, if_exists="append", index=False, chunksize=2000)
        print(f"  {name}: {len(data[name])} rows")

    with admin.begin() as c:
        for host in ("localhost", "%"):
            c.execute(text(f"CREATE USER IF NOT EXISTS '{args.ro_user}'@'{host}' IDENTIFIED BY '{args.ro_password}'"))
            # if the user already existed (e.g. created earlier by hand), force the expected password
            c.execute(text(f"ALTER USER '{args.ro_user}'@'{host}' IDENTIFIED BY '{args.ro_password}'"))
            c.execute(text(f"GRANT SELECT, SHOW VIEW ON `{args.database}`.* TO '{args.ro_user}'@'{host}'"))
        c.execute(text("FLUSH PRIVILEGES"))

    # verify the read-only login really works before telling the user we're done
    host_port = args.admin_url.split("@", 1)[1].rstrip("/")
    agent_url = f"mysql+pymysql://{args.ro_user}:{args.ro_password}@{host_port}/{args.database}"
    try:
        with create_engine(agent_url).connect() as c:
            n = c.execute(text("SELECT COUNT(*) FROM customer_features")).scalar()
        print(f"Verified login as {args.ro_user} ({n} rows in customer_features).")
    except Exception as exc:
        print(f"WARNING: could not log in as {args.ro_user}: {str(exc).splitlines()[0]}")
    print(f"Done. Put this in your .env file:\nDATABASE_URL={agent_url}")


if __name__ == "__main__":
    main()
