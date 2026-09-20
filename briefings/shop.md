# Database briefing: shop (demo database seeded by sample_data/seed_mysql.py)

## What it is
A small synthetic e-commerce database: ~1,500 registered customers, a product catalogue, their orders and order
lines, plus two feature views for machine learning. Generated data, seeded randomly; the "today" used by the
feature views is **2026-06-30**.

## Tables (grain, keys, size)
- **customers** (1 row per customer, PK customer_id; ~1,500 rows): full_name, gender, age, city, signup_date,
  plan (basic | standard | premium), monthly_fee, support_tickets (opened in the last 12 months),
  churned (1 = cancelled membership).
- **products** (1 row per product, PK product_id): product_name, category, price.
- **orders** (1 row per order, PK order_id, FK customer_id -> customers): order_date (DATETIME),
  status (completed | cancelled | returned), payment_method, total_amount.
- **order_items** (1 row per order line, PK order_item_id, FKs order_id -> orders, product_id -> products):
  quantity, unit_price.
- **customer_features** (view, 1 row per customer, for ML): age, gender, city, plan, monthly_fee, signup_date,
  support_tickets, tenure_months (to 2026-06-30), total_orders, total_spent, avg_order_value, returned_orders,
  days_since_last_order, churned. Customers without orders have 0 orders / 0 spent and NULL days_since_last_order.
- **order_features** (view, 1 row per order, for ML): order_hour, payment_method, status, total_amount, n_lines,
  n_units, max_unit_price.

## Joins
orders.customer_id = customers.customer_id; order_items.order_id = orders.order_id;
order_items.product_id = products.product_id.

## Rules of thumb an expert applies
- Revenue questions usually mean **completed** orders only (status = 'completed'); cancelled and returned orders
  inflate totals if not excluded. Say which statuses were included.
- orders.total_amount is the order total; order_items.quantity * unit_price is the line amount - do not sum both.
- Joining customers to orders multiplies customer rows: count customers with COUNT(DISTINCT customer_id).
- Customers with no orders disappear from an inner join; use LEFT JOIN or customer_features for "all customers".
- Churn is a customer attribute (customers.churned), not an order attribute.
- order_date is a DATETIME: group by DATE(order_date) / month via DATE_FORMAT(order_date, '%Y-%m').
- The data is synthetic: patterns are plausible but not real-world evidence.
