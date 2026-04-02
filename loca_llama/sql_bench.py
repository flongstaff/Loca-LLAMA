"""SQL generation benchmark: test how well local LLMs generate SQL from natural language.

Models receive a database schema + natural language question, generate SQL, and
we execute it against an in-memory SQLite database to check correctness.
Includes an agentic retry loop: on failure, the error is fed back to the model.

Inspired by sql-benchmark.nicklothian.com.

Usage (via CLI):
    loca-llama sql                          # All loaded models
    loca-llama sql --model Qwen3.5-35B      # Single model
    loca-llama sql --difficulty easy,medium  # Filter by difficulty
    loca-llama sql --export html             # HTML heatmap report
"""

from __future__ import annotations

import random
import re
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any

from .quality_bench import call_openai_api


# ── Schema DDL ──────────────────────────────────────────────────────────────

SCHEMA_DDL = """\
CREATE TABLE customers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL,
    city TEXT NOT NULL,
    state TEXT,
    country TEXT NOT NULL DEFAULT 'US',
    created_at TEXT NOT NULL
);

CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    subcategory TEXT NOT NULL,
    price REAL NOT NULL,
    cost REAL NOT NULL
);

CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    order_date TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'completed',
    total_amount REAL NOT NULL
);

CREATE TABLE order_items (
    id INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL REFERENCES orders(id),
    product_id INTEGER NOT NULL REFERENCES products(id),
    quantity INTEGER NOT NULL,
    unit_price REAL NOT NULL,
    discount REAL NOT NULL DEFAULT 0.0
);
"""

SCHEMA_DESCRIPTION = """\
Tables:
- customers (id, name, email, city, state, country, created_at)
- products (id, name, category, subcategory, price, cost)
- orders (id, customer_id, order_date, status, total_amount)
- order_items (id, order_id, product_id, quantity, unit_price, discount)

Relationships:
- orders.customer_id → customers.id
- order_items.order_id → orders.id
- order_items.product_id → products.id
"""


# ── Seed Data Generation ────────────────────────────────────────────────────

_FIRST_NAMES = [
    "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry",
    "Iris", "Jack", "Karen", "Leo", "Maya", "Noah", "Olivia", "Pete",
    "Quinn", "Rosa", "Sam", "Tara", "Uma", "Victor", "Wendy", "Xander",
    "Yuki", "Zach",
]

_LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Anderson", "Taylor", "Thomas",
    "Moore", "Jackson", "Martin", "Lee", "Perez", "White", "Harris",
]

_CITIES = [
    ("New York", "NY"), ("Los Angeles", "CA"), ("Chicago", "IL"),
    ("Houston", "TX"), ("Phoenix", "AZ"), ("Philadelphia", "PA"),
    ("San Antonio", "TX"), ("San Diego", "CA"), ("Dallas", "TX"),
    ("Austin", "TX"), ("Portland", "OR"), ("Seattle", "WA"),
    ("Denver", "CO"), ("Boston", "MA"), ("Miami", "FL"),
]

_PRODUCT_CATALOG = [
    # (name, category, subcategory, price, cost)
    ("Laptop Pro 15", "Electronics", "Computers", 1299.99, 850.00),
    ("Laptop Air 13", "Electronics", "Computers", 999.99, 650.00),
    ("Desktop Tower", "Electronics", "Computers", 899.99, 550.00),
    ("Wireless Mouse", "Electronics", "Accessories", 29.99, 12.00),
    ("Mechanical Keyboard", "Electronics", "Accessories", 79.99, 35.00),
    ("USB-C Hub", "Electronics", "Accessories", 49.99, 20.00),
    ("Monitor 27in", "Electronics", "Displays", 449.99, 280.00),
    ("Monitor 32in", "Electronics", "Displays", 599.99, 370.00),
    ("Webcam HD", "Electronics", "Accessories", 69.99, 25.00),
    ("Headphones Pro", "Electronics", "Audio", 249.99, 110.00),
    ("Speaker Set", "Electronics", "Audio", 129.99, 55.00),
    ("Bluetooth Earbuds", "Electronics", "Audio", 89.99, 35.00),
    ("Phone Case", "Electronics", "Accessories", 19.99, 5.00),
    ("Tablet 10in", "Electronics", "Tablets", 499.99, 310.00),
    ("Tablet 8in", "Electronics", "Tablets", 349.99, 210.00),
    ("Running Shoes", "Sports", "Footwear", 129.99, 55.00),
    ("Yoga Mat", "Sports", "Fitness", 39.99, 15.00),
    ("Water Bottle", "Sports", "Accessories", 24.99, 8.00),
    ("Backpack", "Sports", "Bags", 79.99, 30.00),
    ("Sunglasses", "Sports", "Accessories", 59.99, 18.00),
    ("Desk Lamp", "Home", "Lighting", 44.99, 18.00),
    ("Standing Desk", "Home", "Furniture", 549.99, 300.00),
    ("Office Chair", "Home", "Furniture", 399.99, 200.00),
    ("Bookshelf", "Home", "Furniture", 199.99, 90.00),
    ("Coffee Maker", "Home", "Kitchen", 89.99, 40.00),
    ("Blender", "Home", "Kitchen", 69.99, 28.00),
    ("Toaster", "Home", "Kitchen", 39.99, 15.00),
    ("Air Purifier", "Home", "Appliances", 179.99, 80.00),
    ("Vacuum Cleaner", "Home", "Appliances", 249.99, 120.00),
    ("Smart Watch", "Electronics", "Wearables", 299.99, 150.00),
    ("Fiction Novel", "Books", "Fiction", 14.99, 5.00),
    ("Sci-Fi Anthology", "Books", "Fiction", 19.99, 7.00),
    ("Cooking Guide", "Books", "Non-Fiction", 29.99, 10.00),
    ("Programming 101", "Books", "Non-Fiction", 49.99, 18.00),
    ("Art Supplies Kit", "Home", "Crafts", 34.99, 14.00),
    ("Notebook Set", "Home", "Stationery", 12.99, 4.00),
    ("Pen Collection", "Home", "Stationery", 24.99, 8.00),
    ("Camera DSLR", "Electronics", "Photography", 799.99, 480.00),
    ("Camera Lens 50mm", "Electronics", "Photography", 249.99, 130.00),
    ("Tripod", "Electronics", "Photography", 59.99, 22.00),
    ("Memory Card 128GB", "Electronics", "Storage", 24.99, 10.00),
    ("External SSD 1TB", "Electronics", "Storage", 109.99, 55.00),
    ("Flash Drive 64GB", "Electronics", "Storage", 14.99, 5.00),
    ("Wireless Charger", "Electronics", "Accessories", 34.99, 12.00),
    ("Power Bank", "Electronics", "Accessories", 44.99, 18.00),
    ("Fitness Tracker", "Electronics", "Wearables", 99.99, 40.00),
    ("Gaming Mouse", "Electronics", "Gaming", 69.99, 28.00),
    ("Gaming Headset", "Electronics", "Gaming", 119.99, 50.00),
    ("Controller", "Electronics", "Gaming", 59.99, 25.00),
    ("Mousepad XL", "Electronics", "Gaming", 19.99, 6.00),
]


def _generate_seed_sql() -> str:
    """Generate deterministic INSERT statements for seed data."""
    rng = random.Random(42)
    stmts: list[str] = []

    # Customers (~100)
    for i in range(1, 101):
        first = rng.choice(_FIRST_NAMES)
        last = rng.choice(_LAST_NAMES)
        name = f"{first} {last}"
        email = f"{first.lower()}.{last.lower()}{i}@example.com"
        city, state = rng.choice(_CITIES)
        year = rng.choice([2022, 2023, 2024])
        month = rng.randint(1, 12)
        day = rng.randint(1, 28)
        created = f"{year}-{month:02d}-{day:02d}"
        stmts.append(
            f"INSERT INTO customers VALUES ({i}, '{name}', '{email}', "
            f"'{city}', '{state}', 'US', '{created}');"
        )

    # Products (from catalog)
    for i, (name, cat, subcat, price, cost) in enumerate(_PRODUCT_CATALOG, 1):
        stmts.append(
            f"INSERT INTO products VALUES ({i}, '{name}', '{cat}', "
            f"'{subcat}', {price}, {cost});"
        )

    # Orders (~500)
    statuses = ["completed", "completed", "completed", "completed", "shipped", "pending", "cancelled"]
    order_id = 0
    item_id = 0
    for _ in range(500):
        order_id += 1
        cust_id = rng.randint(1, 100)
        year = rng.choice([2023, 2024, 2024, 2024, 2025, 2025])
        month = rng.randint(1, 12)
        day = rng.randint(1, 28)
        order_date = f"{year}-{month:02d}-{day:02d}"
        status = rng.choice(statuses)

        # 1-5 items per order
        num_items = rng.randint(1, 5)
        total = 0.0
        item_stmts: list[str] = []
        used_products: set[int] = set()
        for _ in range(num_items):
            prod_id = rng.randint(1, len(_PRODUCT_CATALOG))
            if prod_id in used_products:
                continue
            used_products.add(prod_id)
            qty = rng.randint(1, 3)
            unit_price = _PRODUCT_CATALOG[prod_id - 1][3]
            discount = rng.choice([0.0, 0.0, 0.0, 0.05, 0.10, 0.15])
            line_total = qty * unit_price * (1 - discount)
            total += line_total
            item_id += 1
            item_stmts.append(
                f"INSERT INTO order_items VALUES ({item_id}, {order_id}, "
                f"{prod_id}, {qty}, {unit_price}, {discount});"
            )

        total = round(total, 2)
        stmts.append(
            f"INSERT INTO orders VALUES ({order_id}, {cust_id}, "
            f"'{order_date}', '{status}', {total});"
        )
        stmts.extend(item_stmts)

    return "\n".join(stmts)


def generate_seed_json() -> dict[str, list[list[Any]]]:
    """Generate deterministic seed data as JSON arrays for DuckDB-WASM.

    Returns dict with table names as keys and lists of row-arrays as values.
    Uses the same Random(42) seed as _generate_seed_sql() for identical data.
    """
    rng = random.Random(42)
    customers: list[list[Any]] = []
    products: list[list[Any]] = []
    orders: list[list[Any]] = []
    order_items: list[list[Any]] = []

    for i in range(1, 101):
        first = rng.choice(_FIRST_NAMES)
        last = rng.choice(_LAST_NAMES)
        name = f"{first} {last}"
        email = f"{first.lower()}.{last.lower()}{i}@example.com"
        city, state = rng.choice(_CITIES)
        year = rng.choice([2022, 2023, 2024])
        month = rng.randint(1, 12)
        day = rng.randint(1, 28)
        created = f"{year}-{month:02d}-{day:02d}"
        customers.append([i, name, email, city, state, "US", created])

    for i, (pname, cat, subcat, price, cost) in enumerate(_PRODUCT_CATALOG, 1):
        products.append([i, pname, cat, subcat, price, cost])

    statuses = ["completed", "completed", "completed", "completed", "shipped", "pending", "cancelled"]
    order_id = 0
    item_id = 0
    for _ in range(500):
        order_id += 1
        cust_id = rng.randint(1, 100)
        year = rng.choice([2023, 2024, 2024, 2024, 2025, 2025])
        month = rng.randint(1, 12)
        day = rng.randint(1, 28)
        order_date = f"{year}-{month:02d}-{day:02d}"
        status = rng.choice(statuses)

        num_items = rng.randint(1, 5)
        total = 0.0
        batch: list[list[Any]] = []
        used_products: set[int] = set()
        for _ in range(num_items):
            prod_id = rng.randint(1, len(_PRODUCT_CATALOG))
            if prod_id in used_products:
                continue
            used_products.add(prod_id)
            qty = rng.randint(1, 3)
            unit_price = _PRODUCT_CATALOG[prod_id - 1][3]
            discount = rng.choice([0.0, 0.0, 0.0, 0.05, 0.10, 0.15])
            line_total = qty * unit_price * (1 - discount)
            total += line_total
            item_id += 1
            batch.append([item_id, order_id, prod_id, qty, unit_price, discount])

        total = round(total, 2)
        orders.append([order_id, cust_id, order_date, status, total])
        order_items.extend(batch)

    return {
        "customers": customers,
        "products": products,
        "orders": orders,
        "order_items": order_items,
    }


# ── Database Creation ───────────────────────────────────────────────────────

def create_benchmark_db() -> sqlite3.Connection:
    """Create a fresh in-memory SQLite database with schema and seed data."""
    conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA_DDL)
    conn.executescript(_generate_seed_sql())
    conn.commit()
    return conn


# ── SQL Questions ───────────────────────────────────────────────────────────

@dataclass
class SQLQuestion:
    """A single SQL benchmark question."""
    id: int
    difficulty: str  # "trivial", "easy", "medium", "hard"
    question: str  # Natural language question
    reference_sql: str  # Reference SQL to compute expected result
    order_matters: bool = False  # Whether row order matters for comparison
    hint: str = ""  # Optional hint shown to model

    # Populated at init time by running reference_sql
    expected_columns: list[str] = field(default_factory=list)
    expected_result: list[tuple] = field(default_factory=list)


_RAW_QUESTIONS: list[dict[str, Any]] = [
    # ── Trivial (4) ──────────────────────────────────────────────────────
    {
        "id": 1, "difficulty": "trivial",
        "question": "How many customers are there in total?",
        "reference_sql": "SELECT COUNT(*) AS customer_count FROM customers;",
    },
    {
        "id": 2, "difficulty": "trivial",
        "question": "What is the most expensive product and its price?",
        "reference_sql": "SELECT name, price FROM products ORDER BY price DESC LIMIT 1;",
        "order_matters": True,
    },
    {
        "id": 3, "difficulty": "trivial",
        "question": "List all distinct product categories.",
        "reference_sql": "SELECT DISTINCT category FROM products ORDER BY category;",
        "order_matters": True,
    },
    {
        "id": 4, "difficulty": "trivial",
        "question": "How many orders have the status 'cancelled'?",
        "reference_sql": "SELECT COUNT(*) AS cancelled_count FROM orders WHERE status = 'cancelled';",
    },

    # ── Easy (5) ─────────────────────────────────────────────────────────
    {
        "id": 5, "difficulty": "easy",
        "question": "List all products in the 'Audio' subcategory, sorted by price descending.",
        "reference_sql": "SELECT name, price FROM products WHERE subcategory = 'Audio' ORDER BY price DESC;",
        "order_matters": True,
    },
    {
        "id": 6, "difficulty": "easy",
        "question": "How many orders were placed in 2024?",
        "reference_sql": "SELECT COUNT(*) AS order_count FROM orders WHERE order_date LIKE '2024%';",
    },
    {
        "id": 7, "difficulty": "easy",
        "question": "What are the top 5 most expensive products? Show name and price.",
        "reference_sql": "SELECT name, price FROM products ORDER BY price DESC LIMIT 5;",
        "order_matters": True,
    },
    {
        "id": 8, "difficulty": "easy",
        "question": "Find all customers from Texas (state = 'TX'). Show name and city.",
        "reference_sql": "SELECT name, city FROM customers WHERE state = 'TX' ORDER BY name;",
        "order_matters": True,
    },
    {
        "id": 9, "difficulty": "easy",
        "question": "What is the average order total amount?",
        "reference_sql": "SELECT ROUND(AVG(total_amount), 2) AS avg_order_total FROM orders;",
    },

    # ── Medium (6) ───────────────────────────────────────────────────────
    {
        "id": 10, "difficulty": "medium",
        "question": "Show the total revenue (sum of total_amount) per order status.",
        "reference_sql": (
            "SELECT status, ROUND(SUM(total_amount), 2) AS total_revenue "
            "FROM orders GROUP BY status ORDER BY total_revenue DESC;"
        ),
        "order_matters": True,
    },
    {
        "id": 11, "difficulty": "medium",
        "question": "List the top 5 customers by total spending. Show customer name and total spent.",
        "reference_sql": (
            "SELECT c.name, ROUND(SUM(o.total_amount), 2) AS total_spent "
            "FROM customers c JOIN orders o ON c.id = o.customer_id "
            "GROUP BY c.id, c.name ORDER BY total_spent DESC LIMIT 5;"
        ),
        "order_matters": True,
    },
    {
        "id": 12, "difficulty": "medium",
        "question": "Which product categories have more than 10 products? Show category and count.",
        "reference_sql": (
            "SELECT category, COUNT(*) AS product_count FROM products "
            "GROUP BY category HAVING COUNT(*) > 10 ORDER BY product_count DESC;"
        ),
        "order_matters": True,
    },
    {
        "id": 13, "difficulty": "medium",
        "question": "Show the number of orders placed per month in 2024. Display month and count.",
        "reference_sql": (
            "SELECT SUBSTR(order_date, 1, 7) AS month, COUNT(*) AS order_count "
            "FROM orders WHERE order_date LIKE '2024%' "
            "GROUP BY SUBSTR(order_date, 1, 7) ORDER BY month;"
        ),
        "order_matters": True,
    },
    {
        "id": 14, "difficulty": "medium",
        "question": "Find customers who have placed more than 5 orders. Show name and order count.",
        "reference_sql": (
            "SELECT c.name, COUNT(o.id) AS order_count "
            "FROM customers c JOIN orders o ON c.id = o.customer_id "
            "GROUP BY c.id, c.name HAVING COUNT(o.id) > 5 "
            "ORDER BY order_count DESC;"
        ),
        "order_matters": True,
    },
    {
        "id": 15, "difficulty": "medium",
        "question": (
            "What is the most frequently ordered product? "
            "Show product name and the total quantity ordered."
        ),
        "reference_sql": (
            "SELECT p.name, SUM(oi.quantity) AS total_qty "
            "FROM order_items oi JOIN products p ON oi.product_id = p.id "
            "GROUP BY p.id, p.name ORDER BY total_qty DESC LIMIT 1;"
        ),
        "order_matters": True,
    },

    # ── Hard (5) ─────────────────────────────────────────────────────────
    {
        "id": 16, "difficulty": "hard",
        "question": (
            "Calculate the revenue and profit margin for each product subcategory. "
            "Revenue = SUM(quantity * unit_price * (1 - discount)). "
            "Margin = (Revenue - Total Cost) / Revenue * 100 where Total Cost = SUM(quantity * product cost). "
            "Show subcategory, revenue (rounded to 2 decimals), and margin percentage (rounded to 1 decimal). "
            "Order by revenue descending."
        ),
        "reference_sql": (
            "SELECT p.subcategory, "
            "ROUND(SUM(oi.quantity * oi.unit_price * (1 - oi.discount)), 2) AS revenue, "
            "ROUND((SUM(oi.quantity * oi.unit_price * (1 - oi.discount)) - SUM(oi.quantity * p.cost)) "
            "/ SUM(oi.quantity * oi.unit_price * (1 - oi.discount)) * 100, 1) AS margin_pct "
            "FROM order_items oi JOIN products p ON oi.product_id = p.id "
            "GROUP BY p.subcategory ORDER BY revenue DESC;"
        ),
        "order_matters": True,
    },
    {
        "id": 17, "difficulty": "hard",
        "question": (
            "Find the month-over-month growth rate of order count for 2024. "
            "Show month, order count, and growth rate as a percentage (rounded to 1 decimal). "
            "The first month should show NULL or 0 for growth rate."
        ),
        "reference_sql": (
            "WITH monthly AS ("
            "  SELECT SUBSTR(order_date, 1, 7) AS month, COUNT(*) AS cnt "
            "  FROM orders WHERE order_date LIKE '2024%' "
            "  GROUP BY SUBSTR(order_date, 1, 7)"
            ") "
            "SELECT m.month, m.cnt AS order_count, "
            "CASE WHEN LAG(m.cnt) OVER (ORDER BY m.month) IS NULL THEN NULL "
            "ELSE ROUND((CAST(m.cnt AS REAL) - LAG(m.cnt) OVER (ORDER BY m.month)) "
            "/ LAG(m.cnt) OVER (ORDER BY m.month) * 100, 1) END AS growth_rate "
            "FROM monthly m ORDER BY m.month;"
        ),
        "order_matters": True,
    },
    {
        "id": 18, "difficulty": "hard",
        "question": (
            "For each customer, calculate their total spend and rank them. "
            "Show customer name, total spend (rounded to 2 decimals), "
            "and their rank (1 = highest spender). Only show top 10."
        ),
        "reference_sql": (
            "SELECT c.name, ROUND(SUM(o.total_amount), 2) AS total_spend, "
            "RANK() OVER (ORDER BY SUM(o.total_amount) DESC) AS spend_rank "
            "FROM customers c JOIN orders o ON c.id = o.customer_id "
            "GROUP BY c.id, c.name ORDER BY total_spend DESC LIMIT 10;"
        ),
        "order_matters": True,
    },
    {
        "id": 19, "difficulty": "hard",
        "question": (
            "Find products that have never been ordered. Show product name and category."
        ),
        "reference_sql": (
            "SELECT p.name, p.category FROM products p "
            "LEFT JOIN order_items oi ON p.id = oi.product_id "
            "WHERE oi.id IS NULL ORDER BY p.name;"
        ),
        "order_matters": True,
    },
    {
        "id": 20, "difficulty": "hard",
        "question": (
            "Calculate the average number of days between a customer's first and last order "
            "for customers who have placed at least 3 orders. "
            "Show customer name, first order date, last order date, days between, and order count. "
            "Order by days between descending."
        ),
        "reference_sql": (
            "SELECT c.name, MIN(o.order_date) AS first_order, MAX(o.order_date) AS last_order, "
            "CAST(JULIANDAY(MAX(o.order_date)) - JULIANDAY(MIN(o.order_date)) AS INTEGER) AS days_between, "
            "COUNT(o.id) AS order_count "
            "FROM customers c JOIN orders o ON c.id = o.customer_id "
            "GROUP BY c.id, c.name HAVING COUNT(o.id) >= 3 "
            "ORDER BY days_between DESC;"
        ),
        "order_matters": True,
    },

    # ── Additional Medium / Hard (5) ──────────────────────────────────────
    {
        "id": 21, "difficulty": "medium",
        "question": (
            "Combine a list of all product names from the 'Electronics' category and all product names "
            "from the 'Books' category into a single list. Show product name and category, ordered by name."
        ),
        "reference_sql": (
            "SELECT name, category FROM products WHERE category IN ('Electronics', 'Books') ORDER BY name;"
        ),
        "order_matters": True,
    },
    {
        "id": 22, "difficulty": "hard",
        "question": (
            "Find customers who have ordered products from at least 3 different product categories. "
            "Show customer name and the number of distinct categories they've ordered from, "
            "ordered by category count descending."
        ),
        "reference_sql": (
            "SELECT c.name, COUNT(DISTINCT p.category) AS category_count "
            "FROM customers c "
            "JOIN orders o ON c.id = o.customer_id "
            "JOIN order_items oi ON o.id = oi.order_id "
            "JOIN products p ON oi.product_id = p.id "
            "GROUP BY c.id, c.name "
            "HAVING COUNT(DISTINCT p.category) >= 3 "
            "ORDER BY category_count DESC;"
        ),
        "order_matters": True,
    },
    {
        "id": 23, "difficulty": "medium",
        "question": (
            "Categorize customers into spending tiers based on their total order amount: "
            "'VIP' for total > $5000, 'Regular' for total $1000-$5000, 'New' for total < $1000. "
            "Show customer name, total spent (rounded to 2 decimals), and tier. "
            "Order by total spent descending."
        ),
        "reference_sql": (
            "SELECT c.name, ROUND(SUM(o.total_amount), 2) AS total_spent, "
            "CASE WHEN SUM(o.total_amount) > 5000 THEN 'VIP' "
            "WHEN SUM(o.total_amount) >= 1000 THEN 'Regular' "
            "ELSE 'New' END AS tier "
            "FROM customers c JOIN orders o ON c.id = o.customer_id "
            "GROUP BY c.id, c.name ORDER BY total_spent DESC;"
        ),
        "order_matters": True,
    },
    {
        "id": 24, "difficulty": "medium",
        "question": (
            "Find pairs of customers from the same city who have both placed orders. "
            "Show both customer names and the city. Each pair should appear only once "
            "(no duplicates like A-B and B-A). Order by city, then first customer name."
        ),
        "reference_sql": (
            "SELECT c1.name AS customer1, c2.name AS customer2, c1.city "
            "FROM customers c1 "
            "JOIN customers c2 ON c1.city = c2.city AND c1.id < c2.id "
            "JOIN orders o1 ON c1.id = o1.customer_id "
            "JOIN orders o2 ON c2.id = o2.customer_id "
            "GROUP BY c1.id, c2.id, c1.name, c2.name, c1.city "
            "ORDER BY c1.city, c1.name;"
        ),
        "order_matters": True,
    },
    {
        "id": 25, "difficulty": "hard",
        "question": (
            "Calculate a running total of order amounts per customer, ordered by order date. "
            "Show customer name, order date, order total, and the cumulative total (running sum) "
            "for that customer. Only include customers with at least 3 orders. "
            "Order by customer name, then order date."
        ),
        "reference_sql": (
            "SELECT c.name, o.order_date, o.total_amount, "
            "SUM(o.total_amount) OVER (PARTITION BY c.id ORDER BY o.order_date, o.id) AS running_total "
            "FROM customers c JOIN orders o ON c.id = o.customer_id "
            "WHERE c.id IN (SELECT customer_id FROM orders GROUP BY customer_id HAVING COUNT(*) >= 3) "
            "ORDER BY c.name, o.order_date, o.id;"
        ),
        "order_matters": True,
    },
]


def _init_questions() -> list[SQLQuestion]:
    """Build SQLQuestion list with pre-computed expected results."""
    conn = create_benchmark_db()
    questions: list[SQLQuestion] = []
    try:
        for raw in _RAW_QUESTIONS:
            cursor = conn.execute(raw["reference_sql"])
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            q = SQLQuestion(
                id=raw["id"],
                difficulty=raw["difficulty"],
                question=raw["question"],
                reference_sql=raw["reference_sql"],
                order_matters=raw.get("order_matters", False),
                hint=raw.get("hint", ""),
                expected_columns=columns,
                expected_result=rows,
            )
            questions.append(q)
    finally:
        conn.close()
    return questions


QUESTIONS: list[SQLQuestion] = _init_questions()


# ── SQL Extraction ──────────────────────────────────────────────────────────

def extract_sql(text: str) -> str:
    """Extract SQL from model response (markdown fences or raw text)."""
    # Strip thinking tags
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Try ```sql ... ``` fences first
    matches = re.findall(r"```(?:sql)?\s*\n(.*?)```", text, re.DOTALL)
    if matches:
        return matches[-1].strip()  # Take last SQL block (most likely the final answer)

    # Try to find SELECT statement
    m = re.search(r"(SELECT\b.+?)(?:;|\Z)", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip().rstrip(";") + ";"

    return text.strip()


# ── Safe Execution ──────────────────────────────────────────────────────────

_BLOCKED_PREFIXES = ("INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "ATTACH", "DETACH", "PRAGMA")


def execute_sql_safe(
    conn: sqlite3.Connection, sql: str, timeout: float = 5.0
) -> tuple[list[str], list[tuple]]:
    """Execute a SQL query safely. Only SELECT is allowed.

    Returns (column_names, rows).
    Raises ValueError for blocked statements, sqlite3.Error for execution errors.
    """
    stripped = sql.strip().upper()
    for prefix in _BLOCKED_PREFIXES:
        if stripped.startswith(prefix):
            raise ValueError(f"Blocked SQL statement: {prefix}")

    # Use a timeout by setting busy_timeout (though in-memory DBs don't really block)
    conn.execute(f"PRAGMA busy_timeout = {int(timeout * 1000)}")

    cursor = conn.execute(sql)
    columns = [desc[0].lower() for desc in cursor.description] if cursor.description else []
    rows = cursor.fetchall()
    return columns, rows


# ── SQL Complexity Analysis ──────────────────────────────────────────────────

def _analyze_sql_complexity(sql: str) -> dict[str, int]:
    """Count SQL complexity indicators."""
    upper = sql.upper()
    return {
        "joins": upper.count("JOIN"),
        "subqueries": upper.count("SELECT") - 1,
        "window_functions": len(re.findall(r"\bOVER\s*\(", upper)),
        "case_when": upper.count("CASE"),
        "ctes": upper.count("WITH"),
        "group_by": 1 if "GROUP BY" in upper else 0,
    }


# ── Result Comparison ───────────────────────────────────────────────────────

# Common column alias equivalences: map alias -> canonical name
_COLUMN_ALIASES: dict[str, str] = {
    "cnt": "count",
    "num": "count",
    "qty": "quantity",
    "amt": "amount",
    "pct": "percent",
}


def _normalize_col_name(col: str) -> str:
    """Normalize a column name for loose comparison.

    Strips underscores and lowercases, then applies common alias mappings.
    """
    normalized = col.lower().replace("_", "")
    return _COLUMN_ALIASES.get(normalized, normalized)


def _normalize_value(v: Any) -> Any:
    """Normalize a value for comparison."""
    if v is None:
        return None
    if isinstance(v, float):
        return round(v, 2)
    if isinstance(v, str):
        return v.strip().lower()
    return v


def _normalize_row(row: tuple) -> tuple:
    return tuple(_normalize_value(v) for v in row)


def _values_close(a: Any, b: Any, rel_tol: float = 0.01) -> bool:
    """Return True if two values are equivalent, using 1% relative tolerance for numerics."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    # Both numeric: use relative tolerance
    try:
        fa, fb = float(a), float(b)
        if fa == fb:
            return True
        if fa == 0.0:
            return abs(fb) <= rel_tol
        return abs(fa - fb) / abs(fa) <= rel_tol
    except (TypeError, ValueError):
        pass
    return a == b


def _rows_close(exp_row: tuple, act_row: tuple) -> bool:
    """Return True if all values in the row are within tolerance."""
    if len(exp_row) != len(act_row):
        return False
    return all(_values_close(e, a) for e, a in zip(exp_row, act_row))


def compare_results(
    expected_cols: list[str],
    expected_rows: list[tuple],
    actual_cols: list[str],
    actual_rows: list[tuple],
    order_matters: bool = False,
) -> tuple[bool, str]:
    """Compare expected and actual SQL results.

    Column name checking is relaxed: if column *count* matches, column names
    are not required to match exactly (models often use different aliases).
    Numeric value comparison uses 1% relative tolerance.

    Returns (match: bool, reason: str).
    """
    # Check column count matches (names are not required to be identical)
    if len(expected_cols) != len(actual_cols):
        return False, f"Column count mismatch: expected {len(expected_cols)}, got {len(actual_cols)}"

    # Check row count
    if len(expected_rows) != len(actual_rows):
        return False, f"Row count mismatch: expected {len(expected_rows)}, got {len(actual_rows)}"

    # Normalize rows for string comparison (lowercased, stripped)
    exp_normalized = [_normalize_row(r) for r in expected_rows]
    act_normalized = [_normalize_row(r) for r in actual_rows]

    if order_matters:
        for i, (exp_row, act_row) in enumerate(zip(exp_normalized, act_normalized)):
            if exp_row != act_row and not _rows_close(exp_row, act_row):
                return False, f"Row {i} mismatch: expected {exp_row}, got {act_row}"
        return True, ""
    else:
        # Unordered: try sorted comparison first, then tolerance-based
        if sorted(str(r) for r in exp_normalized) == sorted(str(r) for r in act_normalized):
            return True, ""
        # Fallback: check each expected row has a matching actual row within tolerance
        unmatched_exp = list(exp_normalized)
        unmatched_act = list(act_normalized)
        for exp_row in exp_normalized:
            for act_row in unmatched_act:
                if _rows_close(exp_row, act_row):
                    unmatched_act.remove(act_row)
                    unmatched_exp.remove(exp_row)
                    break
        if not unmatched_exp and not unmatched_act:
            return True, ""
        return False, "Result sets differ (unordered comparison)"


# ── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class SQLTaskResult:
    """Result of a single SQL benchmark question."""
    question_id: int
    question: str
    difficulty: str
    model: str
    status: str = "error"  # "pass", "fail", "error"
    score: float = 0.0  # 1.0 = pass, 0.5 = partial credit, 0.0 = fail/error
    generated_sql: str = ""
    actual_result: list[tuple] | None = None
    expected_result: list[tuple] | None = None
    expected_columns: list[str] | None = None
    actual_columns: list[str] | None = None
    error_message: str = ""
    mismatch_reason: str = ""
    speed_tps: float = 0.0
    ttft_ms: float = 0.0
    total_ms: float = 0.0
    retries: int = 0
    complexity: dict[str, int] = field(default_factory=dict)


# ── Benchmark Runner ────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a SQL expert. Given a database schema and a natural language question, "
    "generate a single SQLite-compatible SELECT query. Return ONLY the SQL query "
    "inside ```sql``` fences. Do not explain, do not add comments."
)

_RETRY_TEMPLATE = (
    "Your previous SQL query was incorrect.\n\n"
    "Previous SQL:\n```sql\n{prev_sql}\n```\n\n"
    "Error: {error}\n\n"
    "Please generate a corrected SQL query. "
    "Return ONLY the SQL inside ```sql``` fences."
)


def _build_prompt(question: SQLQuestion) -> str:
    """Build the user prompt with schema + question."""
    return (
        f"Database schema:\n```sql\n{SCHEMA_DDL}```\n\n"
        f"{SCHEMA_DESCRIPTION}\n"
        f"Question: {question.question}"
        + (f"\nHint: {question.hint}" if question.hint else "")
    )


def run_sql_question(
    conn: sqlite3.Connection,
    question: SQLQuestion,
    base_url: str,
    model: str,
    api_key: str | None = None,
    max_retries: int = 2,
) -> SQLTaskResult:
    """Run a single SQL question with agentic retry loop."""
    prompt = _build_prompt(question)
    retries = 0
    last_sql = ""
    last_error = ""
    last_tps = 0.0
    last_ttft = 0.0
    last_total_ms = 0.0
    last_actual_cols: list[str] | None = None
    last_actual_rows: list[tuple] | None = None

    for attempt in range(1 + max_retries):
        if attempt > 0:
            retries = attempt
            # Build retry prompt with error context
            prompt = (
                f"Database schema:\n```sql\n{SCHEMA_DDL}```\n\n"
                f"{SCHEMA_DESCRIPTION}\n"
                f"Question: {question.question}\n\n"
                + _RETRY_TEMPLATE.format(prev_sql=last_sql, error=last_error)
            )

        try:
            response, tps, ttft, total_ms = call_openai_api(
                base_url, model, prompt, api_key=api_key,
                max_tokens=1024, temperature=0.3,
                system_prompt=_SYSTEM_PROMPT,
            )
        except Exception as e:
            return SQLTaskResult(
                question_id=question.id, question=question.question,
                difficulty=question.difficulty, model=model,
                status="error", score=0.0, error_message=f"API error: {e}",
                retries=retries,
            )

        sql = extract_sql(response)
        last_sql = sql
        last_tps = tps
        last_ttft = ttft
        last_total_ms = total_ms

        if not sql or not sql.strip().upper().startswith("SELECT"):
            last_error = "No valid SELECT statement found in response"
            continue

        try:
            actual_cols, actual_rows = execute_sql_safe(conn, sql)
            last_actual_cols = actual_cols
            last_actual_rows = actual_rows
        except (ValueError, sqlite3.Error) as e:
            last_error = str(e)
            continue

        match, reason = compare_results(
            question.expected_columns, question.expected_result,
            actual_cols, actual_rows,
            order_matters=question.order_matters,
        )

        if match:
            return SQLTaskResult(
                question_id=question.id, question=question.question,
                difficulty=question.difficulty, model=model,
                status="pass", score=1.0, generated_sql=sql,
                actual_result=actual_rows, actual_columns=actual_cols,
                expected_result=question.expected_result,
                expected_columns=question.expected_columns,
                speed_tps=tps, ttft_ms=ttft, total_ms=total_ms,
                retries=retries,
                complexity=_analyze_sql_complexity(sql),
            )
        else:
            last_error = reason

    # All attempts exhausted — check for partial credit
    status = "fail" if last_sql and last_error else "error"
    score = 0.0
    if status == "fail" and last_actual_rows is not None:
        # Partial credit: row count matches but values differ
        if len(last_actual_rows) == len(question.expected_result):
            score = 0.5

    return SQLTaskResult(
        question_id=question.id, question=question.question,
        difficulty=question.difficulty, model=model,
        status=status, score=score, generated_sql=last_sql,
        actual_result=last_actual_rows,
        actual_columns=last_actual_cols,
        expected_result=question.expected_result,
        expected_columns=question.expected_columns,
        error_message=last_error, mismatch_reason=last_error,
        speed_tps=last_tps, ttft_ms=last_ttft, total_ms=last_total_ms,
        retries=retries,
        complexity=_analyze_sql_complexity(last_sql) if last_sql else {},
    )


def run_sql_benchmark(
    base_url: str,
    models: list[str],
    api_key: str | None = None,
    runtime_name: str = "local",
    difficulties: list[str] | None = None,
    max_retries: int = 2,
    progress_callback: Any = None,
) -> list[SQLTaskResult]:
    """Run SQL benchmark against specified models.

    Args:
        base_url: OpenAI-compatible API base URL.
        models: List of model IDs to test.
        api_key: Optional API key.
        runtime_name: Runtime name for display.
        difficulties: Filter by difficulty tier (None = all).
        max_retries: Max retries per question.
        progress_callback: Called with (model_idx, total_models, question_idx, total_questions).

    Returns:
        List of SQLTaskResult for all models and questions.
    """
    questions = QUESTIONS
    if difficulties:
        valid = {d.lower() for d in difficulties}
        questions = [q for q in QUESTIONS if q.difficulty in valid]

    results: list[SQLTaskResult] = []
    total_q = len(questions)

    for m_idx, model in enumerate(models):
        print(f"\n{'='*60}")
        print(f"  Model: {model}")
        print(f"  Questions: {total_q} ({', '.join(sorted(set(q.difficulty for q in questions)))})")
        print(f"{'='*60}")

        conn = create_benchmark_db()
        try:
            for q_idx, question in enumerate(questions):
                if progress_callback:
                    progress_callback(m_idx, len(models), q_idx, total_q)

                tier_label = f"[{question.difficulty:>7}]"
                print(f"  {tier_label} Q{question.id}: {question.question[:55]}...", end=" ", flush=True)

                result = run_sql_question(
                    conn, question, base_url, model,
                    api_key=api_key, max_retries=max_retries,
                )
                results.append(result)

                status_color = {"pass": "\033[32m", "fail": "\033[31m", "error": "\033[33m"}
                color = status_color.get(result.status, "")
                reset = "\033[0m"
                retry_info = f" (retries={result.retries})" if result.retries > 0 else ""
                speed_info = f" {result.speed_tps:.0f}t/s" if result.speed_tps > 0 else ""
                print(f"{color}{result.status.upper()}{reset}{speed_info}{retry_info}")

                if result.error_message:
                    print(f"         {result.error_message[:80]}")
        finally:
            conn.close()

    return results


# ── Summary ─────────────────────────────────────────────────────────────────

DIFFICULTY_ORDER = {"trivial": 0, "easy": 1, "medium": 2, "hard": 3}


def print_sql_summary(results: list[SQLTaskResult]) -> None:
    """Print SQL benchmark summary table."""
    if not results:
        print("  No results.")
        return

    models = sorted(set(r.model for r in results))
    difficulties = sorted(set(r.difficulty for r in results), key=lambda d: DIFFICULTY_ORDER.get(d, 99))

    print(f"\n{'='*80}")
    print("  SQL BENCHMARK RESULTS")
    print(f"{'='*80}\n")

    # Summary per model
    header = f"{'Model':<40} {'Score':>7}"
    for d in difficulties:
        header += f" {d.capitalize():>8}"
    header += f" {'TPS':>7} {'TTFT':>7}"
    print(header)
    print("-" * len(header))

    for model in models:
        mr = [r for r in results if r.model == model]
        total_pass = sum(1 for r in mr if r.status == "pass")
        total_q = len(mr)

        line = f"{model[:38]:<40} {total_pass}/{total_q:>3}"
        for d in difficulties:
            dr = [r for r in mr if r.difficulty == d]
            d_pass = sum(1 for r in dr if r.status == "pass")
            line += f" {d_pass}/{len(dr):>5}"

        tps_vals = [r.speed_tps for r in mr if r.speed_tps > 0]
        avg_tps = sum(tps_vals) / len(tps_vals) if tps_vals else 0
        ttft_vals = [r.ttft_ms for r in mr if r.ttft_ms > 0]
        avg_ttft = sum(ttft_vals) / len(ttft_vals) if ttft_vals else 0

        line += f" {avg_tps:>6.0f} {avg_ttft:>6.0f}ms"
        print(line)

    # Per-question heatmap (text-based)
    print(f"\n  Question Heatmap:")
    print(f"  {'Q#':<5}", end="")
    for model in models:
        print(f" {model[:15]:>15}", end="")
    print()
    print("  " + "-" * (5 + 16 * len(models)))

    questions_in_results = sorted(
        set((r.question_id, r.difficulty) for r in results),
        key=lambda x: (DIFFICULTY_ORDER.get(x[1], 99), x[0]),
    )
    for qid, diff in questions_in_results:
        diff_short = diff[0].upper()
        print(f"  {diff_short}{qid:<4}", end="")
        for model in models:
            r = next((r for r in results if r.model == model and r.question_id == qid), None)
            if r is None:
                print(f" {'--':>15}", end="")
            else:
                color = {"pass": "\033[32m", "fail": "\033[31m", "error": "\033[33m"}.get(r.status, "")
                reset = "\033[0m"
                print(f" {color}{r.status.upper():>15}{reset}", end="")
        print()


# ── Result Persistence ──────────────────────────────────────────────────────

def results_to_record(
    results: list[SQLTaskResult],
    runtime_name: str,
) -> "BenchmarkRecord":
    """Convert SQLTaskResults to a BenchmarkRecord for saving."""
    from .benchmark_results import BenchmarkRecord

    if not results:
        return BenchmarkRecord(type="sql", model="unknown", runtime=runtime_name)

    model = results[0].model
    total_pass = sum(1 for r in results if r.status == "pass")
    total_fail = sum(1 for r in results if r.status == "fail")
    total_error = sum(1 for r in results if r.status == "error")
    total_q = len(results)

    by_difficulty: dict[str, dict[str, int]] = {}
    for r in results:
        if r.difficulty not in by_difficulty:
            by_difficulty[r.difficulty] = {"pass": 0, "fail": 0, "error": 0, "total": 0}
        by_difficulty[r.difficulty][r.status] = by_difficulty[r.difficulty].get(r.status, 0) + 1
        by_difficulty[r.difficulty]["total"] += 1

    tps_vals = [r.speed_tps for r in results if r.speed_tps > 0]
    avg_tps = sum(tps_vals) / len(tps_vals) if tps_vals else 0

    questions_detail = [
        {
            "id": r.question_id,
            "difficulty": r.difficulty,
            "status": r.status,
            "score": r.score,
            "sql": r.generated_sql,
            "tps": r.speed_tps,
            "ttft_ms": r.ttft_ms,
            "retries": r.retries,
            "error": r.error_message,
        }
        for r in results
    ]

    return BenchmarkRecord(
        type="sql",
        model=model,
        runtime=runtime_name,
        tokens_per_second=avg_tps,
        quality_scores={
            "total_pass": total_pass,
            "total_fail": total_fail,
            "total_error": total_error,
            "total_questions": total_q,
            "pass_rate": total_pass / total_q if total_q else 0,
            "by_difficulty": by_difficulty,
            "questions": questions_detail,
        },
    )
