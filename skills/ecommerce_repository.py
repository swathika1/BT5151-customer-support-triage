from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional


BASE_DIR = Path(__file__).resolve().parent.parent
ECOMMERCE_DATA_DIR = BASE_DIR / "ecommerce_data"
CUSTOMERS_DATA_PATH = ECOMMERCE_DATA_DIR / "ecommerce_customers.csv"
ORDERS_DATA_PATH = ECOMMERCE_DATA_DIR / "ecommerce_orders.csv"
SELLERS_DATA_PATH = ECOMMERCE_DATA_DIR / "ecommerce_sellers.csv"
TRANSPORTERS_DATA_PATH = ECOMMERCE_DATA_DIR / "ecommerce_transporters.csv"
PRODUCTS_DATA_PATH = ECOMMERCE_DATA_DIR / "ecommerce_products.csv"


def normalize_column_name(name: str) -> str:
    """Convert raw CSV column names into stable snake_case keys."""
    normalized = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")
    aliases = {
        "cid": "customer_id",
        "registered_email": "registered_email",
        "registered_phone_number": "registered_phone_number",
        "prime_subscription_flag": "prime_subscription_flag",
        "prime_subscription_flag_1_0": "prime_subscription_flag",
        "order_summary_json": "order_summary",
        "damage_1_0": "damage_flag",
        "applied_for_return_1_0": "applied_for_return",
        "return_claim_accepted_1_0": "return_claim_accepted",
        "eligible_refund_0_to_1": "eligible_refund_ratio",
        "exp_delivery_date": "expected_delivery_date",
    }
    return aliases.get(normalized, normalized)


def clean_csv_value(value: Any) -> str:
    """Normalize CSV values into clean strings."""
    if value is None:
        return ""
    return str(value).strip()


def to_bool_flag(value: Any) -> bool:
    """Interpret common CSV boolean flags."""
    return clean_csv_value(value).lower() in {"1", "true", "yes", "y"}


def to_float_value(value: Any) -> float:
    """Convert numeric CSV values to floats safely."""
    try:
        return float(clean_csv_value(value) or 0)
    except ValueError:
        return 0.0


def parse_flexible_datetime(value: Any) -> Optional[datetime]:
    """Parse dates used across the seeded datasets."""
    raw = clean_csv_value(value)
    if not raw:
        return None

    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%d/%m/%y %H:%M",
        "%d/%m/%y",
        "%m/%d/%Y",
        "%m/%d/%y",
    ):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue

    return None


def format_context_date(value: Any) -> str:
    """Format a CSV date string into a more readable label."""
    parsed = parse_flexible_datetime(value)
    if parsed is None:
        return clean_csv_value(value) or "n/a"
    return parsed.strftime("%d %b %Y")


def _normalize_row(row: dict[str, Any]) -> dict[str, str]:
    return {
        normalize_column_name(key): clean_csv_value(value)
        for key, value in row.items()
    }


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [_normalize_row(row) for row in reader]


@lru_cache(maxsize=1)
def load_products() -> dict[str, dict]:
    """Load product metadata keyed by product ID."""
    products: dict[str, dict] = {}
    for row in _load_csv_rows(PRODUCTS_DATA_PATH):
        product_id = row.get("product_id", "")
        if not product_id:
            continue
        products[product_id] = {
            "product_id": product_id,
            "product_name": row.get("product_name", ""),
            "category": row.get("category", ""),
            "base_price": to_float_value(row.get("base_price")),
        }
    return products


@lru_cache(maxsize=1)
def load_sellers() -> dict[str, dict]:
    """Load seller metadata keyed by seller ID."""
    sellers: dict[str, dict] = {}
    for row in _load_csv_rows(SELLERS_DATA_PATH):
        seller_id = row.get("seller_id", "")
        if not seller_id:
            continue
        sellers[seller_id] = {
            "seller_id": seller_id,
            "seller_name": row.get("seller_name", ""),
            "seller_city": row.get("seller_city", ""),
            "country": row.get("country", ""),
            "website": row.get("website", ""),
            "email": row.get("email", ""),
        }
    return sellers


@lru_cache(maxsize=1)
def load_transporters() -> dict[str, dict]:
    """Load transporter metadata keyed by transporter ID."""
    transporters: dict[str, dict] = {}
    for row in _load_csv_rows(TRANSPORTERS_DATA_PATH):
        transporter_id = row.get("transporter_id", "")
        if not transporter_id:
            continue
        transporters[transporter_id] = {
            "transporter_id": transporter_id,
            "name": row.get("name", ""),
            "website": row.get("website", ""),
            "email": row.get("email", ""),
            "country": row.get("country", ""),
        }
    return transporters


def parse_order_summary(raw_summary: str, products: Optional[dict[str, dict]] = None) -> list[dict]:
    """Parse nested order summary JSON into a stable list of items."""
    if not raw_summary:
        return []

    try:
        summary = json.loads(raw_summary)
    except json.JSONDecodeError:
        return []

    products = products or load_products()
    items = []
    for product_id, payload in summary.items():
        product = products.get(product_id, {})
        items.append({
            "product_id": product_id,
            "product_name": clean_csv_value(payload.get("Prod Name")) or product.get("product_name", product_id),
            "quantity": int(payload.get("Prod Qty", 0) or 0),
            "unit_price": to_float_value(payload.get("Prod Unit Price")),
            "line_amount": to_float_value(payload.get("Line Amount")),
            "category": product.get("category", ""),
        })

    return items


def format_order_summary(order: dict, max_items: int = 3) -> str:
    """Format an order's items into a compact human-readable summary."""
    items = order.get("items", [])
    if not items:
        return clean_csv_value(order.get("order_summary", "")) or "No item summary available"

    parts = [
        f"{item.get('product_name', 'Item')} x{item.get('quantity', 0)}"
        for item in items[:max_items]
    ]
    if len(items) > max_items:
        parts.append(f"+{len(items) - max_items} more")
    return ", ".join(parts)


def get_seller_details(seller_id: str) -> dict:
    """Return seller metadata for the provided seller ID."""
    seller_id = clean_csv_value(seller_id)
    return dict(load_sellers().get(seller_id, {}))


def get_transporter_details(transporter_id: str) -> dict:
    """Return transporter metadata for the provided transporter ID."""
    transporter_id = clean_csv_value(transporter_id)
    return dict(load_transporters().get(transporter_id, {}))


@lru_cache(maxsize=1)
def load_customers() -> dict[str, dict]:
    """Load customer profiles from the structured e-commerce dataset."""
    customers: dict[str, dict] = {}
    for row in _load_csv_rows(CUSTOMERS_DATA_PATH):
        customer_id = row.get("customer_id", "")
        if not customer_id:
            continue

        customers[customer_id] = {
            **row,
            "customer_id": customer_id,
            "name": row.get("name", f"Customer {customer_id}"),
            "prime_subscription_flag": to_bool_flag(row.get("prime_subscription_flag")),
        }

    return customers


def _build_order_record(
    row: dict[str, str],
    products: dict[str, dict],
    sellers: dict[str, dict],
    transporters: dict[str, dict],
) -> dict:
    customer_id = row.get("customer_id", "")
    order_id = row.get("order_id", "")
    seller_id = row.get("seller_id", "")
    transporter_id = row.get("transporter_id", "")
    items = parse_order_summary(row.get("order_summary", ""), products=products)

    seller = dict(sellers.get(seller_id, {}))
    transporter = dict(transporters.get(transporter_id, {}))
    payment = {
        "payment_id": row.get("payment_id", ""),
        "payment_mode": row.get("payment_mode", ""),
        "payment_status": row.get("payment_status", ""),
    }
    delivery = {
        "delivery_id": row.get("delivery_id", ""),
        "seller_dispatch_date": row.get("seller_dispatch_date", ""),
        "expected_delivery_date": row.get("expected_delivery_date", row.get("exp_delivery_date", "")),
        "delivery_status": row.get("delivery_status", ""),
        "actual_delivery_date": row.get("actual_delivery_date", ""),
        "transporter_id": transporter_id,
        "transporter": transporter,
    }
    refund = {
        "refund_id": row.get("refund_id", ""),
        "refund_status": row.get("refund_status", ""),
        "eligible_refund_ratio": to_float_value(row.get("eligible_refund_ratio")),
        "expected_refund_amount": to_float_value(row.get("expected_refund_amount")),
        "expected_refund_date": row.get("expected_refund_date", ""),
        "actual_refund_date": row.get("actual_refund_date", ""),
    }
    return_data = {
        "applied_for_return": to_bool_flag(row.get("applied_for_return")),
        "return_request_date": row.get("return_request_date", ""),
        "return_claim_accepted": to_bool_flag(row.get("return_claim_accepted")),
        "return_status": row.get("return_status", ""),
        "damage_flag": to_bool_flag(row.get("damage_flag")),
        "return_reason": row.get("return_reason", ""),
    }
    replacement = {
        "replacement_flag": to_bool_flag(row.get("replacement_flag")),
        "replacement_id": row.get("replacement_id", ""),
        "replacement_dispatch_date": row.get("replacement_dispatch_date", ""),
        "replacement_delivery_date": row.get("replacement_delivery_date", ""),
        "replacement_status": row.get("replacement_status", ""),
    }

    return {
        **row,
        "customer_id": customer_id,
        "order_id": order_id,
        "seller_id": seller_id,
        "transporter_id": transporter_id,
        "seller": seller,
        "transporter": transporter,
        "items": items,
        "order_amount": to_float_value(row.get("order_amount")),
        "payment": payment,
        "delivery": delivery,
        "refund": refund,
        "return": return_data,
        "replacement": replacement,
        "payment_id": payment["payment_id"],
        "payment_mode": payment["payment_mode"],
        "payment_status": payment["payment_status"],
        "delivery_id": delivery["delivery_id"],
        "seller_dispatch_date": delivery["seller_dispatch_date"],
        "exp_delivery_date": delivery["expected_delivery_date"],
        "delivery_status": delivery["delivery_status"],
        "actual_delivery_date": delivery["actual_delivery_date"],
        "refund_id": refund["refund_id"],
        "refund_status": refund["refund_status"],
        "expected_refund_amount": refund["expected_refund_amount"],
        "eligible_refund_ratio": refund["eligible_refund_ratio"],
        "expected_refund_date": refund["expected_refund_date"],
        "applied_for_return": return_data["applied_for_return"],
        "return_claim_accepted": return_data["return_claim_accepted"],
        "return_status": return_data["return_status"],
        "damage_flag": return_data["damage_flag"],
    }


@lru_cache(maxsize=1)
def load_orders() -> dict[str, list[dict]]:
    """Load orders grouped by customer from the structured e-commerce dataset."""
    products = load_products()
    sellers = load_sellers()
    transporters = load_transporters()
    orders_by_customer: dict[str, list[dict]] = {}

    for row in _load_csv_rows(ORDERS_DATA_PATH):
        customer_id = row.get("customer_id", "")
        order_id = row.get("order_id", "")
        if not customer_id or not order_id:
            continue

        order = _build_order_record(row, products, sellers, transporters)
        orders_by_customer.setdefault(customer_id, []).append(order)

    for customer_id, orders in orders_by_customer.items():
        orders_by_customer[customer_id] = sorted(
            orders,
            key=lambda order: parse_flexible_datetime(order.get("order_date")) or datetime.min,
            reverse=True,
        )

    return orders_by_customer


def get_latest_order_for_customer(customer_id: str) -> Optional[dict]:
    """Return the most recent order for the customer, if available."""
    orders = load_orders().get(clean_csv_value(customer_id), [])
    return dict(orders[0]) if orders else None


def list_chat_users() -> list[dict]:
    """Return selectable chat users from the e-commerce customer dataset."""
    customers = load_customers()
    return [
        {
            "customer_id": customer_id,
            "name": profile["name"],
            "label": f"{customer_id} - {profile['name']}",
        }
        for customer_id, profile in sorted(customers.items(), key=lambda item: int(item[0]))
    ]


def get_customer_scope(customer_id: str) -> tuple[dict, list[dict]]:
    """Return the selected customer's profile and only their own orders."""
    customer_id = clean_csv_value(customer_id)
    customers = load_customers()
    orders_by_customer = load_orders()
    return customers.get(customer_id, {}), orders_by_customer.get(customer_id, [])


def build_user_summary(profile: dict, orders: list[dict]) -> dict:
    """Create a compact summary for the selected user panel."""
    recent_orders = []
    for order in orders[:5]:
        recent_orders.append({
            "order_id": order["order_id"],
            "order_status": order.get("order_status", ""),
            "delivery_status": order.get("delivery_status", ""),
            "order_date": order.get("order_date", ""),
            "order_amount": order.get("order_amount", 0.0),
            "order_currency": order.get("order_currency", ""),
            "seller_name": order.get("seller", {}).get("seller_name", ""),
            "transporter_name": order.get("transporter", {}).get("name", ""),
        })

    return {
        "customer_id": profile.get("customer_id", ""),
        "name": profile.get("name", ""),
        "account_status": profile.get("account_status", ""),
        "registered_email": profile.get("registered_email", ""),
        "registered_phone_number": profile.get("registered_phone_number", ""),
        "subscription_plan_name": profile.get("subscription_plan_name", ""),
        "prime_subscription_flag": bool(profile.get("prime_subscription_flag")),
        "order_count": len(orders),
        "recent_orders": recent_orders,
    }
