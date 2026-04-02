from __future__ import annotations

import re
from datetime import date, datetime
from typing import Optional

from .ecommerce_repository import (
    build_user_summary,
    clean_csv_value,
    format_context_date,
    format_order_summary,
    get_latest_order_for_customer,
    parse_flexible_datetime,
)


ORDER_RELATED_CATEGORIES = {
    "CANCEL",
    "DELIVERY",
    "INVOICE",
    "ORDER",
    "PAYMENT",
    "REFUND",
    "SHIPPING",
}

CATEGORY_OPTIONS = sorted(
    {
        "ACCOUNT",
        "CANCEL",
        "CONTACT",
        "DELIVERY",
        "FEEDBACK",
        "INVOICE",
        "ORDER",
        "PAYMENT",
        "REFUND",
        "SHIPPING",
        "SUBSCRIPTION",
    }
)


def looks_like_subscription_request(text: str) -> bool:
    lowered = (text or "").lower()
    return any(keyword in lowered for keyword in ("subscription", "plan", "membership", "prime"))


def is_delay_or_tracking_query(text: str) -> bool:
    lowered = (text or "").lower()
    return any(
        keyword in lowered
        for keyword in (
            "where is my order",
            "where's my order",
            "why is my order late",
            "late delivery",
            "delayed order",
            "delay",
            "delayed",
            "track my order",
            "order status",
            "delivery status",
            "where is it",
            "where's it",
        )
    )


def detect_explicit_intent_from_query(text: str) -> Optional[str]:
    lowered = (text or "").lower()

    if looks_like_subscription_request(text):
        return "SUBSCRIPTION"
    if any(keyword in lowered for keyword in ("invoice", "receipt", "billing statement")):
        return "INVOICE"
    if any(keyword in lowered for keyword in ("refund", "refunded", "money back", "return", "returned")):
        return "REFUND"
    if any(keyword in lowered for keyword in ("cancel", "cancellation", "cancelled", "canceled")):
        return "CANCEL"
    if any(keyword in lowered for keyword in ("payment", "transaction", "charged", "paynow", "bank transfer")):
        return "PAYMENT"
    if any(
        keyword in lowered
        for keyword in (
            "delivery",
            "shipping",
            "shipped",
            "track",
            "tracking",
            "where is my order",
            "where's my order",
            "havent received",
            "haven't received",
            "have not received",
            "did not receive",
            "didn't receive",
            "not received yet",
            "supposed to come",
            "expected to come",
            "when can i get",
            "out for delivery",
            "in transit",
            "arrive",
            "arrival",
            "late delivery",
            "delayed order",
        )
    ):
        return "DELIVERY"
    return None


def query_requires_order_lookup(category: str, text: str) -> bool:
    lowered = (text or "").lower()
    explicit_reference = bool(
        extract_numeric_candidates(text)
        or extract_identifier_candidates(text, "PAY")
        or extract_identifier_candidates(text, "DLV")
        or extract_identifier_candidates(text, "REF")
        or extract_identifier_candidates(text, "RFD")
        or extract_query_dates(text)
    )

    if category in {"ORDER", "DELIVERY", "SHIPPING", "REFUND", "PAYMENT"}:
        return True
    if category == "CANCEL":
        return not looks_like_subscription_request(text)
    if category == "INVOICE":
        return explicit_reference or any(
            keyword in lowered for keyword in ("order", "payment", "transaction", "invoice for")
        )
    return False


def should_default_to_latest_order(category: str, text: str) -> bool:
    if category not in {"ORDER", "DELIVERY", "SHIPPING"}:
        return False
    return is_delay_or_tracking_query(text)


def extract_numeric_candidates(text: str) -> list[str]:
    return re.findall(r"\b(\d{1,8})\b", text or "")


def extract_identifier_candidates(text: str, prefix: str) -> list[str]:
    pattern = rf"\b{re.escape(prefix.upper())}[A-Z0-9]+\b"
    return re.findall(pattern, (text or "").upper())


def parse_query_date_token(raw_value: str) -> Optional[date]:
    raw = (raw_value or "").strip()
    if not raw:
        return None

    for fmt in (
        "%d/%m/%y",
        "%d/%m/%Y",
        "%m/%d/%y",
        "%m/%d/%Y",
        "%Y-%m-%d",
        "%d-%m-%y",
        "%d-%m-%Y",
        "%m-%d-%y",
        "%m-%d-%Y",
        "%B %d %Y",
        "%b %d %Y",
        "%B %d, %Y",
        "%b %d, %Y",
        "%d %B %Y",
        "%d %b %Y",
    ):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue

    if re.fullmatch(r"\d{1,2}[/-]\d{1,2}", raw):
        separator = "/" if "/" in raw else "-"
        day_text, month_text = raw.split(separator)
        try:
            return datetime(datetime.now().year, int(month_text), int(day_text)).date()
        except ValueError:
            return None

    return None


def classify_query_date_mention(text: str, start: int, end: int) -> str:
    lowered = (text or "").lower()
    before = lowered[max(0, start - 40):start]
    after = lowered[end:min(len(lowered), end + 40)]
    nearby = f"{before} {after}"

    if re.search(r"\b(today|today is|as of|current date|right now|currently|now)\b", nearby):
        return "current_date"
    if any(keyword in nearby for keyword in ("placed", "ordered", "purchased", "bought", "order date")):
        return "order_date"
    if any(
        keyword in nearby
        for keyword in (
            "expected",
            "supposed",
            "delivery",
            "arrive",
            "arrival",
            "come",
            "receive",
            "received",
            "get it",
            "eta",
            "late",
            "delay",
            "delayed",
        )
    ):
        return "delivery_date"
    return "generic"


def extract_query_date_mentions(text: str) -> list[dict]:
    text = text or ""
    patterns = [
        r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b",
        r"\b\d{1,2}-\d{1,2}(?:-\d{2,4})?\b",
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{1,2},?\s+\d{4}\b",
        r"\b\d{1,2}\s+(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{4}\b",
    ]

    mentions: list[dict] = []
    seen = set()
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            raw = match.group(0).strip()
            parsed = parse_query_date_token(raw)
            if parsed is None:
                continue

            mention_kind = classify_query_date_mention(text, match.start(), match.end())
            key = (raw.lower(), parsed.isoformat(), mention_kind)
            if key in seen:
                continue
            seen.add(key)
            mentions.append({"raw": raw, "date": parsed, "kind": mention_kind})

    return mentions


def extract_query_dates(text: str) -> list[date]:
    candidates: list[date] = []
    seen = set()

    for mention in extract_query_date_mentions(text):
        if mention["kind"] == "current_date":
            continue
        parsed = mention["date"]
        if parsed not in seen:
            seen.add(parsed)
            candidates.append(parsed)

    return candidates


def match_orders_from_date_mentions(customer_orders: list[dict], date_mentions: list[dict]) -> tuple[list[dict], str]:
    if not customer_orders or not date_mentions:
        return [], "no_date_mentions"

    usable_mentions = [mention for mention in date_mentions if mention["kind"] != "current_date"]
    if not usable_mentions:
        return [], "current_date_only"

    def matches_field(order: dict, field_name: str, target_dates: set[date]) -> bool:
        field_dt = parse_flexible_datetime(order.get(field_name))
        return bool(field_dt and field_dt.date() in target_dates)

    order_dates = {mention["date"] for mention in usable_mentions if mention["kind"] == "order_date"}
    delivery_dates = {mention["date"] for mention in usable_mentions if mention["kind"] == "delivery_date"}
    generic_dates = {mention["date"] for mention in usable_mentions if mention["kind"] == "generic"}

    if order_dates or delivery_dates:
        candidates = list(customer_orders)
        if order_dates:
            candidates = [order for order in candidates if matches_field(order, "order_date", order_dates)]
        if delivery_dates:
            candidates = [
                order
                for order in candidates
                if matches_field(order, "exp_delivery_date", delivery_dates)
                or matches_field(order, "actual_delivery_date", delivery_dates)
            ]
        if candidates:
            return candidates, "role_aware_date_match"

    if generic_dates:
        candidates = [
            order
            for order in customer_orders
            if matches_field(order, "order_date", generic_dates)
            or matches_field(order, "exp_delivery_date", generic_dates)
            or matches_field(order, "actual_delivery_date", generic_dates)
        ]
        if candidates:
            return candidates, "generic_date_match"

    return [], "no_orders_for_requested_dates"


def find_recent_resolved_order_reference(
    query: str,
    conversation_history: list[dict],
    customer_orders: list[dict],
) -> tuple[Optional[dict], str]:
    lowered = (query or "").lower()
    if not conversation_history or not customer_orders:
        return None, "no_recent_order_context"

    explicit_order_reference = bool(
        re.search(r"order(?:\s*(?:id|number|#))?\s*[:#-]?\s*\d+", lowered)
        or re.search(r"#\s*\d+", lowered)
        or extract_identifier_candidates(query, "PAY")
        or extract_identifier_candidates(query, "DLV")
        or extract_identifier_candidates(query, "REF")
        or extract_identifier_candidates(query, "RFD")
        or extract_query_dates(query)
    )
    if explicit_order_reference:
        return None, "query_has_new_reference"

    if not re.search(r"\b(it|this|that|the order|my order|that order)\b", lowered):
        return None, "no_follow_up_reference"

    orders_by_id = {order["order_id"]: order for order in customer_orders}
    for interaction in reversed(conversation_history):
        resolved_order_id = clean_csv_value(interaction.get("resolved_order_id"))
        if resolved_order_id and resolved_order_id in orders_by_id:
            return orders_by_id[resolved_order_id], "reuse_recent_order_context"

    return None, "no_recent_resolved_order"


def build_order_lookup_entry(order: dict) -> dict:
    return {
        "order_id": order.get("order_id", ""),
        "order_date": order.get("order_date", ""),
        "order_summary": format_order_summary(order),
        "order_amount": order.get("order_amount", 0.0),
        "order_currency": order.get("order_currency", ""),
        "order_status": order.get("order_status", ""),
        "payment_status": order.get("payment_status", ""),
        "delivery_status": order.get("delivery_status", ""),
        "expected_delivery_date": order.get("exp_delivery_date", ""),
    }


def select_relevant_order(
    query: str,
    customer_orders: list[dict],
    pending_interaction: dict,
) -> tuple[Optional[dict], str]:
    if not customer_orders:
        return None, "no_orders"

    lowered = (query or "").lower()
    orders_by_id = {order["order_id"]: order for order in customer_orders}
    orders_by_payment_id = {
        clean_csv_value(order.get("payment_id")).upper(): order
        for order in customer_orders
        if clean_csv_value(order.get("payment_id"))
    }
    orders_by_delivery_id = {
        clean_csv_value(order.get("delivery_id")).upper(): order
        for order in customer_orders
        if clean_csv_value(order.get("delivery_id"))
    }
    orders_by_refund_id = {
        clean_csv_value(order.get("refund_id")).upper(): order
        for order in customer_orders
        if clean_csv_value(order.get("refund_id"))
    }

    explicit_patterns = [
        r"order(?:\s*(?:id|number|#))?\s*[:#-]?\s*(\d+)",
        r"#\s*(\d+)",
    ]
    for pattern in explicit_patterns:
        match = re.search(pattern, lowered)
        if match:
            candidate = match.group(1)
            if candidate in orders_by_id:
                return orders_by_id[candidate], "explicit_order_id"

    for payment_id in extract_identifier_candidates(query, "PAY"):
        if payment_id in orders_by_payment_id:
            return orders_by_payment_id[payment_id], "explicit_payment_id"

    for delivery_id in extract_identifier_candidates(query, "DLV"):
        if delivery_id in orders_by_delivery_id:
            return orders_by_delivery_id[delivery_id], "explicit_delivery_id"

    refund_candidates = extract_identifier_candidates(query, "REF") + extract_identifier_candidates(query, "RFD")
    for refund_id in refund_candidates:
        if refund_id in orders_by_refund_id:
            return orders_by_refund_id[refund_id], "explicit_refund_id"

    if pending_interaction.get("resolved_order_id"):
        pending_order_id = pending_interaction["resolved_order_id"]
        if pending_order_id in orders_by_id:
            return orders_by_id[pending_order_id], "reuse_previous_order"

    if any(keyword in lowered for keyword in ("latest order", "recent order", "last order", "my latest")):
        return customer_orders[0], "latest_order_reference"

    if len(customer_orders) == 1:
        return customer_orders[0], "single_order_scope"

    return None, "ambiguous_order_reference"


def build_recent_order_options(customer_orders: list[dict], limit: int = 5) -> list[dict]:
    return [build_order_lookup_entry(order) for order in customer_orders[:limit]]


def build_context_json(
    profile: dict,
    orders: list[dict],
    category: str,
    selected_order: Optional[dict] = None,
    clarification_needed: bool = False,
    clarification_candidates: Optional[list[dict]] = None,
) -> dict:
    customer_json = {
        "customer_id": profile.get("customer_id", ""),
        "name": profile.get("name", ""),
        "account_status": profile.get("account_status", ""),
        "registered_email": profile.get("registered_email", ""),
        "registered_phone_number": profile.get("registered_phone_number", ""),
        "address": profile.get("address", ""),
        "city": profile.get("city", ""),
        "state": profile.get("state", ""),
        "country": profile.get("country", ""),
        "account_opening_date": profile.get("account_opening_date", ""),
        "prime_subscription_flag": bool(profile.get("prime_subscription_flag")),
        "subscription_plan_name": profile.get("subscription_plan_name", ""),
        "last_subscribed_date": profile.get("last_subscribed_date", ""),
    }

    context = {
        "intent_category": category,
        "customer": customer_json,
        "clarification_needed": clarification_needed,
        "recent_orders": build_recent_order_options(orders),
    }
    if clarification_candidates:
        context["clarification_candidates"] = clarification_candidates

    if selected_order:
        context["selected_order"] = {
            "order_id": selected_order.get("order_id", ""),
            "order_status": selected_order.get("order_status", ""),
            "order_date": selected_order.get("order_date", ""),
            "order_amount": selected_order.get("order_amount", 0.0),
            "order_currency": selected_order.get("order_currency", ""),
            "items": selected_order.get("items", []),
            "seller": selected_order.get("seller", {}),
            "payment": selected_order.get("payment", {}),
            "delivery": selected_order.get("delivery", {}),
            "refund": selected_order.get("refund", {}),
            "return": selected_order.get("return", {}),
            "replacement": selected_order.get("replacement", {}),
        }

    return context


def build_clarification_prompt(
    category: str,
    orders: list[dict],
    candidate_orders: Optional[list[dict]] = None,
    query_dates: Optional[list[date]] = None,
) -> str:
    recent_orders = [
        build_order_lookup_entry(order)
        for order in (candidate_orders if candidate_orders is not None else orders[:5])
    ]
    prompt_map = {
        "ORDER": "I can help with your order, but I need to know which one you mean.",
        "DELIVERY": "I can check the delivery details, but I need the specific order ID first.",
        "SHIPPING": "I can look up the shipping status, but I need the relevant order ID.",
        "REFUND": "I can review the refund information, but I need the order ID you want me to check.",
        "PAYMENT": "I can inspect the payment details, but I need the related order ID.",
        "INVOICE": "I can pull the invoice-relevant details, but I need the order ID or payment reference.",
        "CANCEL": "I can look into cancellation options, but I need to know which order you mean.",
    }
    opener = prompt_map.get(category, "I can help, but I need a bit more detail first.")

    if query_dates:
        requested_dates = ", ".join(sorted({item.strftime("%d %b %Y") for item in query_dates}))
        opener = f"{opener} I found that you referred to {requested_dates}."

    if not recent_orders:
        if query_dates:
            return (
                f"{opener} I could not find any orders on your account for that date. "
                "Please share the exact order ID or another order date."
            )
        return f"{opener} I could not find any orders on your account yet."

    bullet_lines = []
    for order in recent_orders:
        bullet_lines.append(
            "\n".join(
                [
                    f"- Order #{order['order_id']}",
                    f"  Date: {format_context_date(order['order_date'])}",
                    f"  Summary: {order['order_summary']}",
                    f"  Amount: {order['order_currency']} {float(order['order_amount'] or 0):.2f}",
                    f"  Status: {order['order_status']}",
                    f"  Payment: {order['payment_status']}",
                ]
            )
        )

    if query_dates:
        return (
            f"{opener}\n\n"
            "I found these orders on your account for that date:\n"
            + "\n\n".join(bullet_lines)
            + "\n\nPlease choose the correct order ID from this list."
        )

    return (
        f"{opener}\n\n"
        "Here are your recent orders:\n"
        + "\n\n".join(bullet_lines)
        + "\n\nPlease let me know which order you are referring to."
    )


def prepare_contextual_inference_message(
    translated_message: str,
    customer_orders: list[dict],
    pending_interaction: dict,
) -> tuple[str, str]:
    inference_message = translated_message
    prep_reason = "direct_message"

    if pending_interaction:
        short_message = len(translated_message.split()) <= 10
        matched_order, _ = select_relevant_order(
            translated_message,
            customer_orders,
            pending_interaction,
        )
        clarification_words = re.search(
            r"\b(it'?s|this one|that one|the latest|latest order|recent order|order)\b",
            translated_message.lower(),
        )
        likely_follow_up = short_message and (
            matched_order is not None or clarification_words or extract_numeric_candidates(translated_message)
        )
        if likely_follow_up:
            pending_issue = pending_interaction.get("raw_message", "")
            inference_message = f"{pending_issue} Follow-up details: {translated_message}".strip()
            prep_reason = "pending_clarification_follow_up"

    return inference_message, prep_reason


def resolve_customer_context(
    *,
    customer_id: str,
    translated_query: str,
    predicted_label: str,
    customer_profile: dict,
    customer_orders: list[dict],
    conversation_history: list[dict],
    pending_interaction: dict,
) -> dict:
    explicit_intent = detect_explicit_intent_from_query(translated_query)
    effective_label = explicit_intent or predicted_label

    if effective_label == "CANCEL" and looks_like_subscription_request(translated_query):
        effective_label = "SUBSCRIPTION"

    if (
        pending_interaction
        and pending_interaction.get("needs_more_context")
        and pending_interaction.get("predicted_label") in ORDER_RELATED_CATEGORIES
    ):
        short_message = len(translated_query.split()) <= 12
        matched_order, _ = select_relevant_order(
            translated_query,
            customer_orders,
            pending_interaction,
        )
        clarification_words = re.search(
            r"\b(it'?s|this one|that one|the latest|latest order|recent order|order)\b",
            translated_query.lower(),
        )
        if short_message and (
            matched_order is not None
            or clarification_words
            or extract_numeric_candidates(translated_query)
        ):
            effective_label = pending_interaction.get("predicted_label", effective_label)

    date_mentions = extract_query_date_mentions(translated_query)
    query_dates = extract_query_dates(translated_query)
    requires_order_lookup = query_requires_order_lookup(effective_label, translated_query)
    clarification_candidates: list[dict] = []
    selected_order = None
    resolution_reason = "customer_profile_only"
    needs_more_context = False
    clarification_prompt = ""
    resolved_order_id = ""

    if effective_label in ORDER_RELATED_CATEGORIES and requires_order_lookup:
        selected_order, resolution_reason = select_relevant_order(
            translated_query,
            customer_orders,
            pending_interaction,
        )

        if not customer_orders:
            resolution_reason = "no_orders_on_account"
        elif selected_order is None and query_dates:
            matched_orders, date_match_reason = match_orders_from_date_mentions(customer_orders, date_mentions)
            if len(matched_orders) == 1:
                selected_order = matched_orders[0]
                resolution_reason = f"single_order_match_from_{date_match_reason}"
            else:
                needs_more_context = True
                clarification_candidates = build_recent_order_options(
                    matched_orders,
                    limit=max(1, min(len(matched_orders), 10)),
                )
                clarification_prompt = build_clarification_prompt(
                    effective_label,
                    customer_orders,
                    candidate_orders=matched_orders,
                    query_dates=query_dates,
                )
                resolution_reason = (
                    f"multiple_orders_for_{date_match_reason}"
                    if matched_orders
                    else "no_orders_for_requested_dates"
                )
        elif selected_order is None:
            selected_order, resolution_reason = find_recent_resolved_order_reference(
                translated_query,
                conversation_history,
                customer_orders,
            )

        if selected_order is None and not needs_more_context and should_default_to_latest_order(effective_label, translated_query):
            selected_order = get_latest_order_for_customer(customer_id)
            if selected_order:
                resolution_reason = "latest_order_default_for_delay_or_status"

        if customer_orders and selected_order is None and not needs_more_context:
            needs_more_context = True
            clarification_candidates = build_recent_order_options(customer_orders)
            clarification_prompt = build_clarification_prompt(effective_label, customer_orders)
            resolution_reason = "missing_specific_order_reference"

        if selected_order is not None:
            resolved_order_id = selected_order["order_id"]

    elif effective_label in ORDER_RELATED_CATEGORIES:
        resolution_reason = "order_lookup_not_required"

    context_json = build_context_json(
        customer_profile,
        customer_orders,
        effective_label,
        selected_order,
        clarification_needed=needs_more_context,
        clarification_candidates=clarification_candidates,
    )

    return {
        "predicted_label": effective_label,
        "context_json": context_json,
        "resolved_order_id": resolved_order_id,
        "needs_more_context": needs_more_context,
        "clarification_prompt": clarification_prompt,
        "resolution_reason": resolution_reason,
        "requires_order_lookup": requires_order_lookup,
        "date_mentions": date_mentions,
        "query_dates": query_dates,
        "clarification_candidates": clarification_candidates,
        "selected_order": selected_order,
        "user_summary": build_user_summary(customer_profile, customer_orders),
    }

