from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

from .ecommerce_context import ORDER_RELATED_CATEGORIES, is_delay_or_tracking_query
from .ecommerce_repository import clean_csv_value, format_context_date, format_order_summary, parse_flexible_datetime


def build_delivery_timing_note(selected_order: dict) -> str:
    """Summarize whether the selected order is on time or delayed."""
    delivery = selected_order.get("delivery", {})
    expected_delivery_raw = delivery.get("expected_delivery_date")
    actual_delivery_raw = delivery.get("actual_delivery_date")
    expected_delivery_dt = parse_flexible_datetime(expected_delivery_raw)
    actual_delivery_dt = parse_flexible_datetime(actual_delivery_raw)
    today = datetime.now().date()
    today_label = today.strftime("%d %b %Y")
    delivery_status = clean_csv_value(delivery.get("delivery_status"))

    if actual_delivery_dt is not None:
        return f"The latest delivery record shows it was delivered on {actual_delivery_dt.strftime('%d %b %Y')}."

    if expected_delivery_dt is None:
        return "I do not have a newer delivery estimate on file yet."

    expected_label = expected_delivery_dt.strftime("%d %b %Y")
    if expected_delivery_dt.date() < today and delivery_status.lower() != "delivered":
        return (
            f"This looks delayed because the expected delivery date was {expected_label}, "
            f"and today is {today_label}."
        )

    if delivery_status.lower() == "out for delivery":
        return f"It is currently out for delivery, with the latest expected arrival date listed as {expected_label}."

    return f"The latest expected delivery date on file is {expected_label}."


def build_service_recovery_facts(raw_message: str, context_json: dict, customer_orders: list[dict]) -> dict:
    """Describe whether the latest delivery issue likely needs apology or escalation."""
    query = (raw_message or "").lower()
    delivery_issue_reported = any(
        phrase in query
        for phrase in (
            "late",
            "delay",
            "delayed",
            "havent received",
            "haven't received",
            "have not received",
            "not received",
            "why is it late",
            "when will i receive",
            "when can i get",
            "where is my order",
            "where's my order",
        )
    )

    def lookup_order_by_id(order_id: str) -> Optional[dict]:
        order_id = clean_csv_value(order_id)
        if not order_id:
            return None
        for order in customer_orders:
            if clean_csv_value(order.get("order_id")) == order_id:
                return order
        return None

    def order_is_overdue(order: Optional[dict]) -> bool:
        if not order:
            return False
        delivery = order.get("delivery", order)
        actual_delivery = parse_flexible_datetime(delivery.get("actual_delivery_date"))
        if actual_delivery is not None:
            return False

        delivery_status = clean_csv_value(delivery.get("delivery_status", order.get("delivery_status", ""))).lower()
        if delivery_status == "delivered":
            return False

        expected_delivery = parse_flexible_datetime(
            delivery.get(
                "expected_delivery_date",
                order.get("expected_delivery_date", order.get("exp_delivery_date", "")),
            )
        )
        return bool(expected_delivery and expected_delivery.date() < datetime.now().date())

    selected_order = context_json.get("selected_order", {})
    selected_order_live = lookup_order_by_id(selected_order.get("order_id")) if selected_order else None
    selected_order_overdue = order_is_overdue(selected_order_live or selected_order)

    clarification_candidates = context_json.get("clarification_candidates", [])
    overdue_candidate_order_ids = []
    for candidate in clarification_candidates:
        candidate_live = lookup_order_by_id(candidate.get("order_id"))
        if order_is_overdue(candidate_live or candidate):
            overdue_candidate_order_ids.append(clean_csv_value(candidate.get("order_id")))

    escalation_recommended = bool(
        delivery_issue_reported
        and (selected_order_overdue or overdue_candidate_order_ids or context_json.get("clarification_needed"))
    )

    return {
        "delivery_issue_reported": delivery_issue_reported,
        "selected_order_overdue": selected_order_overdue,
        "overdue_candidate_order_ids": overdue_candidate_order_ids,
        "escalation_recommended": escalation_recommended,
        "known_root_cause_available": False,
        "recommended_action": (
            "Apologize for the delay and explain that the issue will be escalated to customer support/logistics for a transporter update."
            if escalation_recommended
            else ""
        ),
    }


def build_relevant_context_slice(
    *,
    raw_message: str,
    predicted_label: str,
    route_decision: str,
    context_json: dict,
    conversation_history: list[dict],
    customer_orders: list[dict],
) -> dict:
    """Create the minimal context slice needed to answer the user's query."""
    context = {
        "customer": context_json.get("customer", {}),
        "intent_category": predicted_label,
        "route_decision": route_decision,
        "query": raw_message,
        "is_first_reply": len(conversation_history) == 0,
        "current_date": datetime.now().strftime("%d %b %Y"),
        "service_recovery": build_service_recovery_facts(raw_message, context_json, customer_orders),
    }

    if context_json.get("clarification_candidates"):
        context["clarification_candidates"] = context_json["clarification_candidates"]

    selected_order = context_json.get("selected_order")
    if selected_order:
        context["selected_order"] = selected_order

    return context


def _format_party_contact(label: str, details: dict) -> str:
    if not details:
        return f"{label} details are not available in the current dataset."

    name = details.get("seller_name") or details.get("name") or label
    parts = [name]
    if details.get("email"):
        parts.append(f"email: {details['email']}")
    if details.get("website"):
        parts.append(f"website: {details['website']}")
    if details.get("seller_city"):
        parts.append(f"city: {details['seller_city']}")
    if details.get("country"):
        parts.append(f"country: {details['country']}")
    return ", ".join(parts)


def build_delay_or_status_response(selected_order: dict, raw_message: str) -> str:
    """Return a response for order-status and delay questions."""
    order_id = selected_order.get("order_id", "")
    order_status = clean_csv_value(selected_order.get("order_status"))
    delivery = selected_order.get("delivery", {})
    delivery_status = clean_csv_value(delivery.get("delivery_status"))
    payment_status = clean_csv_value(selected_order.get("payment", {}).get("payment_status"))
    seller_dispatch_date = parse_flexible_datetime(delivery.get("seller_dispatch_date"))
    actual_delivery_date = parse_flexible_datetime(delivery.get("actual_delivery_date"))
    delivery_id = clean_csv_value(delivery.get("delivery_id"))
    seller_details = selected_order.get("seller", {})
    transporter_details = delivery.get("transporter", selected_order.get("transporter", {}))
    timing_note = build_delivery_timing_note(selected_order)
    order_summary = format_order_summary(selected_order)

    if order_status.lower() in {"cancelled", "payment failed"} or payment_status.lower() in {"failed", "cancelled"}:
        return (
            f"I checked order #{order_id}. This order is not active because the current status is {order_status or payment_status}. "
            f"No active delivery is in progress for this order."
        ).strip()

    if actual_delivery_date is not None or delivery_status.lower() == "delivered" or order_status.lower() == "delivered":
        delivered_on = format_context_date(delivery.get("actual_delivery_date"))
        return (
            f"I checked order #{order_id}. The latest record shows it was delivered on {delivered_on}. "
            f"This order is no longer in transit, and it includes {order_summary}."
        ).strip()

    if seller_dispatch_date is None:
        return (
            f"I checked order #{order_id}. The order has not been dispatched by the seller yet, so the delay is currently on the seller side. "
            f"Seller details: {_format_party_contact('Seller', seller_details)}. "
            "Please contact the seller directly for the dispatch update. We will also get in touch with the seller."
        ).strip()

    transporter_line = _format_party_contact("Transporter", transporter_details)
    tracking_line = f"Track it with delivery ID {delivery_id}." if delivery_id else "The delivery ID is not available yet."
    return (
        f"I checked order #{order_id}. The seller dispatched it on {format_context_date(delivery.get('seller_dispatch_date'))}, "
        f"and it is currently {delivery_status or order_status}. {timing_note} "
        f"Transporter details: {transporter_line}. {tracking_line}"
    ).strip()


def build_refund_response(selected_order: dict, raw_message: str) -> str:
    """Return a response for refund and return questions."""
    order_id = selected_order.get("order_id", "")
    order_currency = selected_order.get("order_currency", "")
    order_status = clean_csv_value(selected_order.get("order_status"))
    payment = selected_order.get("payment", {})
    refund = selected_order.get("refund", {})
    return_data = selected_order.get("return", {})

    payment_mode = clean_csv_value(payment.get("payment_mode"))
    payment_status = clean_csv_value(payment.get("payment_status"))
    refund_status = clean_csv_value(refund.get("refund_status")) or "Not Applicable"
    refund_amount = float(refund.get("expected_refund_amount", 0) or 0)
    expected_refund_date = format_context_date(refund.get("expected_refund_date"))
    actual_refund_date = format_context_date(refund.get("actual_refund_date"))
    return_status = clean_csv_value(return_data.get("return_status")) or "Not Requested"
    return_reason = clean_csv_value(return_data.get("return_reason"))
    return_claim_accepted = bool(return_data.get("return_claim_accepted"))

    if return_status.lower() == "rejected" or refund_status.lower() == "rejected" or (
        clean_csv_value(return_data.get("applied_for_return")).lower() in {"1", "true", "yes"} and not return_claim_accepted
    ):
        rejection_reason = f" Reason noted: {return_reason}." if return_reason else ""
        return (
            f"I checked order #{order_id}. The return request was rejected, so a refund is not allowed for this order.{rejection_reason}"
        ).strip()

    if refund_status.lower() == "pending":
        return (
            f"I checked order #{order_id}. Your refund has been initiated but is still pending completion. "
            f"The expected refund amount is {order_currency} {refund_amount:.2f}, and the expected refund date is {expected_refund_date}."
        ).strip()

    if refund_status.lower() == "processed":
        return (
            f"I checked order #{order_id}. The refund has already been processed for {order_currency} {refund_amount:.2f}, "
            f"and the actual refund date on file is {actual_refund_date}."
        ).strip()

    if payment_mode.lower() == "cod" and order_status.lower() == "cancelled":
        return (
            f"I checked order #{order_id}. No refund is due because this was a cash-on-delivery order and the payment was not captured before cancellation."
        ).strip()

    if order_status.lower() == "payment failed" or payment_status.lower() == "failed":
        return (
            f"I checked order #{order_id}. No refund is applicable because the payment did not go through for this order."
        ).strip()

    if refund_status.lower() == "not applicable":
        if return_status.lower() == "not requested" and order_status.lower() == "delivered":
            return (
                f"I checked order #{order_id}. A refund is not applicable right now because there is no approved return on file for this delivered order."
            ).strip()
        if order_status.lower() in {"placed", "dispatched", "in transit", "out for delivery"}:
            return (
                f"I checked order #{order_id}. A refund is not applicable yet because the order is still active with status {order_status}."
            ).strip()
        return (
            f"I checked order #{order_id}. A refund is not applicable based on the current order, return, and payment status on file."
        ).strip()

    return (
        f"I checked order #{order_id}. The refund status is {refund_status}, with expected refund amount "
        f"{order_currency} {refund_amount:.2f}. The expected refund date on file is {expected_refund_date}."
    ).strip()


def build_policy_response_blueprint(
    *,
    customer_name: str,
    conversation_history: list[dict],
    raw_message: str,
    predicted_label: str,
    context_json: dict,
    needs_more_context: bool,
    clarification_prompt: str,
) -> str:
    """Create a policy-safe response blueprint from structured context."""
    is_first_reply = len(conversation_history) == 0
    intro = f"Hello {customer_name}! I can understand your concern. " if is_first_reply else ""
    customer = context_json.get("customer", {})
    selected_order = context_json.get("selected_order", {})
    category = predicted_label
    is_prime = bool(customer.get("prime_subscription_flag"))

    if needs_more_context:
        service_recovery = build_service_recovery_facts(raw_message, context_json, [])
        if service_recovery.get("delivery_issue_reported"):
            return (
                f"{intro}I’m sorry that this delivery appears delayed. "
                "Our support and logistics teams will check with the transporter as well. "
                f"{clarification_prompt}"
            ).strip()
        return f"{intro}{clarification_prompt}".strip()

    if category == "INVOICE":
        if selected_order:
            return (
                f"{intro}I checked the invoice-related details for order #{selected_order.get('order_id', 'n/a')}. "
                "You can access the invoice any time via My Profile -> Past Orders. "
                f"The payment status currently shows {selected_order.get('payment', {}).get('payment_status', 'n/a')}."
            ).strip()
        return f"{intro}You can access your invoice via My Profile -> Past Orders.".strip()

    if category == "SUBSCRIPTION":
        plan_name = customer.get("subscription_plan_name", "n/a")
        prime_state = "active" if is_prime else "not active"
        return (
            f"{intro}Your current subscription plan is {plan_name}, and Prime access is {prime_state}. "
            "If you'd like, I can help with renewal, upgrade, or cancellation guidance."
        ).strip()

    if category == "ACCOUNT":
        return (
            f"{intro}Your account is currently {customer.get('account_status', 'unknown')} and is registered to "
            f"{customer.get('registered_email', 'the email on file')}. "
            "Please tell me whether you need help with login, password reset, or account verification."
        ).strip()

    if category in ORDER_RELATED_CATEGORIES and not selected_order:
        return (
            f"{intro}I need the exact order reference before I can answer safely. "
            "Please share the order ID or payment ID for the order you mean."
        ).strip()

    if category in {"ORDER", "DELIVERY", "SHIPPING"}:
        status_reply = build_delay_or_status_response(selected_order, raw_message)
        if not is_prime:
            status_reply = (
                f"{status_reply} Prime orders are prioritized for faster delivery, and this account is not currently subscribed to Prime."
            )
        return f"{intro}{status_reply}".strip()

    if category == "PAYMENT":
        payment = selected_order.get("payment", {})
        return (
            f"{intro}I checked the payment details for order #{selected_order.get('order_id', 'n/a')}. "
            f"Payment {payment.get('payment_id', 'n/a')} was made via {payment.get('payment_mode', 'n/a')} and is currently {payment.get('payment_status', 'n/a')}. "
            f"The order total is {selected_order.get('order_currency', '')} {float(selected_order.get('order_amount', 0) or 0):.2f}."
        ).strip()

    if category == "REFUND":
        return f"{intro}{build_refund_response(selected_order, raw_message)}".strip()

    if category == "CANCEL":
        order_status = clean_csv_value(selected_order.get("order_status", ""))
        delivery_status = clean_csv_value(selected_order.get("delivery", {}).get("delivery_status", ""))
        shipped_or_beyond = order_status.lower() in {
            "dispatched",
            "in transit",
            "out for delivery",
            "delivered",
            "return requested",
            "returned",
            "return rejected",
        } or delivery_status.lower() in {"in transit", "out for delivery", "delivered"}
        if shipped_or_beyond:
            return (
                f"{intro}I checked order #{selected_order.get('order_id', 'n/a')}. "
                f"It has already reached the {order_status} stage, so it can no longer be cancelled. "
                "Once you receive the item, you can apply for a return if needed."
            ).strip()
        return (
            f"{intro}I checked order #{selected_order.get('order_id', 'n/a')}, which is currently {order_status}. "
            "If it moves into shipping, it can no longer be cancelled."
        ).strip()

    if category == "CONTACT":
        return f"{intro}You can reach our support team at support@company.com or call 1-800-SUPPORT during business hours.".strip()

    return (
        f"{intro}I have reviewed the details available for your account and can help once you tell me a bit more about what you need."
    ).strip()


def enforce_response_policies(
    *,
    customer_name: str,
    conversation_history: list[dict],
    predicted_label: str,
    raw_message: str,
    context_json: dict,
    needs_more_context: bool,
    blueprint: str,
    drafted_response: str,
) -> str:
    """Ensure the final response respects non-negotiable policy rules."""
    response = (drafted_response or "").strip() or blueprint
    response = re.sub(r"```.*?```", "", response, flags=re.DOTALL).strip()
    if response.startswith("{") or response.startswith("[") or (
        "{" in response and "}" in response and re.search(r'"\w+"\s*:', response)
    ):
        response = blueprint

    response = re.sub(
        rf"(Hello\s+{re.escape(customer_name)}!\s*)+",
        f"Hello {customer_name}! ",
        response,
        flags=re.IGNORECASE,
    )
    response = re.sub(
        r"(I can understand your concern\.\s*){2,}",
        "I can understand your concern. ",
        response,
        flags=re.IGNORECASE,
    )
    response = re.sub(r"\s+", " ", response).strip()

    expected_intro = f"Hello {customer_name}! I can understand your concern."
    if len(conversation_history) == 0 and not response.startswith(expected_intro):
        bare_greeting = f"Hello {customer_name}!"
        if response.startswith(bare_greeting):
            response = expected_intro + " " + response[len(bare_greeting):].lstrip()
        else:
            response = f"{expected_intro} " + response.lstrip()

    if predicted_label == "INVOICE" and "My Profile -> Past Orders" not in response:
        response = response.rstrip() + " You can access the invoice via My Profile -> Past Orders."

    selected_order = context_json.get("selected_order", {})
    if needs_more_context:
        lowered = response.lower()
        asks_for_identifier = any(
            phrase in lowered
            for phrase in (
                "order id",
                "payment id",
                "which order",
                "choose",
                "select",
                "share the order",
                "tell me which order",
            )
        )
        if not asks_for_identifier:
            response = blueprint
    elif selected_order:
        order_id = clean_csv_value(selected_order.get("order_id"))
        if order_id and order_id not in response and f"#{order_id}" not in response:
            response = blueprint

    if predicted_label in {"ORDER", "DELIVERY", "SHIPPING"} and is_delay_or_tracking_query(raw_message):
        if "seller" not in response.lower() and "transporter" not in response.lower() and "delivered" not in response.lower():
            response = blueprint

    return response
