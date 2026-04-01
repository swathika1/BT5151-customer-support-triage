---
name: resolve_context
description: "Restrict reasoning to the selected customer, build scoped context JSON, and request missing order or payment identifiers when needed"
mode: organisational
---

## When to use
After `run_inference_node` and before `confidence_router_node` or `draft_response_node`.

## Purpose
Turn the selected customer's CSV data into structured internal context that the response LLM can safely use.

## How to execute
1. Load only the selected customer's profile from `customers_data.csv`
2. Load only orders whose `CID` matches that customer from `orders_data.csv`
3. Use prior chat history for this same customer to interpret short clarification follow-ups
4. Determine whether the query is about:
   - account
   - subscription
   - invoice
   - shipping or delivery
   - payment
   - refund
   - cancellation
   - a general order query

5. If the query is tied to a particular order or payment:
   - try to resolve the order ID or payment reference from the latest message
   - if missing, reuse the unresolved prior interaction only when the new message is clearly a follow-up
   - if user provides a date but not an ID, find only that user's orders on that date and prepare a clarification list

6. Build `context_json`
   - always include scoped customer profile
   - include only the relevant order and the relevant subsection for the exact query
   - examples:
     - refund query -> refund fields
     - shipping query -> delivery fields
     - payment query -> payment fields
     - invoice query -> billing and order reference fields
   - prepare service-recovery facts when the user is reporting a delay, non-delivery, or missed ETA
     - examples:
       - selected order is overdue
       - one or more candidate orders are overdue
       - the system has no transporter root-cause detail yet
       - escalation to customer support/logistics is recommended

7. Set:
   - `context_json`
   - `resolved_order_id`
   - `needs_more_context`
   - `clarification_prompt`

## Required business rules
- Customer scope must never cross user boundaries
- If exact order or payment ID is missing, do not guess
- If the user gave a date but not an ID, show only matching orders from that date and ask them to choose one
- The disambiguation list should include:
  - order id
  - order date
  - order summary
  - order amount
  - order currency
  - order status
  - payment status
- Internal JSON is for grounding only and must not be shown directly to the user
- If the user reports a missed delivery or asks why an order is late, capture whether the selected or candidate orders look overdue so the downstream response skill can decide whether an apology and escalation note are appropriate
- If the available data does not include a transporter reason, store that as missing context rather than inventing a cause

## Inputs from agent state
- `customer_id`
- `customer_profile`
- `customer_orders`
- `conversation_history`
- `raw_message`
- `translated_message`
- `predicted_label`
- `pending_interaction`

## Outputs to agent state
- `context_json`
- `resolved_order_id`
- `needs_more_context`
- `clarification_prompt`
- optional label adjustment when the customer's wording clearly maps to a different business intent

## Output shape
```python
{
  "context_json": {
    "intent_category": str,
    "customer": dict,
    "recent_orders": list,
    "selected_order": dict | None,
    "clarification_needed": bool,
    "service_recovery": dict | None
  },
  "resolved_order_id": str,
  "needs_more_context": bool,
  "clarification_prompt": str
}
```

## Notes
- This node exists so the response LLM works from facts, not guesses.
- It is acceptable for model confidence to be high while `needs_more_context` is still true.
- If the user asks for invoice details, the downstream response should still direct them to `My Profile -> Past Orders` rather than dumping invoice JSON.
