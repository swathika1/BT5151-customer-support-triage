---
name: confidence_router
description: "Route by confidence and context completeness: AUTO_REPLY if confident and complete, CLARIFY if more identifiers are needed, ESCALATE if confidence is low"
mode: organisational
---

## When to use
Routing node after inference and context resolution. This node decides whether the assistant can answer now, must ask for more context, or should escalate.

## How to execute
1. Read from state:
   - `confidence_score`
   - `tau_high`
   - `tau_low`
   - `predicted_label`
   - `needs_more_context`
   - `clarification_prompt`
   - `resolved_order_id`

2. Apply routing logic in this order:
   - If `needs_more_context == True`: set `route_decision = "CLARIFY"`
     - This overrides model confidence
     - Use when the user has not provided the exact order ID or payment reference needed to answer safely
   - Else if `confidence_score >= tau_high`: set `route_decision = "AUTO_REPLY"`
   - Else if `confidence_score >= tau_low`: set `route_decision = "CLARIFY"`
   - Else: set `route_decision = "ESCALATE"`

3. Log the rationale clearly
   - Example: high confidence answer
   - Example: missing order-specific context
   - Example: low-confidence model prediction

4. Store `route_decision` in state for `draft_response_node`

## Inputs from agent state
- `confidence_score`: maximum model probability
- `tau_high`: high-confidence threshold
- `tau_low`: medium-confidence threshold
- `predicted_label`: current category
- `needs_more_context`: whether the answer is blocked by missing identifiers
- `clarification_prompt`: optional clarification text
- `resolved_order_id`: order id selected from user scope, if any

## Outputs to agent state
- `route_decision`: `AUTO_REPLY` | `CLARIFY` | `ESCALATE`

## Output format
```python
{
  "route_decision": str
}
```

## Decision rules
```text
IF needs_more_context == True:
    route_decision = "CLARIFY"
    reason = "Business context is incomplete, even if model confidence is high"

ELIF confidence_score >= tau_high:
    route_decision = "AUTO_REPLY"
    reason = "Prediction is confident and context is complete"

ELIF confidence_score >= tau_low:
    route_decision = "CLARIFY"
    reason = "Prediction is plausible but model confidence is moderate"

ELSE:
    route_decision = "ESCALATE"
    reason = "Prediction is too uncertain"
```

## Notes
- Missing order or payment identifiers are a customer-context problem, not just a confidence problem.
- A query like "Where is my order?" should normally route to `CLARIFY` if the customer has multiple orders and no exact order ID was given.
- If the customer provides only a date, the system should still route to `CLARIFY` and offer matching orders from that date so the user can choose one.
- This node does not generate the wording of the reply. It only decides the handling tier.
