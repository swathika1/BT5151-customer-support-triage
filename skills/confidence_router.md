---
name: confidence_router
description: "Route based on confidence threshold: AUTO_REPLY if high, CLARIFY if medium, ESCALATE if low"
mode: organisational
---

## When to use
Fifth node in serving pipeline. Pure logic node (no LLM, no ML) that routes customer message to appropriate handling tier.

## How to execute
1. Read from state:
   - confidence_score: Float from run_inference_node (0.0-1.0)
   - tau_high: Routing threshold from artifacts (e.g., 0.85) - HIGH confidence
   - tau_low: Routing threshold from artifacts (e.g., 0.65) - LOW confidence
2. Apply routing logic:
   - **If confidence_score ≥ tau_high**: Set route_decision = "AUTO_REPLY"
     → System sends automated response immediately (high confidence, low escalation risk)
   - **Else if confidence_score ≥ tau_low**: Set route_decision = "CLARIFY"
     → System asks customer for more details to improve confidence (medium confidence)
   - **Else**: Set route_decision = "ESCALATE"
     → System routes to human agent (low confidence, ambiguous message)
3. Log routing decision and rationale
4. Store route_decision in state for draft_response_node

## Inputs from agent state
- confidence_score: Maximum probability from run_inference_node (float, 0.0-1.0)
- tau_high: High confidence threshold (float, typically 0.80-0.95)
- tau_low: Low confidence threshold (float, typically 0.60-0.75)
- predicted_label: Predicted category (for logging context)

## Outputs to agent state
- route_decision: Routing tier (string: "AUTO_REPLY" | "CLARIFY" | "ESCALATE")

## Output format
```python
{
  "route_decision": str  # "AUTO_REPLY" | "CLARIFY" | "ESCALATE"
}
```

## Routing Logic Decision Tree
```
IF confidence_score >= tau_high (e.g., >= 0.85):
    route_decision = "AUTO_REPLY"
    Rationale: High confidence → model is very sure of prediction
    Action: Send prepared response immediately
    
ELSE IF confidence_score >= tau_low (e.g., >= 0.65):
    route_decision = "CLARIFY"
    Rationale: Medium confidence → model is somewhat unsure
    Action: Ask customer for more details to disambiguate
    
ELSE (confidence_score < tau_low):
    route_decision = "ESCALATE"
    Rationale: Low confidence → model is very unsure (ambiguous message)
    Action: Route to human agent for manual handling
```

## Examples
Assume tau_high = 0.85, tau_low = 0.65:

1. Message: "I need a refund for my broken item"
   - Predicted: REFUND, Confidence: 0.92
   - Decision: AUTO_REPLY (0.92 ≥ 0.85)
   - System: Sends refund policy and start process
   
2. Message: "My order is still not here"
   - Predicted: DELIVERY, Confidence: 0.72
   - Decision: CLARIFY (0.72 ≥ 0.65 but < 0.85)
   - System: Asks for order number to look up tracking
   
3. Message: "The thing is broken"
   - Predicted: CANCEL (low confidence 0.48)
   - Decision: ESCALATE (0.48 < 0.65)
   - System: Routes to human agent (could be refund, return, or complaint)

## Notes
- **No ML in this node**: Pure conditional logic, no models or randomness. Highly interpretable.
- **Threshold calibration**: tau_high and tau_low are data-justified (derived from validation set precision-confidence curves), not arbitrary. This ensures AUTO_REPLY predictions have ~90% accuracy, CLARIFY predictions have ~70% accuracy.
- **Three tiers balance efficiency and quality**:
  - AUTO_REPLY: Fast resolution (seconds), high accuracy, low human cost
  - CLARIFY: Moderate resolution (minutes), medium accuracy, forces customer to provide details
  - ESCALATE: Slow resolution (hours), human judgment, highest quality but highest cost
- **Business value**: High-confidence messages resolved instantly (reducing wait time). Low-confidence messages escalated early (avoiding frustrating customer with wrong auto-response).
- **Metric tradeoff**: If tau_high is lowered (e.g., 0.75), more messages get AUTO_REPLY but accuracy drops. If raised (e.g., 0.95), fewer false positives but more unnecessary escalations.
