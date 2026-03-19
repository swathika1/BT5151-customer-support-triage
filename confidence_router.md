---
name: confidence_router

description:
  Route the classified ticket based on model confidence score. High confidence predictions are auto-approved. Medium confidence predictions are held for human review. Low confidence predictions are escalated to a supervisor. This prevents the system from silently misrouting tickets due to uncertain predictions.

tags: [routing, confidence_threshold, decision_making, quality_control, serving]

mode: organisational
---

# Confidence Router Skill

## Role
This is an organisational skill. The Python code applies deterministic threshold-based routing logic to the model's confidence score. This is a critical quality-control gate that ensures low-confidence predictions don't cause incorrect ticket routing.

## When to use
After run_inference_node has produced predicted_label and confidence_score. Before draft_response_node. Triggered once per customer message.

## How to execute
1. Read predicted_label and confidence_score from state
2. Define confidence thresholds:
   - High confidence: confidence_score >= 0.80
   - Medium confidence: 0.60 <= confidence_score < 0.80
   - Low confidence: confidence_score < 0.60
3. Determine routing strategy based on confidence:
   a. HIGH CONFIDENCE (>= 0.80):
      - Status: auto_approved
      - Action: Proceed immediately to draft_response_node without review
      - Rationale: Model is 80%+ certain; trust the prediction
   
   b. MEDIUM CONFIDENCE (0.60-0.79):
      - Status: pending_review
      - Action: Route to draft_response_node BUT flag requires_human_review = True
      - Rationale: Model is 60-80% certain; human should verify before sending
   
   c. LOW CONFIDENCE (< 0.60):
      - Status: escalate_to_supervisor
      - Action: Skip draft_response_node; escalate directly to supervisor queue
      - Rationale: Model is less than 60% certain; too risky to auto-route
4. Consider category-specific overrides:
   - Security Concern: Always escalate (never auto-approve), regardless of confidence
   - Account Suspension: Usually escalate (unless confidence >= 0.95)
   - Payment Problem: Usually escalate (unless confidence >= 0.95)
   - Feature Request: Can auto-approve even at medium confidence (low risk)
5. Store routing decision and rationale in state
6. Log the decision to state.messages

## Inputs from agent state
- predicted_label: str - the ticket category predicted by the model
- confidence_score: float - model's max class probability (0.0-1.0)
- class_probabilities: dict - probabilities for all 10 categories
- customer_message: str - original message (for context if needed)

## Outputs to agent state
- routing_decision: str - one of: "auto_approved", "pending_review", "escalate_to_supervisor"
- confidence_threshold_met: bool - whether confidence_score >= 0.80
- requires_human_review: bool - True if pending_review or escalate
- routing_rationale: str - explanation of routing decision (e.g., "High confidence (0.87 >= 0.80)")
- supervisor_queue_priority: str - if escalating: "HIGH", "MEDIUM", or "LOW" based on category

## Output format
Appends to state.messages:
"[confidence_router] Predicted: Login Issue (0.87 confidence). Routing decision: auto_approved. Rationale: High confidence (0.87 >= 0.80). Proceed to draft_response."
or
"[confidence_router] Predicted: Payment Problem (0.73 confidence). Routing decision: pending_review. Rationale: Medium confidence + payment-critical category. Flag for human review."
or
"[confidence_router] Predicted: Security Concern (0.82 confidence). Routing decision: escalate_to_supervisor. Rationale: Security Concern category always escalates regardless of confidence."

## Routing Thresholds
```
Category-Specific Overrides (take precedence over confidence thresholds):
- Security Concern: Always escalate (critical)
- Account Suspension: Escalate unless confidence >= 0.95
- Payment Problem: Escalate unless confidence >= 0.95
- Refund Request: Escalate unless confidence >= 0.90

Standard Confidence Thresholds:
- High (>= 0.80): Auto-approve
- Medium (0.60-0.79): Pending review (flag for human)
- Low (< 0.60): Escalate to supervisor
```

## Notes
- Confidence thresholds can be adjusted based on business SLA (faster routing = higher thresholds)
- This node is the quality-control gate; use it to prevent bad predictions from causing customer issues
- Log every routing decision for analytics; track which categories are flagged for review most often
- If confidence_score is very high (> 0.95): Skip review, auto-approve immediately (customer satisfaction)
- If confidence_score is ambiguous (0.35-0.50 across multiple classes): Escalate with urgency flag
- For high-volume support teams, pending_review queue is manually processed on an SLA (e.g., within 2 hours)
- Supervisor queue (escalate_to_supervisor) is for senior agents to handle manually
- Monitor: if escalation rate > 20%, retrain model or lower thresholds
