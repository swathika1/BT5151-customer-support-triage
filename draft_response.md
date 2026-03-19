---
name: draft_response

description:
  Take the predicted support category and confidence score from run_inference and translate into a routing recommendation and canned response template. Downstream skill that bridges from raw model output to actionable business output for the support team.

tags: [routing, triage, response_generation, business_logic]

mode: organisational
---

# Draft Response Skill

## Role
This is an organisational skill. The Python code translates the model prediction into business-facing output without any model retraining. All operations are deterministic rule-based logic based on confidence thresholds and category-specific business rules.

## When to use
After run_inference_node has written predicted_label and confidence_score to state. At the end of the serving pipeline, before display in Gradio. Triggered once per customer message in the serving pipeline.

## How to execute
1. Read predicted_label and confidence_score from state
2. Apply routing decision logic based on confidence threshold:
   - High confidence (≥ 0.8): Use predicted category directly
   - Medium confidence (0.6 - 0.79): Flag for manual review before routing
   - Low confidence (< 0.6): Escalate to human agent immediately
3. Assign routing recommendation based on predicted category and confidence:
   - Security Concern + high confidence → "Escalate to Security Team Immediately"
   - Account Suspension + high confidence → "Escalate to Account Specialists"
   - Payment Problem + high confidence → "Escalate to Billing Department"
   - Bug Report + high confidence → "Forward to Product/Engineering"
   - Login Issue / Password Reset + high confidence → "Auto-reply with password reset link"
   - Feature Request + high confidence → "Log to product backlog feedback system"
   - All others + high confidence → "Assign to General Support Queue"
   - Medium confidence (any category) → "Route to senior agent for manual review"
   - Low confidence (any category) → "Escalate to supervisor - unable to classify"
4. Draft a response template appropriate to the predicted category:
   - Include a personalized greeting
   - Reference the ticket category (e.g. "Your password reset request")
   - Provide next steps
5. Build a summary object containing predicted category, confidence, routing decision, and draft response
6. Append trace log entry to state.trace_logs:
           {
             "stage": "draft_response",
             "timestamp": current timestamp,
             "inputs_summary": predicted_label + confidence_score,
             "outputs_summary": routing_recommendation + response_preview,
             "routing_confidence": confidence_score
           }
7. Save all outputs to state

## Inputs from agent state
- predicted_label: str - the predicted support category name
- confidence_score: float - the model's max class probability (0.0 to 1.0)
- class_probabilities: dict - probability distribution across all 10 classes
- customer_message: str - original customer message (for context in response)

## Outputs to agent state
- routing_recommendation: str - where the ticket should be routed (e.g., "Auto-reply: Password Reset")
- response_template: str - draft response to send to customer or support agent
- confidence_level: str - categorical confidence ("High", "Medium", "Low")
- requires_human_review: bool - whether ticket needs human review before routing
- trace_logs: list - one entry appended per node execution

## Output format
Appends to state.messages:
"[draft_response] Predicted: Login Issue. Confidence: 0.87 (High). Routing: Auto-reply with password reset link. Requires review: No."

Returns to Gradio:
{
  "category": "Login Issue",
  "confidence": "87%",
  "routing": "AUTO-REPLY",
  "action": "Send password reset email",
  "draft_response": "Hi there! We see you're having trouble logging in. We've sent a password reset link to your registered email. If it doesn't arrive, please reply to this ticket and we'll help right away."
}

## Notes
- Confidence threshold of 0.8 is based on calibrated model probabilities - it represents a 80% chance the prediction is correct
- Categories like Security Concern and Payment Problem should always escalate even at high confidence because of business risk
- Medium confidence tickets marked for review protect the business from silently misrouting sensitive issues
- Response templates should be professional but friendly, acknowledging the customer's issue category
- This skill does not retrain or modify the model - it only interprets predictions and applies business logic
- The routing_recommendation is what gets displayed to the support team in the Gradio UI
- Human review is required for medium confidence, regardless of category, to ensure escalation of ambiguous cases
