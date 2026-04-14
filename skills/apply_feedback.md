---
name: apply_feedback
description: "Review customer-scoped responses, flag policy or relevance errors, and store corrections for retraining"
mode: organisational
tags: [feedback-loop, human-in-the-loop, quality-control]
---

# Apply Feedback Skill

## When to use
When an admin reviews a customer interaction for accuracy, policy compliance, or relevance, and wishes to flag errors or provide corrections for retraining.

## How to execute
1. Validate feedback:
   - Check `interaction_id` exists
   - Verify `corrected_label` or `suggested_category` is a valid support category
   - Ensure the reviewer is authorized
2. Review the stored interaction holistically:
   - Assess if the assistant used the correct customer context
   - Check if the response followed policy and was appropriate
   - Record corrections and notes for retraining

## Input requirements
```python
state.feedback = {
    "interaction_id": 42,
    "original_message": "Where is my order?",
    "customer_id": "7",
    "original_prediction": "ORDER",
    "corrected_label": "DELIVERY",
    "prediction_correct": False,
    "routing_correct": False,
    "response_appropriate": False,
    "used_correct_context": False,
    "policy_followed": False,
    "feedback_notes": "User asked about shipping. Bot should have requested order ID from the orders on 31/03/26.",
    "suggested_category": "DELIVERY",
    "reviewed_by": "admin_user_id"
}
```

## Output format
- `feedback_flag`: bool
- `feedback_reason`: str
- `feedback_suggested_category`: str
- `feedback_notes`: str

   - `raw_message`
   - `predicted_label`
   - `route_decision`
   - `response`
   - `context_json`
   - `pipeline_trace`

3. Check policy compliance
   - Was the first reply greeted properly when it should have been?
   - If user was not prime, did the response avoid promising fast delivery or prime-only cancellation/return privileges?
   - If invoice-related, did the response direct the user to `My Profile -> Past Orders`?
   - If shipped, did the response avoid offering cancellation and suggest return after delivery instead?
   - If identifier was missing, did the bot ask for the right order or payment reference instead of guessing?
   - Did the bot keep the final reply natural-language only and avoid printing JSON?

4. Store the feedback record
   - Mark the interaction row as flagged or cleared
   - Save feedback notes and suggested category
   - Optionally enqueue the case for future retraining

## Recommended review checklist
- `prediction_correct`
- `routing_correct`
- `response_appropriate`
- `used_correct_context`
- `policy_followed`
- `suggested_category`
- `feedback_notes`

## Output

State or persistence updates:
- mark interaction as flagged or cleared
- save `feedback_reason`
- save `feedback_suggested_category`
- save `feedback_updated_at`
- optionally insert a row into `admin_feedback`

## Example stored feedback record
```json
{
  "interaction_id": 42,
  "flagged": true,
  "reason": "Bot answered with a generic order status but the user actually asked for refund status.",
  "suggested_category": "REFUND",
  "created_at": "2026-04-01T15:20:00Z"
}
```

## Notes
- Feedback should focus on grounded correctness, not just tone.
- The most important question is whether the response matched the selected customer's actual data and the business rules.
- High-value feedback cases for retraining:
  - wrong intent after a short clarification follow-up
  - wrong order chosen when user provided only a date
  - response violated prime, invoice, or shipped-order policy
  - response exposed irrelevant details instead of narrowing to the asked topic
