---
name: apply_feedback
description: "Process human feedback on incorrect predictions and flag for retraining"
mode: organisational
tags: [feedback-loop, human-in-the-loop, quality-control]
---

# Apply Feedback Skill

## Purpose

Handle human correction and validation of model predictions. When the system makes an incorrect prediction or routing decision, admins can provide corrected labels and feedback. This skill records the feedback for potential model retraining and continuous improvement.

## Workflow

### Input Requirements

```python
state.feedback = {
    "interaction_id": 42,                      # Links to original prediction
    "original_message": "I want to cancel",
    "original_prediction": "SUBSCRIPTION",
    "corrected_label": "CANCEL",               # Human's correct label
    "prediction_correct": False,
    "confidence_acceptable": True,
    "routing_correct": False,
    "response_appropriate": False,
    "feedback_notes": "Should be routed to retention team",
    "reviewed_by": "admin_user_id"
}
```

### Processing Steps

1. **Validate Feedback**
   - Check interaction_id exists in database
   - Verify corrected_label is in valid category list (11 categories)
   - Ensure reviewed_by has admin privileges

2. **Calculate Correction Metadata**
   - Is prediction incorrect? (original_prediction ≠ corrected_label)
   - Was routing decision wrong? (route_decision ≠ human expectation)
   - How confident was system? (use original confidence_score)
   - Confidence level of the error (was it a borderline case?)

3. **Store Feedback Record**
   ```json
   {
     "feedback_id": 123,
     "interaction_id": 42,
     "timestamp": "2026-03-22T14:35:00Z",
     "original_message": "I want to cancel",
     "original_prediction": "SUBSCRIPTION",
     "corrected_label": "CANCEL",
     "original_confidence": 0.87,
     "prediction_error": true,
     "routing_error": true,
     "response_error": true,
     "feedback_notes": "Should be routed to retention team",
     "reviewed_by": "admin_user_id",
     "approved_for_retraining": false,
     "status": "pending_review"
   }
   ```

4. **Flag for Retraining (Optional)**
   - If feedback is from trusted admin
   - If pattern indicates systematic error
   - Add to retraining queue

## Output

**State Updates:**
- `state.feedback_recorded = True`
- `state.feedback_id = 123`
- `state.last_feedback_timestamp = "2026-03-22T14:35:00Z"`

**Database Update:**
- Insert into `feedback` table
- Link to original interaction via `interaction_id` (foreign key)

## Error Handling

### Invalid Feedback Cases

```python
# Case 1: Interaction not found
if interaction_id not in database:
    raise FeedbackError("Interaction not found: {interaction_id}")

# Case 2: Invalid category
if corrected_label not in valid_categories:
    raise FeedbackError("Invalid category: {corrected_label}")

# Case 3: Unauthorized reviewer
if reviewed_by not in admin_users:
    raise FeedbackError("User {reviewed_by} not authorized to provide feedback")
```

## Integration Points

**Triggered By:**
- Admin Dashboard (on-demand correction)
- Batch Feedback Upload (CSV import)
- Human Review Queue (systematic errors)

**Feeds Into:**
- Analytics Pipeline (accuracy tracking)
- Retraining Pipeline (model improvement)
- Quality Metrics (category-level performance)

## Metrics Tracked

Per feedback entry:
- Prediction accuracy (binary: correct/incorrect)
- Routing accuracy (binary)
- Response quality (binary)
- Category of error (which category was hardest?)
- Confidence of error (was it a close call?)
- Time to correction (how long before admin caught it?)

## Data Retention

- All feedback stored indefinitely
- Hash original message for PRIVACY compliance
- Keep audit trail of reviewer corrections over time
- Enable retraining from historical feedback

## Future Enhancement

**Retrain Trigger:**
```python
if approved_feedback_count >= 100:
    trigger_retrain_pipeline(
        base_training_data,
        approved_feedback_corrections
    )
```

When approved feedback reaches threshold:
1. Combine original 26,872 training samples with corrected feedback
2. Retrain all 3 models with expanded dataset
3. Re-evaluate on original validation set
4. Compare metrics vs. previous model version
5. Deploy if improvement ≥ 0.5% macro F1

