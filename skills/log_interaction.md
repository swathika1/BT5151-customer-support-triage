---
name: log_interaction
description: "Store customer-scoped chat turns, context JSON, routing trace, and admin feedback fields in SQLite"
mode: organisational
---

## When to use
Final persistence node in the serving pipeline. Logs each chat turn so the support UI, admin UI, and feedback loop all work from the same record.

## How to execute
1. Open a SQLite connection to `artifacts/interactions.db`
2. Ensure the interactions schema exists and includes:
   - customer identity fields
   - model outputs
   - routing outputs
   - response-generation metadata
   - scoped `context_json`
   - clarification state
   - admin feedback fields
3. Insert one row per user turn into `interactions`
4. Persist structured trace data in the same row so the admin UI can reconstruct the full pipeline
5. Handle failures gracefully
   - If logging fails, the customer should still receive the reply
6. Commit and close the connection

## Inputs from agent state
- `timestamp`
- `customer_id`
- `customer_name`
- `raw_message`
- `detected_language`
- `translated_message`
- `predicted_label`
- `confidence_score`
- `route_decision`
- `response_final`
- `class_probabilities`
- `trace_logs`
- `response_generation`
- `context_json`

## Output format
- One row inserted into the `interactions` table in `artifacts/interactions.db` with all fields above.
- If logging fails, the customer still receives a reply, but the interaction may not be persisted.
- `needs_more_context`
- `clarification_prompt`
- `resolved_order_id`

## Outputs to agent state
- `interaction_id`: database id of the stored interaction

## Database schema
```sql
CREATE TABLE interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    raw_message TEXT,
    detected_language TEXT,
    translated_message TEXT,
    customer_id TEXT,
    customer_name TEXT,
    predicted_label TEXT,
    confidence_score REAL,
    route_decision TEXT,
    response TEXT,
    class_probabilities TEXT,
    pipeline_trace TEXT,
    response_generation TEXT,
    context_json TEXT,
    needs_more_context INTEGER DEFAULT 0,
    clarification_prompt TEXT,
    resolved_order_id TEXT,
    feedback_flag INTEGER DEFAULT 0,
    feedback_reason TEXT,
    feedback_suggested_category TEXT,
    feedback_updated_at TEXT
);

CREATE TABLE admin_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    interaction_id INTEGER NOT NULL,
    flagged INTEGER NOT NULL DEFAULT 1,
    reason TEXT,
    suggested_category TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(interaction_id) REFERENCES interactions(id)
);
```

## Logging requirements
- `customer_id` is required so the chat UI can show history for one selected user only
- `context_json` must be stored because the admin reviewer needs to see what grounded the response
- `needs_more_context`, `clarification_prompt`, and `resolved_order_id` must be stored so follow-up turns can continue correctly
- `pipeline_trace` should preserve each node's summary and details for debugging and review

## Notes
- Chat history should be fetched by `customer_id`, ordered by timestamp ascending for replay in the chat UI.
- The assistant must never show raw JSON to the customer, but logging raw `context_json` internally is expected for auditability.
- Admin reviewers should be able to determine whether the assistant followed policies such as:
  - first-turn greeting
  - prime eligibility handling
  - invoice navigation guidance
  - shipped-order cancellation restrictions
- If policy compliance becomes part of retraining, this logged context is the source of truth.
