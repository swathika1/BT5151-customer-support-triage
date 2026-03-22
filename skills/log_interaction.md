---
name: log_interaction
description: "Store customer interaction details and audit trail to SQLite database for compliance and feedback collection"
mode: organisational
---

## When to use
Seventh and final node in serving pipeline. Logs interaction details to database for audit, feedback collection, and optional retraining.

## How to execute
1. Open SQLite connection to artifacts/support_system.db
2. Insert one row into interactions table:
   - conversation_id: UUID from state
   - raw_message: Original customer message
   - detected_language: Language identifier (from detect_language_node)
   - translated_message: English version (from translate_to_english_node)
   - predicted_label: Model's prediction (from run_inference_node)
   - confidence_score: Model's confidence (from run_inference_node)
   - route_decision: Routing tier (from confidence_router_node)
   - response_final: Sent response (from draft_response_node)
   - created_at: Current timestamp
3. Insert N rows into trace_logs table (one per node execution):
   - interaction_id: Foreign key to interactions row
   - active_skill: Node name (detect_language, translate_to_english, etc.)
   - state_snapshot: JSON serialization of full state
   - timestamp: Execution timestamp
4. Handle errors gracefully:
   - If DB write fails, log warning but don't crash serving pipeline
   - Customer still gets response even if logging fails
5. Commit transaction and close connection

## Inputs from agent state
- conversation_id: UUID (from initial state setup)
- raw_message: Original customer message
- detected_language: Language from detect_language_node
- translated_message: English message from translate_to_english_node
- predicted_label: Category from run_inference_node
- confidence_score: Confidence from run_inference_node
- route_decision: Routing decision from confidence_router_node
- response_final: Final response from draft_response_node
- trace_logs: List of logged entries from all nodes

## Outputs to agent state
- (No state modifications; side effect is database write)

## Database Schema
```sql
CREATE TABLE interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT UNIQUE NOT NULL,
    raw_message TEXT NOT NULL,
    detected_language TEXT,
    translated_message TEXT,
    predicted_label TEXT,
    confidence_score REAL,
    route_decision TEXT,
    response_final TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE trace_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    interaction_id INTEGER NOT NULL,
    active_skill TEXT,
    state_snapshot TEXT,  -- JSON
    timestamp TEXT,
    FOREIGN KEY(interaction_id) REFERENCES interactions(id)
);

CREATE TABLE feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    interaction_id INTEGER NOT NULL UNIQUE,
    admin_correction TEXT,  -- Corrected category if prediction was wrong
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(interaction_id) REFERENCES interactions(id)
);

CREATE TABLE model_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT,
    trained_at TIMESTAMP,
    macro_f1 REAL,
    weighted_f1 REAL,
    n_training_samples INTEGER
);
```

## Output format
(No state output; side effects are database writes)

Database file: artifacts/support_system.db

## Notes
- **Audit trail purpose**: Every interaction is logged for compliance, debugging, and customer service investigation. If customer disputes a response, we have full trace of what message was sent and why.
- **Feedback loop setup**: interactions + feedback + model_versions tables enable admin workflow:
  1. Admin reviews interactions (via admin UI)
  2. If model made wrong prediction, admin records correction in feedback table
  3. Optional: retrain_from_feedback_node reads feedback table and retrains model on corrected labels
- **State snapshot**: Complete state object (all 30+ fields) serialized to JSON in trace_logs. This enables full reconstruction of decision path. Example: "Why did the system route this to ESCALATE?" → Look at state_snapshot and see: confidence_score=0.58, tau_low=0.65, so 0.58 < 0.65 triggered escalation.
- **Error handling**: If database write fails (disk full, permission error, etc.), log warning but don't prevent response delivery. Customer service is more important than analytics.
- **Privacy**: Consider storing hashed customer identifiers instead of raw messages in production. Current implementation stores full messages for debugging.
- **Performance**: SQLite is adequate for up to ~1M interactions/month. For higher volume, migrate to PostgreSQL or DynamoDB.
- **Indexing**: Create indexes on conversation_id and created_at for fast lookups by UI:
  ```sql
  CREATE INDEX idx_conversation_id ON interactions(conversation_id);
  CREATE INDEX idx_created_at ON interactions(created_at);
  CREATE INDEX idx_interaction_fk ON trace_logs(interaction_id);
  ```
