# Customer Support Ticket Classification System

## Overview

Complete **multi-agent agentic ML pipeline** for automatically classifying customer support tickets and routing them to the appropriate team. Built with state-based architecture using **Pydantic SupportAgentState** with **14 specialized agents** (5 training + 9 serving + 1 optional admin) working through training, serving, and optional feedback phases. Each serving request is scoped to a selected customer, with e-commerce context (orders, shipping, payments) resolved before generating a grounded, policy-safe reply.

### Core Technologies
- **Pydantic** - SupportAgentState for shared state management across all agents
- **scikit-learn** - ML models (Logistic Regression, LinearSVC, Naive Bayes)
- **OpenAI GPT-4o-mini** - LLM-driven agents (translation, response drafting)
- **TF-IDF** - Text vectorization (1-2 grams, up to 5,000 features)
- **CalibratedClassifierCV** - Probability calibration for confidence scoring
- **SQLite** - Interaction logging and admin feedback storage (`artifacts/interactions.db`)
- **Python `http.server`** - Built-in HTTP server serving the chat UI and REST API on port 7860

---

## 🤖 Agent Pipeline - Complete Implementation

### **PHASE 1: TRAINING PIPELINE** (5 Agents - All Executed ✅)


#### Agent 1: **preprocess_data_node**
- **Skill File:** `preprocess_data.md` (Mode: organisational)
- **Input:** 26,872 customer support tickets (Bitext dataset)
- **Output:** TF-IDF sparse matrices (4,808 features), stratified 70/15/15 split, label encoder
- **Processing Steps:**
  1. Load Bitext customer support dataset from Hugging Face
  2. Clean text: lowercase, remove URLs, emails, hashtags, punctuation, {{placeholders}}
  3. Encode 11 categories → integer labels (ACCOUNT, CANCEL, CONTACT, DELIVERY, FEEDBACK, INVOICE, ORDER, PAYMENT, REFUND, SHIPPING, SUBSCRIPTION)
  4. Stratified split: 18,809 train / 4,032 val / 4,031 test
  5. Fit TF-IDF vectorizer on training only (ngram_range=(1,2), max_features=5000, sublinear_tf=True, min_df=2)
  6. Transform val/test without refitting
- **Artifact:** SupportAgentState with train_texts, val_texts, test_texts, tfidf_vectorizer, label_encoder

---

#### Agent 2: **train_models_node**
- **Skill File:** `train_models.md` (Mode: organisational)
- **Input:** TF-IDF training vectors (18,809 × 4,808), training labels
- **Output:** 3 calibrated, production-ready models in state.trained_models dict
- **Processing Steps:**
  1. **Logistic Regression** with GridSearchCV (C ∈ [0.1, 1.0, 10.0], 5-fold CV, macro F1 scoring)
     - Best C=1.0, cross-validation F1=0.9932
     - Wrapped with CalibratedClassifierCV (isotonic method)
  2. **Linear SVM** with GridSearchCV (C ∈ [0.01, 0.1, 1.0], 5-fold CV, macro F1 scoring)
     - Best C=0.1, cross-validation F1=0.9944
     - Wrapped with CalibratedClassifierCV (sigmoid method) - **ESSENTIAL** for predict_proba()
  3. **Naive Bayes** (MultinomialNB, alpha=1.0, no GridSearchCV)
     - Produces probabilities natively, no calibration wrapper needed
- **Key Feature:** Class weight balancing for imbalanced classes
- **Artifact:** 3 models stored in state.trained_models = {"Logistic Regression": ..., "Linear SVM": ..., "Naive Bayes": ...}

---

#### Agent 3: **evaluate_models_node**
- **Skill File:** `evaluate_models.md` (Mode: llm_driven)
- **Input:** 3 trained models + validation set (4,032 samples)
- **Output:** Metrics dict, confusion matrix PNG, per-class F1 chart PNG, LLM interpretation narrative
- **Processing Steps:**
  1. **Part 1 (Deterministic):** Compute classification metrics for all 3 models
     - Per-model: Macro F1, Weighted F1, per-class F1 scores
     - Generate confusion matrix (11×11 heatmap)
     - Generate per-class F1 bar chart (11 categories × 3 models)
  2. **Part 2 (LLM-driven):** Call GPT-4o-mini with evaluate_models SKILL.md as system prompt
     - Input: Metrics summary, validation sample counts, best model identification
     - Output: Business-language interpretation of model performance
- **Results:**
  - Logistic Regression: Macro F1 = 0.9970, Weighted F1 = 0.9970
  - **Linear SVM (BEST): Macro F1 = 0.9984, Weighted F1 = 0.9985** ⭐
  - Naive Bayes: Macro F1 = 0.9931, Weighted F1 = 0.9933
- **Artifacts:** confusion_matrix.png, per_class_f1.png, evaluation_narrative (text)

---

#### Agent 4: **select_model_node**
- **Skill File:** `select_model.md` (Mode: organisational)
- **Input:** evaluation_results dict with F1 scores for all 3 models
- **Output:** Selected model (Linear SVM), derived routing thresholds, precision-confidence curve PNG
- **Processing Steps:**
  1. Select model with highest Macro F1 → **Linear SVM (0.9984)**
  2. Build precision-confidence curve on validation set:
     - Test 101 confidence thresholds (0.0 to 1.0)
     - For each threshold: count predictions ≥ threshold, measure precision among them
     - Result: Precision ranges from 1.0 (exceptional model quality)
  3. Derive two routing thresholds:
     - **tau_high**: Lowest confidence where precision ≥ 0.90 → **0.0** (all predictions ≥ 0.90)
     - **tau_low**: Highest confidence where precision ≥ 0.70 → **0.99** (extremely selective)
  4. Interpretation: Model is SO good that even lowest-confidence predictions are 90% accurate
- **Artifacts:** Selected model name, tau_high, tau_low, precision_confidence_curve.png

---

#### Agent 5: **persist_artifacts_node**
- **Skill File:** `persist_artifacts.md` (Mode: organisational)
- **Input:** Selected model, vectorizer, encoder, thresholds, evaluation results
- **Output:** 5 serialized files saved to `./artifacts/` directory (2.7 MB total)
- **Files Saved:**
  1. **model.pkl** (2,544,835 bytes) - Calibrated LinearSVC instance, pickle format
  2. **tfidf_vectorizer.pkl** (196,285 bytes) - Fitted TfidfVectorizer with learned vocabulary
  3. **label_encoder.pkl** (357 bytes) - LabelEncoder (maps: 0→ACCOUNT, 1→CANCEL, ..., 10→SUBSCRIPTION)
  4. **thresholds.json** (43 bytes) - JSON: `{"tau_high": 0.0, "tau_low": 0.99}`
  5. **model_info.json** (383 bytes) - Metadata: model_name, trained_at (ISO timestamp), n_classes, classes list, macro_f1, weighted_f1, n_training_samples
- **Ready For:** Serving pipeline production deployment

---

### **PHASE 2: SERVING PIPELINE** (9 Agents - Executed per request ✅)

Each request flows through all 9 agents sequentially. Every agent appends to `state.trace_logs` for a full audit trail.

#### Agent 6: **load_customer_scope_node**
- **Input:** `state.customer_id` (required — a customer must be selected before chatting)
- **Output:** Customer profile, order list, and prior conversation history loaded into state
- **Processing Steps:**
  1. Load customer profile from `ecommerce_data/ecommerce_customers.csv`
  2. Load all orders for that customer from `ecommerce_data/ecommerce_orders.csv`
  3. Fetch last 50 interactions from SQLite (`artifacts/interactions.db`) for this customer
  4. Detect whether the most recent interaction is an unresolved clarification request → stored in `state.pending_interaction`
- **State Updates:** `customer_profile`, `customer_name`, `customer_orders`, `conversation_history`, `pending_interaction`

---

#### Agent 7: **detect_language_node**
- **Skill File:** `detect_language.md` (Mode: organisational)
- **Input:** `state.raw_message`
- **Output:** Detected language code (en, fr, es, de, ja, zh, etc.)
- **Processing:**
  - Short ASCII messages (≤24 chars) are heuristically defaulted to English without calling the library
  - Primary detector: TextBlob; fallback: `langdetect`; final fallback: default to `"en"`
- **State Update:** `state.detected_language`

---

#### Agent 8: **translate_to_english_node**
- **Skill File:** `translate_to_english.md` (Mode: llm_driven)
- **Input:** `state.raw_message` + `state.detected_language`
- **Output:** English translation (or original if already English)
- **Logic:**
  - If `detected_language == "en"`: skip LLM, use original
  - If non-English AND LLM available: call GPT-4o-mini
  - If non-English AND no LLM: use original as-is (graceful fallback)
- **State Update:** `state.translated_message`

---

#### Agent 9: **prepare_contextual_query_node**
- **Input:** `state.translated_message`, `state.customer_orders`, `state.pending_interaction`
- **Output:** Augmented inference text that handles short follow-up messages
- **Processing:** If the current message is a very short follow-up to an unresolved clarification (e.g., "the second one"), the agent enriches `inference_message` with prior context so the ML model classifies the correct intent rather than the bare pronoun.
- **State Update:** `state.inference_message`

---

#### Agent 10: **run_inference_node** ⭐
- **Skill File:** `run_inference.md` (Mode: organisational)
- **Input:** `state.inference_message` (or `state.translated_message`) + saved artifacts from Phase 1
- **Output:** Predicted category, confidence score, class probabilities for all 11 categories
- **Processing Steps:**
  1. Load artifacts from disk on first call (model.pkl, tfidf_vectorizer.pkl, label_encoder.pkl, thresholds.json)
  2. Clean text identically to training (lowercase, regex cleaning)
  3. Vectorize with saved TF-IDF (`transform`, never `refit`)
  4. Call `model.predict_proba()` → shape (1, 11) array
  5. Extract `predicted_label`, `confidence_score`, `class_probabilities`, top-3 predictions
- **State Updates:** `predicted_label`, `confidence_score`, `class_probabilities`, `tau_high`, `tau_low`
- **Critical:** Uses IDENTICAL text cleaning as training (prevents training-serving skew)

---

#### Agent 11: **resolve_context_node**
- **Skill File:** `resolve_context.md` (Mode: organisational)
- **Input:** Customer profile, orders, conversation history, translated query, predicted label
- **Output:** Scoped business context JSON grounded in the selected customer's real data
- **Processing Steps:**
  1. Determine whether query requires order-level context (CANCEL, DELIVERY, INVOICE, ORDER, PAYMENT, REFUND, SHIPPING)
  2. Attempt to resolve the specific order ID from message text, dates, or conversation history
  3. If date mentioned but no ID: list only that customer's matching orders and request disambiguation
  4. For generic delivery/tracking queries with no explicit ID: default to the customer's latest order
  5. Build `context_json` scoped to the exact query type (refund fields, shipping fields, payment fields, etc.)
  6. Set `needs_more_context = True` and a human-readable `clarification_prompt` when the system cannot resolve unambiguously
  7. May adjust `predicted_label` when the customer's wording clearly maps to a different intent (e.g., subscription question classified as CANCEL)
- **State Updates:** `context_json`, `resolved_order_id`, `needs_more_context`, `clarification_prompt`, optionally `predicted_label`
- **Business Rule:** Customer scope never crosses user boundaries — all lookups are scoped to `state.customer_id`

---

#### Agent 12: **confidence_router_node**
- **Skill File:** `confidence_router.md` (Mode: organisational)
- **Input:** `confidence_score`, `tau_high`, `tau_low`, `needs_more_context`, `resolved_order_id`, `predicted_label`
- **Output:** Route decision: `"AUTO_REPLY"` | `"CLARIFY"` | `"ESCALATE"`
- **Routing Logic (priority order):**
  ```
  if needs_more_context:
      route_decision = "CLARIFY"         # Missing order/payment context
  elif resolved_order_id AND category in ORDER_RELATED_CATEGORIES
       AND confidence >= tau_low:
      route_decision = "AUTO_REPLY"      # Grounded order context available
  elif confidence >= tau_high:
      route_decision = "AUTO_REPLY"      # High-confidence non-order query
  elif confidence >= tau_low:
      route_decision = "CLARIFY"         # Medium confidence — ask for clarification
  else:
      route_decision = "ESCALATE"        # Low confidence — route to supervisor
  ```
- **State Update:** `state.route_decision`

---

#### Agent 13: **draft_response_node**
- **Skill File:** `draft_response.md` (Mode: llm_driven)
- **Input:** Predicted category, routing decision, scoped context JSON, conversation history
- **Output:** Policy-safe, customer-facing reply
- **Processing (3-step architecture):**
  1. **Build relevant context slice** (`build_relevant_context_slice`): filters `context_json` to only the fields relevant to the predicted category and routing decision
  2. **Build policy response blueprint** (`build_policy_response_blueprint`): constructs a factual, policy-compliant draft using only data in `context_json`; handles clarification requests, first-greeting logic, and needs_more_context flag
  3. **Polish with LLM** (if OpenAI available): GPT-4o-mini refines the blueprint into a natural reply using the `draft_response.md` skill instructions as the system prompt; if LLM is unavailable the raw blueprint is used as-is
  4. **Enforce response policies** (`enforce_response_policies`): post-LLM guardrails that strip policy metadata, validate greeting rules, and ensure factual accuracy
- **State Updates:** `state.response_final`, `state.response_generation`

---

#### Agent 14: **log_interaction_node** 📊
- **Skill File:** `log_interaction.md` (Mode: organisational)
- **Input:** Complete `SupportAgentState` after all previous agents
- **Output:** Row inserted into `artifacts/interactions.db` (SQLite table: `interactions`)
- **Logged Fields:** timestamp, raw_message, detected_language, translated_message, customer_id, customer_name, predicted_label, confidence_score, route_decision, response, class_probabilities (JSON), pipeline_trace (JSON), response_generation (JSON), context_json (JSON), needs_more_context, clarification_prompt, resolved_order_id
- **State Update:** `state.interaction_id` (auto-increment primary key from SQLite)

---

### **PHASE 3: ADMIN ACTIONS** (1 Agent - Optional, On-Demand)

#### Agent 15: **apply_feedback_node** (Optional Admin Action)
- **Skill File:** `apply_feedback.md` (Mode: organisational)
- **Input:** Human correction + feedback on previous prediction
- **Output:** Feedback record stored for quality tracking & retraining
- **When Triggered:**
  - Admin reviews prediction via the Admin Dashboard (`/admin`) and finds it incorrect
  - Admin corrects the predicted category/routing decision
  - Creates audit trail for model improvement
  - Flags for potential retraining if pattern emerges
- **Processing Steps:**
  1. Validate feedback (interaction exists, category valid)
  2. Insert row into `admin_feedback` table (SQLite)
  3. Update `interactions` table with `feedback_flag`, `feedback_reason`, `feedback_suggested_category`, `feedback_updated_at`
  4. Return updated interaction record
- **API Endpoint:** `POST /api/admin/feedback`
- **Future Integration:** When approved feedback ≥ 100 entries, trigger retraining pipeline combining original dataset + corrected feedback
- **Note:** Not part of main serving pipeline. Triggered by admins through the Admin Dashboard.

---

## 📋 GOVERNANCE LAYER - 13 SKILL.MD FILES + 3 PYTHON MODULES

Each agent corresponds to a SKILL.md file stored in `skills/`. Three additional Python modules provide the business-logic runtime used by the serving pipeline.

### Skill Files

| # | Agent | SKILL.md File | Mode | Purpose |
|---|-------|---------------|------|---------|
| 1 | preprocess_data_node | preprocess_data.md | organisational | Data loading, cleaning, TF-IDF fitting methodology |
| 2 | train_models_node | train_models.md | organisational | Model training, hyperparameter tuning, calibration strategy |
| 3 | evaluate_models_node | evaluate_models.md | **llm_driven** | Metrics computation methodology + LLM interpretation |
| 4 | select_model_node | select_model.md | organisational | Model selection criteria, threshold derivation methodology |
| 5 | persist_artifacts_node | persist_artifacts.md | organisational | Artifact serialization strategy |
| 6 | detect_language_node | detect_language.md | organisational | Language detection algorithm and fallbacks |
| 7 | translate_to_english_node | translate_to_english.md | **llm_driven** | LLM-based translation approach and error handling |
| 8 | run_inference_node | run_inference.md | organisational | Model loading, vectorization, prediction pipeline |
| 9 | resolve_context_node | resolve_context.md | organisational | Customer-scoped context building, order ID resolution, clarification logic |
| 10 | confidence_router_node | confidence_router.md | organisational | Context-aware routing: needs_more_context > grounded order > confidence thresholds |
| 11 | draft_response_node | draft_response.md | **llm_driven** | Policy-blueprint + LLM response generation with enforcement guardrails |
| 12 | log_interaction_node | log_interaction.md | organisational | SQLite audit logging schema and data persistence |
| 13 | apply_feedback_node | apply_feedback.md | organisational | Admin feedback handling, error correction, retraining queue |

### Runtime Python Modules (`skills/`)

| Module | Purpose |
|--------|---------|
| `ecommerce_repository.py` | Loads and normalises data from all 5 e-commerce CSV files (customers, orders, sellers, transporters, products); provides `get_customer_scope`, `build_user_summary`, `list_chat_users` |
| `ecommerce_context.py` | Builds `context_json` for the serving pipeline; resolves order IDs, detects date mentions, prepares disambiguation candidates, sets `needs_more_context` |
| `ecommerce_response.py` | Provides `build_policy_response_blueprint`, `build_relevant_context_slice`, and `enforce_response_policies` — the three policy-enforcement helpers used by `draft_response_node` |

**SKILL.md Format:**
```yaml
---
name: detect_language
description: "Identify if incoming message is English or other language using langdetect"
mode: organisational
tags: [language-detection, preprocessing]
---
# Detect Language Skill

[Markdown body explaining the implementation methodology]
```

---

## 🏗️ STATE-BASED ARCHITECTURE

All agents share a single **SupportAgentState** Pydantic model that flows through the pipeline:

### SupportAgentState Fields

| Category | Field | Type | Purpose |
|----------|-------|------|---------|
| **Control** | messages | list[str] | Log of what each agent did |
| | active_skill | Optional[str] | Current executing skill |
| | mode | str | "train" or "serve" |
| **Training Data** | raw_df | Optional[Any] | Original dataset DataFrame |
| | train_texts, val_texts, test_texts | Optional[Any] | TF-IDF sparse matrices |
| | y_train, y_val, y_test | Optional[Any] | Encoded labels |
| | tfidf_vectorizer | Optional[Any] | Fitted TF-IDF transformer |
| | label_encoder | Optional[Any] | Category encodings |
| **Training Outputs** | trained_models | dict | 3 calibrated models |
| | evaluation_results | dict | F1 scores per model |
| **Model Selection** | selected_model | Optional[Any] | Best model (Linear SVM) |
| | selected_model_name | str | Model name |
| | tau_high, tau_low | float | Routing thresholds (derived from validation set) |
| | artifacts_saved | bool | Persistence flag |
| **Customer Context** | customer_id | str | Selected customer identifier |
| | customer_name | str | Customer display name |
| | customer_profile | dict | Customer record from ecommerce_customers.csv |
| | customer_orders | list[dict] | All orders for this customer |
| | conversation_history | list[dict] | Prior chat turns from SQLite |
| | pending_interaction | dict | Latest unresolved clarification interaction |
| **Serving Inputs** | raw_message | str | Customer's raw input |
| | detected_language | str | Language code (en/fr/es/etc.) |
| | translated_message | str | English translation |
| | inference_message | str | Augmented text used for ML classification |
| **Context Resolution** | context_json | dict | Scoped business context (order, customer, service recovery) |
| | resolved_order_id | str | Order ID resolved from message/history |
| | needs_more_context | bool | True when clarification is required before answering |
| | clarification_prompt | str | Human-readable question to ask the customer |
| **Inference Outputs** | predicted_label | str | Category prediction (e.g., "ORDER") |
| | confidence_score | float | Prediction confidence (0.0-1.0) |
| | class_probabilities | dict | All 11 category probabilities |
| | route_decision | str | "AUTO_REPLY" / "CLARIFY" / "ESCALATE" |
| | response_final | str | Final policy-safe customer reply |
| | response_generation | dict | Metadata: source, LLM used, reason, context_keys |
| | interaction_id | Optional[int] | SQLite primary key of logged interaction |
| **Audit Trail** | trace_logs | list[dict] | Structured execution trace per agent (stage, summary, details, logged_at) |

---

## 📊 TRAINING RESULTS SUMMARY

| Metric | Logistic Regression | Linear SVM | Naive Bayes |
|--------|---------------------|-----------|-------------|
| Macro F1 | 0.9970 | **0.9984** ⭐ | 0.9931 |
| Weighted F1 | 0.9970 | **0.9985** ⭐ | 0.9933 |
| Best Hyperparameter | C=1.0 | C=0.1 | alpha=1.0 |
| Calibration Method | isotonic | sigmoid | native |
| Status | ✅ Trained & Calibrated | ✅ Selected | ✅ Trained & Calibrated |

**Selected Model Justification:**
- Linear SVM achieves highest Macro F1 (0.9984)
- Precision remains ≥0.90 across entire confidence range
- Exceptional quality justifies selective thresholds (tau_low=0.99)

---

## 📁 ARTIFACTS SAVED

**Directory:** `./artifacts/` (generated at runtime, not version-controlled)

| File | Content | Purpose |
|------|---------|---------|
| model.pkl | Calibrated LinearSVC instance | Production inference model |
| tfidf_vectorizer.pkl | Fitted TfidfVectorizer | Text vectorization (fixed vocabulary) |
| label_encoder.pkl | LabelEncoder (0→11 categories) | Category name mapping |
| thresholds.json | `{"tau_high": ..., "tau_low": ...}` | Routing decision boundaries (derived from validation set) |
| model_info.json | Metadata: name, timestamp, classes, F1 scores | Model version tracking |
| interactions.db | SQLite database | Interaction history + admin feedback |

All artifacts are loaded by Agent 10 (`run_inference_node`) at serving time. On first run the full training pipeline executes (~2-3 minutes) and saves these files. Subsequent runs load from disk instantly.

---

## 🗄️ E-COMMERCE DATASET

The serving pipeline resolves queries against real customer data stored in `ecommerce_data/`:

| File | Contents |
|------|---------|
| `ecommerce_customers.csv` | Customer profiles: name, email, phone, prime subscription flag, registered address |
| `ecommerce_orders.csv` | Order records linked by `CID`: order ID, status, amount, currency, dates, payment status, refund eligibility, damage flag |
| `ecommerce_products.csv` | Product catalog linked to orders |
| `ecommerce_sellers.csv` | Seller details enriched into order context |
| `ecommerce_transporters.csv` | Transporter/carrier details used for shipping queries |

This data is loaded by `skills/ecommerce_repository.py` and scoped per customer by Agent 6 (`load_customer_scope_node`). **Customer data never crosses user boundaries** — every lookup is filtered to `state.customer_id`.

---

## 🗃️ DATABASE SCHEMA

`artifacts/interactions.db` contains two tables:

**`interactions`** — one row per chat turn:
```sql
id, timestamp, raw_message, detected_language, translated_message,
customer_id, customer_name, predicted_label, confidence_score,
route_decision, response, class_probabilities (JSON), pipeline_trace (JSON),
response_generation (JSON), context_json (JSON), needs_more_context,
clarification_prompt, resolved_order_id, feedback_flag, feedback_reason,
feedback_suggested_category, feedback_updated_at
```

**`admin_feedback`** — one row per admin review action:
```sql
id, interaction_id (FK), flagged, reason, suggested_category, created_at
```

The schema is created (and migrated with `ALTER TABLE`) automatically on first use by `ensure_database_schema()` — no manual setup required.

---

## 🌐 WEB INTERFACE & REST API

The application runs a native Python HTTP server on **port 7860** (no Gradio dependency required):

```bash
python app.py
# Open http://localhost:7860        ← Customer support chat
# Open http://localhost:7860/admin  ← Admin dashboard
```

### Pages
| URL | Description |
|-----|-------------|
| `/` | Customer-facing support chat — select a customer, type a message, receive a grounded reply |
| `/admin` | Admin dashboard — view all interactions, confidence scores, pipeline traces, submit feedback |

### REST API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/users` | List all selectable customers from the e-commerce dataset |
| `GET` | `/api/user-chat?customer_id=<id>&limit=<n>` | Return user summary + chat history for a customer |
| `GET` | `/api/admin/interactions?limit=<n>` | Return all interactions with summary metrics (admin) |
| `POST` | `/api/classify` | Run the full 9-agent serving pipeline for a customer query |
| `POST` | `/api/admin/feedback` | Submit admin feedback/correction for a logged interaction |

**`POST /api/classify` payload:**
```json
{
  "query": "Where is my order?",
  "customer_id": "C001"
}
```

**`POST /api/admin/feedback` payload:**
```json
{
  "interaction_id": 42,
  "flagged": true,
  "reason": "Wrong category — should be SHIPPING not ORDER",
  "suggested_category": "SHIPPING"
}
```

---

## 🧪 TESTING

### Running Tests
```bash
# Regression tests for ecommerce_repository field normalization
python -m pytest test_ecommerce_repository.py -v

# Verify trained artifacts are intact
python test_model.py

# Smoke test the OpenAI translation integration
python test_openai.py
```

### Test Files
| File | Purpose |
|------|---------|
| `test_ecommerce_repository.py` | Unit tests for `normalize_column_name`, customer/order loading, boolean flag parsing |
| `test_model.py` | Loads `artifacts/` files and verifies model, vectorizer, and encoder are intact |
| `test_backend.py` | Smoke test for `load_trained_pipeline` and `classify_query` pipeline execution |
| `test_openai.py` | Verifies OpenAI API key and translation through GPT-4o-mini |

### Manual API Test
```bash
# Classify a message via the REST API
curl -s -X POST http://localhost:7860/api/classify \
  -H "Content-Type: application/json" \
  -d '{"query": "Where is my order?", "customer_id": "1"}' | python -m json.tool
```

---

## 📝 NOTEBOOK

**File:** `SupportAgentState.ipynb` — exploratory notebook showing all training pipeline agents with executed outputs. The training logic from the notebook has been fully migrated into `app.py` for production use.

---

## OpenAI Integration

### Models Used
- **GPT-4o-mini** - Cost-effective LLM for:
  1. **Translate stage** - Translate non-English tickets to English
  2. **Draft response stage** - Polish the policy blueprint into a natural customer reply

### API Key Setup
```bash
# Option 1: Environment variable
export OPENAI_API_KEY='sk-...'

# Option 2: Create a .env file (loaded automatically on startup)
echo "OPENAI_API_KEY=sk-..." > .env
```

The application detects the API key at startup. If no key is set, LLM features degrade gracefully:
- Translation: falls back to the original (non-English) message
- Response drafting: falls back to the policy blueprint (still factually correct, less polished)

### Cost Estimate
- **Serving**: 1 API call per non-English ticket (~$0.001 per translation) + 1 API call per reply draft

---

## Troubleshooting

**Q: "ModuleNotFoundError: langdetect" or "ModuleNotFoundError: textblob"**
```bash
pip install -r requirements.txt
```

**Q: "ValueError: Selected customer was not found"**
- Ensure `ecommerce_data/ecommerce_customers.csv` exists
- Verify the `customer_id` matches the `CID` column in the CSV

**Q: "OpenAI API key not found"**
```bash
export OPENAI_API_KEY='sk-...'
# OR create a .env file:
echo "OPENAI_API_KEY=sk-..." > .env
```

**Q: Model artifacts not found on startup**
- This is expected on first run — the training pipeline will execute automatically (~2-3 min)
- After training, `artifacts/` will contain `model.pkl`, `tfidf_vectorizer.pkl`, `label_encoder.pkl`, `thresholds.json`, `model_info.json`

**Q: Database error on first interaction**
- The database schema is created automatically — no manual `setup_database.py` required

---

## 📁 File Structure

```
BT5151-customer-support-triage/
├── app.py                              # Main application (3000+ lines):
│                                       #   - All 14 agent functions
│                                       #   - Training pipeline orchestration
│                                       #   - HTTP server + REST API
│                                       #   - Customer chat UI (HTML)
│                                       #   - Admin dashboard UI (HTML)
├── requirements.txt                    # Production dependencies
├── SupportAgentState.ipynb             # Exploratory training notebook
├── bt5151_group_project_2026.pdf       # Project reference document
├── check_dataset.py                    # Dataset inspection utility
├── debug_translation.py                # Translation debugging utility
├── test_ecommerce_repository.py        # Unit tests for ecommerce_repository
├── test_model.py                       # Artifact integrity tests
├── test_backend.py                     # Backend smoke tests
├── test_openai.py                      # OpenAI API integration test
├── ecommerce_data/                     # Structured customer data (CSV)
│   ├── ecommerce_customers.csv
│   ├── ecommerce_orders.csv
│   ├── ecommerce_products.csv
│   ├── ecommerce_sellers.csv
│   └── ecommerce_transporters.csv
├── skills/                             # Skill files & runtime modules
│   ├── __init__.py
│   ├── ecommerce_repository.py         # CSV data access layer
│   ├── ecommerce_context.py            # Context resolution logic
│   ├── ecommerce_response.py           # Policy blueprint & enforcement
│   ├── preprocess_data.md
│   ├── train_models.md
│   ├── evaluate_models.md
│   ├── select_model.md
│   ├── persist_artifacts.md
│   ├── detect_language.md
│   ├── translate_to_english.md
│   ├── run_inference.md
│   ├── resolve_context.md
│   ├── confidence_router.md
│   ├── draft_response.md
│   ├── log_interaction.md
│   └── apply_feedback.md
└── artifacts/                          # Generated at runtime (git-ignored)
    ├── model.pkl
    ├── tfidf_vectorizer.pkl
    ├── label_encoder.pkl
    ├── thresholds.json
    ├── model_info.json
    └── interactions.db
```

---

**Last Updated:** 2025  
**Status:** ✅ Production Ready  
**Pipeline Nodes:** 14 total (5 training + 9 serving) + 1 optional admin  
**Skill Files:** 13 SKILL.md + 3 Python runtime modules  
**Support Categories:** 11 classes (ACCOUNT, CANCEL, CONTACT, DELIVERY, FEEDBACK, INVOICE, ORDER, PAYMENT, REFUND, SHIPPING, SUBSCRIPTION)
