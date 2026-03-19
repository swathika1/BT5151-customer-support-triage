# Customer Support Ticket Classification System

## Overview

This is the complete implementation of a **multi-agent ML pipeline** for automatically classifying customer support tickets and routing them to the appropriate team. The system uses:

- **LangGraph** for multi-agent orchestration
- **scikit-learn** for ML model training (Logistic Regression, LinearSVC, Naive Bayes)
- **OpenAI GPT-4o-mini** for LLM-driven agents (interpretation, translation)
- **SQLite** for persistent logging and audit trails
- **Gradio** for web-based UI

## Project Structure

### 📋 SKILL.md Specifications (11 files)

These document the design of each pipeline stage:

**Training Pipeline (Run Once):**
1. `preprocess_data.md` - Text cleaning, TF-IDF vectorization, 70/15/15 split
2. `train_models.md` - Train 3 models with GridSearchCV and probability calibration
3. `evaluate_models.md` - Calculate F1/Precision/Recall metrics + GPT-4o-mini interpretation
4. `select_model.md` - Choose best model by weighted F1 + LLM justification
5. `persist_artifacts.md` - Save model.pkl, vectorizer.pkl, encoder.pkl, metrics.json

**Serving Pipeline (Run Per-Request):**
6. `run_inference.md` - Predict category + confidence score
7. `detect_language.md` - Identify ticket language (en/fr/es/etc)
8. `translate_to_english.md` - Translate non-English to English via GPT-4o-mini
9. `confidence_router.md` - Route by confidence threshold or category
10. `draft_response.md` - Generate response template
11. `log_interaction.md` - Log to SQLite interactions table

### 🐍 Python Implementation Files

**Main Notebook:**
- `customer_support_pipeline.ipynb` - Complete Jupyter notebook with:
  - Stage 1-6 node definitions
  - Training execution (Phase 1)
  - Inference testing
  - Gradio UI
  - **NEEDS:** Integration of stages 4B, 5B, 7, 8

**Node Functions:**
- `all_priority_nodes.py` - 7 complete Python functions (ready to integrate):
  ```python
  select_model_node()              # Stage 4
  persist_artifacts_node()         # Stage 4B
  detect_language_node()           # Stage 5
  translate_to_english_node()      # Stage 5B
  confidence_router_node()         # Stage 6
  draft_response_node()            # Stage 7
  log_interaction_node()           # Stage 8
  ```

**Setup & Testing:**
- `setup_database.py` - Initialize SQLite with 4 tables:
  - `interactions` - Every ticket classification
  - `feedback` - Human validation of predictions
  - `model_versions` - Trained model metadata
  - `performance_metrics` - Daily/hourly analytics

- `test_system.py` - Validates:
  - All required packages installed
  - Database schema correct
  - Node functions importable
  - State structure compatible with LangGraph
  - OpenAI API configured

- `INTEGRATION_GUIDE.py` - Step-by-step instructions for integrating all components into notebook

## Quick Start

### Step 1: Initialize System
```bash
# Terminal - Initialize SQLite database
python setup_database.py

# Output: Creates data/interactions.db with 4 tables
```

### Step 2: Validate Components
```bash
# Terminal - Run test suite
python test_system.py

# Expected: 6/6 tests pass ✅
```

### Step 3: Integrate into Notebook
```bash
# Open customer_support_pipeline.ipynb
# Follow INTEGRATION_GUIDE.py for step-by-step instructions
# OR view it in terminal:
python INTEGRATION_GUIDE.py
```

### Step 4: Run Pipeline
```
In Jupyter:
1. Cell 1: Install packages (add langdetect)
2. Cell 2-3: Import + configure OpenAI
3. Cell 4-11: Define training pipeline stages
4. Cell 12: Define and compile training graph
5. Cell 13: Execute Phase 1 training ← Validates all fixes
6. Cell 14+: Define serving pipeline stages
7. Cell 15: Define and compile serving graph
8. Cell 16: Test serving pipeline on 5 sample tickets
9. Cell 17: Launch Gradio UI
```

## Architecture Overview

### Two-Pipeline Design

**Training Pipeline (Phase 1 - Runs Once):**
```
Customer Tickets Dataset
        ↓
    [PREPROCESS]  ← Stage 1: Clean text, TF-IDF, 70/15/15 split
        ↓
    [TRAIN]       ← Stage 2: GridSearchCV on 3 models
        ↓
    [EVALUATE]    ← Stage 3: F1/Precision/Recall + GPT interpretation
        ↓
    [SELECT]      ← Stage 4: Best model by weighted F1 + LLM justification
        ↓
    [PERSIST]     ← Stage 4B: Save artifacts (model.pkl, vectorizer.pkl, metrics.json)
        ↓
    Artifact Storage (artifacts/ directory)
```

**Serving Pipeline (Phase 2 - Runs Per Request):**
```
New Support Ticket (customer_message)
        ↓
    [DETECT LANGUAGE]      ← Stage 5: Identify language
        ↓
    [TRANSLATE]            ← Stage 5B: GPT-4o-mini translates to English if needed
        ↓
    [INFERENCE]            ← Stage 6: Load model + predict category + confidence
        ↓
    [CONFIDENCE ROUTER]    ← Stage 7: Route by confidence threshold
        │
        ├─ Confidence ≥ 0.80     → AUTO APPROVED
        ├─ Confidence 0.60-0.79  → PENDING REVIEW
        └─ Confidence < 0.60     → ESCALATE TO SUPERVISOR
        │
        └─ Category="Security Concern" → Always ESCALATE
        ↓
    [DRAFT RESPONSE]       ← Stage 8: Generate template response
        ↓
    [LOG INTERACTION]      ← Stage 9: Insert into SQLite interactions table
        ↓
    Response sent to customer | Audit trail in database
```

### State Flow (Dict-based for LangGraph)

```python
state = {
    # Input
    "customer_message": "I can't login",
    "conversation_id": "conv_12345",
    
    # After detect_language
    "detected_language": "en",
    "requires_translation": False,
    
    # After translate_to_english
    "translated_message": None,  # null if English
    
    # After inference
    "predicted_category": "Login Issue",
    "confidence_score": 0.92,
    "class_probabilities": {
        "Login Issue": 0.92,
        "Billing Issue": 0.05,
        "Technical Support": 0.03
    },
    
    # After confidence_router
    "routing_decision": "auto_approved",
    "routing_rationale": "Confidence 0.92 >= 0.80 threshold",
    
    # After draft_response
    "response_template": "We've sent a password reset link...",
    
    # After log_interaction
    "interaction_id": 42
}
```

## Database Schema

### interactions table
```sql
interaction_id INT PRIMARY KEY
timestamp DATETIME
original_message TEXT
predicted_category TEXT
confidence_score REAL
routing_decision TEXT  -- "auto_approved" | "pending_review" | "escalate_to_supervisor"
response_template TEXT
detected_language TEXT
translated_message TEXT  -- null if English
model_name TEXT
is_correct BOOLEAN  -- null until human feedback provided
```

### feedback table
```sql
feedback_id INT PRIMARY KEY
interaction_id INT (FOREIGN KEY)
human_category TEXT  -- What was the correct category?
is_correct_prediction BOOLEAN
confidence_threshold_ok BOOLEAN
routing_ok BOOLEAN
response_ok BOOLEAN
feedback_timestamp DATETIME
```

## Models & Performance

### Models Trained
1. **Logistic Regression** - Fast, interpretable
2. **LinearSVC** - Good separation boundaries
3. **MultinomialNB** - Robust for text

### Selection Criteria
- **Macro F1** - Equal weight per category (good for imbalanced categories)
- **Weighted F1** - Weighted by class frequency (realistic performance)
- **Weighted F1** is the metric used for model selection

### Example Metrics
```
              Precision  Recall  F1-Score  Support
Login Issue      0.94     0.91      0.92      245
Billing Issue    0.87     0.89      0.88      312
Security Issue   0.96     0.94      0.95      143
Technical Sup    0.90     0.92      0.91      200

Accuracy                          0.91       900
Macro Avg        0.92     0.91      0.91      900
Weighted Avg     0.91     0.91      0.91      900
```

## Confidence Thresholds

The system uses a three-tier routing strategy:

| Confidence | Routing | Action |
|-----------|---------|--------|
| ≥ 0.80 | Auto Approved | Send response immediately |
| 0.60 - 0.79 | Pending Review | Flag for human review |
| < 0.60 | Escalate | Route to supervisor |

**Category Overrides:**
- Security Concern → Always escalate (regardless of confidence)
- Billing Issue → Flag for review (check before approval)

**Example:**
```
Login Issue, confidence 0.92 → AUTO APPROVED
Billing Issue, confidence 0.75 → PENDING REVIEW
Security Issue, confidence 0.98 → ESCALATED (business rule override)
Unknown Issue, confidence 0.45 → ESCALATED (low confidence)
```

## OpenAI Integration

### Models Used
- **GPT-4o-mini** - Cost-effective LLM for:
  1. **Evaluate stage** - Interpret model metrics to human manager
  2. **Select stage** - Generate business justification for model choice
  3. **Translate stage** - Translate non-English tickets to English

### API Key Setup
```bash
# Terminal - Set environment variable
export OPENAI_API_KEY='sk-...'

# OR in notebook - Use Google Secrets (Colab)
from google.colab import userdata
api_key = userdata.get('OPENAI_API_KEY')
```

### Cost Estimate
- **Training**: ~30-50 API calls (evaluate stage) = ~$0.10-0.20
- **Serving**: 1 API call per non-English ticket = ~$0.001 per translation

## Gradio Web UI

The system includes a web interface for manual ticket classification:

```
┌─────────────────────────────────────────────┐
│  Customer Support Ticket Classifier         │
├─────────────────────────────────────────────┤
│ Customer Message:   [________________]      │
│ Conversation ID:    [________________]      │
│                         [CLASSIFY]          │
├─────────────────────────────────────────────┤
│ Results:                                    │
│  Category: Login Issue                      │
│  Confidence: 92%                            │
│  Routing: Auto Approved                     │
│  Response: We've sent a password reset...   │
│  Database ID: 42                            │
└─────────────────────────────────────────────┘
```

## Troubleshooting

**Q: "ModuleNotFoundError: langdetect"**
```bash
pip install langdetect
```

**Q: "sqlite3.OperationalError: no such table: interactions"**
```bash
python setup_database.py
```

**Q: "OpenAI API key not found"**
```bash
export OPENAI_API_KEY='sk-...'
```

**Q: "TypeError: PipelineState is not subscriptable"**
- Ensure state is dict, not dataclass
- Use `create_initial_state()` helper function

**Q: LangGraph errors about state serialization**
- Verify TypedDict import: `from typing import TypedDict`
- Check PipelineState definition uses TypedDict, not @dataclass

## File Checklist

**Before Running Notebook:**
- [ ] `setup_database.py` ✅ (creates data/interactions.db)
- [ ] `test_system.py` ✅ (validates components)
- [ ] `all_priority_nodes.py` ✅ (7 node functions)
- [ ] `INTEGRATION_GUIDE.py` ✅ (step-by-step instructions)

**SKILL.md Specifications:**
- [x] `preprocess_data.md`
- [x] `train_models.md`
- [x] `evaluate_models.md`
- [x] `run_inference.md`
- [x] `select_model.md`
- [x] `persist_artifacts.md`
- [x] `detect_language.md`
- [x] `translate_to_english.md`
- [x] `confidence_router.md`
- [x] `draft_response.md`
- [x] `log_interaction.md`

**After Running Notebook:**
- [ ] `artifacts/model.pkl` (trained model)
- [ ] `artifacts/vectorizer.pkl` (TF-IDF)
- [ ] `artifacts/encoder.pkl` (label encoder)
- [ ] `artifacts/evaluation_results.json` (metrics)
- [ ] `data/interactions.db` (SQLite with logged tickets)

## Next Steps

1. **Run `setup_database.py`** to initialize SQLite
2. **Run `test_system.py`** to validate components
3. **Open `INTEGRATION_GUIDE.py`** for step-by-step notebook integration
4. **Execute `customer_support_pipeline.ipynb`** with all integrated stages
5. **Test serving pipeline** on sample multilingual tickets
6. **Launch Gradio UI** for manual classification

## Support & Debug

For detailed integration instructions, run:
```bash
python INTEGRATION_GUIDE.py
```

This will print the complete step-by-step guide to console. You can also save it:
```bash
python INTEGRATION_GUIDE.py --save
# Creates: INTEGRATION_GUIDE.txt
```

For system validation:
```bash
python test_system.py
# Runs 6 validation tests and reports results
```

---

**Last Updated:** 2024  
**Status:** ✅ Ready for Integration  
**Test Coverage:** 6/6 validation tests  
**Component Count:** 11 SKILL.md + 7 Python nodes + 3 tools + 1 notebook
