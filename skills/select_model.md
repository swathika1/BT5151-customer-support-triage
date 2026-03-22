---
name: select_model
description: "Select best model by Macro F1 and derive confidence thresholds from precision-confidence curves"
mode: organisational
---

## When to use
Executed after evaluate_models_node to select the winning model and compute routing thresholds.

## How to execute

### Step 1: Select by Macro F1
- Identify model with highest Macro F1 score on validation set
- This model becomes the production model

### Step 2: Derive Empirical Routing Thresholds
- For the selected model, get predicted class probabilities on validation set
- For each unique confidence threshold value observed:
  1. Count predictions meeting or exceeding that threshold
  2. Calculate precision among those predictions (% correct)
3. Generate precision-confidence curve: plot precision vs confidence threshold
4. Identify tau_high: **lowest confidence threshold where precision ≥ 0.90**
   - Predictions with confidence ≥ tau_high are high-confidence → AUTO_REPLY
5. Identify tau_low: **highest confidence threshold where precision ≥ 0.70**
   - Predictions with confidence ≥ tau_low but < tau_high are medium-confidence → CLARIFY
   - Predictions with confidence < tau_low are low-confidence → ESCALATE

### Step 3: Store in State
- selected_model: The winning model instance
- selected_model_name: String name ("Logistic Regression", "Linear SVM", or "Naive Bayes")
- tau_high: Confidence threshold for AUTO_REPLY (float)
- tau_low: Confidence threshold for ESCALATE trigger (float)

## Inputs from agent state
- trained_models: All 3 trained models (from train_models_node)
- evaluation_results: Metrics for all 3 models (from evaluate_models_node)
- val_texts: Validation TF-IDF matrix (from preprocess_data_node)
- y_val: Validation labels (from preprocess_data_node)

## Outputs to agent state
- selected_model: Best model instance (ready for inference)
- selected_model_name: String name of selected model
- tau_high: Float threshold for AUTO_REPLY routing (typically 0.80-0.95)
- tau_low: Float threshold for ESCALATE routing (typically 0.60-0.75)

## Output format
```python
{
  "selected_model": <trained model instance>,
  "selected_model_name": str,  # "Logistic Regression" | "Linear SVM" | "Naive Bayes"
  "tau_high": float,  # e.g., 0.85
  "tau_low": float    # e.g., 0.65
}
```

Visualization: precision_confidence_curve.png
- X-axis: Confidence score (0.0 to 1.0)
- Y-axis: Precision (0.0 to 1.0)
- Curve shows how precision changes with confidence threshold
- Horizontal lines at precision = 0.90 and 0.70
- Vertical lines at tau_high and tau_low intersections

## Notes
- **Data-justified thresholds**: tau_high and tau_low are NOT hardcoded. They emerge from validation set behavior, ensuring routing decisions are empirically grounded.
- **Three-tier routing logic**:
  - confidence ≥ tau_high → AUTO_REPLY (send prepared response immediately)
  - tau_low ≤ confidence < tau_high → CLARIFY (ask customer for more details)
  - confidence < tau_low → ESCALATE (route to human agent)
- **Precision at thresholds**: At tau_high, 90% of AUTO_REPLY predictions are correct (low false positive rate). At tau_low, 70% are correct (acceptable for CLARIFY).
- **Example**: If model predicts "REFUND" with 92% confidence, and tau_high=0.85, the system chooses AUTO_REPLY. If confidence was 72% and tau_low=0.70, system chooses CLARIFY.
- **Calibration dependency**: Reliable precision-confidence curves depend on well-calibrated probabilities (hence calibration step in train_models_node).
