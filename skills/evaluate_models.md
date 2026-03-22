---
name: evaluate_models
description: "Evaluate three models on validation set, compute metrics, generate visualizations, and provide LLM interpretation"
mode: llm_driven
---

## When to use
Executed after train_models_node to assess model performance and gain insights before selection.

## How to execute

### Part 1: Sklearn Metric Computation (Deterministic)
1. For each of the 3 trained models:
   - Get predictions on validation set using model.predict(val_texts)
   - Compute full classification_report (Accuracy, Precision, Recall, F1 per class + macro/weighted averages)
   - Extract: Per-class F1 scores, Macro F1, Weighted F1
2. Store results in state.evaluation_results dictionary

### Part 2: Visualizations (Required by Rubric)
1. Select best model by Macro F1 score
2. Generate Confusion Matrix heatmap:
   - True labels (rows) vs predicted labels (columns)
   - Annotate cells with prediction counts
   - Save to confusion_matrix.png
3. Generate Per-Class F1 Bar Chart:
   - Show all 11 complaint categories on x-axis
   - Show F1 scores (0-1) on y-axis
   - Group bars by model (LR, SVM, NB) for comparison
   - Save to per_class_f1.png

### Part 3: LLM Interpretation (GPT-4o-mini)
1. Prepare metrics summary string with all results
2. Call GPT-4o-mini with this skill's body as system prompt
3. Ask LLM to interpret metrics in business language
4. Store narrative in state.evaluation_narrative

## Inputs from agent state
- trained_models: Dictionary of 3 trained models (from train_models_node)
- val_texts: Validation TF-IDF matrix (from preprocess_data_node)
- y_val: Validation labels (from preprocess_data_node)
- label_encoder: For class name mapping (from preprocess_data_node)

## Outputs to agent state
- evaluation_results: Dictionary with per-model metrics:
  ```python
  {
    "Logistic Regression": {
      "macro_f1": 0.96,
      "weighted_f1": 0.97,
      "per_class_f1": {"ACCOUNT": 0.94, "CANCEL": 0.95, ...},
      "report": full classification_report dict
    },
    "Linear SVM": {...},
    "Naive Bayes": {...}
  }
  ```
- evaluation_narrative: LLM-generated business interpretation (string)

## Output format
```python
{
  "evaluation_results": {
    "<model_name>": {
      "macro_f1": float,
      "weighted_f1": float,
      "per_class_f1": dict[str, float],
      "report": dict (sklearn classification_report)
    }
  },
  "evaluation_narrative": str  # LLM interpretation
}
```

Visualizations:
- confusion_matrix.png: Heatmap showing classification accuracy and confusion patterns
- per_class_f1.png: Bar chart comparing F1 scores across models and categories

## Notes
- **Validation set only**: Models are evaluated on validation set (used for hyperparameter tuning). Test set remains held out for final reporting.
- **Macro F1 emphasis**: This metric is the primary selection criterion because it weights all classes equally. This prevents the model from optimizing for common categories at the expense of rarer ones (e.g., FEEDBACK has fewer samples than ORDER).
- **LLM interpretation**: The narrative explains metrics in business terms. For example: "Model X achieved 95% F1 on billing inquiries, indicating strong performance on high-volume requests. Performance on refunds (80% F1) suggests room for improvement on more complex cases."
- **Visualizations are required**: Confusion matrix and per-class comparison are mandatory project deliverables showing that model evaluation was rigorous and interpretable.
