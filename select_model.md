---
name: select_model

description:
  Compare evaluation metrics across three candidate models and select the best performer using weighted F1 as the primary criterion. Document the selection rationale including trade-offs and business implications for the chosen model.

tags: [model_selection, comparison, decision, justification]

mode: llm_driven
---

# Select Model Skill

## Role
This is an LLM-driven skill. The metrics comparison is performed deterministically by sklearn and Python. The LLM then receives the evaluation results and produces a plain-English justification for the final model selection, explaining the trade-offs to a business audience.

## When to use
After evaluate_models_node has written evaluation_results to state. Before run_inference_node. Triggered once during the training pipeline.

## How to execute
1. Read evaluation_results from state (contains macro F1, weighted F1, and per-class F1 for all 3 models)
2. Extract weighted F1 scores for each model (primary selection criterion):
   - Weighted F1 balances per-class performance and accounts for class distribution
   - It is more realistic for real-world triage where some categories are more common
3. Identify the model with the highest weighted F1 score
4. If models are within 1 percentage point, examine per-class F1 for critical categories:
   - Security Concern: highest priority (security risk)
   - Account Suspension: high priority (customer retention)
   - Payment Problem: high priority (revenue impact)
5. Use the LLM to produce a 3-5 sentence plain-English justification that:
   - Names the selected model and its weighted F1 score
   - Compares it to the other two models
   - Explains why this choice is best for customer support triage
   - Acknowledges any trade-offs (e.g., "Linear SVM performs better on Security Concerns but is slower")
6. Save the selected model name and justification to state

## Inputs from agent state
- evaluation_results: dict - raw metrics for all 3 models
      Structure:
      {
        "Logistic Regression": {
            "macro_f1": float,
            "weighted_f1": float,
            "per_class_f1": dict  (class name → f1 score)
        },
        "Linear SVM": {
            "macro_f1": float,
            "weighted_f1": float,
            "per_class_f1": dict
        },
        "Naive Bayes": {
            "macro_f1": float,
            "weighted_f1": float,
            "per_class_f1": dict
        }
      }
- trained_models: dict - the 3 trained, calibrated models

## Outputs to agent state
- selected_model_name: str - the name of the chosen model (e.g., "Linear SVM")
- selected_model: sklearn model - the chosen trained model (unwrapped from trained_models)
- selection_justification: str - plain-English explanation of the decision
- model_comparison_summary: dict - side-by-side metrics for all 3 models for audit trail

## Output format
Appends to state.messages:
"[select_model] Selected: Linear SVM (Weighted F1: 0.89). Justification: Linear SVM achieves the highest weighted F1 score at 0.89, outperforming Logistic Regression (0.87) and Naive Bayes (0.81). Its strength on Security Concerns (0.94) and Payment Problems (0.91) makes it optimal for high-risk triage. Trade-off: slightly slower inference than Logistic Regression, but acceptable for support workflow."

## Notes
- Weighted F1 is the primary metric because it reflects real-world ticket distribution
- Per-class F1 trade-offs should be noted if they are substantial (>5 percentage points)
- The selected model will be saved to disk at the end of the pipeline for inference time use
- Do not use training accuracy - only use validation set metrics
