---
name: run_inference
description: "Load trained model from disk, vectorize translated message, predict category and confidence score"
mode: organisational
---

## When to use
Fourth node in serving pipeline. Runs trained model on English message to predict support category and confidence.

## How to execute
1. Load artifacts from disk (if not already in state):
   - artifacts/model.pkl → trained classifier
   - artifacts/tfidf_vectorizer.pkl → fitted TfidfVectorizer
   - artifacts/label_encoder.pkl → fitted LabelEncoder
   - artifacts/thresholds.json → tau_high, tau_low
2. Clean translated_message using identical text preprocessing as training:
   - Lowercase
   - Remove {{placeholders}}, URLs, emails, hashtags, punctuation
   - Collapse whitespace
3. Vectorize cleaned message:
   - Call tfidf_vectorizer.transform([cleaned_text]) (NOT fit_transform!)
   - Produces sparse matrix of shape (1, n_features)
4. Get predictions:
   - model.predict() → single predicted class index (integer 0-10)
   - model.predict_proba() → probability distribution across all 11 classes
5. Extract outputs:
   - predicted_label: Decode class index to category name
   - confidence_score: Maximum probability (highest class probability)
   - class_probabilities: Dictionary of all class probabilities
6. Log confidence score, top 3 predictions, and decision reasoning
7. Store all outputs in state for confidence_router_node

## Inputs from agent state
- translated_message: English version of customer message (from translate_to_english_node)
- tfidf_vectorizer: Fitted vectorizer (loaded from artifacts/tfidf_vectorizer.pkl)
- label_encoder: Fitted encoder (loaded from artifacts/label_encoder.pkl)

## Outputs to agent state
- predicted_label: Predicted support category (string, e.g., "REFUND")
- confidence_score: Maximum probability (float, 0.0-1.0)
- class_probabilities: Dictionary mapping category → probability:
  ```python
  {
    "ACCOUNT": 0.05,
    "CANCEL": 0.12,
    "CONTACT": 0.08,
    "DELIVERY": 0.02,
    "FEEDBACK": 0.01,
    "INVOICE": 0.04,
    "ORDER": 0.15,
    "PAYMENT": 0.08,
    "REFUND": 0.38,
    "SHIPPING": 0.04,
    "SUBSCRIPTION": 0.03
  }
  ```
- tau_high, tau_low: Routing thresholds (loaded from artifacts/thresholds.json)

## Output format
```python
{
  "predicted_label": str,  # e.g., "REFUND"
  "confidence_score": float,  # e.g., 0.92 (max probability in class_probabilities)
  "class_probabilities": dict  # {category: probability}
}
```

## Notes
- **Training-serving consistency**: The text cleaning function MUST BE IDENTICAL to preprocessing during training. Any difference (punctuation handling, case conversion, etc.) introduces training-serving skew and degrades predictions.
- **Vectorizer reuse**: Vocabulary is fixed from training data. transform() NOT fit_transform(). If vectorizer is retrained during serving, it adapts to new words—breaking consistency.
- **Confidence score definition**: The maximum probability among all 11 classes. High confidence (e.g., 0.92) indicates model is very sure of prediction. Low confidence (e.g., 0.45) indicates ambiguous message.
- **Class probabilities interpretation**: All probabilities sum to 1.0. Useful for understanding model uncertainty. If top 2 classes are similar (e.g., 0.40 vs 0.38), message is ambiguous → should route to CLARIFY.
- **Artifact loading**: Artifacts may be loaded once and cached in state to avoid repeated disk I/O. Current implementation reloads every inference (simple but slower). Future optimization: load once and keep in state.
- **Error handling**: If model or vectorizer loading fails, raise clear error with artifact location. Don't attempt graceless fallbacks (no "reset to default prediction" - that would confuse customers).
