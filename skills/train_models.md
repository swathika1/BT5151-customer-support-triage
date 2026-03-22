---
name: train_models
description: "Train three candidate classifiers (LR, SVM, NB) with hyperparameter tuning and calibration"
mode: organisational
---

## When to use
Executed after preprocess_data_node to train multiple models for comparison and selection.

## How to execute
1. Read preprocessed training data from state (train_texts TF-IDF matrix, y_train labels)
2. Train Logistic Regression:
   - Use GridSearchCV to tune regularization parameter C ∈ {0.1, 1.0, 10.0}
   - Use 5-fold cross-validation with macro F1 scoring
   - Wrap with CalibratedClassifierCV (isotonic method) for reliable probability estimates
3. Train Linear SVM:
   - Use GridSearchCV to tune C ∈ {0.01, 0.1, 1.0}
   - Use 5-fold cross-validation with macro F1 scoring
   - LinearSVC has no native predict_proba() → calibration wrapper is mandatory
   - Use sigmoid calibration method (better for SVM decision scores)
4. Train Multinomial Naive Bayes:
   - No hyperparameter tuning (alpha=1.0 default for text)
   - NB produces probabilities natively, no calibration wrapper needed
   - Handles sparse TF-IDF features directly
5. Store all three models in state.trained_models dictionary

## Inputs from agent state
- train_texts: Sparse TF-IDF matrix (from preprocess_data_node)
- y_train: Integer-encoded labels (from preprocess_data_node)
- label_encoder: For class count and names (from preprocess_data_node)

## Outputs to agent state
- trained_models: Dictionary with 3 entries:
  ```python
  {
    "Logistic Regression": CalibratedClassifierCV(LogisticRegression),
    "Linear SVM": CalibratedClassifierCV(LinearSVC),
    "Naive Bayes": MultinomialNB
  }
  ```

## Output format
Each model instance supports:
- model.predict(X) → numpy array of predicted class indices
- model.predict_proba(X) → numpy array of shape (n_samples, n_classes) with probabilities per class
- model.classes_ → array of unique class labels

## Notes
- **Calibration is critical**: Only calibrated models produce probability estimates (confidence scores) that reflect true correctness likelihood. Without calibration, SVM confidence scores would be meaningless.
- **GridSearchCV scoring**: Using macro F1 (average F1 across all classes) as scoring metric matches downstream model selection criteria. This ensures selected model optimizes for balanced performance across rare and common categories.
- **No test set**: Training nodes never touch test_texts. Strict separation ensures test set remains held out for final evaluation only.
- **Class weights**: class_weight='balanced' for LR and SVM automatically adjusts loss for imbalanced classes (some categories have fewer samples).
- **Random state**: Set to 42 for reproducibility across notebook reruns.
