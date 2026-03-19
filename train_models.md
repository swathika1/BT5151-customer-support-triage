---
name: train_models

description: 
  Train three candidate classifiers (Logistic Regression, Linear SVM, Naive Bayes) on TF-IDF training features. All models are wrapped with probability calibration for reliable confidence scores at inference time.

tags: [training, logistic_regression, linear_svm, naive_bayes, calibration]

mode: organisational
---

# Train Models Skill

## Role
This is an organisational skill. The Python code in train_models_node was written to satisfy this specification. The LLM does not read this file at runtime. All operations are deterministic.

## When to use
After preprocess_data_node has written train_texts and y_train to state. Before evaluate_models_node runs. Triggered once during the training pipeline.

## How to execute
1. Read train_texts, y_train, val_texts, y_val and label_encoder from state (val_texts and y_val are not used for training just passed through for evaluate_models_node)
2. Train Logistic Regression with GridSearchCV:
   - Param grid: C = [0.1, 1.0, 10.0]
   - GridSearchCV(lr, param_grid, cv=5, scoring='f1_macro')
   - Fit on train_texts, y_train - best estimator selected automatically
   - Wrap best estimator with CalibratedClassifierCV(cv=5, method='isotonic')
3. Train Linear SVM with GridSearchCV:
   - Param grid: C = [0.01, 0.1, 1.0]
   - GridSearchCV(svm, param_grid, cv=5, scoring='f1_macro')
   - Fit on train_texts, y_train - best estimator selected automatically
   - LinearSVC does not produce probabilities natively
   - Wrap best estimator with CalibratedClassifierCV(cv=5, method='sigmoid')
4. Train Naive Bayes:
   - MultinomialNB(alpha=1.0) - no GridSearchCV needed
   - alpha=1.0 is the standard default for text classification
   - Already produces probabilities natively - no calibration wrapper needed
   - Fit on train_texts, y_train
   - Requires non-negative features - TF-IDF satisfies this
5. Store all three trained models in state.trained_models dictionary
6. Do not touch test_texts at any point - test data is only used in evaluation

## Inputs from agent state
- train_texts: sparse matrix - TF-IDF features for training set
- val_texts: sparse matrix - TF-IDF features for validation set
- y_train: numpy array - encoded labels for training set
- y_val: numpy array - encoded labels for validation set
- label_encoder: fitted LabelEncoder - needed to log class names in message

## Outputs to agent state
- trained_models: dict - three trained, calibrated classifiers keyed by name:
    {
      "Logistic Regression": calibrated LR model,
      "Linear SVM":          calibrated SVM model,
      "Naive Bayes":         MultinomialNB model
    }

## Output format
Appends to state.messages:
"[train_models] Trained 3 models: Logistic Regression, Linear SVM, Naive Bayes. All calibrated for probability output. Training samples: 23486, Classes: 10"

## Notes
- class_weight='balanced' used for LR and SVM because even though the dataset is balanced, it is good practice to document this
- random_state=42 throughout for reproducibility
- CalibratedClassifierCV wraps models that do not natively produce reliable probabilities, this is essential for confidence-based routing (AUTO_REPLY/CLARIFY/ESCALATE)
- MultinomialNB requires non-negative input TF-IDF values are always >= 0 so this condition is satisfied
- val_texts is passed into state but not used for training - it is available here so evaluate_models_node can use it next
- test_texts is never touched in this node - strict separation
- GridSearchCV uses scoring='f1_macro' to match the primary evaluation metric - ensures tuning optimises for the same criterion used in model selection
- GridSearchCV is applied only to LR and SVM - NB has one meaningful parameter (alpha) and default works well for text
- Naive Bayes is appropriate for this dataset given short average text length (~66 chars). For datasets with longer texts (>200 chars), LR or SVM should be preferred as they handle complex word relationships better than the Naive Bayes independence assumption