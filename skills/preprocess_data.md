---
name: preprocess_data
description: "Load customer support dataset, clean text, encode labels, split into train/val/test sets"
mode: organisational
---

## When to use
Executed once at the start of the training pipeline to load and prepare raw data for model training.

## How to execute
1. Load Bitext customer support dataset from Hugging Face (26k+ samples, 11 categories)
2. Select relevant columns (instruction text, category label)
3. Drop rows with missing values
4. Clean text: lowercase, remove URLs, emails, hashtags, punctuation, collapse whitespace
5. Remove rows that became empty after cleaning
6. Encode category labels using LabelEncoder (maps category name → integer 0-10)
7. Split data: 70% train, 15% validation, 15% test (stratified split)
8. Fit TfidfVectorizer ONLY on training data (1-2 grams, 5000 features max)
9. Transform validation and test data using fitted vectorizer (no refitting)
10. Store all outputs to state for downstream nodes

## Inputs from agent state
- None required (uses default raw_data_path)

## Outputs to agent state
- raw_df: Original pandas DataFrame from dataset
- train_texts: Sparse TF-IDF matrix for training samples (shape: n_train × n_features)
- val_texts: Sparse TF-IDF matrix for validation samples
- test_texts: Sparse TF-IDF matrix for test samples (held out, only for final evaluation)
- y_train: Integer-encoded labels for training samples
- y_val: Integer-encoded labels for validation samples
- y_test: Integer-encoded labels for test samples
- tfidf_vectorizer: Fitted TfidfVectorizer instance (reused at inference time)
- label_encoder: Fitted LabelEncoder instance (maps int ↔ category name)

## Output format
```python
{
  "raw_df": pandas.DataFrame,  # shape (26872, 5)
  "train_texts": scipy.sparse._matrix.csr_matrix,  # shape (~18809, 4808)
  "val_texts": scipy.sparse._matrix.csr_matrix,    # shape (~4032, 4808)
  "test_texts": scipy.sparse._matrix.csr_matrix,   # shape (~4031, 4808)
  "y_train": numpy.ndarray,  # shape (18809,), dtype int
  "y_val": numpy.ndarray,    # shape (4032,), dtype int
  "y_test": numpy.ndarray,   # shape (4031,), dtype int
  "tfidf_vectorizer": TfidfVectorizer instance,
  "label_encoder": LabelEncoder instance
}
```

## Notes
- **Critical**: TF-IDF is fit ONLY on training data. Validation and test data are transformed using the training vocabulary. This prevents data leakage and ensures proper evaluation.
- **PyArrow compatibility**: When loading from Hugging Face datasets, convert to native numpy arrays before train_test_split to avoid sklearn compatibility issues.
- **Text cleaning is consistent**: The exact same cleaning function must be used at inference time (run_inference_node) to prevent training-serving skew.
- **Stratified split**: Class distribution is maintained across train/val/test to ensure representative samples.
- **Classes**: 11 complaint categories: ACCOUNT, CANCEL, CONTACT, DELIVERY, FEEDBACK, INVOICE, ORDER, PAYMENT, REFUND, SHIPPING, SUBSCRIPTION
