---
name: preprocess_data

description:
  Load the customer support dataset, filter English-language rows, clean issue_description text, encode category labels, split into train, validation and test datasets. Fit TF-IDF vectoriser on training data only.

tags: [preprocessing, tf-idf, cleaning, split, encoding]

mode: organisational
---

# Preprocess Data Skill

## Role
This is an organisational skill. The Python code in preprocess_data_node was written to satisfy this specification. The LLM does not read this file at runtime. All operations are deterministic.

## When to use
At the start of the training pipeline before any model training. Triggered once when the training graph is invoked.

## How to execute
1. Load CSV from state.raw_data_path using pd.read_csv()
2. Filter rows where language == 'English' (33553 rows retained)
3. Clean issue_description text:
   - Lowercase all text
   - Remove URLs and email addresses
   - Remove punctuation and special characters
   - Collapse multiple spaces into one
   - Remove ticket or order numbers (e.g. #12345)
   - Strip leading and trailing whitespace
4. Encode category labels using LabelEncoder (10 classes)
5. Split into train/validation/test: 70% / 15% / 15%, stratified, random_state=42
6. Fit TfidfVectorizer on training data only:
   - ngram_range=(1,2), max_features=5000, min_df=2, sublinear_tf=True
7. Transform validation and test using the fitted vectoriser (no re-fitting) 
8. Save all outputs to state

## Inputs from agent state
- raw_data_path: str — path to customer_support_tickets.csv

## Outputs to agent state
- raw_df: DataFrame - full loaded dataset (200000 rows)
- train_texts: sparse matrix - TF-IDF features for training
- val_texts: sparse matrix - TF-IDF features for validation
- test_texts: sparse matrix - TF-IDF features for testing
- y_train: numpy array - encoded labels for training
- y_val: numpy array - encoded labels for validation
- y_test: numpy array - encoded labels for testing
- tfidf_vectorizer: fitted TfidfVectorizer - for use at inference time
- label_encoder: fitted LabelEncoder - maps integers back to class names

## Output format
Appends to state.messages:
"[preprocess_data] Train: 23486, Val: 5034, Test: 5033.
 TF-IDF shape: (23486, N).
 Classes: ['Account Suspension', 'Bug Report', 'Data Sync Issue', 'Feature Request', 'Login Issue', 'Payment Problem', 'Performance Issue', 'Refund Request', 'Security Concern', 'Subscription Cancellation'].
 Encoded: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]"

## Notes
- TF-IDF must never be fit on validation or test data — this causes data leakage
- random_state=42 used throughout for full reproducibility
- Only English rows used for training; non-English handled at inference time via the translate-to-english node in the serving pipeline
- 10 original category labels are kept as it is (no collapsing needed data is already clean and balanced at ~3300 rows per class)
- Word n-grams ngram_range=(1,2) are sufficient for this synthetic dataset which contains no typos or informal language. For real complaint datasets with informal writing, adding character n-grams analyzer='char_wb', ngram_range=(3,5) would improve robustness to typos and abbreviations
