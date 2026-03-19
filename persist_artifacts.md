---
name: persist_artifacts

description:
  Save the selected trained model, TF-IDF vectorizer, label encoder, and evaluation metrics to disk as pickle and JSON files. Also save SKILL.md metadata for audit and reproducibility. This enables the inference pipeline to load pre-trained artifacts without retraining.

tags: [artifact_persistence, model_serialization, deployment, reproducibility]

mode: organisational
---

# Persist Artifacts Skill

## Role
This is an organisational skill. The Python code saves trained models and metadata to disk deterministically. This is essential for deploying the model to production inference without retraining.

## When to use
After select_model_node has chosen the best model. Before the training pipeline completes. Triggered once during the training pipeline.

## How to execute
1. Create an `artifacts/` directory if it doesn't exist
2. Save the selected trained model:
   - Use pickle.dump() to serialize the sklearn model
   - File: artifacts/model.pkl
3. Save the TF-IDF vectorizer:
   - Use pickle.dump() to serialize the TfidfVectorizer
   - File: artifacts/tfidf_vectorizer.pkl
4. Save the label encoder:
   - Use pickle.dump() to serialize the LabelEncoder
   - File: artifacts/label_encoder.pkl
5. Save evaluation metrics as JSON:
   - File: artifacts/evaluation_results.json
   - Structure: { "Logistic Regression": {...}, "Linear SVM": {...}, "Naive Bayes": {...} }
6. Save model comparison summary:
   - File: artifacts/model_comparison.json
   - Include accuracy, macro_f1, weighted_f1 for each model
7. Save selection justification as text:
   - File: artifacts/selection_justification.txt
   - Include selected model name and business rationale
8. Create artifacts/METADATA.json with pipeline info:
   - Timestamp of training
   - Git commit hash (if available)
   - Selected model name and version
   - Data preprocessing parameters (ngram_range, max_features, min_df)
   - Training set size
   - Validation set size
9. Log artifact paths and checksums to state.messages

## Inputs from agent state
- selected_model: sklearn model - the chosen trained model
- selected_model_name: str - name of selected model (e.g., "Linear SVM")
- selection_justification: str - business justification for selection
- tfidf_vectorizer: fitted TfidfVectorizer
- label_encoder: fitted LabelEncoder
- evaluation_results: dict - metrics for all 3 models
- model_comparison_summary: dict - comparison table
- y_train, y_val, y_test: numpy arrays for data split sizes

## Outputs to agent state
- artifacts_path: str - path to artifacts directory
- artifact_files: dict - mapping of artifact names to file paths
     {
       "model": "artifacts/model.pkl",
       "vectorizer": "artifacts/tfidf_vectorizer.pkl",
       "encoder": "artifacts/label_encoder.pkl",
       "evaluation_results": "artifacts/evaluation_results.json",
       "model_comparison": "artifacts/model_comparison.json",
       "selection_justification": "artifacts/selection_justification.txt",
       "metadata": "artifacts/METADATA.json"
     }
- artifact_checksums: dict - SHA256 checksums of saved files (for verification)

## Output format
Appends to state.messages:
"[persist_artifacts] Saved artifacts to artifacts/. Model: model.pkl (4.2 MB), Vectorizer: tfidf_vectorizer.pkl (1.1 MB), Encoder: label_encoder.pkl (0.05 MB). Checksums: model=a1b2c3..., vectorizer=d4e5f6..., encoder=g7h8i9..."

## Notes
- All artifacts must be saved before the training pipeline ends
- Pickle format chosen for maximum sklearn compatibility (binary, fast)
- JSON used for metrics to allow inspection without deserialization
- METADATA.json is NOT used at inference time but valuable for audit and reproducibility
- File sizes will vary based on TF-IDF vocabulary size (typically 100KB - 10MB for TF-IDF)
- Artifacts directory should be .gitignored in version control (model files are large binary objects)
- At inference time, loading: model = pickle.load(open('artifacts/model.pkl', 'rb'))
- Ensures no data leakage: only training data was used to fit the model and vectorizer during training
