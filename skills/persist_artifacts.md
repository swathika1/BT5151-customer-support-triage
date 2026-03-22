---
name: persist_artifacts
description: "Serialize trained model, vectorizer, encoder, and thresholds to disk for serving pipeline inference"
mode: organisational
---

## When to use
Executed at the end of training pipeline to save artifacts needed for the serving pipeline.

## How to execute
1. Create artifacts directory (if not exists)
2. Pickle and save model to artifacts/model.pkl
   - This is the selected, trained, calibrated classifier
3. Pickle and save TF-IDF vectorizer to artifacts/tfidf_vectorizer.pkl
   - Required for text preprocessing at inference time
4. Pickle and save label encoder to artifacts/label_encoder.pkl
   - Required to decode predicted class indices back to category names
5. Save routing thresholds to artifacts/thresholds.json:
   ```json
   {
     "tau_high": 0.85,
     "tau_low": 0.65
   }
   ```
6. Save model metadata to artifacts/model_info.json:
   ```json
   {
     "model_name": "Naive Bayes",
     "training_date": "2026-03-22",
     "n_classes": 11,
     "classes": ["ACCOUNT", "CANCEL", ...],
     "macro_f1": 0.96,
     "weighted_f1": 0.97
   }
   ```
7. Set state.artifacts_saved = True

## Inputs from agent state
- selected_model: Trained model from select_model_node
- selected_model_name: Name of selected model
- tfidf_vectorizer: Fitted vectorizer from preprocess_data_node
- label_encoder: Fitted encoder from preprocess_data_node
- tau_high, tau_low: Routing thresholds from select_model_node
- evaluation_results: Metrics used to populate model_info.json

## Outputs to agent state
- artifacts_saved: Boolean flag set to True

## Output format
Files created in artifacts/ directory:
```
artifacts/
├── model.pkl              (binary pickle - trained classifier)
├── tfidf_vectorizer.pkl   (binary pickle - TfidfVectorizer)
├── label_encoder.pkl      (binary pickle - LabelEncoder)
├── thresholds.json        (JSON - tau_high, tau_low)
└── model_info.json        (JSON - metadata)
```

No state output structure (aside from artifacts_saved flag).

## Notes
- **Artifact location**: All files saved to ./artifacts/ directory in notebook root. This directory is accessed by serving_pipeline nodes (run_inference_node loads from here).
- **Pickle choice**: Models are pickled (binary format) for speed and exact reproduction of Python objects. JSON reserved for human-readable thresholds and metadata.
- **Metadata purpose**: model_info.json serves as documentation of which model and thresholds were used. Useful for audit trails and debugging.
- **Serving pipeline dependency**: The entire serving pipeline (detect_language → translate → infer → route → draft → log) depends on these artifacts. If artifacts are missing, serving fails gracefully with clear error messages.
- **Model version tracking**: In future iterations, could extend this step to store versioned models (model_v1.pkl, model_v2.pkl, ...) for A/B testing or rollback.
