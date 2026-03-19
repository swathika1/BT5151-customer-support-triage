---
name: run_inference

description: Clean and vectorise customer message. Classify the message using the saved model.

tags: [inference, classification, tfidf, prediction, confidence, probability, serving, sklearn]

mode: organisational
---

# Run Inference Skill

## Role
This is an organisational skill. The Python code in run_inference_node was written to satisfy this specification. The LLM does not read this file at runtime. All operations are deterministic.

## When to use
After translate_to_english_node has written translated_message to state. Before confidence_router_node runs. Triggered once per customer message in the serving pipeline.

## How to execute
1. Load saved artifacts from disk:
   model      = pickle.load("artifacts/model.pkl")
   tfidf      = pickle.load("artifacts/tfidf_vectorizer.pkl")
   encoder    = pickle.load("artifacts/label_encoder.pkl")
   thresholds = json.load("artifacts/thresholds.json")
2. Clean issue_description text:
   - Lowercase all text
   - Remove URLs and email addresses
   - Remove punctuation and special characters
   - Collapse multiple spaces into one
   - Remove ticket or order numbers (e.g. #12345)
   - Strip leading and trailing whitespace
3. Vectorise the cleaned message using the tfidf.transform([cleaned_message]) (no re-fitting)
4. Get predicted probabilities using probs = model.predict_proba(vectorised_message)[0] (10 classes)
5. Extract predicted label and confidence score:
           predicted_index = probs.argmax()
           predicted_label = encoder.inverse_transform([predicted_index])[0]
           confidence_score = probs.max()
6. Build class_probabilities dictionary mapping each class name to its probability - {"Account Suspension": 0.02, "Bug Report": 0.05, ...}
7. Append trace log entry to state.trace_logs:
           {
             "stage": "run_inference",
             "timestamp": current timestamp,
             "inputs_summary": first 100 chars of translated_message,
             "outputs_summary": predicted_label + confidence_score,
             "top_3_classes": top 3 class names and probabilities
           }
8. Save all outputs to state

## Inputs from agent state
- translated_message: str - written by translate_to_english_node
- conversation_id: str - written when serving pipeline was triggered

## Outputs to agent state
- predicted_label: str - the predicted support category name e.g. "Login Issue", "Bug Report", "Refund Request"
- confidence_score: float - the model's max class probability. Range: 0.0 to 1.0. e.g. 0.87 means 87% confident. Routing decision is made by confidence_router_node, not here
- class_probabilities: dict - probability for each of the 10 classes e.g. {"Account Suspension": 0.02, "Bug Report": 0.05, ...}
- trace_logs: list - one entry appended per node execution contains stage, timestamp, inputs_summary, outputs_summary, top_3_classes

## Output format
Appends to state.messages:
"[run_inference] Predicted: Login Issue. Confidence: 0.87. Top 3: Login Issue (0.87), Account Suspension (0.06), Security Concern (0.04).
Model: Linear SVM."

## Notes
- Same clean_text() function must be used as training and inference must use identical preprocessing. If preprocessing differs, the TF-IDF features will not match the vocabulary the model was trained on.
- Since the vocabulary is fixed at training time use tfidf.transform() not tfidf.fit_transform(). Re-fitting at inference would create a different vocabulary and produce meaningless features.
- Confidence_score = probs.max() is not a hardcoded threshold as the raw max probability is stored here. The routing decision happens in confidence_router_node, not here. This node only classifies - it does not route.
- Admin UI needs to show full probability distribution so class_probabilities should store all 10 classes. It is also useful for detecting ambiguous cases where two classes have similar probabilities.
- This node never retrains the model as model.fit() is never called here. Training only happens in the training pipeline. This node is read-only with respect to the model.
