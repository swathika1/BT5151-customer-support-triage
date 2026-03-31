import os
import json
import pickle
import re
import sqlite3
import warnings
from pathlib import Path
from typing import Any, Optional
from datetime import datetime

warnings.filterwarnings('ignore')

# Essential data libraries
import pandas as pd
import numpy as np

# UI
from http.server import HTTPServer, BaseHTTPRequestHandler
import json as json_module
from urllib.parse import urlparse, parse_qs

# ML
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support
)

# LangChain
from openai import OpenAI

# Pydantic
from pydantic import BaseModel, Field, ConfigDict

# Try to load datasets for Bitext
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except:
    HAS_DATASETS = False
    print("⚠ datasets library not available - will use fallback data")

# Try to load language detection
try:
    from textblob import TextBlob
    from langdetect import detect
    HAS_LANGDETECT = True
except:
    HAS_LANGDETECT = False
    print("⚠ Language detection not available")

# Check for OpenAI API
api_key = os.getenv("OPENAI_API_KEY", "").strip()
HAS_OPENAI_API_KEY = bool(api_key and api_key != "sk-your-openai-api-key-here")
HAS_LLM = False
llm_client = None
if HAS_OPENAI_API_KEY:
    try:
        # Initialize with explicit API key
        llm_client = OpenAI(api_key=api_key)
        HAS_LLM = True
        print("✓ OpenAI API key detected - LLM features enabled")
    except Exception as e:
        print(f"⚠ LLM initialization failed: {str(e)[:100]}")
        HAS_LLM = False
        llm_client = None

# Paths
ARTIFACTS_DIR = "artifacts"
MODEL_PATH = f"{ARTIFACTS_DIR}/model.pkl"
VECTORIZER_PATH = f"{ARTIFACTS_DIR}/tfidf_vectorizer.pkl"
ENCODER_PATH = f"{ARTIFACTS_DIR}/label_encoder.pkl"
THRESHOLDS_PATH = f"{ARTIFACTS_DIR}/thresholds.json"
MODEL_INFO_PATH = f"{ARTIFACTS_DIR}/model_info.json"
DB_PATH = f"{ARTIFACTS_DIR}/interactions.db"

# Response templates for each category
RESPONSE_TEMPLATES = {
    "ACCOUNT": {
        "AUTO_REPLY": "We've sent password reset instructions to your email. If you don't see it, check your spam folder. You'll have access restored within minutes.",
        "CLARIFY": "I can help with your account. Can you confirm the email address associated with your account?",
        "ESCALATE": "For security reasons, account issues require manual verification. A specialist will contact you within 2 hours."
    },
    "CANCEL": {
        "AUTO_REPLY": "Your cancellation request has been processed. You may be eligible for a refund depending on your plan. Check your confirmation email.",
        "CLARIFY": "I'd like to help. Can you tell me which service or subscription you'd like to cancel?",
        "ESCALATE": "Your request needs special review. An agent will be in touch within 24 hours to discuss options."
    },
    "CONTACT": {
        "AUTO_REPLY": "You can reach our support team at support@company.com or call 1-800-SUPPORT. Our hours are 9 AM - 6 PM EST, Monday-Friday.",
        "CLARIFY": "What issue can I help direct you to the right department for?",
        "ESCALATE": "Your inquiry requires specialist attention. A representative will reach you shortly."
    },
    "DELIVERY": {
        "AUTO_REPLY": "Your order is on its way! Check your confirmation email for a tracking link. Delivery typically takes 3-5 business days.",
        "CLARIFY": "I can help track your order. Could you provide your order number?",
        "ESCALATE": "We're looking into your delivery issue. A specialist will contact you with an update within 24 hours."
    },
    "FEEDBACK": {
        "AUTO_REPLY": "Thank you for your feedback! We genuinely appreciate your insights and use them to improve our service.",
        "CLARIFY": "We'd love to hear more about your experience. What can we improve?",
        "ESCALATE": "Your feedback has been escalated to our management team for priority review."
    },
    "INVOICE": {
        "AUTO_REPLY": "Your latest invoice is available in your account dashboard. Log in to view charges and payment history.",
        "CLARIFY": "I can help with your invoice. What date or amount are you looking for?",
        "ESCALATE": "Our billing team will investigate your invoice inquiry and contact you within 24 hours."
    },
    "ORDER": {
        "AUTO_REPLY": "Your order has been confirmed! Check your email for order details and expected delivery date.",
        "CLARIFY": "I can help with your order. Can you provide your order number?",
        "ESCALATE": "We're reviewing your order issue. A specialist will reach out within 24 hours."
    },
    "PAYMENT": {
        "AUTO_REPLY": "Your payment has been processed successfully. You should see the transaction reflected in your account within 1-2 business days.",
        "CLARIFY": "I'd like to help with the payment issue. Can you describe what went wrong?",
        "ESCALATE": "Our payments team is investigating this. You'll hear from us with a resolution within 24 hours."
    },
    "REFUND": {
        "AUTO_REPLY": "We've initiated your refund! You should see the credit to your original payment method within 5-10 business days.",
        "CLARIFY": "I can process your refund. Can you confirm the reason and order number?",
        "ESCALATE": "Your refund request needs manual review. An agent will contact you within 24 hours with next steps."
    },
    "SHIPPING": {
        "AUTO_REPLY": "Your package has been shipped! Click the link in your confirmation email to track your delivery in real-time.",
        "CLARIFY": "I'd be happy to help. Can you provide your order or tracking number?",
        "ESCALATE": "We're investigating your shipping issue. A specialist will contact you with an update within 24 hours."
    },
    "SUBSCRIPTION": {
        "AUTO_REPLY": "Your subscription is active and current. You can manage your plan anytime in your account settings.",
        "CLARIFY": "What would you like to do with your subscription?",
        "ESCALATE": "Your subscription request needs review. A specialist will be in touch within 24 hours."
    }
}


# ============================================================================
# PYDANTIC STATE MODEL
# ============================================================================

class SupportAgentState(BaseModel):
    """Shared state passed between all agents in the pipeline."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # Pipeline control
    messages: list[str] = Field(default_factory=list)
    active_skill: Optional[str] = None
    mode: str = "serve"  # "train" or "serve"
    
    # Training data
    raw_data_path: str = "customer_support_tickets.csv"
    raw_df: Optional[Any] = None
    train_texts: Optional[Any] = None
    val_texts: Optional[Any] = None
    test_texts: Optional[Any] = None
    y_train: Optional[Any] = None
    y_val: Optional[Any] = None
    y_test: Optional[Any] = None
    
    # Artifacts
    tfidf_vectorizer: Optional[Any] = None
    label_encoder: Optional[Any] = None
    trained_models: dict = Field(default_factory=dict)
    selected_model: Optional[Any] = None
    selected_model_name: str = ""
    evaluation_results: dict = Field(default_factory=dict)
    
    # Thresholds
    tau_high: float = 0.90
    tau_low: float = 0.70
    
    # Serving inputs
    raw_message: str = ""
    detected_language: str = "en"
    translated_message: str = ""
    
    # Inference outputs
    predicted_label: str = ""
    confidence_score: float = 0.0
    class_probabilities: dict = Field(default_factory=dict)
    route_decision: str = "AUTO_REPLY"
    response_final: str = ""
    
    # Logging
    trace_logs: list[dict] = Field(default_factory=list)
    artifacts_saved: bool = False


# ============================================================================
# LOAD BITEXT DATASET
# ============================================================================

def load_bitext_dataset():
    """Load the Bitext customer support dataset from Hugging Face."""
    print("[Dataset] Loading Bitext customer support dataset...")
    
    if not HAS_DATASETS:
        print("[Dataset] ⚠ datasets library not available - using fallback")
        # Fallback to minimal data
        df = pd.DataFrame([
            ("I need help with my account", "ACCOUNT"),
            ("How do I reset my password?", "ACCOUNT"),
            ("I want to cancel my order", "CANCEL"),
            ("Can I get a refund?", "REFUND"),
            ("Where is my package?", "DELIVERY"),
            ("Track my order", "DELIVERY"),
        ], columns=['instruction', 'category'])
        print(f"[Dataset] Loaded {len(df)} fallback examples")
        return df
    
    try:
        dataset = load_dataset(
            "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
        )
        df = dataset['train'].to_pandas()
        print(f"[Dataset] ✓ Loaded {len(df)} examples from Bitext dataset")
        print(f"[Dataset] Categories: {df['category'].nunique()}")
        print(f"[Dataset] Sample categories: {list(df['category'].unique()[:5])}")
        return df
    except Exception as e:
        print(f"[Dataset] ⚠ Failed to load from Hugging Face: {str(e)}")
        print("[Dataset] Using fallback data")
        df = pd.DataFrame([
            ("I need help with my account", "ACCOUNT"),
            ("How do I reset my password?", "ACCOUNT"),
            ("I want to cancel my order", "CANCEL"),
            ("Can I get a refund?", "REFUND"),
            ("Where is my package?", "DELIVERY"),
        ], columns=['instruction', 'category'])
        return df


def load_trained_pipeline():
    """Load trained model and vectorizer from artifacts, or train if not available."""
    print("[Pipeline] Checking for trained artifacts...")
    
    # Check if artifacts exist
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        try:
            print(f"[Pipeline] Loading model from {MODEL_PATH}...")
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            
            print(f"[Pipeline] Loading vectorizer from {VECTORIZER_PATH}...")
            with open(VECTORIZER_PATH, "rb") as f:
                vectorizer = pickle.load(f)
            
            print("[Pipeline] ✓ Artifacts loaded successfully")
            return model, vectorizer
        except Exception as e:
            print(f"[Pipeline] ⚠ Failed to load artifacts: {str(e)}")
            print("[Pipeline] Running training pipeline instead...")
    else:
        print("[Pipeline] ⚠ Trained artifacts not found")
        print("[Pipeline] Running training pipeline to generate artifacts...")
    
    # If artifacts don't exist, run the full training pipeline
    print("\n" + "="*80)
    print("RUNNING FULL TRAINING PIPELINE")
    print("="*80 + "\n")
    
    # Load data
    df = load_bitext_dataset()
    state = SupportAgentState(raw_data=df, messages=[])
    
    # Run all training nodes
    state = preprocess_data_node(state)
    state = train_models_node(state)
    state = evaluate_models_node(state)
    state = select_model_node(state)
    state = persist_artifacts_node(state)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE - artifacts saved to disk")
    print("="*80 + "\n")
    
    # Return the trained model and vectorizer
    return state.selected_model, state.tfidf_vectorizer


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def preprocess_text(text):
    """Preprocess text for model input (matching training preprocessing)"""
    text = str(text).lower()
    text = re.sub(r'\{\{.*?\}\}', '', text)  # Remove {{placeholders}}
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'\S+@\S+', '', text)  # Remove emails
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    text = re.sub(r'[^a-z0-9\s]', ' ', text)  # Keep only alphanumeric
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    return text

CONFIDENCE_THRESHOLD = 0.70  # Threshold for high/low confidence responses


# ============================================================================
# TRAINING PIPELINE NODES
# ============================================================================

def preprocess_data_node(state: SupportAgentState) -> SupportAgentState:
    """Load and preprocess Bitext dataset."""
    print("[PreprocessNode] Starting preprocessing...")
    
    df = load_bitext_dataset()
    state.raw_df = df
    
    # Clean text
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'\{\{.*?\}\}', '', text)
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'#\w+', '', text)
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    df_clean = df[['instruction', 'category']].copy()
    df_clean = df_clean.dropna(subset=['instruction', 'category'])
    df_clean['clean_text'] = df_clean['instruction'].apply(clean_text)
    df_clean = df_clean[df_clean['clean_text'].str.len() > 3]
    
    print(f"[PreprocessNode] Rows after cleaning: {len(df_clean)}")
    
    # Encode labels
    le = LabelEncoder()
    df_clean['label_encoded'] = le.fit_transform(df_clean['category'])
    print(f"[PreprocessNode] Classes ({len(le.classes_)}): {list(le.classes_)}")
    
    # Split data
    X = [str(x) for x in df_clean['clean_text'].to_list()]
    y = [int(yi) for yi in df_clean['label_encoded'].to_list()]
    X = np.array(X, dtype=object)
    y = np.array(y, dtype=int)
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1765, stratify=y_temp, random_state=42
    )
    
    # TF-IDF
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        sublinear_tf=True,
        min_df=2,
        strip_accents='unicode'
    )
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)
    X_test_tfidf = tfidf.transform(X_test)
    
    state.train_texts = X_train_tfidf
    state.val_texts = X_val_tfidf
    state.test_texts = X_test_tfidf
    state.y_train = y_train
    state.y_val = y_val
    state.y_test = y_test
    state.tfidf_vectorizer = tfidf
    state.label_encoder = le
    
    message = f"[preprocess_data] Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}. TF-IDF: {X_train_tfidf.shape}"
    state.messages.append(message)
    print(f"[PreprocessNode] Done.")
    
    return state


def train_models_node(state: SupportAgentState) -> SupportAgentState:
    """Train 3 models using GridSearchCV."""
    print("[TrainModelsNode] Training models...")
    
    models_config = {
        'Logistic Regression': (
            CalibratedClassifierCV(LogisticRegression(max_iter=1000, random_state=42)),
            {'base_estimator__C': [0.1, 1.0, 10.0]}
        ),
        'Linear SVM': (
            CalibratedClassifierCV(LinearSVC(max_iter=2000, random_state=42, dual=False)),
            {'base_estimator__C': [0.1, 1.0, 10.0]}
        ),
        'Naive Bayes': (
            MultinomialNB(),
            {'alpha': [0.1, 1.0]}
        ),
    }
    
    state.trained_models = {}
    for name, (model, params) in models_config.items():
        print(f"[TrainModelsNode] GridSearchCV for {name}...")
        grid = GridSearchCV(model, params, cv=3, scoring='f1_weighted', n_jobs=-1)
        grid.fit(state.train_texts, state.y_train)
        state.trained_models[name] = grid.best_estimator_
        print(f"[TrainModelsNode]   Best params: {grid.best_params_}")
    
    state.messages.append("[train_models] Trained 3 models with GridSearchCV")
    print(f"[TrainModelsNode] Done.")
    return state


def evaluate_models_node(state: SupportAgentState) -> SupportAgentState:
    """Evaluate models on validation set."""
    print("[EvaluateModelsNode] Evaluating models...")
    
    state.evaluation_results = {}
    for name, model in state.trained_models.items():
        y_pred = model.predict(state.val_texts)
        _, _, macro_f1, _ = precision_recall_fscore_support(
            state.y_val, y_pred, average='macro'
        )
        _, _, weighted_f1, _ = precision_recall_fscore_support(
            state.y_val, y_pred, average='weighted'
        )
        state.evaluation_results[name] = {
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1
        }
        print(f"[EvaluateModelsNode] {name}: Macro F1={macro_f1:.4f}, Weighted F1={weighted_f1:.4f}")
    
    state.messages.append("[evaluate_models] Evaluated 3 models on validation set")
    print(f"[EvaluateModelsNode] Done.")
    return state


def select_model_node(state: SupportAgentState) -> SupportAgentState:
    """Select best model and derive confidence thresholds."""
    print("[SelectModelNode] Selecting best model...")
    
    best_model_name = max(
        state.evaluation_results,
        key=lambda x: state.evaluation_results[x]['macro_f1']
    )
    best_macro_f1 = state.evaluation_results[best_model_name]['macro_f1']
    
    state.selected_model = state.trained_models[best_model_name]
    state.selected_model_name = best_model_name
    
    print(f"[SelectModelNode] Selected: {best_model_name} (Macro F1: {best_macro_f1:.4f})")
    
    # Derive thresholds from validation set
    y_pred = state.selected_model.predict(state.val_texts)
    y_proba = state.selected_model.predict_proba(state.val_texts)
    max_probs = np.max(y_proba, axis=1)
    
    thresholds = np.linspace(0.0, 1.0, 101)
    precisions = []
    
    for threshold in thresholds:
        mask = max_probs >= threshold
        if mask.sum() == 0:
            precisions.append(np.nan)
            continue
        correct = (y_pred[mask] == state.y_val[mask]).astype(int).sum()
        precision = correct / mask.sum()
        precisions.append(precision)
    
    precisions = np.array(precisions)
    valid = ~np.isnan(precisions)
    
    high_mask = precisions >= 0.90
    if (valid & high_mask).any():
        state.tau_high = float(thresholds[valid & high_mask][0])
    else:
        state.tau_high = 0.95
    
    low_mask = precisions >= 0.70
    if (valid & low_mask).any():
        state.tau_low = float(thresholds[valid & low_mask][-1])
    else:
        state.tau_low = 0.50
    
    print(f"[SelectModelNode] Thresholds: tau_high={state.tau_high:.4f}, tau_low={state.tau_low:.4f}")
    state.messages.append(f"[select_model] Selected {best_model_name}. Thresholds: tau_high={state.tau_high:.4f}, tau_low={state.tau_low:.4f}")
    print(f"[SelectModelNode] Done.")
    return state


def persist_artifacts_node(state: SupportAgentState) -> SupportAgentState:
    """Save all artifacts to disk."""
    print("[PersistNode] Saving artifacts...")
    
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    
    # Save model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(state.selected_model, f)
    print(f"[PersistNode] Saved model")
    
    # Save vectorizer
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(state.tfidf_vectorizer, f)
    print(f"[PersistNode] Saved vectorizer")
    
    # Save encoder
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(state.label_encoder, f)
    print(f"[PersistNode] Saved label encoder")
    
    # Save thresholds
    with open(THRESHOLDS_PATH, "w") as f:
        json.dump({
            "tau_high": state.tau_high,
            "tau_low": state.tau_low
        }, f, indent=2)
    print(f"[PersistNode] Saved thresholds")
    
    # Save metadata
    best_result = state.evaluation_results[state.selected_model_name]
    with open(MODEL_INFO_PATH, "w") as f:
        json.dump({
            "model_name": state.selected_model_name,
            "trained_at": datetime.now().isoformat(),
            "n_classes": len(state.label_encoder.classes_),
            "classes": list(state.label_encoder.classes_),
            "macro_f1": best_result['macro_f1'],
            "weighted_f1": best_result['weighted_f1'],
            "n_training_samples": len(state.y_train)
        }, f, indent=2)
    print(f"[PersistNode] Saved metadata")
    
    state.artifacts_saved = True
    state.messages.append("[persist_artifacts] All artifacts saved to disk")
    print(f"[PersistNode] Done.")
    return state


# ============================================================================
# SERVING PIPELINE NODES
# ============================================================================

def detect_language_node(state: SupportAgentState) -> SupportAgentState:
    """Detect language of incoming message."""
    print(f"[DetectLanguageNode] Detecting language...")
    
    if not HAS_LANGDETECT:
        state.detected_language = 'en'
        print("[DetectLanguageNode] No language detection available - defaulting to English")
    else:
        try:
            from textblob import TextBlob
            blob = TextBlob(state.raw_message)
            state.detected_language = blob.detect_language()
            print(f"[DetectLanguageNode] Detected: {state.detected_language}")
        except Exception as e:
            try:
                from langdetect import detect
                state.detected_language = detect(state.raw_message)
                print(f"[DetectLanguageNode] Detected (fallback): {state.detected_language}")
            except:
                state.detected_language = 'en'
                print("[DetectLanguageNode] Detection failed - defaulting to English")
    
    state.messages.append(f"[detect_language] Detected language: {state.detected_language}")
    return state


def translate_to_english_node(state: SupportAgentState) -> SupportAgentState:
    """Translate non-English messages to English if LLM available."""
    print(f"[TranslateNode] Translating...")
    
    if state.detected_language == 'en':
        state.translated_message = state.raw_message
        print("[TranslateNode] Message already in English")
    elif not HAS_LLM:
        state.translated_message = state.raw_message
        print("[TranslateNode] No LLM available - using original")
    else:
        try:
            response = llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": f"Translate to English, return ONLY the translation:\n\n{state.raw_message}"}
                ],
                temperature=0.0
            )
            state.translated_message = response.choices[0].message.content.strip()
            print("[TranslateNode] Translation complete")
        except Exception as e:
            state.translated_message = state.raw_message
            print(f"[TranslateNode] Translation failed: {str(e)[:50]} - using original")
    
    state.messages.append(f"[translate_to_english] Message: {state.translated_message[:80]}...")
    return state


def run_inference_node(state: SupportAgentState) -> SupportAgentState:
    """Run inference using trained model."""
    print(f"[InferenceNode] Running inference...")
    
    # Load artifacts if not already loaded
    if state.selected_model is None:
        try:
            with open(MODEL_PATH, "rb") as f:
                state.selected_model = pickle.load(f)
            with open(VECTORIZER_PATH, "rb") as f:
                state.tfidf_vectorizer = pickle.load(f)
            with open(ENCODER_PATH, "rb") as f:
                state.label_encoder = pickle.load(f)
            with open(THRESHOLDS_PATH, "r") as f:
                thresholds = json.load(f)
                state.tau_high = thresholds['tau_high']
                state.tau_low = thresholds['tau_low']
            print("[InferenceNode] Artifacts loaded from disk")
        except Exception as e:
            print(f"[InferenceNode] ⚠ Failed to load artifacts: {str(e)}")
            raise
    
    # Clean and vectorize
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'\{\{.*?\}\}', '', text)
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'#\w+', '', text)
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    cleaned = clean_text(state.translated_message)
    vec = state.tfidf_vectorizer.transform([cleaned])
    
    # Predict
    probs = state.selected_model.predict_proba(vec)[0]
    pred_idx = probs.argmax()
    pred_label = state.label_encoder.inverse_transform([pred_idx])[0]
    confidence = float(probs.max())
    
    state.predicted_label = pred_label
    state.confidence_score = confidence
    state.class_probabilities = {
        cls: round(float(p), 4)
        for cls, p in zip(state.label_encoder.classes_, probs)
    }
    
    print(f"[InferenceNode] Predicted: {pred_label} ({confidence:.4f})")
    state.messages.append(f"[run_inference] Predicted: {pred_label} ({confidence:.4f})")
    return state


def confidence_router_node(state: SupportAgentState) -> SupportAgentState:
    """Route based on confidence scores."""
    print(f"[RouterNode] Routing...")
    
    if state.confidence_score >= state.tau_high:
        state.route_decision = "AUTO_REPLY"
        rationale = f"High confidence ({state.confidence_score:.4f} >= {state.tau_high:.4f})"
    elif state.confidence_score >= state.tau_low:
        state.route_decision = "CLARIFY"
        rationale = f"Medium confidence ({state.confidence_score:.4f} >= {state.tau_low:.4f})"
    else:
        state.route_decision = "ESCALATE"
        rationale = f"Low confidence ({state.confidence_score:.4f} < {state.tau_low:.4f})"
    
    print(f"[RouterNode] Route: {state.route_decision}")
    state.messages.append(f"[confidence_router] {state.route_decision} decision. {rationale}")
    return state


def draft_response_node(state: SupportAgentState) -> SupportAgentState:
    """Generate customer response based on category and confidence."""
    print(f"[DraftResponseNode] Drafting response...")
    
    category = state.predicted_label
    confidence = state.confidence_score
    route = state.route_decision
    
    # Generate contextual responses based on category and actual query
    category_responses = {
        "ACCOUNT": {
            "HIGH": f"I understand you need help with your account. For security reasons, I can assist with password resets and account access issues. Can you clarify what specifically you need help with?",
            "MID": "I can help with your account-related inquiry. Could you provide more details about what you need assistance with?",
            "LOW": "Your inquiry seems to be account-related. To provide the best assistance, could you describe your issue in more detail?"
        },
        "CANCEL": {
            "HIGH": "I understand you'd like to cancel a service or subscription. I can help process that for you. Which service would you like to cancel?",
            "MID": "It sounds like you want to cancel something. Can you tell me which service or subscription you're referring to?",
            "LOW": "Your request involves cancellation. Could you provide more details about what you'd like to cancel?"
        },
        "CONTACT": {
            "HIGH": "I can help you get in touch with our support team. Our support team is available 9 AM - 6 PM EST, Monday-Friday. What's your inquiry about?",
            "MID": "You're looking to contact support. Can you tell me what your issue is so I can direct you properly?",
            "LOW": "You'd like to contact our team. Please describe your issue so I can help or route you appropriately."
        },
        "DELIVERY": {
            "HIGH": "I can help you track your delivery or resolve any shipping issues. Do you have your order or tracking number handy?",
            "MID": "It sounds like you have a delivery-related question. Can you provide your order number so I can assist?",
            "LOW": "Your inquiry is about delivery or shipping. Could you provide more information about your order?"
        },
        "FEEDBACK": {
            "HIGH": "Thank you for taking the time to share your feedback with us. We genuinely value your insights to help us improve.",
            "MID": "We appreciate your feedback. Could you share more details about your experience?",
            "LOW": "Thank you for reaching out with feedback. We'd like to understand your experience better."
        },
        "INVOICE": {
            "HIGH": "I can help you with your invoice. Your latest invoices are available in your account dashboard. What specific information do you need?",
            "MID": "You need help with an invoice. Can you provide the invoice date or order number?",
            "LOW": "Your inquiry is about billing or invoices. Please provide more details about what you need."
        },
        "ORDER": {
            "HIGH": "I can assist with your order inquiry. Do you need help tracking it, modifying it, or have another question?",
            "MID": "You have a question about an order. Can you provide your order number and what you need help with?",
            "LOW": "Your inquiry involves an order. Could you provide more details about what you need?"
        },
        "PAYMENT": {
            "HIGH": "I can help with payment-related issues. Can you describe what payment problem you're experiencing?",
            "MID": "It sounds like you have a payment question or issue. Can you provide more details?",
            "LOW": "Your inquiry is payment-related. Could you explain what you're experiencing?"
        },
        "REFUND": {
            "HIGH": "I understand you'd like to request a refund. I can help process that. Which order or purchase does this relate to?",
            "MID": "You're asking about a refund. Can you provide the order number or purchase details?",
            "LOW": "Your request is about a refund. Could you provide more information about which order you're referring to?"
        },
        "SHIPPING": {
            "HIGH": "I can help you with shipping information and tracking. Can you provide your tracking number or order number?",
            "MID": "You have a shipping-related question. Can you share your order number?",
            "LOW": "Your inquiry is about shipping. Please provide details about your order."
        },
        "SUBSCRIPTION": {
            "HIGH": "I can assist with your subscription. Would you like to modify, cancel, or get information about your plan?",
            "MID": "You have a subscription-related question. What would you like to do with your subscription?",
            "LOW": "Your inquiry involves your subscription. Could you clarify what you need help with?"
        }
    }
    
    # Determine confidence level
    if confidence >= 0.85:
        conf_level = "HIGH"
    elif confidence >= 0.65:
        conf_level = "MID"
    else:
        conf_level = "LOW"
    
    # Get response from category-specific templates
    if category in category_responses:
        state.response_final = category_responses[category].get(conf_level, "Thank you for contacting support. How can I assist you today?")
    else:
        state.response_final = "Thank you for reaching out. Our support team will be happy to help you. Could you provide more details about your inquiry?"
    
    print(f"[DraftResponseNode] Response: {state.response_final[:80]}...")
    state.messages.append(f"[draft_response] {state.response_final[:100]}...")
    return state


def log_interaction_node(state: SupportAgentState) -> SupportAgentState:
    """Log interaction to database."""
    print(f"[LogInteractionNode] Logging...")
    
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                raw_message TEXT,
                detected_language TEXT,
                translated_message TEXT,
                predicted_label TEXT,
                confidence_score REAL,
                route_decision TEXT,
                response TEXT
            )
        """)
        
        cursor.execute("""
            INSERT INTO interactions 
            (timestamp, raw_message, detected_language, translated_message, predicted_label, confidence_score, route_decision, response)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            state.raw_message,
            state.detected_language,
            state.translated_message,
            state.predicted_label,
            state.confidence_score,
            state.route_decision,
            state.response_final
        ))
        
        conn.commit()
        conn.close()
        print(f"[LogInteractionNode] Logged to database")
    except Exception as e:
        print(f"[LogInteractionNode] ⚠ Logging failed: {str(e)}")
    
    state.messages.append("[log_interaction] Interaction logged")
    print(f"[LogInteractionNode] Done.")
    return state


# ============================================================================
# TRAINING ORCHESTRATION
# ============================================================================

def train_pipeline():
    """Run full training pipeline."""
    print("\n" + "="*70)
    print("STARTING TRAINING PIPELINE")
    print("="*70 + "\n")
    
    state = SupportAgentState(mode="train")
    
    state = preprocess_data_node(state)
    state = train_models_node(state)
    state = evaluate_models_node(state)
    state = select_model_node(state)
    state = persist_artifacts_node(state)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70 + "\n")
    
    return state

def predict_with_confidence(query, model, vectorizer):
    processed = preprocess_text(query)
    vec = vectorizer.transform([processed])
    pred = model.predict(vec)[0]
    probs = model.predict_proba(vec)[0]
    conf = float(np.max(probs))
    print(f"→ {pred} ({conf:.0%})")
    return pred, conf, dict(zip(model.classes_, probs))

def generate_response(predicted_class, confidence):
    level = "high" if confidence >= CONFIDENCE_THRESHOLD else "low"
    resp = RESPONSE_TEMPLATES.get(predicted_class, {}).get(level, "Thank you for contacting support.")
    if confidence < CONFIDENCE_THRESHOLD:
        resp += f"\n\n⚠️ Low confidence ({confidence:.0%})"
    return resp

def classify_query(query):
    """Full serving pipeline with all nodes executed."""
    if not query or len(query.strip()) < 3:
        return {"error": "Query too short"}, 0.0
    
    print("\n" + "="*80)
    print(f"PROCESSING NEW QUERY: {query[:100]}")
    print("="*80 + "\n")
    
    # Create state for this query
    state = SupportAgentState(raw_message=query)
    
    # Execute full serving pipeline (all 11 nodes)
    print(">>> STARTING FULL SERVING PIPELINE <<<\n")
    state = detect_language_node(state)
    state = translate_to_english_node(state)
    state = run_inference_node(state)
    state = confidence_router_node(state)
    state = draft_response_node(state)
    state = log_interaction_node(state)
    print("\n>>> PIPELINE COMPLETE <<<\n")
    
    # Print all messages from pipeline
    print("Pipeline trace:")
    for msg in state.messages:
        print(f"  {msg}")
    print()
    
    return {
        "category": state.predicted_label,
        "confidence": round(state.confidence_score, 4),
        "response": state.response_final,
        "route": state.route_decision,
        "language": state.detected_language
    }, state.confidence_score


class SupportTriageHandler(BaseHTTPRequestHandler):
    """HTTP request handler for support triage UI and API."""
    
    def do_GET(self):
        """Serve HTML interface."""
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            
            html = """<!DOCTYPE html>
<html>
<head>
    <title>Support Triage System</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        .container {
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            max-width: 900px;
            width: 100%;
            padding: 40px;
        }
        h1 {
            color: #333;
            margin-bottom: 10px;
            font-size: 28px;
        }
        .subtitle {
            color: #666;
            margin-bottom: 30px;
            font-size: 14px;
        }
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 20px;
        }
        .column {
            display: flex;
            flex-direction: column;
        }
        @media (max-width: 768px) {
            .grid { grid-template-columns: 1fr; }
        }
        label {
            font-weight: 600;
            color: #333;
            margin-bottom: 10px;
            display: block;
        }
        textarea {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-family: inherit;
            font-size: 14px;
            resize: vertical;
            transition: border-color 0.3s;
        }
        textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 14px 28px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            margin-top: 10px;
            transition: transform 0.2s;
        }
        button:hover { transform: translateY(-2px); }
        button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        .output-section {
            background: #f5f5f5;
            padding: 20px;
            border-radius: 8px;
            margin-top: 20px;
        }
        .result-item {
            display: flex;
            justify-content: space-between;
            padding: 12px 0;
            border-bottom: 1px solid #ddd;
        }
        .result-item:last-child { border-bottom: none; }
        .result-label {
            font-weight: 600;
            color: #666;
        }
        .result-value {
            color: #333;
            font-weight: 500;
        }
        .confidence-bar {
            background: #e0e0e0;
            height: 8px;
            border-radius: 4px;
            margin: 5px 0;
            overflow: hidden;
        }
        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            border-radius: 4px;
            transition: width 0.5s ease;
        }
        .error {
            color: #d32f2f;
            padding: 12px;
            background: #ffebee;
            border-radius: 8px;
            margin-top: 10px;
        }
        .success {
            color: #388e3c;
            padding: 12px;
            background: #e8f5e9;
            border-radius: 8px;
            margin-top: 10px;
        }
        .loading {
            display: none;
            text-align: center;
            color: #666;
            margin-top: 10px;
        }
        .loading.active {
            display: block;
        }
        .spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #e0e0e0;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .examples {
            margin-top: 20px;
            padding: 15px;
            background: #f0f7ff;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        .examples-title {
            font-weight: 600;
            color: #667eea;
            margin-bottom: 10px;
            font-size: 13px;
        }
        .example-btn {
            display: inline-block;
            background: white;
            color: #667eea;
            border: 1px solid #667eea;
            padding: 6px 12px;
            border-radius: 20px;
            margin: 4px 5px 4px 0;
            cursor: pointer;
            font-size: 12px;
            transition: all 0.3s;
        }
        .example-btn:hover {
            background: #667eea;
            color: white;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Support Ticket Triage</h1>
        <p class="subtitle">Classify customer queries and generate responses</p>
        
        <div class="grid">
            <div class="column">
                <label for="query">💬 Customer Query</label>
                <textarea id="query" placeholder="Enter customer query..." rows="8"></textarea>
                
                <div class="examples">
                    <div class="examples-title">Try these examples:</div>
                    <button class="example-btn" onclick="setExample('I need to reset my password')">Reset password</button>
                    <button class="example-btn" onclick="setExample('I want to cancel my order 1234')">Cancel order</button>
                    <button class="example-btn" onclick="setExample('Where is my package?')">Track package</button>
                    <button class="example-btn" onclick="setExample('Can I get a refund?')">Request refund</button>
                    <button class="example-btn" onclick="setExample('How do I contact support?')">Contact info</button>
                </div>
                
                <button onclick="analyzeQuery()" id="submitBtn">🚀 ANALYZE</button>
                <div class="loading" id="loading"><div class="spinner"></div> Processing...</div>
            </div>
            
            <div class="column">
                <label>📊 Results</label>
                <div id="results" style="min-height: 180px;"></div>
                <div id="messageContainer"></div>
            </div>
        </div>
        
        <div class="output-section">
            <label>💬 Suggested Response</label>
            <div id="response" style="background: white; padding: 15px; border-radius: 6px; min-height: 60px; color: #333; line-height: 1.6;"></div>
        </div>
    </div>
    
    <script>
        function setExample(text) {
            document.getElementById('query').value = text;
        }
        
        async function analyzeQuery() {
            const query = document.getElementById('query').value.trim();
            if (!query) {
                alert('Please enter a query');
                return;
            }
            
            const btn = document.getElementById('submitBtn');
            const loading = document.getElementById('loading');
            btn.disabled = true;
            loading.classList.add('active');
            
            try {
                const response = await fetch('/api/classify', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query })
                });
                
                const data = await response.json();
                
                if (data.error) {
                    showError(data.error);
                    clearResults();
                } else {
                    displayResults(data);
                }
            } catch (err) {
                showError('Server error: ' + err.message);
                clearResults();
            } finally {
                btn.disabled = false;
                loading.classList.remove('active');
            }
        }
        
        function displayResults(data) {
            const resultsHtml = `
                <div class="output-section">
                    <div class="result-item">
                        <span class="result-label">Category:</span>
                        <span class="result-value" style="font-size: 18px; color: #667eea;">${escapeHtml(data.category)}</span>
                    </div>
                    <div class="result-item">
                        <span class="result-label">Confidence:</span>
                        <span class="result-value">${(data.confidence * 100).toFixed(1)}%</span>
                    </div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${data.confidence * 100}%"></div>
                    </div>
                    <div class="result-item">
                        <span class="result-label">Route:</span>
                        <span class="result-value">${escapeHtml(data.route)}</span>
                    </div>
                    <div class="result-item">
                        <span class="result-label">Language:</span>
                        <span class="result-value">${escapeHtml(data.language)}</span>
                    </div>
                </div>
            `;
            document.getElementById('results').innerHTML = resultsHtml;
            document.getElementById('response').textContent = data.response;
            document.getElementById('messageContainer').innerHTML = '';
        }
        
        function clearResults() {
            document.getElementById('results').innerHTML = '';
            document.getElementById('response').textContent = '';
        }
        
        function showError(msg) {
            document.getElementById('results').innerHTML = `<div class="error">⚠️ ${escapeHtml(msg)}</div>`;
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
    </script>
</body>
</html>"""
            self.wfile.write(html.encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        """Handle API requests."""
        if self.path == "/api/classify":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                if content_length == 0:
                    self.send_json({"error": "No request body"}, 400)
                    return
                    
                body = self.rfile.read(content_length).decode('utf-8')
                if not body:
                    self.send_json({"error": "Empty request body"}, 400)
                    return
                
                data = json_module.loads(body)
                query = data.get("query", "").strip()
                
                if not query:
                    self.send_json({"error": "Empty query"}, 400)
                    return
                
                result, confidence = classify_query(query)
                
                if "error" in result:
                    self.send_json(result, 400)
                else:
                    self.send_json(result, 200)
                    
            except json_module.JSONDecodeError as e:
                print(f"[Server] JSON decode error: {str(e)}")
                self.send_json({"error": f"Invalid JSON: {str(e)}"}, 400)
            except Exception as e:
                print(f"[Server] Error processing request: {str(e)}")
                self.send_json({"error": f"Server error: {str(e)}"}, 500)
        else:
            self.send_response(404)
            self.end_headers()
    
    def send_json(self, data, code):
        """Send JSON response."""
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json_module.dumps(data).encode())
    
    def log_message(self, format, *args):
        """Suppress HTTP server logs."""
        pass


if __name__ == "__main__":
    print("\n🤖 AI SUPPORT TRIAGE\n")
    model, vectorizer = load_trained_pipeline()
    
    print("\n" + "="*80)
    print("HTTP SERVER STARTING")
    print("🌐 Open your browser: http://localhost:7860")
    print("="*80 + "\n")
    
    server = HTTPServer(("0.0.0.0", 7860), SupportTriageHandler)
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\nShutting down server...")
        server.shutdown()
