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
import gradio as gr

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
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

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
llm = None
if HAS_OPENAI_API_KEY:
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        HAS_LLM = True
        print("✓ OpenAI API key detected - LLM features enabled")
    except Exception as e:
        print(f"⚠ LLM initialization failed: {str(e)[:100]}")

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
            response = llm.invoke([
                HumanMessage(content=f"Translate to English, return ONLY the translation:\n\n{state.raw_message}")
            ])
            state.translated_message = response.content.strip()
            print("[TranslateNode] Translation complete")
        except Exception as e:
            state.translated_message = state.raw_message
            print(f"[TranslateNode] Translation failed - using original")
    
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
    """Generate customer response."""
    print(f"[DraftResponseNode] Drafting response...")
    
    category = state.predicted_label
    route = state.route_decision
    
    if category not in RESPONSE_TEMPLATES:
        base_response = f"Thank you for contacting us about {category.lower()}. A specialist will assist you shortly."
    else:
        base_response = RESPONSE_TEMPLATES[category].get(
            route,
            "Thank you for your inquiry. A specialist will assist you shortly."
        )
    
    state.response_final = base_response
    print(f"[DraftResponseNode] Response: {base_response[:80]}...")
    state.messages.append(f"[draft_response] {base_response[:100]}...")
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
    if not query or len(query.strip()) < 3:
        return "<div style='padding:30px;background:#fee;border-radius:20px;text-align:center;'><h2 style='color:#c00;margin:0;'>⚠️ Invalid</h2></div>", "", 0.0
    
    pred, conf, probs = predict_with_confidence(query, model, vectorizer)
    resp = generate_response(pred, conf)
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    
    gradients = ["linear-gradient(135deg, #667eea 0%, #764ba2 100%)", "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)", "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)"]
    
    bars = ""
    for i, (cat, prob) in enumerate(sorted_probs):
        pct = prob * 100
        grad = gradients[i] if i < len(gradients) else gradients[-1]
        emoji = "🚚" if "ship" in cat else ("🔐" if "account" in cat else "💰")
        bars += f"<div style='margin-bottom:24px;'><div style='display:flex;justify-content:space-between;margin-bottom:10px;'><span style='font-size:22px;font-weight:700;'>{emoji} {cat.upper()}</span><span style='font-size:30px;font-weight:800;background:{grad};-webkit-background-clip:text;-webkit-text-fill-color:transparent;'>{pct:.1f}%</span></div><div style='background:#e2e8f0;border-radius:12px;height:36px;overflow:hidden;'><div style='background:{grad};height:100%;width:{pct}%;border-radius:12px;'></div></div></div>"
    
    html = f"<div style='font-family:system-ui;'><div style='background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);padding:60px 40px;border-radius:24px;box-shadow:0 20px 60px rgba(102,126,234,0.5);margin-bottom:30px;text-align:center;'><p style='color:rgba(255,255,255,0.9);font-size:14px;font-weight:600;text-transform:uppercase;letter-spacing:3px;margin:0 0 15px 0;'>PREDICTED CATEGORY</p><h1 style='color:white;font-size:72px;margin:0;font-weight:900;text-shadow:0 4px 20px rgba(0,0,0,0.3);'>🎯 {pred.upper()}</h1><div style='margin-top:25px;display:inline-block;background:rgba(255,255,255,0.2);padding:15px 45px;border-radius:50px;'><p style='color:white;font-size:36px;margin:0;font-weight:700;'>💯 {conf:.0%}</p></div></div><div style='background:white;padding:40px;border-radius:24px;box-shadow:0 15px 50px rgba(0,0,0,0.12);'><h2 style='color:#1e293b;margin:0 0 30px 0;font-size:32px;font-weight:700;border-bottom:3px solid #e2e8f0;padding-bottom:20px;'>📊 Probability Breakdown</h2>{bars}</div></div>"
    
    return html, resp, conf

if __name__ == "__main__":
    print("\n🤖 AI SUPPORT TRIAGE\n")
    model, vectorizer = load_trained_pipeline()
    
    css = ".gradio-container{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%)!important;}button.primary{background:linear-gradient(135deg,#f093fb 0%,#f5576c 100%)!important;border:none!important;color:white!important;font-size:24px!important;font-weight:800!important;padding:26px 50px!important;border-radius:16px!important;box-shadow:0 15px 45px rgba(245,87,108,0.7)!important;}button.primary:hover{transform:translateY(-6px)!important;box-shadow:0 25px 65px rgba(245,87,108,0.9)!important;}textarea{font-size:18px!important;border:3px solid #e2e8f0!important;border-radius:16px!important;padding:20px!important;}textarea:focus{border-color:#667eea!important;box-shadow:0 0 0 4px rgba(102,126,234,0.2)!important;}"
    
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.HTML("<div style='text-align:center;padding:60px 30px;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);border-radius:28px;margin-bottom:40px;box-shadow:0 20px 70px rgba(102,126,234,0.6);'><h1 style='color:white;font-size:80px;margin:0 0 20px 0;font-weight:900;text-shadow:0 6px 20px rgba(0,0,0,0.3);'>🤖 AI SUPPORT TRIAGE</h1><p style='color:rgba(255,255,255,0.95);font-size:28px;margin:0;font-weight:600;'>Real-Time ML Classification</p></div>")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 💬 Query")
                query_input = gr.Textbox(placeholder="Enter question...", lines=8, show_label=False)
                submit_btn = gr.Button("🚀 ANALYZE", variant="primary", size="lg")
                gr.Examples(examples=[["Package lost"], ["Can't login"], ["Broken product"]], inputs=query_input)
            with gr.Column(scale=1):
                prediction_output = gr.HTML()
                gr.Markdown("### 📈 Confidence")
                confidence_bar = gr.Slider(minimum=0, maximum=1, interactive=False, show_label=False)
        
        gr.HTML("<div style='height:40px;background:linear-gradient(90deg,#667eea 0%,#764ba2 100%);margin:40px 0;border-radius:10px;'></div>")
        gr.Markdown("### 💬 Response")
        response_output = gr.Textbox(lines=6, show_label=False)
        
        submit_btn.click(fn=classify_query, inputs=query_input, outputs=[prediction_output, response_output, confidence_bar])
    
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, css=css)
