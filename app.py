import argparse
import html
import os
import json
import pickle
import re
import socket
import sqlite3
import warnings
from pathlib import Path
from typing import Any, Optional
from datetime import datetime
from functools import lru_cache

warnings.filterwarnings('ignore')

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"


def load_local_env(env_path: Path) -> bool:
    """Load environment variables from a local .env file."""
    if not env_path.exists():
        return False

    if load_dotenv is not None:
        load_dotenv(env_path)
        return True

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)

    return True


if load_local_env(ENV_PATH):
    print("✓ Loaded environment from .env")

# Essential data libraries
import pandas as pd
import numpy as np

# UI
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
from skills.ecommerce_context import (
    CATEGORY_OPTIONS,
    ORDER_RELATED_CATEGORIES,
    prepare_contextual_inference_message,
    resolve_customer_context,
)
from skills.ecommerce_repository import (
    build_user_summary as build_ecommerce_user_summary,
    clean_csv_value,
    format_context_date,
    format_order_summary,
    get_customer_scope as get_ecommerce_customer_scope,
    list_chat_users as list_ecommerce_chat_users,
    parse_flexible_datetime,
)
from skills.ecommerce_response import (
    build_policy_response_blueprint,
    build_relevant_context_slice,
    enforce_response_policies,
)

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
SKILLS_DIR = BASE_DIR / "skills"


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
    inference_message: str = ""
    customer_id: str = ""
    customer_name: str = ""
    customer_profile: dict = Field(default_factory=dict)
    customer_orders: list[dict] = Field(default_factory=list)
    conversation_history: list[dict] = Field(default_factory=list)
    pending_interaction: dict = Field(default_factory=dict)
    context_json: dict = Field(default_factory=dict)
    resolved_order_id: str = ""
    needs_more_context: bool = False
    clarification_prompt: str = ""
    
    # Inference outputs
    predicted_label: str = ""
    confidence_score: float = 0.0
    class_probabilities: dict = Field(default_factory=dict)
    route_decision: str = "AUTO_REPLY"
    response_final: str = ""
    response_generation: dict = Field(default_factory=dict)
    interaction_id: Optional[int] = None
    
    # Logging
    trace_logs: list[dict] = Field(default_factory=list)
    artifacts_saved: bool = False


def add_trace_log(state: SupportAgentState, stage: str, summary: str, details: Optional[dict] = None) -> None:
    """Append a structured pipeline trace entry to state."""
    state.trace_logs.append({
        "stage": stage,
        "summary": summary,
        "details": details or {},
        "logged_at": datetime.now().isoformat(timespec="seconds")
    })


@lru_cache(maxsize=16)
def load_skill_instructions(skill_name: str) -> str:
    """Load a skill markdown file so the runtime can use repo-authored guidance."""
    skill_path = SKILLS_DIR / f"{skill_name}.md"
    if not skill_path.exists():
        return ""

    raw_text = skill_path.read_text(encoding="utf-8")
    if raw_text.startswith("---"):
        parts = raw_text.split("---", 2)
        if len(parts) == 3:
            return parts[2].strip()
    return raw_text.strip()


def normalize_thresholds(tau_high: float, tau_low: float) -> tuple[float, float]:
    """Ensure confidence thresholds are usable for routing."""
    default_high, default_low = 0.85, 0.65

    try:
        tau_high = float(tau_high)
        tau_low = float(tau_low)
    except (TypeError, ValueError):
        print("[Thresholds] Invalid threshold values - falling back to defaults")
        return default_high, default_low

    if not (0.0 <= tau_low <= 1.0 and 0.0 <= tau_high <= 1.0):
        print("[Thresholds] Thresholds out of range - falling back to defaults")
        return default_high, default_low

    if tau_high < tau_low:
        print(f"[Thresholds] Invalid threshold order (tau_high={tau_high:.4f}, tau_low={tau_low:.4f}) - falling back to defaults")
        return default_high, default_low

    return tau_high, tau_low


def json_loads_safe(raw_value: Optional[str], default):
    """Safely parse JSON payloads stored in SQLite."""
    if not raw_value:
        return default

    try:
        return json.loads(raw_value)
    except (TypeError, json.JSONDecodeError):
        return default


def get_db_connection() -> sqlite3.Connection:
    """Open a SQLite connection with schema migrations applied."""
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    ensure_database_schema(conn)
    return conn


def ensure_database_schema(conn: sqlite3.Connection) -> None:
    """Create or migrate database tables used by the serving and admin UIs."""
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

    existing_columns = {
        row["name"] if isinstance(row, sqlite3.Row) else row[1]
        for row in cursor.execute("PRAGMA table_info(interactions)").fetchall()
    }

    required_columns = {
        "class_probabilities": "TEXT",
        "pipeline_trace": "TEXT",
        "response_generation": "TEXT",
        "customer_id": "TEXT",
        "customer_name": "TEXT",
        "context_json": "TEXT",
        "needs_more_context": "INTEGER DEFAULT 0",
        "clarification_prompt": "TEXT",
        "resolved_order_id": "TEXT",
        "feedback_flag": "INTEGER DEFAULT 0",
        "feedback_reason": "TEXT",
        "feedback_suggested_category": "TEXT",
        "feedback_updated_at": "TEXT"
    }

    for column_name, definition in required_columns.items():
        if column_name not in existing_columns:
            cursor.execute(f"ALTER TABLE interactions ADD COLUMN {column_name} {definition}")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS admin_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            interaction_id INTEGER NOT NULL,
            flagged INTEGER NOT NULL DEFAULT 1,
            reason TEXT,
            suggested_category TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(interaction_id) REFERENCES interactions(id)
        )
    """)

    conn.commit()


def interaction_row_to_dict(row: sqlite3.Row) -> dict:
    """Convert an interaction row into API-friendly JSON."""
    return {
        "id": row["id"],
        "timestamp": row["timestamp"],
        "raw_message": row["raw_message"],
        "detected_language": row["detected_language"],
        "translated_message": row["translated_message"],
        "predicted_label": row["predicted_label"],
        "confidence_score": float(row["confidence_score"] or 0.0),
        "route_decision": row["route_decision"],
        "response": row["response"],
        "class_probabilities": json_loads_safe(row["class_probabilities"], {}),
        "pipeline_trace": json_loads_safe(row["pipeline_trace"], []),
        "response_generation": json_loads_safe(row["response_generation"], {}),
        "customer_id": row["customer_id"] or "",
        "customer_name": row["customer_name"] or "",
        "context_json": json_loads_safe(row["context_json"], {}),
        "needs_more_context": bool(row["needs_more_context"]),
        "clarification_prompt": row["clarification_prompt"] or "",
        "resolved_order_id": row["resolved_order_id"] or "",
        "feedback": {
            "flagged": bool(row["feedback_flag"]),
            "reason": row["feedback_reason"] or "",
            "suggested_category": row["feedback_suggested_category"] or "",
            "updated_at": row["feedback_updated_at"] or ""
        }
    }


def interaction_row_to_user_dict(row: sqlite3.Row) -> dict:
    """Convert an interaction row into a safe user-facing chat payload."""
    data = interaction_row_to_dict(row)
    return {
        "id": data["id"],
        "timestamp": data["timestamp"],
        "raw_message": data["raw_message"],
        "predicted_label": data["predicted_label"],
        "confidence_score": data["confidence_score"],
        "route_decision": data["route_decision"],
        "response": data["response"],
        "needs_more_context": data["needs_more_context"],
        "resolved_order_id": data["resolved_order_id"],
    }


def fetch_admin_dashboard_data(limit: int = 100) -> dict:
    """Return interaction history and summary metrics for the admin UI."""
    limit = max(1, min(int(limit), 500))

    conn = get_db_connection()
    cursor = conn.cursor()

    summary_row = cursor.execute("""
        SELECT
            COUNT(*) AS total_interactions,
            SUM(CASE WHEN feedback_flag = 1 THEN 1 ELSE 0 END) AS flagged_interactions,
            SUM(CASE WHEN route_decision = 'AUTO_REPLY' THEN 1 ELSE 0 END) AS auto_replies,
            SUM(CASE WHEN route_decision = 'CLARIFY' THEN 1 ELSE 0 END) AS clarifications,
            SUM(CASE WHEN route_decision = 'ESCALATE' THEN 1 ELSE 0 END) AS escalations
        FROM interactions
    """).fetchone()

    rows = cursor.execute("""
        SELECT *
        FROM interactions
        ORDER BY datetime(timestamp) DESC, id DESC
        LIMIT ?
    """, (limit,)).fetchall()

    conn.close()

    return {
        "summary": {
            "total_interactions": int(summary_row["total_interactions"] or 0),
            "flagged_interactions": int(summary_row["flagged_interactions"] or 0),
            "auto_replies": int(summary_row["auto_replies"] or 0),
            "clarifications": int(summary_row["clarifications"] or 0),
            "escalations": int(summary_row["escalations"] or 0),
        },
        "interactions": [interaction_row_to_dict(row) for row in rows],
        "category_options": CATEGORY_OPTIONS,
    }


def save_admin_feedback(
    interaction_id: int,
    flagged: bool,
    reason: str = "",
    suggested_category: str = ""
) -> dict:
    """Persist admin review feedback for a logged interaction."""
    cleaned_reason = reason.strip()
    cleaned_category = suggested_category.strip().upper()

    if cleaned_category and cleaned_category not in CATEGORY_OPTIONS:
        raise ValueError(f"Suggested category must be one of: {', '.join(CATEGORY_OPTIONS)}")

    timestamp = datetime.now().isoformat(timespec="seconds")
    conn = get_db_connection()
    cursor = conn.cursor()

    row = cursor.execute("SELECT id FROM interactions WHERE id = ?", (interaction_id,)).fetchone()
    if row is None:
        conn.close()
        raise ValueError("Interaction not found")

    cursor.execute("""
        INSERT INTO admin_feedback (interaction_id, flagged, reason, suggested_category, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (
        interaction_id,
        1 if flagged else 0,
        cleaned_reason,
        cleaned_category,
        timestamp
    ))

    cursor.execute("""
        UPDATE interactions
        SET
            feedback_flag = ?,
            feedback_reason = ?,
            feedback_suggested_category = ?,
            feedback_updated_at = ?
        WHERE id = ?
    """, (
        1 if flagged else 0,
        cleaned_reason,
        cleaned_category,
        timestamp,
        interaction_id
    ))

    updated_row = cursor.execute("SELECT * FROM interactions WHERE id = ?", (interaction_id,)).fetchone()
    conn.commit()
    conn.close()
    return interaction_row_to_dict(updated_row)


def list_chat_users() -> list[dict]:
    """Return selectable chat users from the structured e-commerce dataset."""
    return list_ecommerce_chat_users()


def get_customer_scope(customer_id: str) -> tuple[dict, list[dict]]:
    """Return the selected customer's profile and only their own orders."""
    return get_ecommerce_customer_scope(customer_id)


def fetch_customer_chat_history(customer_id: str, limit: int = 100) -> list[dict]:
    """Return chat history for one selected customer."""
    limit = max(1, min(int(limit), 200))
    conn = get_db_connection()
    rows = conn.execute("""
        SELECT *
        FROM (
            SELECT *
            FROM interactions
            WHERE customer_id = ?
            ORDER BY datetime(timestamp) DESC, id DESC
            LIMIT ?
        ) recent
        ORDER BY datetime(timestamp) ASC, id ASC
    """, (str(customer_id), limit)).fetchall()
    conn.close()
    return [interaction_row_to_dict(row) for row in rows]


def fetch_customer_chat_history_for_user(customer_id: str, limit: int = 100) -> list[dict]:
    """Return sanitized chat history for the customer-facing UI."""
    limit = max(1, min(int(limit), 200))
    conn = get_db_connection()
    rows = conn.execute("""
        SELECT *
        FROM (
            SELECT *
            FROM interactions
            WHERE customer_id = ?
            ORDER BY datetime(timestamp) DESC, id DESC
            LIMIT ?
        ) recent
        ORDER BY datetime(timestamp) ASC, id ASC
    """, (str(customer_id), limit)).fetchall()
    conn.close()
    return [interaction_row_to_user_dict(row) for row in rows]


def fetch_latest_customer_interaction(customer_id: str) -> dict:
    """Return the latest stored interaction for one customer."""
    conn = get_db_connection()
    row = conn.execute("""
        SELECT *
        FROM interactions
        WHERE customer_id = ?
        ORDER BY datetime(timestamp) DESC, id DESC
        LIMIT 1
    """, (str(customer_id),)).fetchone()
    conn.close()
    return interaction_row_to_dict(row) if row else {}


def build_user_summary(profile: dict, orders: list[dict]) -> dict:
    """Create a compact summary for the selected user panel."""
    return build_ecommerce_user_summary(profile, orders)


def fetch_user_chat_payload(customer_id: str, limit: int = 100) -> dict:
    """Return user summary plus chat history for the support chat UI."""
    profile, orders = get_customer_scope(customer_id)
    if not profile:
        raise ValueError("Selected customer was not found in ecommerce_data/ecommerce_customers.csv")

    history = fetch_customer_chat_history_for_user(customer_id, limit=limit)
    return {
        "user": build_user_summary(profile, orders),
        "history": history,
    }


def polish_response_with_llm(state: SupportAgentState, relevant_context: dict, blueprint: str) -> str:
    """Use the LLM to refine the response while preserving all policy facts."""
    if not HAS_LLM or llm_client is None:
        raise RuntimeError("LLM unavailable")

    is_first_reply = len(state.conversation_history) == 0
    skill_instructions = load_skill_instructions("draft_response")
    system_prompt = (
        "You are a customer support assistant. "
        "The repository skill instructions below are authoritative for how to handle the response. "
        "Use only the provided context and response blueprint. "
        "Never reveal JSON, internal reasoning, or policy metadata. "
        "Keep the reply polished, empathetic, and concise. "
        "Do not add facts that are not present in the blueprint or context. "
        "Do not repeat the greeting or opening reassurance twice. "
        "If this is the first reply, the response must begin with "
        f"'Hello {state.customer_name}!'."
    )
    if skill_instructions:
        system_prompt = (
            f"{system_prompt}\n\n"
            "Repository skill instructions for `draft_response`:\n"
            f"{skill_instructions}"
        )

    user_prompt = (
        f"Latest user message:\n{state.raw_message}\n\n"
        f"Route decision: {state.route_decision}\n"
        f"First reply in chat: {is_first_reply}\n\n"
        f"Relevant context JSON:\n{json.dumps(relevant_context, ensure_ascii=False, indent=2)}\n\n"
        f"Required policy-safe content:\n{blueprint}\n\n"
        "Rewrite this as a refined customer-facing reply in plain text only. "
        "Do not output JSON, bullet lists, or markdown code fences."
    )

    response = llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()

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
        state.tau_low = float(thresholds[valid & low_mask][0])
    else:
        state.tau_low = 0.50

    state.tau_high, state.tau_low = normalize_thresholds(state.tau_high, state.tau_low)
    
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

def load_customer_scope_node(state: SupportAgentState) -> SupportAgentState:
    """Load the selected customer's profile, orders, and recent chat history."""
    print("[CustomerScopeNode] Loading selected customer scope...")

    if not state.customer_id:
        raise ValueError("A customer must be selected before starting the chat.")

    profile, orders = get_customer_scope(state.customer_id)
    if not profile:
        raise ValueError("Selected customer was not found in ecommerce_data/ecommerce_customers.csv")

    state.customer_profile = profile
    state.customer_name = profile.get("name", f"Customer {state.customer_id}")
    state.customer_orders = orders
    state.conversation_history = fetch_customer_chat_history(state.customer_id, limit=50)

    latest = state.conversation_history[-1] if state.conversation_history else {}
    if latest.get("needs_more_context"):
        state.pending_interaction = latest
    else:
        state.pending_interaction = {}

    summary = (
        f"Customer {state.customer_name} (ID {state.customer_id}) with "
        f"{len(state.customer_orders)} orders and {len(state.conversation_history)} prior chat turns."
    )
    state.messages.append(f"[customer_scope] {summary}")
    add_trace_log(
        state,
        "customer_scope",
        summary,
        {
            "customer_id": state.customer_id,
            "customer_name": state.customer_name,
            "order_count": len(state.customer_orders),
            "prior_turns": len(state.conversation_history),
            "pending_clarification": bool(state.pending_interaction),
        }
    )
    return state


def detect_language_node(state: SupportAgentState) -> SupportAgentState:
    """Detect language of incoming message."""
    print(f"[DetectLanguageNode] Detecting language...")

    short_ascii_reference = (
        len(state.raw_message.strip()) <= 24
        and re.fullmatch(r"[A-Za-z0-9\s#:\-'\?\!\.,/]+", state.raw_message.strip() or "") is not None
    )
    if short_ascii_reference:
        state.detected_language = "en"
        detector_used = "short_ascii_heuristic"
        print("[DetectLanguageNode] Short ASCII message detected - defaulting to English")
        state.messages.append(f"[detect_language] Detected language: {state.detected_language}")
        add_trace_log(
            state,
            "detect_language",
            f"Detected `{state.detected_language}` using {detector_used}",
            {
                "detected_language": state.detected_language,
                "detector": detector_used,
                "requires_translation": False
            }
        )
        return state
    
    if not HAS_LANGDETECT:
        state.detected_language = 'en'
        detector_used = "default"
        print("[DetectLanguageNode] No language detection available - defaulting to English")
    else:
        try:
            from textblob import TextBlob
            blob = TextBlob(state.raw_message)
            state.detected_language = blob.detect_language()
            detector_used = "textblob"
            print(f"[DetectLanguageNode] Detected: {state.detected_language}")
        except Exception as e:
            try:
                from langdetect import detect
                state.detected_language = detect(state.raw_message)
                detector_used = "langdetect"
                print(f"[DetectLanguageNode] Detected (fallback): {state.detected_language}")
            except:
                state.detected_language = 'en'
                detector_used = "default"
                print("[DetectLanguageNode] Detection failed - defaulting to English")
    
    state.messages.append(f"[detect_language] Detected language: {state.detected_language}")
    add_trace_log(
        state,
        "detect_language",
        f"Detected `{state.detected_language}` using {detector_used}",
        {
            "detected_language": state.detected_language,
            "detector": detector_used,
            "requires_translation": state.detected_language != "en"
        }
    )
    return state


def translate_to_english_node(state: SupportAgentState) -> SupportAgentState:
    """Translate non-English messages to English if LLM available."""
    print(f"[TranslateNode] Translating...")
    
    if state.detected_language == 'en':
        state.translated_message = state.raw_message
        translation_mode = "skipped_english"
        print("[TranslateNode] Message already in English")
    elif not HAS_LLM:
        state.translated_message = state.raw_message
        translation_mode = "no_llm_fallback"
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
            translation_mode = "llm_translation"
            print("[TranslateNode] Translation complete")
        except Exception as e:
            state.translated_message = state.raw_message
            translation_mode = "translation_error_fallback"
            print(f"[TranslateNode] Translation failed: {str(e)[:50]} - using original")
    
    state.messages.append(f"[translate_to_english] Message: {state.translated_message[:80]}...")
    add_trace_log(
        state,
        "translate_to_english",
        f"Translation mode: {translation_mode}",
        {
            "source_language": state.detected_language,
            "translation_mode": translation_mode,
            "used_llm": translation_mode == "llm_translation",
            "translated_message": state.translated_message
        }
    )
    return state


def prepare_contextual_query_node(state: SupportAgentState) -> SupportAgentState:
    """Augment short clarification follow-ups with prior pending context."""
    print("[ContextPrepNode] Preparing customer-scoped inference text...")

    state.inference_message, prep_reason = prepare_contextual_inference_message(
        state.translated_message,
        state.customer_orders,
        state.pending_interaction,
    )

    state.messages.append(f"[context_prep] Inference text prepared via {prep_reason}")
    add_trace_log(
        state,
        "context_prep",
        f"Prepared inference text using {prep_reason}",
        {
            "prep_reason": prep_reason,
            "inference_message": state.inference_message
        }
    )
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
                state.tau_high, state.tau_low = normalize_thresholds(
                    thresholds.get('tau_high', state.tau_high),
                    thresholds.get('tau_low', state.tau_low)
                )
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
    
    cleaned = clean_text(state.inference_message or state.translated_message)
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

    top_predictions = [
        {"label": label, "score": score}
        for label, score in sorted(
            state.class_probabilities.items(),
            key=lambda item: item[1],
            reverse=True
        )[:3]
    ]
    
    print(f"[InferenceNode] Predicted: {pred_label} ({confidence:.4f})")
    state.messages.append(f"[run_inference] Predicted: {pred_label} ({confidence:.4f})")
    add_trace_log(
        state,
        "run_inference",
        f"Predicted {pred_label} with {confidence:.2%} confidence",
        {
            "predicted_label": pred_label,
            "confidence_score": round(confidence, 4),
            "top_predictions": top_predictions,
            "inference_input": state.inference_message or state.translated_message,
            "tau_high": round(state.tau_high, 4),
            "tau_low": round(state.tau_low, 4)
        }
    )
    return state


def resolve_context_node(state: SupportAgentState) -> SupportAgentState:
    """Resolve user-scoped business context needed to answer the query."""
    print("[ContextResolveNode] Resolving customer and order context...")
    translated_query = state.translated_message or state.raw_message
    original_label = state.predicted_label
    resolved = resolve_customer_context(
        customer_id=state.customer_id,
        translated_query=translated_query,
        predicted_label=state.predicted_label,
        customer_profile=state.customer_profile,
        customer_orders=state.customer_orders,
        conversation_history=state.conversation_history,
        pending_interaction=state.pending_interaction,
    )

    state.predicted_label = resolved["predicted_label"]
    state.context_json = resolved["context_json"]
    state.resolved_order_id = resolved["resolved_order_id"]
    state.needs_more_context = resolved["needs_more_context"]
    state.clarification_prompt = resolved["clarification_prompt"]
    resolution_reason = resolved["resolution_reason"]
    date_mentions = resolved["date_mentions"]
    query_dates = resolved["query_dates"]
    clarification_candidates = resolved["clarification_candidates"]
    requires_order_lookup = resolved["requires_order_lookup"]

    if state.predicted_label != original_label:
        add_trace_log(
            state,
            "context_resolve",
            f"Adjusted intent from {original_label} to {state.predicted_label} based on structured query analysis.",
            {
                "original_label": original_label,
                "adjusted_label": state.predicted_label,
                "trigger_query": translated_query,
            }
        )

    state.messages.append(f"[context_resolve] Context reason: {resolution_reason}")
    add_trace_log(
        state,
        "context_resolve",
        f"Resolved context using {resolution_reason}",
        {
            "predicted_label": state.predicted_label,
            "requires_order_lookup": requires_order_lookup,
            "date_mentions": [
                {
                    "raw": mention["raw"],
                    "date": mention["date"].isoformat(),
                    "kind": mention["kind"],
                }
                for mention in date_mentions
            ],
            "query_dates": [date.isoformat() for date in query_dates],
            "needs_more_context": state.needs_more_context,
            "resolved_order_id": state.resolved_order_id,
            "clarification_prompt": state.clarification_prompt,
            "clarification_candidate_ids": [item.get("order_id", "") for item in clarification_candidates],
        }
    )
    return state


def confidence_router_node(state: SupportAgentState) -> SupportAgentState:
    """Route based on confidence scores."""
    print(f"[RouterNode] Routing...")
    grounded_order_context = bool(state.resolved_order_id) and state.predicted_label in ORDER_RELATED_CATEGORIES

    if state.needs_more_context:
        state.route_decision = "CLARIFY"
        rationale = "Missing the order-specific context needed to answer safely"
    elif grounded_order_context and state.confidence_score >= state.tau_low:
        state.route_decision = "AUTO_REPLY"
        rationale = (
            f"Resolved order {state.resolved_order_id} provides grounded context "
            f"({state.confidence_score:.4f} >= {state.tau_low:.4f})"
        )
    elif state.confidence_score >= state.tau_high:
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
    add_trace_log(
        state,
        "confidence_router",
        f"Routed to {state.route_decision}",
        {
            "route_decision": state.route_decision,
            "rationale": rationale,
            "confidence_score": round(state.confidence_score, 4),
            "tau_high": round(state.tau_high, 4),
            "tau_low": round(state.tau_low, 4)
        }
    )
    return state


def draft_response_node(state: SupportAgentState) -> SupportAgentState:
    """Generate a policy-safe customer reply from internal context."""
    print(f"[DraftResponseNode] Drafting response...")

    state.active_skill = "draft_response"
    relevant_context = build_relevant_context_slice(
        raw_message=state.raw_message,
        predicted_label=state.predicted_label,
        route_decision=state.route_decision,
        context_json=state.context_json,
        conversation_history=state.conversation_history,
        customer_orders=state.customer_orders,
    )
    blueprint = build_policy_response_blueprint(
        customer_name=state.customer_name,
        conversation_history=state.conversation_history,
        raw_message=state.raw_message,
        predicted_label=state.predicted_label,
        context_json=state.context_json,
        needs_more_context=state.needs_more_context,
        clarification_prompt=state.clarification_prompt,
    )
    llm_error = ""

    if HAS_LLM and llm_client is not None:
        try:
            drafted_response = polish_response_with_llm(state, relevant_context, blueprint)
            response_source = "skill_guided_customer_context_llm"
            generation_reason = (
                "Used the LLM with the repo's draft_response skill instructions, the scoped internal JSON context, "
                "and the enforced policy blueprint."
            )
        except Exception as e:
            drafted_response = blueprint
            response_source = "policy_blueprint_fallback"
            llm_error = str(e)[:200]
            generation_reason = (
                "LLM refinement was unavailable, so the reply was generated directly from the scoped "
                "policy blueprint."
            )
    else:
        drafted_response = blueprint
        response_source = "policy_blueprint"
        generation_reason = (
            "Generated the reply directly from the scoped policy blueprint because no LLM was available."
        )

    state.response_final = enforce_response_policies(
        customer_name=state.customer_name,
        conversation_history=state.conversation_history,
        predicted_label=state.predicted_label,
        raw_message=state.raw_message,
        context_json=state.context_json,
        needs_more_context=state.needs_more_context,
        blueprint=blueprint,
        drafted_response=drafted_response,
    )
    state.response_generation = {
        "source": response_source,
        "active_skill": state.active_skill,
        "category": state.predicted_label,
        "route_decision": state.route_decision,
        "confidence_score": round(state.confidence_score, 4),
        "reason": generation_reason,
        "used_llm": response_source == "skill_guided_customer_context_llm",
        "resolved_order_id": state.resolved_order_id,
        "needs_more_context": state.needs_more_context,
        "context_keys": sorted(relevant_context.keys()),
    }
    if llm_error:
        state.response_generation["llm_error"] = llm_error

    print(f"[DraftResponseNode] Response: {state.response_final[:80]}...")
    state.messages.append(f"[draft_response] {state.response_final[:100]}...")
    add_trace_log(
        state,
        "draft_response",
        state.response_generation["reason"],
        {
            **state.response_generation,
            "blueprint_preview": blueprint[:160],
            "response_preview": state.response_final[:160]
        }
    )
    return state


def log_interaction_node(state: SupportAgentState) -> SupportAgentState:
    """Log interaction to database."""
    print(f"[LogInteractionNode] Logging...")
    
    add_trace_log(
        state,
        "feedback_loop",
        "Stored interaction for admin review and future retraining feedback.",
        {
            "feedback_flagged": False,
            "feedback_status": "awaiting_review"
        }
    )
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO interactions 
            (
                timestamp,
                raw_message,
                detected_language,
                translated_message,
                customer_id,
                customer_name,
                predicted_label,
                confidence_score,
                route_decision,
                response,
                class_probabilities,
                pipeline_trace,
                response_generation,
                context_json,
                needs_more_context,
                clarification_prompt,
                resolved_order_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            state.raw_message,
            state.detected_language,
            state.translated_message,
            state.customer_id,
            state.customer_name,
            state.predicted_label,
            state.confidence_score,
            state.route_decision,
            state.response_final,
            json.dumps(state.class_probabilities, ensure_ascii=False),
            json.dumps(state.trace_logs, ensure_ascii=False),
            json.dumps(state.response_generation, ensure_ascii=False),
            json.dumps(state.context_json, ensure_ascii=False),
            1 if state.needs_more_context else 0,
            state.clarification_prompt,
            state.resolved_order_id,
        ))

        state.interaction_id = int(cursor.lastrowid)
        conn.commit()
        conn.close()
        print(f"[LogInteractionNode] Logged to database as interaction #{state.interaction_id}")
    except Exception as e:
        print(f"[LogInteractionNode] ⚠ Logging failed: {str(e)}")
    
    state.messages.append(f"[log_interaction] Interaction logged as #{state.interaction_id}")
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

def classify_query(query, customer_id: str):
    """Full serving pipeline with all nodes executed for one selected customer."""
    if not query or len(query.strip()) < 3:
        return {"error": "Query too short"}, 0.0
    
    print("\n" + "="*80)
    print(f"PROCESSING NEW QUERY FOR CUSTOMER {customer_id}: {query[:100]}")
    print("="*80 + "\n")
    
    # Create state for this query
    state = SupportAgentState(raw_message=query, customer_id=str(customer_id))
    
    # Execute full serving pipeline (all 11 nodes)
    print(">>> STARTING FULL SERVING PIPELINE <<<\n")
    state = load_customer_scope_node(state)
    state = detect_language_node(state)
    state = translate_to_english_node(state)
    state = prepare_contextual_query_node(state)
    state = run_inference_node(state)
    state = resolve_context_node(state)
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
        "interaction_id": state.interaction_id,
        "customer_id": state.customer_id,
        "customer_name": state.customer_name,
        "category": state.predicted_label,
        "confidence": round(state.confidence_score, 4),
        "response": state.response_final,
        "route": state.route_decision,
        "language": state.detected_language,
        "needs_more_context": state.needs_more_context,
        "resolved_order_id": state.resolved_order_id,
    }, state.confidence_score


def parse_runtime_args() -> argparse.Namespace:
    """Parse runtime options for launching the support UI."""
    parser = argparse.ArgumentParser(description="Run the customer support triage app.")
    parser.add_argument(
        "--host",
        default=os.getenv("HOST", "0.0.0.0"),
        help="Host interface to bind the Gradio app to.",
    )
    parser.add_argument(
        "--gradio-port",
        type=int,
        default=int(os.getenv("GRADIO_PORT", "7861")),
        help="Port used by the Gradio app.",
    )
    return parser.parse_args()


def resolve_gradio_port(host: str, preferred_port: int, max_attempts: int = 20) -> int:
    """Find an available local port for Gradio, starting from the preferred port."""
    host = "127.0.0.1" if host in {"0.0.0.0", ""} else host
    for offset in range(max_attempts):
        candidate = preferred_port + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, candidate))
            except OSError:
                continue
        return candidate
    raise OSError(
        f"Could not find an available port in range {preferred_port}-{preferred_port + max_attempts - 1}."
    )


def _display_text(value: Any, default: str = "") -> str:
    text = "" if value is None else str(value).strip()
    return text or default


def _escape_text(value: Any, default: str = "") -> str:
    return html.escape(_display_text(value, default))


def _format_timestamp(value: Any) -> str:
    raw = _display_text(value)
    if not raw:
        return "Unknown time"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return raw
    return parsed.strftime("%d %b %Y, %I:%M %p")


def _pretty_key(value: str) -> str:
    words = value.replace("_", " ").strip().split()
    return " ".join(word.capitalize() for word in words) or "Value"


def _format_trace_value(value: Any) -> str:
    if value in (None, ""):
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def render_gradio_banner(message: str = "", tone: str = "error") -> str:
    message = _display_text(message)
    if not message:
        return ""
    tone_class = {
        "error": "ui-banner--error",
        "success": "ui-banner--success",
        "info": "ui-banner--info",
    }.get(tone, "ui-banner--info")
    return f'<div class="ui-banner {tone_class}">{_escape_text(message)}</div>'


def render_support_hero_html() -> str:
    return """
    <section class="hero-panel hero-panel--support">
        <div class="hero-panel__copy">
            <h1>Customer Support Chat</h1>
            <p>Chat as one selected user at a time. Every reply is restricted to that customer's account and structured e-commerce order data, and the agent grounds delivery, refund, return, seller, and transporter answers from the <code>ecommerce_data/</code> datasets.</p>
            <div class="hero-panel__note">Current context is limited to the selected customer only.</div>
        </div>
    </section>
    """


def render_support_status_html(message: str) -> str:
    return f'<div class="support-status-line">{_escape_text(message)}</div>'


def render_support_user_summary_html(user: Optional[dict]) -> str:
    if not user:
        return """
        <div class="support-kv-grid">
            <div class="support-kv"><span>Status</span><div>No user selected.</div></div>
        </div>
        """

    rows = [
        ("Customer", f"{_display_text(user.get('name'), 'n/a')} (#{_display_text(user.get('customer_id'), 'n/a')})"),
        ("Account", _display_text(user.get("account_status"), "n/a")),
        ("Email", _display_text(user.get("registered_email"), "n/a")),
        ("Phone", _display_text(user.get("registered_phone_number"), "n/a")),
        ("Plan", _display_text(user.get("subscription_plan_name"), "n/a")),
        ("Prime", "Yes" if user.get("prime_subscription_flag") else "No"),
        ("Orders", str(user.get("order_count", 0) or 0)),
    ]
    rows_html = "".join(
        f'<div class="support-kv"><span>{_escape_text(label)}</span><div>{_escape_text(value)}</div></div>'
        for label, value in rows
    )
    return f'<div class="support-kv-grid">{rows_html}</div>'


def render_support_recent_orders_html(orders: list[dict]) -> str:
    if not orders:
        return '<div class="support-order-card">No orders found for this customer.</div>'

    cards = []
    for order in orders:
        amount = float(order.get("order_amount", 0.0) or 0.0)
        cards.append(
            f"""
            <article class="support-order-card">
                <strong>Order #{_escape_text(order.get('order_id'), 'n/a')}</strong>
                <div>{_escape_text(order.get('order_status'), 'n/a')} · {_escape_text(order.get('delivery_status'), 'n/a')}</div>
                <div>{_escape_text(order.get('order_currency'))} {amount:.2f}</div>
                <div>{_escape_text(order.get('order_date'), 'n/a')}</div>
            </article>
            """
        )
    return "".join(cards)


def render_support_chat_history_html(history: list[dict]) -> str:
    if not history:
        return """
        <div class="support-chat-scroll">
            <div class="support-empty-state">No previous chat history for this customer yet. Start the conversation below.</div>
        </div>
        """

    turns = []
    for turn in history:
        meta_pills = [
            _display_text(turn.get("predicted_label"), "n/a"),
            f"{float(turn.get('confidence_score', 0.0) or 0.0) * 100:.1f}% confidence",
            _display_text(turn.get("route_decision"), "n/a"),
        ]
        if turn.get("needs_more_context"):
            meta_pills.append("Needs more context")
        meta_pills.append(_format_timestamp(turn.get("timestamp")))
        pills_html = "".join(
            f'<span class="support-pill">{_escape_text(pill)}</span>'
            for pill in meta_pills
        )
        turns.append(
            f"""
            <div class="support-turn-group">
                <div class="support-bubble-row support-bubble-row--user">
                    <div class="support-bubble support-bubble--user">
                        <div class="support-bubble-label">Customer</div>
                        <p class="support-bubble-copy">{_escape_text(turn.get('raw_message'))}</p>
                    </div>
                </div>
                <div class="support-bubble-row support-bubble-row--assistant">
                    <div class="support-bubble support-bubble--assistant">
                        <div class="support-bubble-label">Assistant</div>
                        <p class="support-bubble-copy">{_escape_text(turn.get('response'))}</p>
                        <div class="support-meta-row">{pills_html}</div>
                    </div>
                </div>
            </div>
            """
        )
    return f'<div class="support-chat-scroll">{"".join(turns)}</div>'


def build_support_view_outputs(
    customer_id: str,
    *,
    status_override: Optional[str] = None,
    error_message: str = "",
) -> tuple[str, str, str, str, str]:
    customer_id = _display_text(customer_id)
    if not customer_id:
        return (
            render_support_chat_history_html([]),
            render_support_user_summary_html(None),
            render_support_recent_orders_html([]),
            render_support_status_html(status_override or "No customer records were found."),
            render_gradio_banner(error_message, tone="error"),
        )

    try:
        payload = fetch_user_chat_payload(customer_id, limit=100)
    except Exception as exc:
        return (
            render_support_chat_history_html([]),
            render_support_user_summary_html(None),
            render_support_recent_orders_html([]),
            render_support_status_html(status_override or "Unable to load customer chat."),
            render_gradio_banner(error_message or str(exc), tone="error"),
        )

    user = payload.get("user") or {}
    return (
        render_support_chat_history_html(payload.get("history") or []),
        render_support_user_summary_html(user),
        render_support_recent_orders_html(user.get("recent_orders") or []),
        render_support_status_html(
            status_override
            or f"Chatting as {_display_text(user.get('name'), 'Customer')} (#{_display_text(user.get('customer_id'), customer_id)})"
        ),
        render_gradio_banner(error_message, tone="error"),
    )


def initialize_support_tab(gr_module):
    try:
        users = list_chat_users()
    except Exception as exc:
        return (
            gr_module.update(choices=[], value=None),
            render_support_chat_history_html([]),
            render_support_user_summary_html(None),
            render_support_recent_orders_html([]),
            render_support_status_html("Unable to load users."),
            render_gradio_banner(str(exc), tone="error"),
        )

    if not users:
        return (
            gr_module.update(choices=[], value=None),
            render_support_chat_history_html([]),
            render_support_user_summary_html(None),
            render_support_recent_orders_html([]),
            render_support_status_html("No customer records were found."),
            "",
        )

    choices = [(user["label"], user["customer_id"]) for user in users]
    selected_id = users[0]["customer_id"]
    chat_html, summary_html, orders_html, status_html, error_html = build_support_view_outputs(selected_id)
    return (
        gr_module.update(choices=choices, value=selected_id),
        chat_html,
        summary_html,
        orders_html,
        status_html,
        error_html,
    )


def handle_support_customer_change(customer_id: str):
    return build_support_view_outputs(customer_id)


def handle_support_message_submit(customer_id: str, query: str):
    query = _display_text(query)
    customer_id = _display_text(customer_id)

    if not customer_id:
        chat_html, summary_html, orders_html, status_html, error_html = build_support_view_outputs(
            customer_id,
            status_override="Please choose a customer first.",
            error_message="Please choose a customer first.",
        )
        return query, chat_html, summary_html, orders_html, status_html, error_html

    if not query:
        chat_html, summary_html, orders_html, status_html, error_html = build_support_view_outputs(
            customer_id,
            error_message="Please enter a message before sending.",
        )
        return query, chat_html, summary_html, orders_html, status_html, error_html

    try:
        result, _confidence = classify_query(query, customer_id)
        if "error" in result:
            raise ValueError(result["error"])
    except Exception as exc:
        chat_html, summary_html, orders_html, status_html, error_html = build_support_view_outputs(
            customer_id,
            status_override="Unable to send message.",
            error_message=str(exc),
        )
        return query, chat_html, summary_html, orders_html, status_html, error_html

    chat_html, summary_html, orders_html, status_html, error_html = build_support_view_outputs(customer_id)
    return "", chat_html, summary_html, orders_html, status_html, error_html


def render_admin_hero_html() -> str:
    return """
    <section class="hero-panel hero-panel--admin">
        <div class="hero-panel__copy">
            <h1>Admin Pipeline Dashboard</h1>
            <p>Review every customer interaction end to end: language handling, model classification, confidence-based routing, response generation, and the feedback loop that helps improve retraining data.</p>
            <div class="hero-panel__note">Switch between the interaction history and review editor below to inspect traces and save feedback.</div>
        </div>
    </section>
    """


def render_admin_last_updated_html(timestamp: Optional[datetime] = None) -> str:
    timestamp = timestamp or datetime.now()
    return (
        '<div class="admin-last-updated">'
        f'Last refreshed: {_escape_text(timestamp.strftime("%d %b %Y, %I:%M:%S %p"))}'
        '</div>'
    )


def render_admin_summary_html(summary: dict) -> str:
    metrics = [
        (
            "Total interactions",
            int(summary.get("total_interactions", 0) or 0),
            "All logged requests across the serving pipeline",
        ),
        (
            "Flagged responses",
            int(summary.get("flagged_interactions", 0) or 0),
            "Admin-reviewed cases captured for feedback",
        ),
        (
            "Auto replies",
            int(summary.get("auto_replies", 0) or 0),
            "High-confidence responses sent automatically",
        ),
        (
            "Clarify + escalate",
            int(summary.get("clarifications", 0) or 0) + int(summary.get("escalations", 0) or 0),
            "Cases needing extra detail or human review",
        ),
    ]
    cards = "".join(
        f"""
        <article class="admin-metric">
            <div class="admin-metric__label">{_escape_text(label)}</div>
            <div class="admin-metric__value">{value}</div>
            <div class="admin-metric__subtitle">{_escape_text(subtitle)}</div>
        </article>
        """
        for label, value, subtitle in metrics
    )
    return f'<div class="admin-summary-grid">{cards}</div>'


def render_admin_pipeline_coverage_html() -> str:
    steps = [
        (
            "1",
            "Language handling",
            "See detected language, whether translation was skipped or invoked, and the English text passed into the classifier.",
        ),
        (
            "2",
            "Classification model",
            "Inspect the predicted category, confidence score, and top class probabilities for each interaction.",
        ),
        (
            "3",
            "Decision routing",
            "Review how thresholds drove AUTO_REPLY, CLARIFY, or ESCALATE, plus the reason used to pick the response pattern.",
        ),
        (
            "4",
            "Admin feedback loop",
            "Flag poor responses, add notes, and store suggested labels so future retraining can learn from reviewed cases.",
        ),
    ]
    rows = "".join(
        f"""
        <div class="admin-pipeline-step">
            <div class="admin-step-badge">{_escape_text(index)}</div>
            <div class="admin-step-copy">
                <strong>{_escape_text(title)}</strong>
                <span>{_escape_text(copy)}</span>
            </div>
        </div>
        """
        for index, title, copy in steps
    )
    return f'<div class="admin-pipeline-flow">{rows}</div>'


def render_admin_probability_html(probabilities: dict) -> str:
    items = sorted((probabilities or {}).items(), key=lambda entry: entry[1], reverse=True)[:4]
    if not items:
        return '<div class="admin-subtle">No probability breakdown stored.</div>'

    rows = []
    for label, score in items:
        score = float(score or 0.0)
        rows.append(
            f"""
            <div class="admin-prob-row">
                <strong>{_escape_text(label)}</strong>
                <div class="admin-prob-bar"><div class="admin-prob-fill" style="width:{max(0.0, min(100.0, score * 100)):.1f}%"></div></div>
                <span>{score * 100:.1f}%</span>
            </div>
            """
        )
    return f'<div class="admin-probability-list">{"".join(rows)}</div>'


def render_admin_trace_html(trace: list[dict]) -> str:
    if not trace:
        return '<div class="admin-subtle">No trace data stored for this interaction.</div>'

    entries = []
    for index, entry in enumerate(trace, start=1):
        details = entry.get("details") or {}
        details_html = ""
        if details:
            rows = "".join(
                f"""
                <div class="admin-trace-kv">
                    <span>{_escape_text(_pretty_key(key))}</span>
                    <code>{_escape_text(_format_trace_value(value))}</code>
                </div>
                """
                for key, value in details.items()
            )
            details_html = f'<div class="admin-trace-details">{rows}</div>'

        entries.append(
            f"""
            <article class="admin-trace-entry">
                <h4>{index}. {_escape_text(_pretty_key(_display_text(entry.get('stage'), 'stage')))}</h4>
                <p>{_escape_text(entry.get('summary'))}</p>
                {details_html}
            </article>
            """
        )
    return f'<div class="admin-trace-list">{"".join(entries)}</div>'


def render_admin_feedback_badge_html(feedback: dict) -> str:
    if feedback.get("flagged"):
        return '<span class="admin-pill admin-pill--flagged">Flagged for retraining</span>'
    return '<span class="admin-pill admin-pill--reviewed">Awaiting or cleared</span>'


def render_admin_feedback_summary_html(feedback: dict) -> str:
    updated_at = feedback.get("updated_at")
    rows = [
        ("Current state", "Flagged for retraining" if feedback.get("flagged") else "Awaiting or cleared"),
        ("Suggested category", _display_text(feedback.get("suggested_category"), "Keep predicted category")),
        ("Admin note", _display_text(feedback.get("reason"), "No admin note saved yet.")),
        ("Last update", _format_timestamp(updated_at) if updated_at else "No admin feedback saved yet."),
    ]
    rows_html = "".join(
        f'<div class="admin-trace-kv"><span>{_escape_text(label)}</span><code>{_escape_text(value)}</code></div>'
        for label, value in rows
    )
    return f'<div class="admin-feedback-summary">{rows_html}</div>'


def render_admin_interaction_card_html(item: dict) -> str:
    feedback = item.get("feedback") or {}
    translated_message = item.get("translated_message") or item.get("raw_message") or ""
    translated_note = (
        "Translation changed the text before classification."
        if translated_message != (item.get("raw_message") or "")
        else "Original text was used directly for classification."
    )
    generation = item.get("response_generation") or {}
    route_class = _display_text(item.get("route_decision"), "AUTO_REPLY").lower()
    customer_line = _display_text(item.get("customer_name")) or f"Customer #{_display_text(item.get('customer_id'), 'n/a')}"
    return f"""
    <article class="admin-interaction-card">
        <div class="admin-interaction-head">
            <div>
                <div class="admin-interaction-title">
                    <h3>Interaction #{_escape_text(item.get('id'), 'n/a')}</h3>
                    <span class="admin-pill admin-pill--route admin-pill--{_escape_text(route_class)}">{_escape_text(item.get('route_decision'), 'n/a')}</span>
                    {render_admin_feedback_badge_html(feedback)}
                </div>
                <div class="admin-subtle">{_escape_text(customer_line)} · {_escape_text(_format_timestamp(item.get('timestamp')))}</div>
            </div>
            <div class="admin-subtle">Language: {_escape_text(item.get('detected_language'), 'en')}</div>
        </div>

        <div class="admin-message-grid">
            <section class="admin-box">
                <div class="admin-box-label">Original customer message</div>
                <p class="admin-box-copy">{_escape_text(item.get('raw_message'))}</p>
            </section>
            <section class="admin-box">
                <div class="admin-box-label">Translated message used for inference</div>
                <p class="admin-box-copy">{_escape_text(translated_message)}</p>
                <div class="admin-subtle admin-subtle--spaced">{_escape_text(translated_note)}</div>
            </section>
        </div>

        <section class="admin-box">
            <div class="admin-box-label">Classification and routing</div>
            <div class="admin-analysis-grid">
                <div class="admin-analysis-stat">
                    <div class="admin-subtle">Predicted category</div>
                    <strong>{_escape_text(item.get('predicted_label'), 'n/a')}</strong>
                </div>
                <div class="admin-analysis-stat">
                    <div class="admin-subtle">Confidence score</div>
                    <strong>{float(item.get('confidence_score', 0.0) or 0.0) * 100:.1f}%</strong>
                </div>
                <div class="admin-analysis-stat">
                    <div class="admin-subtle">Response route</div>
                    <strong>{_escape_text(item.get('route_decision'), 'n/a')}</strong>
                </div>
            </div>
            <div class="admin-generation-note">{_escape_text(generation.get('reason'), 'No response-generation note stored for this interaction.')}</div>
            {render_admin_probability_html(item.get('class_probabilities') or {})}
        </section>

        <section class="admin-box admin-box--response">
            <div class="admin-box-label">Automated reply</div>
            <p class="admin-box-copy">{_escape_text(item.get('response'))}</p>
        </section>

        <section class="admin-box">
            <div class="admin-box-label">Admin feedback summary</div>
            {render_admin_feedback_summary_html(feedback)}
        </section>

        <details class="admin-trace">
            <summary>Full pipeline trace</summary>
            {render_admin_trace_html(item.get('pipeline_trace') or [])}
        </details>
    </article>
    """


def render_admin_interactions_html(interactions: list[dict]) -> str:
    if not interactions:
        return '<div class="admin-empty-state">No interactions have been logged yet. Run a customer query from the support tab to populate this dashboard.</div>'
    cards = "".join(render_admin_interaction_card_html(item) for item in interactions)
    return f'<div class="admin-interactions-list">{cards}</div>'


def render_admin_selected_interaction_html(item: Optional[dict]) -> str:
    if not item:
        return '<div class="admin-empty-state admin-empty-state--compact">Select an interaction from the reviewer dropdown to inspect its full details and update the feedback state.</div>'

    customer_line = _display_text(item.get("customer_name")) or f"Customer #{_display_text(item.get('customer_id'), 'n/a')}"
    return (
        '<div class="admin-selection-banner">'
        f'Editing feedback for Interaction #{_escape_text(item.get("id"), "n/a")} · {_escape_text(customer_line)} '
        f'· {_escape_text(_display_text(item.get("route_decision"), "n/a"))}'
        '</div>'
        + render_admin_interaction_card_html(item)
    )


def render_admin_feedback_status_html(
    feedback: Optional[dict],
    *,
    message: str = "",
    tone: str = "info",
) -> str:
    feedback = feedback or {}
    if message:
        return render_gradio_banner(message, tone=tone)

    updated_at = feedback.get("updated_at")
    if updated_at:
        return f'<div class="admin-feedback-status">Last feedback update: {_escape_text(_format_timestamp(updated_at))}</div>'
    return '<div class="admin-feedback-status">No admin feedback saved yet.</div>'


def _admin_interaction_choices(interactions: list[dict]) -> list[tuple[str, str]]:
    choices = []
    for item in interactions:
        customer_line = _display_text(item.get("customer_name")) or f"Customer #{_display_text(item.get('customer_id'), 'n/a')}"
        label = (
            f"Interaction #{item['id']} · "
            f"{customer_line} · "
            f"{_display_text(item.get('route_decision'), 'n/a')} · "
            f"{_format_timestamp(item.get('timestamp'))}"
        )
        choices.append((label, str(item["id"])))
    return choices


def _admin_category_choices() -> list[tuple[str, str]]:
    return [("Keep predicted category", "")] + [(category, category) for category in CATEGORY_OPTIONS]


def _empty_admin_dashboard_outputs(
    gr_module,
    *,
    error_message: str = "",
    feedback_message: str = "",
    feedback_tone: str = "info",
) -> tuple[str, str, str, str, dict, str, str, dict, str, dict]:
    return (
        render_gradio_banner(error_message, tone="error"),
        render_admin_last_updated_html(),
        render_admin_summary_html({}),
        render_admin_interactions_html([]),
        gr_module.update(choices=[], value=None),
        render_admin_selected_interaction_html(None),
        "",
        gr_module.update(choices=_admin_category_choices(), value=""),
        render_admin_feedback_status_html({}, message=feedback_message, tone=feedback_tone),
        {},
    )


def build_admin_dashboard_outputs(
    gr_module,
    selected_interaction_id: Optional[str] = None,
    *,
    error_message: str = "",
    feedback_message: str = "",
    feedback_tone: str = "info",
) -> tuple[str, str, str, str, dict, str, str, dict, str, dict]:
    try:
        dashboard = fetch_admin_dashboard_data(limit=100)
    except Exception as exc:
        return _empty_admin_dashboard_outputs(
            gr_module,
            error_message=error_message or str(exc),
            feedback_message=feedback_message,
            feedback_tone=feedback_tone,
        )

    interactions = dashboard.get("interactions") or []
    interaction_map = {str(item["id"]): item for item in interactions}
    if not interactions:
        return _empty_admin_dashboard_outputs(
            gr_module,
            error_message=error_message,
            feedback_message=feedback_message,
            feedback_tone=feedback_tone,
        )

    selected_id = _display_text(selected_interaction_id)
    if selected_id not in interaction_map:
        selected_id = str(interactions[0]["id"])
    selected_item = interaction_map[selected_id]
    feedback_status_html = render_admin_feedback_status_html(
        selected_item.get("feedback"),
        message=feedback_message,
        tone=feedback_tone,
    )
    return (
        render_gradio_banner(error_message, tone="error"),
        render_admin_last_updated_html(),
        render_admin_summary_html(dashboard.get("summary") or {}),
        render_admin_interactions_html(interactions),
        gr_module.update(choices=_admin_interaction_choices(interactions), value=selected_id),
        render_admin_selected_interaction_html(selected_item),
        _display_text(selected_item.get("feedback", {}).get("reason")),
        gr_module.update(
            choices=_admin_category_choices(),
            value=_display_text(selected_item.get("feedback", {}).get("suggested_category")),
        ),
        feedback_status_html,
        interaction_map,
    )


def initialize_admin_dashboard(gr_module):
    return build_admin_dashboard_outputs(gr_module)


def handle_admin_interaction_change(gr_module, interaction_id: str, interaction_map: dict):
    interaction_id = _display_text(interaction_id)
    item = (interaction_map or {}).get(interaction_id)
    if not item:
        return (
            render_admin_selected_interaction_html(None),
            "",
            gr_module.update(choices=_admin_category_choices(), value=""),
            render_admin_feedback_status_html({}, message="Select an interaction to review.", tone="info"),
            "",
        )

    return (
        render_admin_selected_interaction_html(item),
        _display_text(item.get("feedback", {}).get("reason")),
        gr_module.update(
            choices=_admin_category_choices(),
            value=_display_text(item.get("feedback", {}).get("suggested_category")),
        ),
        render_admin_feedback_status_html(item.get("feedback")),
        "",
    )


def save_admin_feedback_from_gradio(
    gr_module,
    interaction_id: str,
    reason: str,
    suggested_category: str,
    flagged: bool,
):
    interaction_id = _display_text(interaction_id)
    if not interaction_id:
        return build_admin_dashboard_outputs(
            gr_module,
            feedback_message="Select an interaction to review first.",
            feedback_tone="error",
        )

    try:
        save_admin_feedback(
            interaction_id=int(interaction_id),
            flagged=flagged,
            reason=_display_text(reason),
            suggested_category=_display_text(suggested_category),
        )
    except Exception as exc:
        return build_admin_dashboard_outputs(
            gr_module,
            selected_interaction_id=interaction_id,
            feedback_message=str(exc),
            feedback_tone="error",
        )

    return build_admin_dashboard_outputs(
        gr_module,
        selected_interaction_id=interaction_id,
        feedback_message=(
            "Flagged and stored for retraining review."
            if flagged
            else "Flag cleared and review state updated."
        ),
        feedback_tone="success",
    )


def build_gradio_demo():
    """Create a native Gradio implementation of the support and admin UIs."""
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError(
            "Gradio is not installed. Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    app_css = """
    body, .gradio-container {
        background:
            radial-gradient(circle at top left, rgba(20, 184, 166, 0.18), transparent 28%),
            radial-gradient(circle at top right, rgba(14, 116, 144, 0.16), transparent 24%),
            linear-gradient(180deg, #eef6fb 0%, #f8fbfd 100%);
    }
    .gradio-container {
        max-width: 1460px !important;
        padding-top: 24px !important;
    }
    .gradio-container .tabs {
        border: none !important;
        background: transparent !important;
    }
    .gradio-container .tab-nav {
        gap: 10px;
        margin-bottom: 16px;
        border-bottom: none !important;
    }
    .gradio-container .tab-nav button {
        border: 1px solid rgba(184, 206, 222, 0.9) !important;
        border-radius: 16px 16px 0 0 !important;
        background: rgba(255, 255, 255, 0.44) !important;
        color: #607286 !important;
        font-weight: 700 !important;
        padding: 12px 18px !important;
        transition: all 0.2s ease;
    }
    .gradio-container .tab-nav button.selected {
        background: white !important;
        color: #102033 !important;
        box-shadow: 0 14px 30px rgba(15, 23, 42, 0.1);
        border-color: rgba(216, 228, 239, 1) !important;
    }
    footer {
        display: none !important;
    }
    .hero-panel {
        border-radius: 28px;
        padding: 28px 30px;
        margin-bottom: 16px;
        box-shadow: 0 20px 56px rgba(15, 23, 42, 0.14);
        position: relative;
        overflow: hidden;
    }
    .hero-panel h1 {
        margin: 0 0 10px;
        font-size: 34px;
        letter-spacing: -0.03em;
    }
    .hero-panel p {
        margin: 0;
        line-height: 1.7;
        max-width: 920px;
    }
    .hero-panel__note {
        margin-top: 12px;
        font-size: 13px;
        font-weight: 700;
    }
    .hero-panel code {
        background: rgba(255, 255, 255, 0.22);
        padding: 2px 6px;
        border-radius: 8px;
    }
    .hero-panel--support {
        color: #102033;
        background:
            radial-gradient(circle at top left, rgba(20, 184, 166, 0.16), transparent 28%),
            linear-gradient(180deg, rgba(255,255,255,0.98) 0%, rgba(244,250,255,0.98) 100%);
        border: 1px solid rgba(216, 228, 239, 0.9);
    }
    .hero-panel--support .hero-panel__note {
        color: #0f766e;
    }
    .hero-panel--admin {
        color: white;
        background:
            radial-gradient(circle at top left, rgba(14, 165, 233, 0.28), transparent 32%),
            radial-gradient(circle at top right, rgba(245, 158, 11, 0.24), transparent 28%),
            linear-gradient(140deg, #07111f 0%, #0b1730 48%, #10263f 100%);
        border: 1px solid rgba(191, 219, 254, 0.16);
    }
    .hero-panel--admin::before {
        content: "";
        position: absolute;
        inset: 0;
        background:
            linear-gradient(115deg, rgba(14, 165, 233, 0.16), transparent 42%),
            radial-gradient(circle at 88% 18%, rgba(255, 255, 255, 0.08), transparent 24%);
        pointer-events: none;
    }
    .hero-panel--admin .hero-panel__copy {
        position: relative;
        z-index: 1;
    }
    .hero-panel--admin h1 {
        color: #f8fbff !important;
        text-shadow: 0 10px 28px rgba(0, 0, 0, 0.22);
    }
    .hero-panel--admin p {
        color: rgba(226, 232, 240, 0.92) !important;
    }
    .hero-panel--admin .hero-panel__note {
        color: rgba(191, 219, 254, 0.96) !important;
        text-shadow: 0 6px 18px rgba(0, 0, 0, 0.18);
    }
    .surface-card {
        background: rgba(255, 255, 255, 0.96);
        border: 1px solid rgba(255, 255, 255, 0.78) !important;
        border-radius: 24px;
        box-shadow: 0 18px 48px rgba(15, 23, 42, 0.12);
        padding: 22px !important;
    }
    .surface-card-dark {
        background: rgba(255, 255, 255, 0.96);
        border: 1px solid rgba(255, 255, 255, 0.4) !important;
        border-radius: 24px;
        box-shadow: 0 24px 60px rgba(8, 15, 35, 0.18);
        padding: 22px !important;
    }
    .section-heading h2 {
        margin: 0 0 6px;
        font-size: 20px;
    }
    .section-heading p {
        margin: 0;
        color: #607286;
        line-height: 1.6;
    }
    .ui-banner {
        border-radius: 16px;
        padding: 14px 16px;
        margin-bottom: 12px;
        font-weight: 600;
    }
    .ui-banner--error {
        color: #b42318;
        background: #fef3f2;
        border: 1px solid #fecdca;
    }
    .ui-banner--success {
        color: #166534;
        background: #dcfce7;
        border: 1px solid #bbf7d0;
    }
    .ui-banner--info {
        color: #0f3e5d;
        background: #edf7ff;
        border: 1px solid #cde7fb;
    }
    .support-chat-scroll {
        width: 100%;
        min-height: 58vh;
        max-height: 58vh;
        overflow-y: auto;
        padding: 22px;
        border-radius: 22px;
        background: linear-gradient(180deg, rgba(255,255,255,0.9) 0%, rgba(244,250,255,0.96) 100%);
        border: 1px solid #d8e4ef;
        display: flex;
        flex-direction: column;
        gap: 18px;
    }
    .support-empty-state, .admin-empty-state {
        text-align: center;
        color: #607286;
        padding: 28px;
    }
    .admin-empty-state--compact {
        padding: 20px 18px;
        border: 1px dashed #d9e2ec;
        border-radius: 16px;
        background: #f8fafc;
    }
    .support-turn-group {
        display: flex;
        flex-direction: column;
        gap: 10px;
    }
    .support-bubble-row {
        display: flex;
    }
    .support-bubble-row--user {
        justify-content: flex-end;
    }
    .support-bubble-row--assistant {
        justify-content: flex-start;
    }
    .support-bubble {
        max-width: min(78%, 720px);
        border-radius: 20px;
        padding: 14px 16px;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
    }
    .support-bubble--user {
        background: #0f766e;
        color: white;
        border-bottom-right-radius: 8px;
    }
    .support-bubble--assistant {
        background: white;
        border: 1px solid #d8e4ef;
        color: #102033;
        border-bottom-left-radius: 8px;
    }
    .support-bubble-label {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 700;
        opacity: 0.82;
        margin-bottom: 8px;
    }
    .support-bubble-copy {
        white-space: pre-wrap;
        line-height: 1.6;
        margin: 0;
    }
    .support-meta-row {
        margin-top: 12px;
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }
    .support-pill {
        display: inline-flex;
        align-items: center;
        padding: 6px 10px;
        border-radius: 999px;
        background: #f8fbff;
        border: 1px solid #d8e4ef;
        font-size: 12px;
        font-weight: 700;
        color: #115e59;
    }
    .support-kv-grid {
        display: grid;
        gap: 10px;
    }
    .support-kv {
        display: grid;
        grid-template-columns: 120px 1fr;
        gap: 10px;
        align-items: start;
        font-size: 14px;
    }
    .support-kv span {
        color: #607286;
        font-weight: 600;
    }
    .support-order-card {
        border: 1px solid #d8e4ef;
        background: #f8fbff;
        border-radius: 16px;
        padding: 12px 14px;
        margin-top: 10px;
    }
    .support-order-card strong {
        display: block;
        margin-bottom: 6px;
    }
    .support-side-note {
        margin-top: 16px;
        color: #607286;
        font-size: 13px;
        line-height: 1.6;
    }
    .support-status-line {
        display: flex;
        align-items: center;
        min-height: 52px;
        padding: 12px 14px;
        border-radius: 16px;
        background: linear-gradient(180deg, #f8fbff 0%, #eef5fc 100%);
        border: 1px solid #d8e4ef;
        color: #47617d;
        font-size: 13px;
        line-height: 1.5;
        font-weight: 600;
    }
    .support-dropdown label,
    .support-composer label,
    .admin-review-dropdown label,
    .admin-feedback-text label,
    .admin-feedback-dropdown label {
        font-weight: 700 !important;
        color: #516074 !important;
    }
    .support-dropdown, .support-composer, .admin-review-dropdown, .admin-feedback-text, .admin-feedback-dropdown {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
    }
    .support-dropdown input,
    .support-dropdown .wrap,
    .support-composer textarea,
    .admin-review-dropdown input,
    .admin-review-dropdown .wrap,
    .admin-feedback-text textarea,
    .admin-feedback-dropdown input,
    .admin-feedback-dropdown .wrap {
        border-radius: 16px !important;
        border: 1px solid #d8e4ef !important;
        background: white !important;
    }
    .support-composer textarea {
        min-height: 104px !important;
        padding: 16px 18px !important;
        line-height: 1.6 !important;
        box-shadow: inset 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    .support-composer textarea::placeholder {
        color: #7b8ba4 !important;
    }
    .support-suggestion-row {
        gap: 10px !important;
        margin-top: 14px !important;
        flex-wrap: wrap !important;
        align-items: center !important;
    }
    .support-suggestion-btn {
        flex: 0 0 auto !important;
        min-width: 0 !important;
    }
    .support-suggestion-btn button {
        border: 1px solid #d8e4ef !important;
        background: #f8fbff !important;
        color: #115e59 !important;
        border-radius: 999px !important;
        font-weight: 600 !important;
        width: auto !important;
        min-width: 0 !important;
        padding: 10px 16px !important;
        box-shadow: 0 8px 18px rgba(15, 23, 42, 0.04);
    }
    .support-suggestion-btn button:hover {
        background: #ecfdf5 !important;
        border-color: #9fd9cf !important;
    }
    .support-composer-footer {
        gap: 14px !important;
        margin-top: 14px !important;
        align-items: stretch !important;
    }
    .support-status-block {
        flex: 1 1 320px !important;
        min-width: 0 !important;
    }
    .support-send-block {
        flex: 0 0 220px !important;
        min-width: 220px !important;
    }
    .support-send-btn button {
        border: none !important;
        background: linear-gradient(135deg, #0f766e, #115e59) !important;
        color: white !important;
        border-radius: 14px !important;
        font-weight: 700 !important;
        min-height: 48px !important;
        box-shadow: 0 14px 28px rgba(15, 118, 110, 0.22);
        letter-spacing: 0.01em;
    }
    .support-send-btn button:hover {
        filter: brightness(1.02);
    }
    .admin-last-updated {
        color: rgba(15, 23, 42, 0.7);
        font-size: 13px;
        margin: 4px 0 14px;
    }
    .admin-summary-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 14px;
    }
    .admin-metric {
        border-radius: 18px;
        background: linear-gradient(180deg, #f8fbff 0%, #eef6ff 100%);
        border: 1px solid #d9ebfb;
        padding: 16px;
    }
    .admin-metric__label {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #516074;
        margin-bottom: 10px;
    }
    .admin-metric__value {
        font-size: 30px;
        font-weight: 700;
        color: #0369a1;
    }
    .admin-metric__subtitle {
        margin-top: 6px;
        color: #516074;
        font-size: 13px;
        line-height: 1.5;
    }
    .admin-pipeline-flow {
        display: grid;
        gap: 12px;
    }
    .admin-pipeline-step {
        display: grid;
        grid-template-columns: auto 1fr;
        gap: 12px;
        align-items: start;
    }
    .admin-step-badge {
        width: 34px;
        height: 34px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, #0ea5e9, #22d3ee);
        color: white;
        font-weight: 700;
    }
    .admin-step-copy strong {
        display: block;
        margin-bottom: 4px;
    }
    .admin-step-copy span {
        color: #516074;
        line-height: 1.5;
        font-size: 14px;
    }
    .admin-interactions-list {
        display: grid;
        gap: 18px;
    }
    .admin-interaction-card {
        padding: 22px;
        border-radius: 22px;
        border: 1px solid #d9e2ec;
        background: linear-gradient(180deg, rgba(255,255,255,1) 0%, rgba(248,250,252,1) 100%);
    }
    .admin-interaction-head {
        display: flex;
        justify-content: space-between;
        gap: 16px;
        align-items: flex-start;
        margin-bottom: 16px;
    }
    .admin-interaction-title {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        align-items: center;
        margin-bottom: 8px;
    }
    .admin-interaction-title h3 {
        margin: 0;
        font-size: 20px;
    }
    .admin-pill {
        display: inline-flex;
        align-items: center;
        padding: 6px 10px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.02em;
    }
    .admin-pill--auto_reply {
        background: #dbeafe;
        color: #1d4ed8;
    }
    .admin-pill--clarify {
        background: #fef3c7;
        color: #b45309;
    }
    .admin-pill--escalate {
        background: #fee2e2;
        color: #b91c1c;
    }
    .admin-pill--flagged {
        background: #fee2e2;
        color: #b91c1c;
    }
    .admin-pill--reviewed {
        background: #dcfce7;
        color: #166534;
    }
    .admin-message-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 16px;
        margin-bottom: 16px;
    }
    .admin-box {
        background: #f8fafc;
        border: 1px solid #d9e2ec;
        border-radius: 16px;
        padding: 16px;
    }
    .admin-box--response {
        margin: 16px 0;
    }
    .admin-box-label {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #516074;
        margin-bottom: 10px;
    }
    .admin-box-copy {
        margin: 0;
        white-space: pre-wrap;
        line-height: 1.6;
    }
    .admin-analysis-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 12px;
        margin-bottom: 14px;
    }
    .admin-analysis-stat {
        background: white;
        border: 1px solid #d9e2ec;
        border-radius: 14px;
        padding: 12px;
    }
    .admin-analysis-stat strong {
        display: block;
        font-size: 20px;
        margin-top: 4px;
    }
    .admin-generation-note {
        background: #edf7ff;
        border: 1px solid #cde7fb;
        padding: 12px;
        border-radius: 12px;
        color: #0f3e5d;
        line-height: 1.5;
    }
    .admin-probability-list {
        display: grid;
        gap: 8px;
        margin-top: 14px;
    }
    .admin-prob-row {
        display: grid;
        grid-template-columns: 110px 1fr 54px;
        gap: 10px;
        align-items: center;
    }
    .admin-prob-bar {
        height: 8px;
        border-radius: 999px;
        background: #dbeafe;
        overflow: hidden;
    }
    .admin-prob-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #0ea5e9, #22c55e);
    }
    .admin-trace {
        margin-top: 16px;
        border: 1px solid #d9e2ec;
        border-radius: 16px;
        overflow: hidden;
        background: white;
    }
    .admin-trace summary {
        cursor: pointer;
        padding: 16px;
        font-weight: 700;
    }
    .admin-trace-list {
        display: grid;
        gap: 12px;
        padding: 0 16px 16px;
    }
    .admin-trace-entry {
        border: 1px solid #d9e2ec;
        border-radius: 14px;
        padding: 14px;
        background: #f8fafc;
    }
    .admin-trace-entry h4 {
        margin: 0 0 6px;
        font-size: 15px;
    }
    .admin-trace-entry p {
        margin: 0 0 10px;
        color: #516074;
        line-height: 1.5;
    }
    .admin-trace-details {
        display: grid;
        gap: 8px;
    }
    .admin-trace-kv {
        display: grid;
        grid-template-columns: 180px 1fr;
        gap: 10px;
        font-size: 13px;
    }
    .admin-trace-kv span {
        color: #516074;
    }
    .admin-trace-kv code {
        background: white;
        border: 1px solid #e2e8f0;
        padding: 6px 8px;
        border-radius: 10px;
        white-space: pre-wrap;
        word-break: break-word;
    }
    .admin-feedback-summary {
        display: grid;
        gap: 8px;
    }
    .admin-feedback-status {
        margin-top: 8px;
        color: #516074;
        font-size: 13px;
    }
    .admin-subtle {
        color: #516074;
        font-size: 13px;
    }
    .admin-subtle--spaced {
        margin-top: 10px;
    }
    .admin-selection-banner {
        margin-bottom: 12px;
        padding: 12px 14px;
        border-radius: 14px;
        background: #edf7ff;
        border: 1px solid #cde7fb;
        color: #0f3e5d;
        font-weight: 600;
    }
    .admin-refresh-btn button {
        border: 1px solid rgba(191, 219, 254, 0.28) !important;
        background: linear-gradient(135deg, #0ea5e9, #0369a1) !important;
        color: white !important;
        border-radius: 999px !important;
        font-weight: 700 !important;
    }
    .admin-flag-btn button {
        background: linear-gradient(135deg, #ef4444, #dc2626) !important;
        color: white !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
    }
    .admin-clear-btn button {
        background: white !important;
        color: #102033 !important;
        border: 1px solid #d9e2ec !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
    }
    @media (max-width: 1120px) {
        .admin-summary-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }
    @media (max-width: 900px) {
        .admin-message-grid,
        .admin-analysis-grid,
        .admin-summary-grid {
            grid-template-columns: 1fr;
        }
        .admin-interaction-head {
            flex-direction: column;
        }
        .admin-trace-kv,
        .support-kv {
            grid-template-columns: 1fr;
        }
    }
    @media (max-width: 720px) {
        .hero-panel {
            padding: 22px 20px;
        }
        .support-bubble {
            max-width: 92%;
        }
        .support-composer-footer {
            flex-direction: column !important;
        }
        .support-send-block {
            min-width: 100% !important;
            flex-basis: 100% !important;
        }
    }
    """

    with gr.Blocks(
        title="Customer Support Triage",
        css=app_css,
        analytics_enabled=False,
    ) as demo:
        with gr.Tabs():
            with gr.Tab("Support Chat"):
                gr.HTML(render_support_hero_html())
                support_error = gr.HTML()
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5):
                        with gr.Group(elem_classes=["surface-card"]):
                            gr.HTML(
                                """
                                <div class="section-heading">
                                    <h2>Scoped Conversation</h2>
                                    <p>Previous chat turns for the selected customer are shown below.</p>
                                </div>
                                """
                            )
                            customer_dropdown = gr.Dropdown(
                                label="Current User",
                                choices=[],
                                value=None,
                                interactive=True,
                                elem_classes=["support-dropdown"],
                            )
                            chat_history_html = gr.HTML(render_support_chat_history_html([]))
                            query_input = gr.Textbox(
                                show_label=False,
                                placeholder="Ask a question about the selected customer's account, order, refund, payment, or subscription.",
                                lines=4,
                                elem_classes=["support-composer"],
                            )
                            with gr.Row(elem_classes=["support-suggestion-row"]):
                                suggest_order_btn = gr.Button("Where is my order?", elem_classes=["support-suggestion-btn"])
                                suggest_refund_btn = gr.Button("I want a refund", elem_classes=["support-suggestion-btn"])
                                suggest_subscription_btn = gr.Button("Check subscription", elem_classes=["support-suggestion-btn"])
                            with gr.Row(equal_height=True, elem_classes=["support-composer-footer"]):
                                with gr.Column(scale=5, elem_classes=["support-status-block"]):
                                    support_status = gr.HTML(render_support_status_html("Choose a customer to start chatting."))
                                with gr.Column(scale=2, min_width=220, elem_classes=["support-send-block"]):
                                    send_btn = gr.Button("Send message", elem_classes=["support-send-btn"])
                    with gr.Column(scale=3):
                        with gr.Group(elem_classes=["surface-card"]):
                            gr.HTML("<div class=\"section-heading\"><h2>Selected User</h2></div>")
                            user_summary_html = gr.HTML(render_support_user_summary_html(None))
                            recent_orders_html = gr.HTML("")
                            gr.HTML(
                                '<div class="support-side-note">Replies in this chat are limited to the selected user\'s account and order records. Internal JSON context stays hidden from the customer view.</div>'
                            )
            with gr.Tab("Admin UI"):
                gr.HTML(render_admin_hero_html())
                admin_error = gr.HTML()
                admin_last_updated = gr.HTML(render_admin_last_updated_html())
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5):
                        with gr.Group(elem_classes=["surface-card-dark"]):
                            gr.HTML('<div class="section-heading"><h2>Operational Snapshot</h2></div>')
                            admin_summary_html = gr.HTML(render_admin_summary_html({}))
                    with gr.Column(scale=3):
                        with gr.Group(elem_classes=["surface-card-dark"]):
                            gr.HTML('<div class="section-heading"><h2>Pipeline Coverage</h2></div>')
                            gr.HTML(render_admin_pipeline_coverage_html())
                with gr.Group(elem_classes=["surface-card-dark"]):
                    gr.HTML(
                        """
                        <div class="section-heading">
                            <h2>Review Interaction</h2>
                            <p>Select a logged interaction, inspect its full details, and update the admin feedback state.</p>
                        </div>
                        """
                    )
                    admin_interaction_dropdown = gr.Dropdown(
                        label="Interaction",
                        choices=[],
                        value=None,
                        interactive=True,
                        elem_classes=["admin-review-dropdown"],
                    )
                    admin_selected_interaction_html = gr.HTML(render_admin_selected_interaction_html(None))
                    feedback_reason_text = gr.Textbox(
                        label="Admin feedback notes",
                        placeholder="Why is this response inappropriate or useful for retraining?",
                        lines=4,
                        elem_classes=["admin-feedback-text"],
                    )
                    feedback_category_dropdown = gr.Dropdown(
                        label="Suggested category",
                        choices=_admin_category_choices(),
                        value="",
                        interactive=True,
                        elem_classes=["admin-feedback-dropdown"],
                    )
                    with gr.Row():
                        flag_feedback_btn = gr.Button("Flag response", elem_classes=["admin-flag-btn"])
                        clear_feedback_btn = gr.Button("Clear flag", elem_classes=["admin-clear-btn"])
                    admin_feedback_status = gr.HTML(render_admin_feedback_status_html({}))
                    admin_state = gr.State({})
                with gr.Group(elem_classes=["surface-card-dark"]):
                    with gr.Row(equal_height=False):
                        gr.HTML(
                            """
                            <div class="section-heading">
                                <h2>Interaction History</h2>
                                <p>Newest interactions appear first. Review the full trace for each interaction and then use the review editor above to save feedback.</p>
                            </div>
                            """
                        )
                        refresh_admin_btn = gr.Button("Refresh Data", elem_classes=["admin-refresh-btn"])
                    admin_interactions_html = gr.HTML(render_admin_interactions_html([]))

        demo.load(
            fn=lambda: initialize_support_tab(gr),
            outputs=[
                customer_dropdown,
                chat_history_html,
                user_summary_html,
                recent_orders_html,
                support_status,
                support_error,
            ],
        )
        demo.load(
            fn=lambda: initialize_admin_dashboard(gr),
            outputs=[
                admin_error,
                admin_last_updated,
                admin_summary_html,
                admin_interactions_html,
                admin_interaction_dropdown,
                admin_selected_interaction_html,
                feedback_reason_text,
                feedback_category_dropdown,
                admin_feedback_status,
                admin_state,
            ],
        )

        customer_dropdown.change(
            fn=handle_support_customer_change,
            inputs=[customer_dropdown],
            outputs=[
                chat_history_html,
                user_summary_html,
                recent_orders_html,
                support_status,
                support_error,
            ],
        )

        send_btn.click(
            fn=handle_support_message_submit,
            inputs=[customer_dropdown, query_input],
            outputs=[
                query_input,
                chat_history_html,
                user_summary_html,
                recent_orders_html,
                support_status,
                support_error,
            ],
        )
        query_input.submit(
            fn=handle_support_message_submit,
            inputs=[customer_dropdown, query_input],
            outputs=[
                query_input,
                chat_history_html,
                user_summary_html,
                recent_orders_html,
                support_status,
                support_error,
            ],
        )

        suggest_order_btn.click(lambda: "Where is my order?", outputs=[query_input])
        suggest_refund_btn.click(lambda: "I want a refund", outputs=[query_input])
        suggest_subscription_btn.click(lambda: "Can you check my subscription?", outputs=[query_input])

        refresh_admin_btn.click(
            fn=lambda selected_id: build_admin_dashboard_outputs(gr, selected_id),
            inputs=[admin_interaction_dropdown],
            outputs=[
                admin_error,
                admin_last_updated,
                admin_summary_html,
                admin_interactions_html,
                admin_interaction_dropdown,
                admin_selected_interaction_html,
                feedback_reason_text,
                feedback_category_dropdown,
                admin_feedback_status,
                admin_state,
            ],
        )

        admin_interaction_dropdown.change(
            fn=lambda interaction_id, interaction_map: handle_admin_interaction_change(gr, interaction_id, interaction_map),
            inputs=[admin_interaction_dropdown, admin_state],
            outputs=[
                admin_selected_interaction_html,
                feedback_reason_text,
                feedback_category_dropdown,
                admin_feedback_status,
                admin_error,
            ],
        )

        flag_feedback_btn.click(
            fn=lambda interaction_id, reason, category: save_admin_feedback_from_gradio(
                gr,
                interaction_id,
                reason,
                category,
                True,
            ),
            inputs=[admin_interaction_dropdown, feedback_reason_text, feedback_category_dropdown],
            outputs=[
                admin_error,
                admin_last_updated,
                admin_summary_html,
                admin_interactions_html,
                admin_interaction_dropdown,
                admin_selected_interaction_html,
                feedback_reason_text,
                feedback_category_dropdown,
                admin_feedback_status,
                admin_state,
            ],
        )

        clear_feedback_btn.click(
            fn=lambda interaction_id, reason, category: save_admin_feedback_from_gradio(
                gr,
                interaction_id,
                reason,
                category,
                False,
            ),
            inputs=[admin_interaction_dropdown, feedback_reason_text, feedback_category_dropdown],
            outputs=[
                admin_error,
                admin_last_updated,
                admin_summary_html,
                admin_interactions_html,
                admin_interaction_dropdown,
                admin_selected_interaction_html,
                feedback_reason_text,
                feedback_category_dropdown,
                admin_feedback_status,
                admin_state,
            ],
        )

    return demo

def run_gradio_ui(host: str, gradio_port: int) -> None:
    """Launch the native Gradio support and admin frontend."""
    launch_host = "127.0.0.1" if host == "0.0.0.0" else host
    resolved_port = resolve_gradio_port(launch_host, gradio_port)

    print("\n" + "=" * 80)
    print("GRADIO UI STARTING")
    if resolved_port != gradio_port:
        print(f"ℹ Requested port {gradio_port} was busy; using {resolved_port} instead.")
    print(f"🌐 Gradio app: http://localhost:{resolved_port}")
    print("=" * 80 + "\n")

    demo = build_gradio_demo()
    demo.launch(
        server_name=launch_host,
        server_port=resolved_port,
        show_error=True,
    )


if __name__ == "__main__":
    args = parse_runtime_args()

    print("\n🤖 AI SUPPORT TRIAGE\n")
    model, vectorizer = load_trained_pipeline()
    run_gradio_ui(
        host=args.host,
        gradio_port=args.gradio_port,
    )
