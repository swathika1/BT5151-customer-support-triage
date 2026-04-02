import os
import json
import pickle
import re
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


def render_support_ui() -> str:
    """Render the customer-facing chat UI."""
    return """<!DOCTYPE html>
<html>
<head>
    <title>Support Chat</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        :root {
            --bg: #f3f7fb;
            --panel: rgba(255, 255, 255, 0.96);
            --panel-soft: #f8fbff;
            --border: #d8e4ef;
            --text: #102033;
            --muted: #607286;
            --accent: #0f766e;
            --accent-strong: #115e59;
            --accent-soft: #dff7f3;
            --user-bubble: #0f766e;
            --assistant-bubble: #ffffff;
            --danger: #b42318;
            --shadow: 0 18px 48px rgba(15, 23, 42, 0.12);
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            color: var(--text);
            background:
                radial-gradient(circle at top left, rgba(20, 184, 166, 0.18), transparent 28%),
                radial-gradient(circle at top right, rgba(14, 116, 144, 0.16), transparent 24%),
                linear-gradient(180deg, #eef6fb 0%, #f8fbfd 100%);
            min-height: 100vh;
        }
        .page {
            max-width: 1380px;
            margin: 0 auto;
            padding: 28px 18px 32px;
        }
        .topbar {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 20px;
            margin-bottom: 20px;
        }
        .hero h1 {
            margin: 0 0 8px;
            font-size: 34px;
            letter-spacing: -0.03em;
        }
        .hero p {
            margin: 0;
            color: var(--muted);
            line-height: 1.6;
            max-width: 760px;
        }
        .hero-note {
            margin-top: 10px;
            font-size: 13px;
            color: #0f766e;
            font-weight: 600;
        }
        .top-actions {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        .link-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 12px 16px;
            border-radius: 999px;
            border: 1px solid var(--border);
            background: white;
            color: var(--text);
            text-decoration: none;
            font-weight: 700;
        }
        .layout {
            display: grid;
            grid-template-columns: minmax(0, 1.25fr) minmax(320px, 0.75fr);
            gap: 18px;
            align-items: start;
        }
        .panel {
            background: var(--panel);
            border-radius: 24px;
            border: 1px solid rgba(255, 255, 255, 0.7);
            box-shadow: var(--shadow);
        }
        .chat-shell {
            display: flex;
            flex-direction: column;
            min-height: 78vh;
            overflow: hidden;
        }
        .chat-header {
            padding: 22px 22px 16px;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            gap: 16px;
            align-items: end;
            flex-wrap: wrap;
        }
        .chat-header h2 {
            margin: 0 0 4px;
            font-size: 18px;
        }
        .chat-header p {
            margin: 0;
            color: var(--muted);
            font-size: 14px;
        }
        .selector-group {
            display: flex;
            flex-direction: column;
            gap: 8px;
            min-width: 280px;
        }
        .selector-group label {
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--muted);
            font-weight: 700;
        }
        select, textarea {
            width: 100%;
            border: 1px solid var(--border);
            border-radius: 16px;
            font: inherit;
            background: white;
            color: var(--text);
        }
        select {
            padding: 12px 14px;
            font-weight: 600;
        }
        .chat-history {
            flex: 1;
            padding: 22px;
            overflow-y: auto;
            background:
                linear-gradient(180deg, rgba(255,255,255,0.9) 0%, rgba(244,250,255,0.96) 100%);
            display: flex;
            flex-direction: column;
            gap: 18px;
        }
        .turn-group {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .bubble-row {
            display: flex;
        }
        .bubble-row.user {
            justify-content: flex-end;
        }
        .bubble-row.assistant {
            justify-content: flex-start;
        }
        .bubble {
            max-width: min(78%, 720px);
            border-radius: 20px;
            padding: 14px 16px;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
        }
        .bubble.user {
            background: var(--user-bubble);
            color: white;
            border-bottom-right-radius: 8px;
        }
        .bubble.assistant {
            background: var(--assistant-bubble);
            border: 1px solid var(--border);
            border-bottom-left-radius: 8px;
        }
        .bubble-label {
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 700;
            opacity: 0.82;
            margin-bottom: 8px;
        }
        .bubble-copy {
            white-space: pre-wrap;
            line-height: 1.6;
            margin: 0;
        }
        .assistant-meta {
            margin-top: 12px;
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }
        .pill {
            display: inline-flex;
            align-items: center;
            padding: 6px 10px;
            border-radius: 999px;
            background: var(--panel-soft);
            border: 1px solid var(--border);
            font-size: 12px;
            font-weight: 700;
            color: var(--accent-strong);
        }
        .empty-state {
            margin: auto 0;
            text-align: center;
            color: var(--muted);
            padding: 28px;
        }
        .composer {
            padding: 18px 22px 22px;
            border-top: 1px solid var(--border);
            background: white;
        }
        .composer textarea {
            min-height: 92px;
            resize: vertical;
            padding: 14px 16px;
            line-height: 1.5;
        }
        .composer-actions {
            display: flex;
            justify-content: space-between;
            gap: 12px;
            align-items: center;
            margin-top: 12px;
            flex-wrap: wrap;
        }
        .suggestions {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }
        .suggestion-btn {
            border: 1px solid var(--border);
            background: var(--panel-soft);
            color: var(--accent-strong);
            border-radius: 999px;
            padding: 8px 12px;
            font-weight: 600;
            cursor: pointer;
        }
        .send-btn {
            border: none;
            background: linear-gradient(135deg, #0f766e, #115e59);
            color: white;
            border-radius: 14px;
            padding: 13px 18px;
            font-weight: 700;
            cursor: pointer;
        }
        .send-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }
        .status-line {
            color: var(--muted);
            font-size: 13px;
        }
        .side-stack {
            display: grid;
            gap: 18px;
        }
        .side-card {
            padding: 20px;
        }
        .side-card h3 {
            margin: 0 0 14px;
            font-size: 17px;
        }
        .kv-grid {
            display: grid;
            gap: 10px;
        }
        .kv {
            display: grid;
            grid-template-columns: 120px 1fr;
            gap: 10px;
            align-items: start;
            font-size: 14px;
        }
        .kv span {
            color: var(--muted);
            font-weight: 600;
        }
        .recent-orders {
            display: grid;
            gap: 10px;
            margin-top: 16px;
        }
        .order-card {
            border: 1px solid var(--border);
            background: var(--panel-soft);
            border-radius: 16px;
            padding: 12px 14px;
        }
        .order-card strong {
            display: block;
            margin-bottom: 6px;
        }
        .error-banner {
            display: none;
            margin-bottom: 14px;
            padding: 12px 14px;
            border-radius: 16px;
            color: var(--danger);
            background: #fef3f2;
            border: 1px solid #fecdca;
        }
        .error-banner.active {
            display: block;
        }
        @media (max-width: 1080px) {
            .layout {
                grid-template-columns: 1fr;
            }
            .chat-shell {
                min-height: auto;
            }
        }
        @media (max-width: 720px) {
            .topbar, .chat-header, .composer-actions {
                flex-direction: column;
                align-items: stretch;
            }
            .bubble {
                max-width: 92%;
            }
            .kv {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="page">
        <div class="topbar">
            <div class="hero">
                <h1>Customer Support Chat</h1>
                <p>Chat as one selected user at a time. Every reply is restricted to that customer's account and structured e-commerce order data, and the agent now grounds delivery, refund, return, seller, and transporter answers from the `ecommerce_data/` datasets.</p>
                <div class="hero-note">Current context is limited to the selected customer only.</div>
            </div>
            <div class="top-actions">
                <a class="link-pill" href="/admin">Open Admin UI</a>
            </div>
        </div>

        <div class="error-banner" id="errorBanner"></div>

        <div class="layout">
            <section class="panel chat-shell">
                <div class="chat-header">
                    <div>
                        <h2>Scoped Conversation</h2>
                        <p>Previous chat turns for the selected customer are shown below.</p>
                    </div>
                    <div class="selector-group">
                        <label for="currentUser">Current User</label>
                        <select id="currentUser"></select>
                    </div>
                </div>

                <div class="chat-history" id="chatHistory">
                    <div class="empty-state">Loading users and chat history...</div>
                </div>

                <div class="composer">
                    <textarea id="queryInput" placeholder="Ask a question about the selected customer's account, order, refund, payment, or subscription."></textarea>
                    <div class="composer-actions">
                        <div class="suggestions">
                            <button class="suggestion-btn" type="button" onclick="setSuggestion('Where is my order?')">Where is my order?</button>
                            <button class="suggestion-btn" type="button" onclick="setSuggestion('I want a refund')">I want a refund</button>
                            <button class="suggestion-btn" type="button" onclick="setSuggestion('Can you check my subscription?')">Check subscription</button>
                        </div>
                        <div style="display:flex; gap:12px; align-items:center; flex-wrap:wrap;">
                            <span class="status-line" id="composerStatus">Choose a customer to start chatting.</span>
                            <button class="send-btn" type="button" id="sendBtn" onclick="sendMessage()">Send message</button>
                        </div>
                    </div>
                </div>
            </section>

            <aside class="side-stack">
                <section class="panel side-card">
                    <h3>Selected User</h3>
                    <div class="kv-grid" id="userSummary">
                        <div class="kv"><span>Status</span><div>Loading...</div></div>
                    </div>
                    <div class="recent-orders" id="recentOrders"></div>
                    <div style="margin-top:16px; color: var(--muted); font-size: 13px; line-height: 1.6;">
                        Replies in this chat are limited to the selected user's account and order records. Internal JSON context stays hidden from the customer view.
                    </div>
                </section>
            </aside>
        </div>
    </div>

    <script>
        let users = [];
        let currentUserId = '';

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text ?? '';
            return div.innerHTML;
        }

        function showError(message) {
            const banner = document.getElementById('errorBanner');
            if (!message) {
                banner.classList.remove('active');
                banner.textContent = '';
                return;
            }
            banner.classList.add('active');
            banner.textContent = message;
        }

        function setSuggestion(text) {
            document.getElementById('queryInput').value = text;
            document.getElementById('queryInput').focus();
        }

        function renderUserSummary(user) {
            const summary = document.getElementById('userSummary');
            const recentOrders = document.getElementById('recentOrders');

            if (!user) {
                summary.innerHTML = '<div class="kv"><span>Status</span><div>No user selected.</div></div>';
                recentOrders.innerHTML = '';
                return;
            }

            summary.innerHTML = `
                <div class="kv"><span>Customer</span><div>${escapeHtml(user.name)} (#${escapeHtml(user.customer_id)})</div></div>
                <div class="kv"><span>Account</span><div>${escapeHtml(user.account_status || 'n/a')}</div></div>
                <div class="kv"><span>Email</span><div>${escapeHtml(user.registered_email || 'n/a')}</div></div>
                <div class="kv"><span>Phone</span><div>${escapeHtml(user.registered_phone_number || 'n/a')}</div></div>
                <div class="kv"><span>Plan</span><div>${escapeHtml(user.subscription_plan_name || 'n/a')}</div></div>
                <div class="kv"><span>Prime</span><div>${user.prime_subscription_flag ? 'Yes' : 'No'}</div></div>
                <div class="kv"><span>Orders</span><div>${escapeHtml(String(user.order_count || 0))}</div></div>
            `;

            const orders = user.recent_orders || [];
            if (!orders.length) {
                recentOrders.innerHTML = '<div class="order-card">No orders found for this customer.</div>';
                return;
            }

            recentOrders.innerHTML = orders.map((order) => `
                <article class="order-card">
                    <strong>Order #${escapeHtml(order.order_id)}</strong>
                    <div>${escapeHtml(order.order_status || 'n/a')} · ${escapeHtml(order.delivery_status || 'n/a')}</div>
                    <div>${escapeHtml(order.order_currency || '')} ${Number(order.order_amount || 0).toFixed(2)}</div>
                    <div>${escapeHtml(order.order_date || 'n/a')}</div>
                </article>
            `).join('');
        }

        function renderHistory(history) {
            const container = document.getElementById('chatHistory');
            if (!history.length) {
                container.innerHTML = '<div class="empty-state">No previous chat history for this customer yet. Start the conversation below.</div>';
                return;
            }

            container.innerHTML = history.map((turn) => `
                <div class="turn-group">
                    <div class="bubble-row user">
                        <div class="bubble user">
                            <div class="bubble-label">Customer</div>
                            <p class="bubble-copy">${escapeHtml(turn.raw_message || '')}</p>
                        </div>
                    </div>
                    <div class="bubble-row assistant">
                        <div class="bubble assistant">
                            <div class="bubble-label">Assistant</div>
                            <p class="bubble-copy">${escapeHtml(turn.response || '')}</p>
                            <div class="assistant-meta">
                                <span class="pill">${escapeHtml(turn.predicted_label || 'n/a')}</span>
                                <span class="pill">${((turn.confidence_score || 0) * 100).toFixed(1)}% confidence</span>
                                <span class="pill">${escapeHtml(turn.route_decision || 'n/a')}</span>
                                ${turn.needs_more_context ? '<span class="pill">Needs more context</span>' : ''}
                                <span class="pill">${escapeHtml(turn.timestamp || '')}</span>
                            </div>
                        </div>
                    </div>
                </div>
            `).join('');

            container.scrollTop = container.scrollHeight;
        }

        async function loadUsers() {
            const response = await fetch('/api/users');
            const payload = await response.json();
            if (!response.ok || payload.error) {
                throw new Error(payload.error || 'Unable to load users');
            }

            users = payload.users || [];
            const select = document.getElementById('currentUser');
            select.innerHTML = users.map((user) => `
                <option value="${escapeHtml(user.customer_id)}">${escapeHtml(user.label)}</option>
            `).join('');

            if (users.length) {
                currentUserId = users[0].customer_id;
                select.value = currentUserId;
            } else {
                currentUserId = '';
            }
        }

        async function loadChatForSelectedUser() {
            if (!currentUserId) {
                renderUserSummary(null);
                renderHistory([]);
                document.getElementById('composerStatus').textContent = 'No customer records were found.';
                return;
            }

            showError('');
            const response = await fetch(`/api/user-chat?customer_id=${encodeURIComponent(currentUserId)}&limit=100`);
            const payload = await response.json();
            if (!response.ok || payload.error) {
                throw new Error(payload.error || 'Unable to load customer chat');
            }

            renderUserSummary(payload.user);
            renderHistory(payload.history || []);
            document.getElementById('composerStatus').textContent = `Chatting as ${payload.user.name} (#${payload.user.customer_id})`;
        }

        async function sendMessage() {
            const input = document.getElementById('queryInput');
            const sendBtn = document.getElementById('sendBtn');
            const query = input.value.trim();

            if (!currentUserId) {
                showError('Please choose a customer first.');
                return;
            }
            if (!query) {
                showError('Please enter a message before sending.');
                return;
            }

            showError('');
            sendBtn.disabled = true;
            document.getElementById('composerStatus').textContent = 'Sending message...';

            try {
                const response = await fetch('/api/classify', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        customer_id: currentUserId,
                        query
                    })
                });

                const payload = await response.json();
                if (!response.ok || payload.error) {
                    throw new Error(payload.error || 'Unable to send message');
                }

                input.value = '';
                await loadChatForSelectedUser();
            } catch (error) {
                showError(error.message);
                document.getElementById('composerStatus').textContent = 'Unable to send message.';
            } finally {
                sendBtn.disabled = false;
            }
        }

        document.getElementById('currentUser').addEventListener('change', async (event) => {
            currentUserId = event.target.value;
            try {
                await loadChatForSelectedUser();
            } catch (error) {
                showError(error.message);
            }
        });

        document.getElementById('queryInput').addEventListener('keydown', async (event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                await sendMessage();
            }
        });

        window.addEventListener('load', async () => {
            try {
                await loadUsers();
                await loadChatForSelectedUser();
            } catch (error) {
                showError(error.message);
            }
        });
    </script>
</body>
</html>"""


def render_admin_ui() -> str:
    """Render the admin dashboard for reviewing pipeline history."""
    html = """<!DOCTYPE html>
<html>
<head>
    <title>Support Triage Admin</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        :root {
            --bg-deep: #07111f;
            --bg-mid: #10263f;
            --panel: rgba(255, 255, 255, 0.96);
            --panel-alt: #f8fafc;
            --text: #0f172a;
            --muted: #516074;
            --border: #d9e2ec;
            --accent: #0ea5e9;
            --accent-strong: #0369a1;
            --warm: #f59e0b;
            --danger: #dc2626;
            --success: #15803d;
            --shadow: 0 24px 60px rgba(8, 15, 35, 0.18);
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            color: var(--text);
            background:
                radial-gradient(circle at top left, rgba(14, 165, 233, 0.28), transparent 32%),
                radial-gradient(circle at top right, rgba(245, 158, 11, 0.24), transparent 28%),
                linear-gradient(140deg, var(--bg-deep) 0%, #0b1730 48%, var(--bg-mid) 100%);
            min-height: 100vh;
        }
        .page {
            max-width: 1320px;
            margin: 0 auto;
            padding: 32px 20px 56px;
        }
        .topbar {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 20px;
            margin-bottom: 24px;
        }
        .hero h1 {
            margin: 0 0 8px;
            color: white;
            font-size: 36px;
            letter-spacing: -0.03em;
        }
        .hero p {
            margin: 0;
            color: rgba(226, 232, 240, 0.88);
            max-width: 760px;
            line-height: 1.6;
        }
        .hero-meta {
            margin-top: 12px;
            font-size: 13px;
            color: rgba(191, 219, 254, 0.88);
        }
        .top-actions {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }
        .ghost-link, .refresh-btn {
            border: 1px solid rgba(191, 219, 254, 0.28);
            background: rgba(255, 255, 255, 0.08);
            color: white;
            text-decoration: none;
            padding: 12px 16px;
            border-radius: 999px;
            font-weight: 600;
            cursor: pointer;
            backdrop-filter: blur(10px);
        }
        .refresh-btn { font-size: 14px; }
        .dashboard-grid {
            display: grid;
            grid-template-columns: 1.25fr 0.75fr;
            gap: 18px;
            margin-bottom: 18px;
        }
        .card {
            background: var(--panel);
            border-radius: 22px;
            box-shadow: var(--shadow);
            border: 1px solid rgba(255, 255, 255, 0.4);
            padding: 22px;
        }
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 14px;
        }
        .metric {
            border-radius: 18px;
            background: linear-gradient(180deg, #f8fbff 0%, #eef6ff 100%);
            border: 1px solid #d9ebfb;
            padding: 16px;
        }
        .metric-label {
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--muted);
            margin-bottom: 10px;
        }
        .metric-value {
            font-size: 30px;
            font-weight: 700;
            color: var(--accent-strong);
        }
        .metric-subtitle {
            margin-top: 6px;
            color: var(--muted);
            font-size: 13px;
        }
        .card h2 {
            margin: 0 0 14px;
            font-size: 18px;
        }
        .pipeline-flow {
            display: grid;
            gap: 12px;
        }
        .pipeline-step {
            display: grid;
            grid-template-columns: auto 1fr;
            gap: 12px;
            align-items: start;
        }
        .step-badge {
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
        .step-copy strong {
            display: block;
            margin-bottom: 4px;
        }
        .step-copy span {
            color: var(--muted);
            line-height: 1.5;
            font-size: 14px;
        }
        .interactions-card {
            padding: 0;
            overflow: hidden;
        }
        .interactions-header {
            padding: 22px 22px 14px;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            gap: 12px;
            align-items: baseline;
        }
        .interactions-header p {
            margin: 6px 0 0;
            color: var(--muted);
            font-size: 14px;
        }
        .interactions-list {
            display: grid;
            gap: 0;
        }
        .interaction-card {
            padding: 22px;
            border-top: 1px solid var(--border);
            background: linear-gradient(180deg, rgba(255,255,255,1) 0%, rgba(248,250,252,1) 100%);
        }
        .interaction-card:first-child {
            border-top: none;
        }
        .interaction-head {
            display: flex;
            justify-content: space-between;
            gap: 16px;
            align-items: flex-start;
            margin-bottom: 16px;
        }
        .interaction-title {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            align-items: center;
            margin-bottom: 8px;
        }
        .interaction-title h3 {
            margin: 0;
            font-size: 20px;
        }
        .subtle {
            color: var(--muted);
            font-size: 13px;
        }
        .pill {
            display: inline-flex;
            align-items: center;
            padding: 6px 10px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.02em;
        }
        .pill.route-auto_reply { background: #dbeafe; color: #1d4ed8; }
        .pill.route-clarify { background: #fef3c7; color: #b45309; }
        .pill.route-escalate { background: #fee2e2; color: #b91c1c; }
        .pill.flagged { background: #fee2e2; color: #b91c1c; }
        .pill.reviewed { background: #dcfce7; color: #166534; }
        .message-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            margin-bottom: 16px;
        }
        .message-box, .analysis-box, .response-box, .feedback-box {
            background: var(--panel-alt);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 16px;
        }
        .box-label {
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--muted);
            margin-bottom: 10px;
        }
        .box-copy {
            margin: 0;
            white-space: pre-wrap;
            line-height: 1.6;
        }
        .analysis-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
            margin-bottom: 14px;
        }
        .analysis-stat {
            background: white;
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 12px;
        }
        .analysis-stat strong {
            display: block;
            font-size: 20px;
            margin-top: 4px;
        }
        .generation-note {
            background: #edf7ff;
            border: 1px solid #cde7fb;
            padding: 12px;
            border-radius: 12px;
            color: #0f3e5d;
            line-height: 1.5;
        }
        .probability-list {
            display: grid;
            gap: 8px;
            margin-top: 14px;
        }
        .prob-row {
            display: grid;
            grid-template-columns: 90px 1fr 54px;
            gap: 10px;
            align-items: center;
        }
        .prob-bar {
            height: 8px;
            border-radius: 999px;
            background: #dbeafe;
            overflow: hidden;
        }
        .prob-fill {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, #0ea5e9, #22c55e);
        }
        .response-box {
            margin-bottom: 16px;
        }
        details.trace {
            margin-top: 16px;
            border: 1px solid var(--border);
            border-radius: 16px;
            overflow: hidden;
            background: white;
        }
        details.trace summary {
            cursor: pointer;
            padding: 16px;
            font-weight: 700;
            list-style: none;
        }
        details.trace summary::-webkit-details-marker { display: none; }
        .trace-list {
            display: grid;
            gap: 12px;
            padding: 0 16px 16px;
        }
        .trace-entry {
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 14px;
            background: #f8fafc;
        }
        .trace-entry h4 {
            margin: 0 0 6px;
            font-size: 15px;
        }
        .trace-entry p {
            margin: 0 0 10px;
            color: var(--muted);
            line-height: 1.5;
        }
        .trace-details {
            display: grid;
            gap: 8px;
        }
        .trace-kv {
            display: grid;
            grid-template-columns: 180px 1fr;
            gap: 10px;
            font-size: 13px;
        }
        .trace-kv span {
            color: var(--muted);
        }
        .trace-kv code {
            background: white;
            border: 1px solid #e2e8f0;
            padding: 6px 8px;
            border-radius: 10px;
            white-space: pre-wrap;
            word-break: break-word;
        }
        .feedback-grid {
            display: grid;
            grid-template-columns: 1.1fr 0.45fr auto auto;
            gap: 12px;
            align-items: start;
        }
        .feedback-box textarea, .feedback-box select {
            width: 100%;
            padding: 12px;
            border-radius: 12px;
            border: 1px solid var(--border);
            font: inherit;
            background: white;
        }
        .feedback-box textarea {
            min-height: 96px;
            resize: vertical;
        }
        .btn {
            border: none;
            border-radius: 12px;
            padding: 13px 16px;
            font-weight: 700;
            cursor: pointer;
            min-width: 120px;
        }
        .btn-flag {
            background: linear-gradient(135deg, #ef4444, #dc2626);
            color: white;
        }
        .btn-clear {
            background: white;
            border: 1px solid var(--border);
            color: var(--text);
        }
        .feedback-status {
            margin-top: 12px;
            color: var(--muted);
            font-size: 13px;
        }
        .empty-state {
            padding: 36px 22px 42px;
            text-align: center;
            color: var(--muted);
        }
        .error-banner {
            background: rgba(254, 226, 226, 0.96);
            color: #b91c1c;
            border: 1px solid rgba(239, 68, 68, 0.3);
            padding: 14px 16px;
            border-radius: 16px;
            margin-bottom: 18px;
            display: none;
        }
        .error-banner.active {
            display: block;
        }
        @media (max-width: 1120px) {
            .dashboard-grid {
                grid-template-columns: 1fr;
            }
        }
        @media (max-width: 900px) {
            .summary-grid,
            .message-grid,
            .analysis-grid,
            .feedback-grid {
                grid-template-columns: 1fr;
            }
            .topbar,
            .interaction-head {
                flex-direction: column;
            }
            .trace-kv {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="page">
        <div class="topbar">
            <div class="hero">
                <h1>Admin Pipeline Dashboard</h1>
                <p>Review every customer interaction end to end: language handling, model classification, confidence-based routing, response generation, and the feedback loop that helps improve retraining data.</p>
                <div class="hero-meta">Last refreshed: <span id="lastUpdated">never</span></div>
            </div>
            <div class="top-actions">
                <a class="ghost-link" href="/">Back to Support UI</a>
                <button class="refresh-btn" type="button" onclick="loadDashboard()">Refresh Data</button>
            </div>
        </div>

        <div class="error-banner" id="errorBanner"></div>

        <div class="dashboard-grid">
            <section class="card">
                <h2>Operational Snapshot</h2>
                <div class="summary-grid" id="summaryGrid"></div>
            </section>

            <section class="card">
                <h2>Pipeline Coverage</h2>
                <div class="pipeline-flow">
                    <div class="pipeline-step">
                        <div class="step-badge">1</div>
                        <div class="step-copy">
                            <strong>Language handling</strong>
                            <span>See detected language, whether translation was skipped or invoked, and the English text passed into the classifier.</span>
                        </div>
                    </div>
                    <div class="pipeline-step">
                        <div class="step-badge">2</div>
                        <div class="step-copy">
                            <strong>Classification model</strong>
                            <span>Inspect the predicted category, confidence score, and top class probabilities for each interaction.</span>
                        </div>
                    </div>
                    <div class="pipeline-step">
                        <div class="step-badge">3</div>
                        <div class="step-copy">
                            <strong>Decision routing</strong>
                            <span>Review how thresholds drove AUTO_REPLY, CLARIFY, or ESCALATE, plus the reason used to pick the response pattern.</span>
                        </div>
                    </div>
                    <div class="pipeline-step">
                        <div class="step-badge">4</div>
                        <div class="step-copy">
                            <strong>Admin feedback loop</strong>
                            <span>Flag poor responses, add notes, and store suggested labels so future retraining can learn from reviewed cases.</span>
                        </div>
                    </div>
                </div>
            </section>
        </div>

        <section class="card interactions-card">
            <div class="interactions-header">
                <div>
                    <h2>Interaction History</h2>
                    <p>Newest interactions appear first. Expand a record to inspect the full trace and add admin feedback.</p>
                </div>
            </div>
            <div class="interactions-list" id="interactionsList">
                <div class="empty-state">Loading interactions...</div>
            </div>
        </section>
    </div>

    <script>
        const CATEGORY_OPTIONS = __CATEGORY_OPTIONS__;

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text ?? '';
            return div.innerHTML;
        }

        function prettyKey(key) {
            return key.replace(/_/g, ' ').replace(/\\b\\w/g, (char) => char.toUpperCase());
        }

        function formatValue(value) {
            if (value === null || value === undefined || value === '') {
                return 'n/a';
            }
            if (typeof value === 'number') {
                return Number.isInteger(value) ? String(value) : value.toFixed(4);
            }
            if (Array.isArray(value) || typeof value === 'object') {
                return JSON.stringify(value);
            }
            return String(value);
        }

        function routeClass(route) {
            return `route-${String(route || '').toLowerCase()}`;
        }

        function feedbackBadge(feedback) {
            if (feedback && feedback.flagged) {
                return '<span class="pill flagged">Flagged for retraining</span>';
            }
            return '<span class="pill reviewed">Awaiting or cleared</span>';
        }

        function formatTimestamp(value) {
            if (!value) {
                return 'Unknown time';
            }
            const parsed = new Date(value);
            return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleString();
        }

        function renderSummary(summary) {
            const metrics = [
                ['Total interactions', summary.total_interactions, 'All logged requests across the serving pipeline'],
                ['Flagged responses', summary.flagged_interactions, 'Admin-reviewed cases captured for feedback'],
                ['Auto replies', summary.auto_replies, 'High-confidence responses sent automatically'],
                ['Clarify + escalate', summary.clarifications + summary.escalations, 'Cases needing extra detail or human review']
            ];

            document.getElementById('summaryGrid').innerHTML = metrics.map(([label, value, subtitle]) => `
                <article class="metric">
                    <div class="metric-label">${escapeHtml(label)}</div>
                    <div class="metric-value">${escapeHtml(String(value))}</div>
                    <div class="metric-subtitle">${escapeHtml(subtitle)}</div>
                </article>
            `).join('');
        }

        function renderProbabilities(probabilities) {
            const entries = Object.entries(probabilities || {})
                .sort((a, b) => b[1] - a[1])
                .slice(0, 4);

            if (!entries.length) {
                return '<div class="subtle">No probability breakdown stored.</div>';
            }

            return `<div class="probability-list">${entries.map(([label, score]) => `
                <div class="prob-row">
                    <strong>${escapeHtml(label)}</strong>
                    <div class="prob-bar"><div class="prob-fill" style="width:${Math.max(0, Math.min(100, score * 100))}%"></div></div>
                    <span>${(score * 100).toFixed(1)}%</span>
                </div>
            `).join('')}</div>`;
        }

        function renderTrace(trace) {
            if (!Array.isArray(trace) || !trace.length) {
                return '<div class="subtle">No trace data stored for this interaction.</div>';
            }

            return `<div class="trace-list">${trace.map((entry, index) => `
                <article class="trace-entry">
                    <h4>${index + 1}. ${escapeHtml(prettyKey(entry.stage || 'stage'))}</h4>
                    <p>${escapeHtml(entry.summary || '')}</p>
                    ${entry.details && Object.keys(entry.details).length ? `
                        <div class="trace-details">
                            ${Object.entries(entry.details).map(([key, value]) => `
                                <div class="trace-kv">
                                    <span>${escapeHtml(prettyKey(key))}</span>
                                    <code>${escapeHtml(formatValue(value))}</code>
                                </div>
                            `).join('')}
                        </div>
                    ` : ''}
                </article>
            `).join('')}</div>`;
        }

        function renderCategoryOptions(selected) {
            const base = '<option value="">Keep predicted category</option>';
            const options = CATEGORY_OPTIONS.map((category) => {
                const isSelected = category === selected ? 'selected' : '';
                return `<option value="${escapeHtml(category)}" ${isSelected}>${escapeHtml(category)}</option>`;
            }).join('');
            return base + options;
        }

        function renderInteractionCard(item) {
            const feedback = item.feedback || {};
            const generation = item.response_generation || {};
            const translatedDiffers = item.translated_message && item.translated_message !== item.raw_message;

            return `
                <article class="interaction-card">
                    <div class="interaction-head">
                        <div>
                            <div class="interaction-title">
                                <h3>Interaction #${item.id}</h3>
                                <span class="pill ${routeClass(item.route_decision)}">${escapeHtml(item.route_decision)}</span>
                                ${feedbackBadge(feedback)}
                            </div>
                            <div class="subtle">${escapeHtml(formatTimestamp(item.timestamp))}</div>
                        </div>
                        <div class="subtle">Language: ${escapeHtml(item.detected_language || 'en')}</div>
                    </div>

                    <div class="message-grid">
                        <section class="message-box">
                            <div class="box-label">Original customer message</div>
                            <p class="box-copy">${escapeHtml(item.raw_message || '')}</p>
                        </section>
                        <section class="message-box">
                            <div class="box-label">Translated message used for inference</div>
                            <p class="box-copy">${escapeHtml(item.translated_message || item.raw_message || '')}</p>
                            <div class="subtle" style="margin-top:10px;">
                                ${translatedDiffers ? 'Translation changed the text before classification.' : 'Original text was used directly for classification.'}
                            </div>
                        </section>
                    </div>

                    <section class="analysis-box">
                        <div class="box-label">Classification and routing</div>
                        <div class="analysis-grid">
                            <div class="analysis-stat">
                                <div class="subtle">Predicted category</div>
                                <strong>${escapeHtml(item.predicted_label || 'n/a')}</strong>
                            </div>
                            <div class="analysis-stat">
                                <div class="subtle">Confidence score</div>
                                <strong>${((item.confidence_score || 0) * 100).toFixed(1)}%</strong>
                            </div>
                            <div class="analysis-stat">
                                <div class="subtle">Response route</div>
                                <strong>${escapeHtml(item.route_decision || 'n/a')}</strong>
                            </div>
                        </div>
                        <div class="generation-note">
                            ${escapeHtml(generation.reason || 'No response-generation note stored for this interaction.')}
                        </div>
                        ${renderProbabilities(item.class_probabilities)}
                    </section>

                    <section class="response-box">
                        <div class="box-label">Automated reply</div>
                        <p class="box-copy">${escapeHtml(item.response || '')}</p>
                    </section>

                    <section class="feedback-box">
                        <div class="box-label">Admin feedback loop</div>
                        <div class="feedback-grid">
                            <textarea id="feedback-reason-${item.id}" placeholder="Why is this response inappropriate or useful for retraining?">${escapeHtml(feedback.reason || '')}</textarea>
                            <select id="feedback-category-${item.id}">
                                ${renderCategoryOptions(feedback.suggested_category || '')}
                            </select>
                            <button class="btn btn-flag" type="button" onclick="submitFeedback(${item.id}, true)">Flag response</button>
                            <button class="btn btn-clear" type="button" onclick="submitFeedback(${item.id}, false)">Clear flag</button>
                        </div>
                        <div class="feedback-status" id="feedback-status-${item.id}">
                            ${feedback.updated_at ? `Last feedback update: ${escapeHtml(formatTimestamp(feedback.updated_at))}` : 'No admin feedback saved yet.'}
                        </div>
                    </section>

                    <details class="trace">
                        <summary>Full pipeline trace</summary>
                        ${renderTrace(item.pipeline_trace)}
                    </details>
                </article>
            `;
        }

        function renderInteractions(interactions) {
            const container = document.getElementById('interactionsList');
            if (!interactions.length) {
                container.innerHTML = '<div class="empty-state">No interactions have been logged yet. Run a customer query from the support UI to populate this dashboard.</div>';
                return;
            }

            container.innerHTML = interactions.map(renderInteractionCard).join('');
        }

        function showError(message) {
            const banner = document.getElementById('errorBanner');
            if (!message) {
                banner.classList.remove('active');
                banner.textContent = '';
                return;
            }
            banner.classList.add('active');
            banner.textContent = message;
        }

        async function loadDashboard() {
            showError('');

            try {
                const response = await fetch('/api/admin/interactions?limit=100');
                const data = await response.json();

                if (!response.ok || data.error) {
                    throw new Error(data.error || 'Unable to load admin data');
                }

                renderSummary(data.summary || {});
                renderInteractions(data.interactions || []);
                document.getElementById('lastUpdated').textContent = new Date().toLocaleString();
            } catch (error) {
                showError(error.message);
            }
        }

        async function submitFeedback(interactionId, flagged) {
            const reason = document.getElementById(`feedback-reason-${interactionId}`).value.trim();
            const suggestedCategory = document.getElementById(`feedback-category-${interactionId}`).value;
            const status = document.getElementById(`feedback-status-${interactionId}`);

            status.textContent = 'Saving feedback...';

            try {
                const response = await fetch('/api/admin/feedback', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        interaction_id: interactionId,
                        flagged,
                        reason,
                        suggested_category: suggestedCategory
                    })
                });

                const data = await response.json();
                if (!response.ok || data.error) {
                    throw new Error(data.error || 'Unable to save feedback');
                }

                status.textContent = flagged
                    ? 'Flagged and stored for retraining review.'
                    : 'Flag cleared and review state updated.';
                await loadDashboard();
            } catch (error) {
                status.textContent = `Error: ${error.message}`;
            }
        }

        window.addEventListener('load', loadDashboard);
    </script>
</body>
</html>"""
    return html.replace("__CATEGORY_OPTIONS__", json.dumps(CATEGORY_OPTIONS))


class SupportTriageHandler(BaseHTTPRequestHandler):
    """HTTP request handler for support triage UI and API."""
    
    def do_GET(self):
        """Serve HTML interfaces and admin APIs."""
        parsed = urlparse(self.path)

        if parsed.path == "/":
            self.send_html(render_support_ui())
        elif parsed.path == "/admin":
            self.send_html(render_admin_ui())
        elif parsed.path == "/api/users":
            self.send_json({"users": list_chat_users()}, 200)
        elif parsed.path == "/api/user-chat":
            params = parse_qs(parsed.query)
            customer_id = str(params.get("customer_id", [""])[0]).strip()
            if not customer_id:
                self.send_json({"error": "customer_id is required"}, 400)
                return

            try:
                limit = int(params.get("limit", ["100"])[0])
            except (TypeError, ValueError):
                limit = 100

            try:
                self.send_json(fetch_user_chat_payload(customer_id, limit=limit), 200)
            except ValueError as e:
                self.send_json({"error": str(e)}, 400)
        elif parsed.path == "/api/admin/interactions":
            try:
                params = parse_qs(parsed.query)
                limit = int(params.get("limit", ["100"])[0])
            except (TypeError, ValueError):
                limit = 100

            self.send_json(fetch_admin_dashboard_data(limit=limit), 200)
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        """Handle support and admin API requests."""
        parsed = urlparse(self.path)

        if parsed.path == "/api/classify":
            try:
                data = self.read_json_body()
                query = str(data.get("query", "")).strip()
                customer_id = str(data.get("customer_id", "")).strip()
                
                if not query:
                    self.send_json({"error": "Empty query"}, 400)
                    return
                if not customer_id:
                    self.send_json({"error": "customer_id is required"}, 400)
                    return
                
                result, confidence = classify_query(query, customer_id)
                
                if "error" in result:
                    self.send_json(result, 400)
                else:
                    self.send_json(result, 200)
                    
            except ValueError as e:
                self.send_json({"error": str(e)}, 400)
            except json_module.JSONDecodeError as e:
                print(f"[Server] JSON decode error: {str(e)}")
                self.send_json({"error": f"Invalid JSON: {str(e)}"}, 400)
            except Exception as e:
                print(f"[Server] Error processing request: {str(e)}")
                self.send_json({"error": f"Server error: {str(e)}"}, 500)
        elif parsed.path == "/api/admin/feedback":
            try:
                data = self.read_json_body()
                interaction_id = int(data.get("interaction_id", 0))
                if interaction_id <= 0:
                    raise ValueError("A valid interaction_id is required")

                raw_flagged = data.get("flagged", True)
                if isinstance(raw_flagged, str):
                    flagged = raw_flagged.lower() in {"1", "true", "yes", "on"}
                else:
                    flagged = bool(raw_flagged)

                updated_interaction = save_admin_feedback(
                    interaction_id=interaction_id,
                    flagged=flagged,
                    reason=str(data.get("reason", "")),
                    suggested_category=str(data.get("suggested_category", ""))
                )
                self.send_json({"interaction": updated_interaction}, 200)
            except ValueError as e:
                self.send_json({"error": str(e)}, 400)
            except json_module.JSONDecodeError as e:
                self.send_json({"error": f"Invalid JSON: {str(e)}"}, 400)
            except Exception as e:
                print(f"[Server] Admin feedback error: {str(e)}")
                self.send_json({"error": f"Server error: {str(e)}"}, 500)
        else:
            self.send_response(404)
            self.end_headers()

    def read_json_body(self) -> dict:
        """Read and decode a JSON request body."""
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            raise ValueError("No request body")

        body = self.rfile.read(content_length).decode("utf-8")
        if not body:
            raise ValueError("Empty request body")

        return json_module.loads(body)
    
    def send_html(self, html: str, code: int = 200):
        """Send an HTML response."""
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))
    
    def send_json(self, data, code):
        """Send JSON response."""
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(json_module.dumps(data).encode("utf-8"))
    
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
