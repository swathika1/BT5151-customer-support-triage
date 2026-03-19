# ============================================
# ALL PRIORITY 1, 2, 3 PYTHON NODE FUNCTIONS
# Add these to your notebook
# ============================================

import sqlite3
import json
import hashlib
from datetime import datetime
from pathlib import Path
import os

try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    print("⚠️  langdetect not installed. Install with: pip install langdetect")


# ============================================
# PRIORITY 1: SELECT_MODEL_NODE
# ============================================

def select_model_node(state: dict) -> dict:
    """
    STAGE 4: SELECT MODEL (Training Pipeline)
    Implements: select_model.md
    
    Compare all trained models by weighted F1 and select the best one
    with business-focused justification using LLM
    """
    if state.get("mode") != "training":
        return state
    
    print("[select_model_node] Starting...")
    
    # Find best model by weighted F1
    best_model_name = max(
        state["evaluation_results"].keys(),
        key=lambda x: state["evaluation_results"][x]['weighted_f1']
    )
    
    best_weighted_f1 = state["evaluation_results"][best_model_name]['weighted_f1']
    
    # Build comparison summary
    model_comparison = {}
    for model_name, results in state["evaluation_results"].items():
        model_comparison[model_name] = {
            "accuracy": results['accuracy'],
            "weighted_f1": results['weighted_f1'],
            "macro_f1": results['macro_f1'],
            "per_class_f1": results['per_class_f1']
        }
    
    # Generate LLM-based justification
    try:
        client = get_llm_client()
        
        comparison_text = "Model Comparison:\n\n"
        for model_name, results in state["evaluation_results"].items():
            comparison_text += f"{model_name}:\n"
            comparison_text += f"  Weighted F1: {results['weighted_f1']:.4f}\n"
            comparison_text += f"  Macro F1: {results['macro_f1']:.4f}\n"
            comparison_text += f"  Accuracy: {results['accuracy']:.4f}\n\n"
        
        prompt = f"""You are selecting the best ML model for customer support ticket classification.

{comparison_text}

Selected Model: {best_model_name}

In 2-3 business-focused sentences, explain WHY {best_model_name} was selected:
- Focus on weighted F1 as primary metric
- Mention critical category performance (Security, Payment, Account)
- Write for business stakeholders, not ML experts"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an ML engineer justifying model selection to business stakeholders."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=250,
            temperature=0.7
        )
        
        justification = response.choices[0].message.content
    except:
        # Fallback
        justification = f"{best_model_name} achieves the best weighted F1 score of {best_weighted_f1:.4f}, " \
                       f"making it optimal for multi-class customer support triage across all categories."
    
    # Save to state
    state["selected_model_name"] = best_model_name
    state["selected_model"] = state["trained_models"][best_model_name]
    state["selection_justification"] = justification
    state["model_comparison_summary"] = model_comparison
    
    msg = f"[select_model_node] Selected: {best_model_name} (Weighted F1: {best_weighted_f1:.4f})"
    state["messages"].append(msg)
    print(msg)
    print(f"✨ Justification: {justification}\n")
    
    return state


# ============================================
# PRIORITY 1: PERSIST_ARTIFACTS_NODE
# ============================================

def persist_artifacts_node(state: dict) -> dict:
    """
    Save trained model, vectorizer, encoder, and metrics to disk
    Implements: persist_artifacts.md
    """
    if state.get("mode") != "training":
        return state
    
    print("[persist_artifacts_node] Starting...")
    
    # Create artifacts directory
    artifacts_dir = "artifacts"
    Path(artifacts_dir).mkdir(exist_ok=True)
    
    artifact_files = {}
    artifact_checksums = {}
    
    try:
        # 1. Save model
        model_path = f"{artifacts_dir}/model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(state["selected_model"], f)
        artifact_files["model"] = model_path
        artifact_checksums["model"] = hashlib.sha256(open(model_path, "rb").read()).hexdigest()[:8]
        print(f"  ✅ Model saved: {model_path}")
        
        # 2. Save vectorizer
        vectorizer_path = f"{artifacts_dir}/tfidf_vectorizer.pkl"
        with open(vectorizer_path, "wb") as f:
            pickle.dump(state["tfidf_vectorizer"], f)
        artifact_files["vectorizer"] = vectorizer_path
        artifact_checksums["vectorizer"] = hashlib.sha256(open(vectorizer_path, "rb").read()).hexdigest()[:8]
        print(f"  ✅ Vectorizer saved: {vectorizer_path}")
        
        # 3. Save label encoder
        encoder_path = f"{artifacts_dir}/label_encoder.pkl"
        with open(encoder_path, "wb") as f:
            pickle.dump(state["label_encoder"], f)
        artifact_files["encoder"] = encoder_path
        artifact_checksums["encoder"] = hashlib.sha256(open(encoder_path, "rb").read()).hexdigest()[:8]
        print(f"  ✅ Label encoder saved: {encoder_path}")
        
        # 4. Save evaluation results
        results_path = f"{artifacts_dir}/evaluation_results.json"
        with open(results_path, "w") as f:
            json.dump(state["evaluation_results"], f, indent=2)
        artifact_files["evaluation_results"] = results_path
        print(f"  ✅ Evaluation results saved: {results_path}")
        
        # 5. Save model comparison
        comparison_path = f"{artifacts_dir}/model_comparison.json"
        with open(comparison_path, "w") as f:
            json.dump(state["model_comparison_summary"], f, indent=2)
        artifact_files["model_comparison"] = comparison_path
        print(f"  ✅ Model comparison saved: {comparison_path}")
        
        # 6. Save selection justification
        justification_path = f"{artifacts_dir}/selection_justification.txt"
        with open(justification_path, "w") as f:
            f.write(f"Selected Model: {state['selected_model_name']}\n\n")
            f.write(f"Justification:\n{state['selection_justification']}\n")
        artifact_files["selection_justification"] = justification_path
        print(f"  ✅ Selection justification saved: {justification_path}")
        
        # 7. Save metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "selected_model": state["selected_model_name"],
            "training_samples": int(len(state["y_train"])),
            "validation_samples": int(len(state["y_val"])),
            "test_samples": int(len(state["y_test"])),
            "tfidf_features": int(state["train_texts"].shape[1]),
            "n_classes": int(len(state["label_encoder"].classes_)),
            "classes": list(state["label_encoder"].classes_),
            "model_metrics": {
                state["selected_model_name"]: state["evaluation_results"][state["selected_model_name"]]
            }
        }
        metadata_path = f"{artifacts_dir}/METADATA.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        artifact_files["metadata"] = metadata_path
        print(f"  ✅ Metadata saved: {metadata_path}")
        
        # Save to state
        state["artifacts_path"] = artifacts_dir
        state["artifact_files"] = artifact_files
        state["artifact_checksums"] = artifact_checksums
        
        msg = f"[persist_artifacts_node] Saved {len(artifact_files)} artifacts to {artifacts_dir}/. " \
              f"Selected model: {state['selected_model_name']}"
        state["messages"].append(msg)
        print(f"\n✅ {msg}")
        
    except Exception as e:
        print(f"❌ Error persisting artifacts: {str(e)}")
        state["messages"].append(f"[persist_artifacts_node] ERROR: {str(e)}")
    
    return state


# ============================================
# PRIORITY 2: DETECT_LANGUAGE_NODE
# ============================================

def detect_language_node(state: dict) -> dict:
    """
    Detect the language of incoming customer message
    Implements: detect_language.md
    """
    if not state.get("customer_message"):
        return state
    
    print("[detect_language_node] Starting...")
    
    message = state["customer_message"]
    
    if not LANGDETECT_AVAILABLE:
        # Fallback to English
        detected_language = "en"
        confidence = 1.0
        print("  ⚠️  langdetect not available; defaulting to English")
    else:
        try:
            # Skip very short messages
            if len(message.strip()) < 5:
                detected_language = "en"
                confidence = 0.5
            else:
                detected_language = detect(message)
                # langdetect doesn't return confidence, approximate it
                confidence = 0.85
        except:
            # Default to English if detection fails
            detected_language = "en"
            confidence = 0.5
    
    state["detected_language"] = detected_language
    state["language_confidence"] = confidence
    state["requires_translation"] = (detected_language != "en")
    state["original_message"] = message
    
    msg = f"[detect_language_node] Language: {detected_language} (confidence: {confidence:.2f}). " \
          f"Requires translation: {state['requires_translation']}"
    state["messages"].append(msg)
    print(msg)
    
    return state


# ============================================
# PRIORITY 2: TRANSLATE_TO_ENGLISH_NODE
# ============================================

def translate_to_english_node(state: dict) -> dict:
    """
    Translate non-English messages to English using GPT-4o-mini
    Implements: translate_to_english.md
    """
    if state.get("detected_language") == "en":
        # No translation needed
        state["translated_message"] = state["customer_message"]
        state["translation_performed"] = False
        return state
    
    if not state.get("requires_translation"):
        state["translated_message"] = state["customer_message"]
        state["translation_performed"] = False
        return state
    
    print("[translate_to_english_node] Starting translation...")
    
    try:
        client = get_llm_client()
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional translator. Translate the following customer support "
                              "message to English. Preserve the meaning, tone, and intent. Keep the translation concise."
                },
                {"role": "user", "content": state["customer_message"]}
            ],
            max_tokens=500,
            temperature=0.3
        )
        
        translated = response.choices[0].message.content
        state["translated_message"] = translated
        state["translation_performed"] = True
        state["translation_source_language"] = state["detected_language"]
        state["translation_confidence"] = 0.85
        
        msg = f"[translate_to_english_node] Translated from {state['detected_language']} to en. " \
              f"Original: '{state['customer_message'][:50]}...' → Translated: '{translated[:50]}...'"
        state["messages"].append(msg)
        print(msg)
        
    except Exception as e:
        print(f"  ⚠️  Translation failed: {str(e)}. Using original message.")
        state["translated_message"] = state["customer_message"]
        state["translation_performed"] = False
        msg = f"[translate_to_english_node] Translation failed; using original message"
        state["messages"].append(msg)
    
    return state


# ============================================
# PRIORITY 2: CONFIDENCE_ROUTER_NODE
# ============================================

def confidence_router_node(state: dict) -> dict:
    """
    Route ticket based on model confidence score
    Implements: confidence_router.md
    """
    if not state.get("predicted_label"):
        return state
    
    print("[confidence_router_node] Starting routing logic...")
    
    confidence = state["confidence_score"]
    category = state["predicted_label"]
    
    # Determine confidence level
    if confidence >= 0.80:
        confidence_level = "High"
    elif confidence >= 0.60:
        confidence_level = "Medium"
    else:
        confidence_level = "Low"
    
    state["confidence_threshold_met"] = (confidence >= 0.80)
    state["confidence_level"] = confidence_level
    
    # Category-specific overrides
    critical_categories = ["Security Concern"]
    high_risk_categories = ["Account Suspension", "Payment Problem", "Refund Request"]
    
    if category in critical_categories:
        routing_decision = "escalate_to_supervisor"
        requires_review = True
        priority = "CRITICAL"
        rationale = f"{category} category always escalates regardless of confidence"
    elif category in high_risk_categories and confidence < 0.95:
        routing_decision = "escalate_to_supervisor"
        requires_review = True
        priority = "HIGH"
        rationale = f"{category} requires high confidence (≥0.95); current: {confidence:.2f}"
    elif confidence >= 0.80:
        routing_decision = "auto_approved"
        requires_review = False
        priority = "AUTO"
        rationale = f"High confidence ({confidence:.2f} >= 0.80)"
    elif confidence >= 0.60:
        routing_decision = "pending_review"
        requires_review = True
        priority = "MEDIUM"
        rationale = f"Medium confidence ({confidence:.2f}, 0.60-0.80); flag for human review"
    else:
        routing_decision = "escalate_to_supervisor"
        requires_review = True
        priority = "HIGH"
        rationale = f"Low confidence ({confidence:.2f} < 0.60); escalate to supervisor"
    
    state["routing_decision"] = routing_decision
    state["requires_human_review"] = requires_review
    state["routing_rationale"] = rationale
    state["supervisor_queue_priority"] = priority
    
    msg = f"[confidence_router_node] Predicted: {category} ({confidence:.2f}). " \
          f"Routing: {routing_decision}. Rationale: {rationale}"
    state["messages"].append(msg)
    print(msg)
    
    return state


# ============================================
# PRIORITY 2: DRAFT_RESPONSE_NODE
# ============================================

def draft_response_node(state: dict) -> dict:
    """
    Draft customer response based on predicted category and routing
    Implements: draft_response.md (downstream skill)
    """
    if not state.get("predicted_label"):
        return state
    
    print("[draft_response_node] Drafting response...")
    
    category = state["predicted_label"]
    requires_review = state.get("requires_human_review", False)
    
    # Response templates
    templates = {
        'Account Suspension': {
            'routing': 'Account Specialists',
            'response': "Hi! We understand your account suspension. Our specialist team will review your case "
                       "and contact you within 24 hours."
        },
        'Bug Report': {
            'routing': 'Product/Engineering',
            'response': "Thanks for reporting this bug! We've logged it and our engineering team will investigate."
        },
        'Data Sync Issue': {
            'routing': 'Technical Support',
            'response': "We see you're experiencing sync issues. Please ensure you're using the latest app version. "
                       "Our team will help troubleshoot."
        },
        'Feature Request': {
            'routing': 'Product Feedback',
            'response': "Thanks for the suggestion! Your request has been logged in our product roadmap."
        },
        'Login Issue': {
            'routing': 'Auto-Reply: Password Reset',
            'response': "We've sent a password reset link to your registered email. Click it to regain access."
        },
        'Payment Problem': {
            'routing': 'Billing Department',
            'response': "We've received your report. Our billing team will review your account and reach out within 4 hours."
        },
        'Performance Issue': {
            'routing': 'Technical Support',
            'response': "Sorry to hear about the slowness! Try clearing your cache. Our team can help troubleshoot."
        },
        'Refund Request': {
            'routing': 'Billing Department',
            'response': "We've recorded your refund request. Our billing specialist will contact you within 1 business day."
        },
        'Security Concern': {
            'routing': 'Security Team (URGENT)',
            'response': "Your security concern is our top priority. We're escalating to our security team immediately. "
                       "Do not share passwords. We'll contact you within 1 hour."
        },
        'Subscription Cancellation': {
            'routing': 'Retention Specialist',
            'response': "We're sorry to see you go. Before you cancel, a retention specialist will reach out."
        }
    }
    
    template = templates.get(category, {
        'routing': 'Support Manager',
        'response': 'Our support team will help you shortly.'
    })
    
    # Adjust based on review requirement
    if requires_review:
        routing_recommendation = f"REVIEW NEEDED: {template['routing']}"
        response = f"[Flagged for Review] {template['response']}"
    else:
        routing_recommendation = template['routing']
        response = template['response']
    
    state["routing_recommendation"] = routing_recommendation
    state["response_template"] = response
    
    msg = f"[draft_response_node] Drafted response for {category}. Routing: {routing_recommendation}"
    state["messages"].append(msg)
    print(msg)
    
    return state


# ============================================
# PRIORITY 3: LOG_INTERACTION_NODE
# ============================================

def log_interaction_node(state: dict) -> dict:
    """
    Log ticket classification interaction to SQLite database
    Implements: log_interaction.md
    """
    print("[log_interaction_node] Logging interaction...")
    
    try:
        # Ensure database exists
        db_path = "data/interactions.db"
        Path("data").mkdir(exist_ok=True)
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                original_message TEXT,
                detected_language TEXT,
                translated_message TEXT,
                predicted_category TEXT,
                confidence_score REAL,
                class_probabilities_json TEXT,
                routing_decision TEXT,
                routing_rationale TEXT,
                requires_human_review BOOLEAN,
                response_template TEXT,
                model_name TEXT,
                agent_feedback TEXT,
                is_correct BOOLEAN
            )
        """)
        
        # Prepare data
        conversation_id = state.get("conversation_id", f"conv_{int(datetime.now().timestamp())}")
        timestamp = datetime.now().isoformat()
        original_message = state.get("customer_message", "")
        detected_language = state.get("detected_language", "en")
        translated_message = state.get("translated_message", "")
        predicted_category = state.get("predicted_label", "")
        confidence_score = state.get("confidence_score", 0.0)
        class_probs_json = json.dumps(state.get("class_probabilities", {}))
        routing_decision = state.get("routing_decision", "unknown")
        routing_rationale = state.get("routing_rationale", "")
        requires_human_review = state.get("requires_human_review", False)
        response_template = state.get("response_template", "")
        model_name = state.get("selected_model_name", "unknown")
        
        # Insert
        cursor.execute("""
            INSERT INTO interactions (
                conversation_id, timestamp, original_message, detected_language,
                translated_message, predicted_category, confidence_score,
                class_probabilities_json, routing_decision, routing_rationale,
                requires_human_review, response_template, model_name
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            conversation_id, timestamp, original_message, detected_language,
            translated_message, predicted_category, confidence_score,
            class_probs_json, routing_decision, routing_rationale,
            requires_human_review, response_template, model_name
        ))
        
        interaction_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        state["interaction_id"] = interaction_id
        state["conversation_id"] = conversation_id
        
        msg = f"[log_interaction_node] Logged interaction_id={interaction_id} to {db_path}"
        state["messages"].append(msg)
        print(msg)
        
    except Exception as e:
        print(f"  ❌ Error logging interaction: {str(e)}")
        state["messages"].append(f"[log_interaction_node] ERROR: {str(e)}")
    
    return state


# ============================================
# HELPER FUNCTION: GET LLM CLIENT
# ============================================

def get_llm_client():
    """Get OpenAI client"""
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")
    return OpenAI(api_key=api_key)


print("✅ All Priority 1, 2, 3 node functions loaded!")
print("   - select_model_node")
print("   - persist_artifacts_node")
print("   - detect_language_node")
print("   - translate_to_english_node")
print("   - confidence_router_node")
print("   - draft_response_node")
print("   - log_interaction_node")
