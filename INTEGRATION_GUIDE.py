#!/usr/bin/env python
"""
INTEGRATION GUIDE: Customer Support Classification System
Complete step-by-step instructions for integrating all components into the Jupyter notebook
"""

INTEGRATION_GUIDE = """
═════════════════════════════════════════════════════════════════════════════════════
  CUSTOMER SUPPORT TICKET CLASSIFICATION SYSTEM
  Complete Integration Guide for customer_support_pipeline.ipynb
═════════════════════════════════════════════════════════════════════════════════════

## COMPONENTS CREATED
═════════════════════════════════════════════════════════════════════════════════════

✅ SPECIFICATION FILES (SKILL.md):

  1. preprocess_data.md        (Given by teammate Swathika)
  2. train_models.md            (Given by teammate Swathika)
  3. evaluate_models.md         (Given by teammate Swathika)
  4. run_inference.md           (Given by teammate Swathika)
  5. select_model.md            (Created for training pipeline stage 4)
  6. persist_artifacts.md       (Created for artifact persistence stage 4B)
  7. detect_language.md         (Created for serving pipeline stage 5)
  8. translate_to_english.md    (Created for serving pipeline stage 5B)
  9. confidence_router.md       (Created for serving pipeline stage 6)
  10. draft_response.md         (Created for serving pipeline stage 7)
  11. log_interaction.md        (Created for serving pipeline stage 8)

✅ PYTHON IMPLEMENTATION FILES:

  1. all_priority_nodes.py      (7 complete node functions)
     - select_model_node()
     - persist_artifacts_node()
     - detect_language_node()
     - translate_to_english_node()
     - confidence_router_node()
     - draft_response_node()
     - log_interaction_node()

  2. setup_database.py          (SQLite initialization)
     - Creates 4 tables: interactions, feedback, model_versions, performance_metrics
     - Creates indexes for fast querying

  3. test_system.py             (Validation test suite)
     - Tests imports, database, artifacts, nodes, state, OpenAI config

  4. customer_support_pipeline.ipynb (Main notebook - needs integration)
     - Currently has stages 1-6
     - Needs stages 4B-8 added


## INTEGRATION STEPS
═════════════════════════════════════════════════════════════════════════════════════

### STEP 1: PRE-INTEGRATION VALIDATION (Run in Terminal)
────────────────────────────────────────────────────────────────────────────────────

Before integrating into the notebook, validate that all components are ready:

    cd c:\\Users\\Lenovo\\Downloads\\5151-share\\General
    
    # Initialize the database
    python setup_database.py
    
    # Run the test suite
    python test_system.py

Expected output:
   - Database initialized at data/interactions.db (4 tables created)
   - 6/6 tests should pass (or 5/6 if database not yet created by setup)


### STEP 2: ADD DEPENDENCIES TO NOTEBOOK
────────────────────────────────────────────────────────────────────────────────────

In the first notebook cell (install packages), ADD THESE PACKAGES:

    # Add to existing pip install cell
    !pip install langdetect
    
    # Verify sqlite3 is available (should be standard library)


### STEP 3: IMPORT NODE FUNCTIONS IN NOTEBOOK
────────────────────────────────────────────────────────────────────────────────────

After the existing imports section, ADD THIS IMPORT BLOCK:

    # Import all node functions
    import sys
    sys.path.insert(0, '/path/to/all_priority_nodes.py')
    
    from all_priority_nodes import (
        select_model_node,
        persist_artifacts_node,
        detect_language_node,
        translate_to_english_node,
        confidence_router_node,
        draft_response_node,
        log_interaction_node
    )

Replace '/path/to/' with the actual directory path.


### STEP 4: INTEGRATE TRAINING GRAPH (PHASE 1)
────────────────────────────────────────────────────────────────────────────────────

ADD these nodes to the existing training graph:

    # ADD AFTER the select_model_agent function definition
    # Stage 4: Select best model (UPDATED to use node)
    def select_model_agent(state: dict) -> dict:
        return select_model_node(state)
    
    # ADD NEW: Stage 4B: Persist artifacts
    def persist_artifacts_agent(state: dict) -> dict:
        return persist_artifacts_node(state)
    
    # MODIFY the graph building section to add new nodes:
    workflow.add_node("select_model", select_model_agent)
    workflow.add_node("persist_artifacts", persist_artifacts_agent)
    
    # ADD edge from select_model to persist_artifacts
    workflow.add_edge("select_model", "persist_artifacts")
    
    # MODIFY: Change final edge to point to persist_artifacts instead of END
    workflow.add_edge("persist_artifacts", END)

Expected flow after this step:
   START → preprocess → train → evaluate → select_model → persist_artifacts → END


### STEP 5: BUILD AND TEST TRAINING PIPELINE (PHASE 1)
────────────────────────────────────────────────────────────────────────────────────

IN PHASE 1 EXECUTION CELL (cell 13), the training should now:

    1. Preprocess: TF-IDF vectorization, 70/15/15 split
    2. Train: LogisticRegression, LinearSVC, MultinomialNB
    3. Evaluate: F1/Precision/Recall metrics + GPT-4o-mini interpretation
    4. Select: Best model by weighted F1 + LLM justification
    5. Persist: Save model.pkl, vectorizer.pkl, encoder.pkl, evaluation_results.json

Check artifacts/ directory should have 4 files after execution.


### STEP 6: BUILD SERVING PIPELINE (PHASE 2)
────────────────────────────────────────────────────────────────────────────────────

ADD this new section AFTER the training pipeline completion:

    # ═══════════════════════════════════════════════════════════════════════════════
    # PHASE 2: SERVING PIPELINE (Process incoming tickets in production)
    # ═══════════════════════════════════════════════════════════════════════════════
    
    from langgraph.graph import StateGraph, START, END
    
    # Stage 5: Detect language
    def detect_language_agent(state: dict) -> dict:
        return detect_language_node(state)
    
    # Stage 5B: Translate if needed
    def translate_to_english_agent(state: dict) -> dict:
        if state.get("requires_translation", False):
            return translate_to_english_node(state)
        return state
    
    # Stage 6: Run inference (already exists - reuse)
    def run_inference_agent(state: dict) -> dict:
        # Load the trained model from artifacts
        import pickle
        model = pickle.load(open("artifacts/model.pkl", "rb"))
        vectorizer = pickle.load(open("artifacts/vectorizer.pkl", "rb"))
        
        message = state.get("translated_message") or state.get("customer_message")
        tfidf = vectorizer.transform([message])
        probabilities = model.predict_proba(tfidf)[0]
        
        state["class_probabilities"] = {
            model.classes_[i]: float(probabilities[i])
            for i in range(len(model.classes_))
        }
        state["predicted_category"] = model.classes_[np.argmax(probabilities)]
        state["confidence_score"] = float(np.max(probabilities))
        
        return state
    
    # Stage 7: Confidence router
    def confidence_router_agent(state: dict) -> dict:
        return confidence_router_node(state)
    
    # Stage 8: Draft response
    def draft_response_agent(state: dict) -> dict:
        return draft_response_node(state)
    
    # Stage 9: Log interaction
    def log_interaction_agent(state: dict) -> dict:
        return log_interaction_node(state)
    
    # Build serving workflow
    serving_workflow = StateGraph(PipelineState)
    
    serving_workflow.add_node("detect_language", detect_language_agent)
    serving_workflow.add_node("translate_english", translate_to_english_agent)
    serving_workflow.add_node("inference", run_inference_agent)
    serving_workflow.add_node("router", confidence_router_agent)
    serving_workflow.add_node("draft_response", draft_response_agent)
    serving_workflow.add_node("log_interaction", log_interaction_agent)
    
    # Connect nodes in sequence
    serving_workflow.set_entry_point("detect_language")
    serving_workflow.add_edge("detect_language", "translate_english")
    serving_workflow.add_edge("translate_english", "inference")
    serving_workflow.add_edge("inference", "router")
    serving_workflow.add_edge("router", "draft_response")
    serving_workflow.add_edge("draft_response", "log_interaction")
    serving_workflow.add_edge("log_interaction", END)
    
    # Compile serving pipeline
    serving_app = serving_workflow.compile()
    print("✅ Serving pipeline compiled successfully")


### STEP 7: TEST SERVING PIPELINE (PHASE 2)
────────────────────────────────────────────────────────────────────────────────────

ADD a new cell to test the serving pipeline:

    # Test serving pipeline with diverse examples
    test_tickets = [
        {
            "customer_message": "I can't login to my account",
            "conversation_id": "test_001"
        },
        {
            "customer_message": "Je ne peux pas accéder à mon compte",  # French
            "conversation_id": "test_002"
        },
        {
            "customer_message": "This is a security issue with your payment system",
            "conversation_id": "test_003"
        }
    ]
    
    print("Testing Serving Pipeline:")
    print("=" * 100)
    
    for i, ticket in enumerate(test_tickets, 1):
        print(f"\n📋 Test {i}: {ticket['customer_message'][:50]}...")
        
        # Create initial state
        state = create_initial_state()
        state.update(ticket)
        
        # Run serving pipeline
        result = serving_app.invoke(state)
        
        print(f"   Language: {result.get('detected_language', 'unknown')}")
        if result.get('translated_message'):
            print(f"   Translated: {result['translated_message']}")
        print(f"   Category: {result.get('predicted_category', 'unknown')}")
        print(f"   Confidence: {result.get('confidence_score', 0):.3f}")
        print(f"   Routing: {result.get('routing_decision', 'unknown')}")
        print(f"   Response: {result.get('response_template', 'N/A')[:50]}...")
        print(f"   DB ID: {result.get('interaction_id', 'N/A')}")


### STEP 8: BUILD GRADIO UI (OPTIONAL)
────────────────────────────────────────────────────────────────────────────────────

ADD a new cell for the web interface:

    import gradio as gr
    
    def classify_ticket(message: str, conversation_id: str = None):
        \"\"\"Classify a support ticket and return results\"\"\"
        if not conversation_id:
            import uuid
            conversation_id = str(uuid.uuid4())[:8]
        
        # Create state and run serving pipeline
        state = create_initial_state()
        state["customer_message"] = message
        state["conversation_id"] = conversation_id
        
        result = serving_app.invoke(state)
        
        # Format output for UI
        return {
            "Category": result.get("predicted_category", "Unknown"),
            "Confidence": f"{result.get('confidence_score', 0):.1%}",
            "Routing": result.get("routing_decision", "Unknown").replace("_", " ").title(),
            "Response": result.get("response_template", "No response generated"),
            "Database ID": result.get("interaction_id", "N/A")
        }
    
    # Create Gradio interface
    interface = gr.Interface(
        fn=classify_ticket,
        inputs=[
            gr.Textbox(label="Customer Message", lines=3, max_length=500),
            gr.Textbox(label="Conversation ID (optional)", max_length=50)
        ],
        outputs=gr.JSON(label="Classification Result"),
        title="Customer Support Ticket Classifier",
        description="Classify support tickets and route to appropriate teams",
        theme="soft",
        examples=[
            ["I forgot my password and can't log in", "test_001"],
            ["Your website has a security vulnerability", "test_002"],
            ["I was charged twice for my subscription", "test_003"]
        ]
    )
    
    # Launch interface
    interface.launch(share=False)  # Set share=True for shareable link


### STEP 9: VISUALIZATION AND ANALYTICS (OPTIONAL)
────────────────────────────────────────────────────────────────────────────────────

ADD a cell to generate reports from logged interactions:

    import sqlite3
    import matplotlib.pyplot as plt
    
    conn = sqlite3.connect("data/interactions.db")
    
    # Query category distribution
    df_categories = pd.read_sql_query("""
        SELECT predicted_category, COUNT(*) as count, AVG(confidence_score) as avg_confidence
        FROM interactions
        GROUP BY predicted_category
    """, conn)
    
    # Query routing distribution
    df_routing = pd.read_sql_query("""
        SELECT routing_decision, COUNT(*) as count
        FROM interactions
        GROUP BY routing_decision
    """, conn)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Categories
    df_categories.plot(x='predicted_category', y='count', kind='bar', ax=axes[0, 0], color='skyblue')
    axes[0, 0].set_title('Tickets by Category')
    axes[0, 0].set_xlabel('Category')
    axes[0, 0].set_ylabel('Count')
    
    # Confidence distribution
    df_routing.plot(x='routing_decision', y='count', kind='bar', ax=axes[0, 1], color='lightcoral')
    axes[0, 1].set_title('Routing Decision Distribution')
    axes[0, 1].set_xlabel('Routing Decision')
    axes[0, 1].set_ylabel('Count')
    
    # Confidence scores
    df_confidence = pd.read_sql_query("""
        SELECT confidence_score FROM interactions
    """, conn)
    axes[1, 0].hist(df_confidence['confidence_score'], bins=20, color='lightgreen', edgecolor='black')
    axes[1, 0].set_title('Confidence Score Distribution')
    axes[1, 0].set_xlabel('Confidence Score')
    axes[1, 0].set_ylabel('Frequency')
    
    # Time series
    df_timeline = pd.read_sql_query("""
        SELECT DATE(timestamp) as date, COUNT(*) as count
        FROM interactions
        GROUP BY DATE(timestamp)
        ORDER BY date
    """, conn)
    axes[1, 1].plot(df_timeline['date'], df_timeline['count'], marker='o', color='purple')
    axes[1, 1].set_title('Classification Volume Over Time')
    axes[1, 1].set_xlabel('Date')
    axes[1, 1].set_ylabel('Classifications')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    conn.close()


## DIRECTORY STRUCTURE AFTER INTEGRATION
═════════════════════════════════════════════════════════════════════════════════════

General/
├── preprocess_data.md               (Given by Swathika)
├── train_models.md                  (Given by Swathika)
├── evaluate_models.md               (Given by Swathika)
├── run_inference.md                 (Given by Swathika)
├── select_model.md                  (NEW - Training stage 4)
├── persist_artifacts.md             (NEW - Training stage 4B)
├── detect_language.md               (NEW - Serving stage 5)
├── translate_to_english.md          (NEW - Serving stage 5B)
├── confidence_router.md             (NEW - Serving stage 6)
├── draft_response.md                (NEW - Serving stage 7)
├── log_interaction.md               (NEW - Serving stage 8)
├── customer_support_pipeline.ipynb  (Main notebook - INTEGRATE)
├── all_priority_nodes.py            (NEW - Node implementations)
├── setup_database.py                (NEW - Database initialization)
├── test_system.py                   (NEW - Validation suite)
├── artifacts/                       (Created after training)
│   ├── model.pkl
│   ├── vectorizer.pkl
│   ├── encoder.pkl
│   └── evaluation_results.json
└── data/                            (Created after setup_database.py)
    └── interactions.db              (SQLite database)


## VALIDATION CHECKLIST
═════════════════════════════════════════════════════════════════════════════════════

Before considering the system complete, verify:

  [ ] Database initialized: data/interactions.db with 4 tables
  [ ] Node imports successful: all_priority_nodes.py functions imported
  [ ] Langdetect installed: pip install langdetect
  [ ] Training pipeline runs: cell 13 executes without errors
  [ ] Artifacts created: 4 files in artifacts/ directory
  [ ] Serving pipeline runs: All 5 test tickets classified successfully
  [ ] Gradio UI launches: Interface opens in browser
  [ ] SQLite logging works: data/interactions.db has entries after serving tests
  [ ] Visualizations generate: No errors in analytics cell

## TROUBLESHOOTING
═════════════════════════════════════════════════════════════════════════════════════

Problem: "ModuleNotFoundError: No module named 'langdetect'"
  → Solution: pip install langdetect

Problem: "sqlite3.OperationalError: no such table: interactions"
  → Solution: Run setup_database.py to initialize the database

Problem: "TypeError: 'type' object is not subscriptable"
  → Solution: Ensure Python 3.9+ (use from __future__ import annotations for 3.8)

Problem: "OpenAI API key not found"
  → Solution: Set environment variable: export OPENAI_API_KEY='sk-...'

Problem: "LangGraph state serialization error"
  → Solution: Ensure state is dict, not dataclass. Use create_initial_state() helper.


═════════════════════════════════════════════════════════════════════════════════════
END OF INTEGRATION GUIDE
═════════════════════════════════════════════════════════════════════════════════════
"""

if __name__ == "__main__":
    print(INTEGRATION_GUIDE)
    
    # Optionally save to file
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--save":
        with open("INTEGRATION_GUIDE.txt", "w") as f:
            f.write(INTEGRATION_GUIDE)
        print("\n✅ Guide saved to INTEGRATION_GUIDE.txt")
