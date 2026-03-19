#!/usr/bin/env python
"""
SQLite Database Setup for Customer Support Ticket Classification System
Run this once to initialize the database schema
"""

import sqlite3
from pathlib import Path
import json


def setup_database(db_path="data/interactions.db"):
    """
    Create SQLite database and tables for logging ticket interactions
    """
    # Create data directory if it doesn't exist
    Path("data").mkdir(exist_ok=True)
    
    print(f"Setting up SQLite database at {db_path}...")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # ============================================
    # TABLE 1: interactions
    # ============================================
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS interactions (
            interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            original_message TEXT NOT NULL,
            detected_language TEXT DEFAULT 'en',
            translated_message TEXT,
            predicted_category TEXT NOT NULL,
            confidence_score REAL,
            class_probabilities_json TEXT,
            routing_decision TEXT,
            routing_rationale TEXT,
            requires_human_review BOOLEAN DEFAULT 0,
            response_template TEXT,
            model_name TEXT DEFAULT 'unknown',
            agent_feedback TEXT,
            is_correct BOOLEAN,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create indexes for fast querying
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON interactions(timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_category ON interactions(predicted_category)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_routing ON interactions(routing_decision)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversation ON interactions(conversation_id)")
    
    print("  ✅ Created table: interactions")
    print("  ✅ Created indexes: timestamp, category, routing, conversation_id")
    
    # ============================================
    # TABLE 2: feedback
    # ============================================
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
            interaction_id INTEGER NOT NULL,
            human_feedback_text TEXT,
            human_category TEXT,
            agent_id TEXT,
            feedback_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            is_correct_prediction BOOLEAN,
            confidence_threshold_ok BOOLEAN,
            routing_ok BOOLEAN,
            response_ok BOOLEAN,
            notes TEXT,
            FOREIGN KEY (interaction_id) REFERENCES interactions(interaction_id)
        )
    """)
    
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_interaction_id ON feedback(interaction_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON feedback(feedback_timestamp)")
    
    print("  ✅ Created table: feedback")
    print("  ✅ Created indexes: interaction_id, feedback_timestamp")
    
    # ============================================
    # TABLE 3: model_versions
    # ============================================
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_versions (
            model_id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            model_type TEXT NOT NULL,
            artifact_path TEXT NOT NULL,
            training_timestamp DATETIME,
            training_samples INTEGER,
            validation_samples INTEGER,
            test_samples INTEGER,
            accuracy REAL,
            macro_f1 REAL,
            weighted_f1 REAL,
            metrics_json TEXT,
            is_active BOOLEAN DEFAULT 1,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_model_active ON model_versions(is_active)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_model_timestamp ON model_versions(training_timestamp)")
    
    print("  ✅ Created table: model_versions")
    print("  ✅ Created indexes: is_active, training_timestamp")
    
    # ============================================
    # TABLE 4: performance_metrics
    # ============================================
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS performance_metrics (
            metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            model_id INTEGER,
            total_classifications INTEGER,
            auto_approved INTEGER,
            pending_review INTEGER,
            escalated INTEGER,
            human_validated_count INTEGER,
            correct_predictions INTEGER,
            accuracy_rate REAL,
            avg_confidence REAL,
            FOREIGN KEY (model_id) REFERENCES model_versions(model_id)
        )
    """)
    
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_metric_timestamp ON performance_metrics(timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_metric_model ON performance_metrics(model_id)")
    
    print("  ✅ Created table: performance_metrics")
    print("  ✅ Created indexes: timestamp, model_id")
    
    # Commit and close
    conn.commit()
    conn.close()
    
    print(f"\n✨ Database initialized successfully at {db_path}")
    print("\nTables created:")
    print("  1. interactions - logs every ticket classification")
    print("  2. feedback - stores human feedback on predictions")
    print("  3. model_versions - tracks trained model versions")
    print("  4. performance_metrics - daily/hourly performance analytics")
    
    return db_path


def query_interactions(db_path="data/interactions.db"):
    """
    Example: Query recent interactions
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT interaction_id, timestamp, predicted_category, confidence_score, routing_decision
        FROM interactions
        ORDER BY timestamp DESC
        LIMIT 10
    """)
    
    print("\nRecent Interactions:")
    print("─" * 100)
    for row in cursor.fetchall():
        print(f"ID: {row[0]} | Time: {row[1]} | Category: {row[2]} | Conf: {row[3]:.2f} | Routing: {row[4]}")
    
    conn.close()


def get_category_stats(db_path="data/interactions.db"):
    """
    Example: Get statistics by category
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT predicted_category, COUNT(*) as count, AVG(confidence_score) as avg_confidence
        FROM interactions
        GROUP BY predicted_category
        ORDER BY count DESC
    """)
    
    print("\nCategory Statistics:")
    print("─" * 70)
    for row in cursor.fetchall():
        print(f"{row[0]:30} | Count: {row[1]:5} | Avg Confidence: {row[2]:.3f}")
    
    conn.close()


def get_accuracy_metrics(db_path="data/interactions.db"):
    """
    Example: Get accuracy from feedback data
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            COUNT(f.feedback_id) as total_feedback,
            SUM(CASE WHEN f.is_correct_prediction = 1 THEN 1 ELSE 0 END) as correct,
            ROUND(CAST(SUM(CASE WHEN f.is_correct_prediction = 1 THEN 1 ELSE 0 END) AS FLOAT) 
                  / COUNT(f.feedback_id) * 100, 2) as accuracy_pct
        FROM feedback f
    """)
    
    row = cursor.fetchone()
    if row[0] > 0:
        print(f"\nAccuracy Metrics (from {row[0]} feedback entries):")
        print(f"  Correct: {row[1]} / {row[0]}")
        print(f"  Accuracy: {row[2]}%")
    else:
        print("\nNo feedback data yet.")
    
    conn.close()


if __name__ == "__main__":
    # Set up database
    db_path = setup_database()
    
    # For demo, insert some example data
    print("\n" + "="*80)
    print("DATABASE READY FOR USE")
    print("="*80)
    print("\nYou can now use the log_interaction_node to log ticket classifications.")
    print(f"Database location: {db_path}")
    print("\nExample usage in Python:")
    print("  from log_interaction_node import log_interaction_node")
    print("  state = {...}  # Your state dict")
    print("  state = log_interaction_node(state)")
    print("  print(f'Logged interaction_id={state[\"interaction_id\"]}')")
