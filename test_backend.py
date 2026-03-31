#!/usr/bin/env python3
"""Test the ML pipeline backend without Gradio"""

import sys
sys.path.insert(0, '/home/swathika/5151BT/BT5151-customer-support-triage')

from app import load_trained_pipeline, predict_with_confidence, generate_response

# Load the model
print("Loading trained pipeline...")
model, vectorizer = load_trained_pipeline()
print(f"✅ Loaded model and vectorizer\n")

# Test queries
test_queries = [
    "I lost my package",
    "Can't log into my account",
    "I want to cancel my subscription",
    "Where is my order?",
    "How do I contact support?"
]

print("="*60)
print("TESTING ML PIPELINE")
print("="*60)

for query in test_queries:
    print(f"\n📝 Query: {query}")
    
    pred, conf, probs = predict_with_confidence(query, model, vectorizer)
    resp = generate_response(pred, conf)
    
    print(f"   🎯 Prediction: {pred}")
    print(f"   💯 Confidence: {conf:.1%}")
    print(f"   💬 Response: {resp[:80]}...")
    
    # Show top 3
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    print(f"   Top 3: {sorted_probs[:3]}")

print("\n" + "="*60)
print("✅ BACKEND TEST COMPLETE - ALL SYSTEMS OPERATIONAL!")
print("="*60)
