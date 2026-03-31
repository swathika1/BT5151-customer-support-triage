#!/usr/bin/env python3
"""Test script to verify model and dataset are working correctly."""

import pickle
import json
from pathlib import Path

# Load artifacts
MODEL_PATH = "artifacts/model.pkl"
VECTORIZER_PATH = "artifacts/tfidf_vectorizer.pkl"
ENCODER_PATH = "artifacts/label_encoder.pkl"

print("\n=== VERIFYING MODEL & ARTIFACTS ===\n")

# Load model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
print(f"✓ Model loaded: {type(model)}")

# Load vectorizer
with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)
print(f"✓ Vectorizer loaded: {vectorizer.get_feature_names_out().shape[0]} features")

# Load encoder
with open(ENCODER_PATH, "rb") as f:
    encoder = pickle.load(f)
print(f"✓ Label encoder loaded")

# Show all categories the model knows
print(f"\n📊 CATEGORIES IN MODEL ({len(encoder.classes_)}):")
for i, cls in enumerate(encoder.classes_):
    print(f"   {i+1}. {cls}")

# Test predictions
test_queries = [
    "I am trying to cancel purchase 9384",
    "I want to cancel my order",
    "How do I cancel my subscription?",
    "I can't access my account",
    "Where is my package?",
    "I need a refund",
]

print(f"\n🧪 TESTING PREDICTIONS:\n")
for query in test_queries:
    vec = vectorizer.transform([query])
    probs = model.predict_proba(vec)[0]
    pred_idx = probs.argmax()
    pred_label = encoder.classes_[pred_idx]
    confidence = probs.max()
    
    print(f"Q: {query}")
    print(f"   → {pred_label} ({confidence:.1%})")
    
    # Show top 3 predictions
    top_3_idx = (-probs).argsort()[:3]
    print(f"   Top 3:")
    for idx in top_3_idx:
        print(f"      {encoder.classes_[idx]}: {probs[idx]:.1%}")
    print()
