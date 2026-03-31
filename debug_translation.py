#!/usr/bin/env python3
"""Debug translation and prediction pipeline."""

import os
import pickle
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from textblob import TextBlob
from langdetect import detect

# Check if API key exists
api_key = os.getenv("OPENAI_API_KEY", "").strip()
print(f"OpenAI API Key available: {bool(api_key)}")

# Test query
query = "Je m'attends à une indemnisation de 243 $."
print(f"\n📝 Original query: {query}")

# Step 1: Detect language
print(f"\n1️⃣ LANGUAGE DETECTION")
try:
    lang = detect(query)
    print(f"   Detected: {lang}")
except Exception as e:
    print(f"   ERROR: {e}")

# Step 2: Translate
print(f"\n2️⃣ TRANSLATION")
if not api_key:
    print("   ⚠️  No API key - skipping translation")
else:
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        response = llm.invoke([
            HumanMessage(content=f"Translate to English, return ONLY the translation:\n\n{query}")
        ])
        translated = response.content.strip()
        print(f"   Translated: {translated}")
    except Exception as e:
        print(f"   ERROR: {e}")
        translated = query

# Step 3: Preprocess
print(f"\n3️⃣ PREPROCESSING")
import re
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'\{\{.*?\}\}', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

processed = preprocess_text(translated)
print(f"   Preprocessed: {processed}")

# Step 4: Load model and predict
print(f"\n4️⃣ MODEL PREDICTION")
with open("artifacts/model.pkl", "rb") as f:
    model = pickle.load(f)
with open("artifacts/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
with open("artifacts/label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

vec = vectorizer.transform([processed])
probs = model.predict_proba(vec)[0]
pred_idx = probs.argmax()
pred_label = encoder.classes_[pred_idx]
confidence = probs.max()

print(f"   Predicted: {pred_label} ({confidence:.1%})")
print(f"\n   Top 3 predictions:")
top_3_idx = (-probs).argsort()[:3]
for idx in top_3_idx:
    print(f"      {encoder.classes_[idx]}: {probs[idx]:.1%}")

# Step 5: Check what the model sees
print(f"\n5️⃣ WHAT MODEL RECEIVES")
print(f"   Vector shape: {vec.shape}")
print(f"   Vector density: {vec.nnz / (vec.shape[0] * vec.shape[1]):.2%}")

# Test with direct English
print(f"\n6️⃣ TESTING WITH ENGLISH EQUIVALENTS")
test_queries = [
    "I expect a 243 dollar compensation",
    "I need a refund for 243 dollars",
    "Request compensation 243 dollars",
    processed  # What the model actually got
]
for q in test_queries:
    vec = vectorizer.transform([q])
    probs = model.predict_proba(vec)[0]
    pred_idx = probs.argmax()
    pred_label = encoder.classes_[pred_idx]
    confidence = probs.max()
    print(f"   '{q}'")
    print(f"      → {pred_label} ({confidence:.1%})")
