#!/usr/bin/env python3
"""Check actual Bitext dataset structure and real responses."""

from datasets import load_dataset

print("\n=== LOADING BITEXT DATASET ===\n")

dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
df = dataset['train'].to_pandas()

print(f"Dataset shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nFirst few rows:\n")
print(df.head(3))

print(f"\n\nSample rows for each category:\n")
for category in df['category'].unique()[:5]:
    sample = df[df['category'] == category].iloc[0]
    print(f"Category: {category}")
    print(f"  Instruction: {sample['instruction'][:80]}")
    print(f"  Response: {sample['response'][:100]}")
    print()
