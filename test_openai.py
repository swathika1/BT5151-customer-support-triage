#!/usr/bin/env python3
import os
import sys

print(f"API Key from env: {os.getenv('OPENAI_API_KEY')[:20] if os.getenv('OPENAI_API_KEY') else 'NOT SET'}...")

try:
    from openai import OpenAI
    print("✓ OpenAI import successful")
    
    # Create client with minimal config
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    print("✓ OpenAI client initialized")
    
    # Direct test without models.list()
    result = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Translate to English only: Je m'attends à une indemnisation de 243 $."}],
        temperature=0.0,
        max_tokens=50
    )
    print(f"✓ Translation result: {result.choices[0].message.content}")
    
except Exception as e:
    import traceback
    print(f"✗ Error: {str(e)}")
    traceback.print_exc()
    sys.exit(1)
