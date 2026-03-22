---
name: translate_to_english
description: "Translate non-English messages to English using GPT-4o-mini, skip if already English"
mode: llm_driven
---

## When to use
Second node in serving pipeline. Translates incoming messages to English for consistent model inference.

## How to execute
1. Check state.detected_language from detect_language_node
2. If detected_language == 'en': skip LLM call, set translated_message = raw_message, log message
3. Else:
   - Build prompt: "Translate this [detected_language] message to English, preserving intent and tone: [raw_message]"
   - Call GPT-4o-mini via LangChain ChatOpenAI
   - Extract response.content as translated_message
4. Handle errors gracefully:
   - If translation fails, fallback to raw_message with warning
   - Log the error but don't crash serving pipeline
5. Store translated_message in state for run_inference_node

## Inputs from agent state
- raw_message: Customer's original message (string)
- detected_language: ISO 639-1 code from detect_language_node (e.g., 'en', 'fr', 'zh')

## Outputs to agent state
- translated_message: Message in English (string). If already English, equals raw_message. If translation fails, equals raw_message with warning.

## Output format
```python
{
  "translated_message": str  # English version of message (or original if already English)
}
```

## Prompt Template
```
You are a professional translator for a customer support system. 
Translate the following customer message to English, preserving the original intent, tone, and meaning. 
Return ONLY the translated text with no explanation.

[Input message in original language]
```

## Notes
- **Conditional execution**: LLM is only invoked if message is non-English. This saves API costs for English-language customers (typically 80%+ of traffic).
- **GPT-4o-mini choice**: Lightweight and fast (important for latency-critical serving). Sufficient quality for customer support message translation.
- **Error handling**: Translation failures don't crash the pipeline. If LLM unavailable, system falls back to raw message and logs warning. Downstream nodes proceed with best-effort translation.
- **Intent preservation**: Prompt emphasizes preserving customer intent and tone (e.g., if frustrated, translation should maintain that sentiment).
- **Example flows**:
  - Input: "Je veux annuler ma commande" (French) → Output: "I want to cancel my order"
  - Input: "我想要退款" (Chinese) → Output: "I want a refund"
  - Input: "I need help logging in" (English) → Skip LLM, pass through
- **Performance**: Average latency ~1-2 seconds per translation. For batch processing, could be optimized with batching or caching common phrases.
