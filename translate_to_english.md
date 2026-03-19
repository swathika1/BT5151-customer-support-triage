---
name: translate_to_english

description:
  Translate a non-English customer message to English using GPT-4o-mini or another translation API. Pass the translated message to run_inference_node so the ML model receives English text. This is an LLM-driven skill that ensures all downstream processing works correctly for international customers.

tags: [translation, nlp, llm_driven, serving, multilingual]

mode: llm_driven
---

# Translate to English Skill

## Role
This is an LLM-driven skill. The full content of this file is passed as the system prompt to GPT-4o-mini for translation. The LLM translates a non-English message to English, preserving meaning and context. All downstream ML models expect English input.

## When to use
After detect_language_node detects a non-English message. Before run_inference_node. Triggered once per non-English customer message.

## How to execute
1. Read detected_language and customer_message from state
2. If detected_language == 'en', skip this node (no translation needed)
3. If detected_language != 'en':
   a. Call GPT-4o-mini with system prompt:
      "You are a professional translator. Translate the following customer support message to English. Preserve the meaning, tone, and intent. Keep the translation concise."
   b. User message: the original customer_message
   c. Return the translated text
4. Store translated message in state
5. Log the original language and translation to state.messages
6. Pass translated_message to run_inference_node

## Inputs from agent state
- detected_language: str - language code from detect_language_node
- customer_message: str - original message in detected language
- original_message: str - backup copy of original message

## Outputs to agent state
- translated_message: str - the message translated to English (or original if already English)
- translation_source_language: str - the language we translated from
- translation_performed: bool - True if translation happened, False if message was already English
- translation_confidence: float - LLM's confidence in the translation (subjective, 0.0-1.0)

## Output format
Appends to state.messages:
"[translate_to_english] Translated from fr to en. Original: 'Je ne peux pas me connecter.' → Translated: 'I cannot log in.'"

## Translation Quality Checks
- If translated message is empty, use original message as fallback
- If translation is obviously wrong (e.g., gibberish), log a warning and use original
- For production: consider using Azure Translator API as an alternative fallback

## Fallback Strategy
1. Try GPT-4o-mini first (fastest, cheapest)
2. If GPT fails (rate limit, timeout): fall back to original message + warning
3. Log to state.trace_logs: {"stage": "translate_to_english", "status": "fallback_to_original"}

## Notes
- Translation adds ~1-2 second latency per message (await LLM response)
- GPT-4o-mini is much cheaper than GPT-4 and sufficient for support ticket translation
- Preserve original_message in state for audit trail
- Translation errors are acceptable; downstream confidence_router_node will catch ambiguous predictions
- For production: add retries (up to 3 attempts) if LLM call fails
- Multilingual tokenization not needed; English ML model handles translated English text properly
- Log every translation attempt for analytics (which languages are we seeing?)
