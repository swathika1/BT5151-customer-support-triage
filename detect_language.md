---
name: detect_language

description:
  Detect the language of an incoming customer support message. Return the language code (e.g., 'en', 'fr', 'es'). If English, pass to next stage. If non-English, route to translate_to_english_node. This ensures the ML model (trained on English) receives English-language text at inference time.

tags: [language_detection, nlp, preprocessing, serving]

mode: organisational
---

# Detect Language Skill

## Role
This is an organisational skill. The Python code uses langdetect or textblob to identify the language of a customer message deterministically. This is a fast, lightweight preprocessing step that routes messages to translation if needed.

## When to use
At the start of the serving pipeline, immediately after a customer message arrives. Before run_inference_node. Triggered once per customer message.

## How to execute
1. Receive customer_message from state
2. Use langdetect.detect(customer_message) to identify language
   - Returns language code: 'en', 'fr', 'es', 'de', 'pt', 'zh', 'ja', etc.
   - If message is too short (<5 characters), default to 'en'
   - If detection fails, default to 'en'
3. Check if the detected language is English ('en'):
   - If yes: set detected_language = 'en', skip translation
   - If no: set detected_language to the detected code, will translate next
4. Store language code and confidence in state
5. Log the detection to state.messages

## Inputs from agent state
- customer_message: str - the original customer support message (may be in any language)

## Outputs to agent state
- detected_language: str - ISO 639-1 language code (e.g., 'en', 'fr')
- language_confidence: float - confidence score of detection (0.0-1.0)
- requires_translation: bool - True if language is not English, False otherwise
- original_message: str - unchanged copy of original message for later reference

## Output format
Appends to state.messages:
"[detect_language] Language: en (confidence: 0.98). Requires translation: No."
or
"[detect_language] Language: fr (confidence: 0.95). Requires translation: Yes. Routing to translate_to_english_node."

## Routing Logic
- If detected_language == 'en' AND language_confidence >= 0.8: Skip translation, proceed to inference
- If detected_language != 'en' OR language_confidence < 0.8: Route to translate_to_english_node

## Notes
- langdetect is fast (<50ms for typical support messages)
- Short messages (<10 chars) often have low confidence; default to 'en' for very short messages
- For ambiguous messages, always translate rather than fail silently
- This node is essential for serving real customers who may write in their native language
- Supported languages: en, fr, es, de, it, pt, nl, ru, ja, zh, ko, ar, hi, and 50+ others
- Language detection happens before any ML model inference to prevent garbage-in-garbage-out
