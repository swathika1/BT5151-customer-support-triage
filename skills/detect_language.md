---
name: detect_language
description: "Identify language of customer message using textblob or langdetect, return ISO 639-1 code"
mode: organisational
---

## When to use
First node in serving pipeline. Identifies language of incoming customer message to determine if translation is needed.

## How to execute
1. Read raw_message from state
2. Attempt language detection using textblob.TextBlob.detect_language():
   - Lightweight, pure Python, no dependencies
   - Returns ISO 639-1 code (e.g., 'en', 'fr', 'zh', 'es', 'de', 'ja')
3. If textblob fails or returns uncertain result, fallback to langdetect.detect():
   - More accurate for minority languages
   - Also returns ISO 639-1 code
4. If both detection methods fail, default to 'en' (assume English with warning)
5. Store detected_language in state for downstream translate_to_english_node

## Inputs from agent state
- raw_message: Customer's original message (string)

## Outputs to agent state
- detected_language: ISO 639-1 language code (string, e.g., 'en', 'fr', 'zh', 'ja', 'es', 'de')

## Output format
```python
{
  "detected_language": str  # e.g., "en", "fr", "de", "zh", "ja", "es"
}
```

## Notes
- **ISO 639-1 standard**: Two-letter language codes (en=English, fr=French, de=German, zh=Chinese, ja=Japanese, es=Spanish, etc.)
- **Supported languages**: Any language supported by textblob/langdetect; system gracefully defaults to English if detection fails
- **Fallback chain**: textblob → langdetect → default 'en'. This ensures serving pipeline never crashes on language detection.
- **Common languages handled**:
  - en = English
  - fr = French
  - de = German
  - es = Spanish
  - it = Italian
  - pt = Portuguese
  - zh = Chinese (Simplified)
  - ja = Japanese
  - ko = Korean
  - ru = Russian
- **Next step**: translate_to_english_node will skip translation if detected_language == 'en', otherwise will call GPT-4o-mini for translation.
