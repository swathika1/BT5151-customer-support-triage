---
name: prepare_contextual_query
description: "Augment short clarification follow-ups with prior pending context to prepare the inference message."
mode: organisational
tags: [context, query, clarification]
---

# Prepare Contextual Query Skill

## When to use
After translation and before running inference, especially when the user message is a short follow-up or clarification.

## How to execute
1. Check if there is a pending clarification from the previous interaction.
2. If so, combine the current message with relevant prior context (orders, pending interaction, etc.).
3. Prepare the final inference message for the model.

## Output format
- `inference_message`: str
- `prep_reason`: str (explanation of how the message was prepared)
