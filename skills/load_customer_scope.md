---
name: load_customer_scope
description: "Load the selected customer's profile, orders, and recent chat history for context-aware support."
mode: organisational
tags: [customer, context, profile, orders]
---

# Load Customer Scope Skill

## When to use
At the start of the serving pipeline, before any language detection or inference, to ensure all subsequent steps have access to the correct customer context.

## How to execute
1. Retrieve the selected customer's profile from the customer database.
2. Fetch all orders associated with the customer.
3. Load recent chat history for the customer (e.g., last 50 turns).
4. Store all retrieved data in the agent state for downstream nodes.

## Output format
- `customer_profile`: dict
- `customer_orders`: list of dict
- `conversation_history`: list of dict
- `pending_interaction`: dict (if clarification is pending)

