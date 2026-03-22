---
name: draft_response
description: "Generate customer response based on predicted category and routing tier, using templates and LLM polishing"
mode: llm_driven
---

## When to use
Sixth node in serving pipeline. Generates appropriate response for customer based on their issue category and confidence tier.

## How to execute
1. Set up response templates for each complaint category:
   - ACCOUNT: Password reset, login issue instructions
   - CANCEL: Cancellation confirmation, refund timeline
   - CONTACT: Contact information for specialized support
   - DELIVERY: Shipping status, delivery estimate
   - FEEDBACK: Thank you, note feedback is valued
   - INVOICE: Billing details, dispute process
   - ORDER: Order confirmation details, tracking info
   - PAYMENT: Payment status, issue resolution
   - REFUND: Refund policy, timeline, authorization process
   - SHIPPING: Tracking link, estimated delivery
   - SUBSCRIPTION: Subscription management, renewal info

2. Select base template based on predicted_label (from run_inference_node)

3. Adapt template based on route_decision:
   - **AUTO_REPLY**: Full, confident response (2-3 sentences, actionable)
     Example: "We've initiated a refund to your original payment method. You should see it within 3-5 business days."
   - **CLARIFY**: Questions to gather more info (1-2 sentences, open-ended)
     Example: "To help you better, could you provide your order number? This will allow us to check shipping status."
   - **ESCALATE**: Escalation notice (1 sentence)
     Example: "Your inquiry requires specialist attention. A human agent will contact you within 24 hours."

4. Call GPT-4o-mini to polish final response:
   - Input: Adapted template
   - Prompt: "Polish this customer support response to be professional, empathetic, and concise (max 3 sentences). No technical jargon."
   - Output: response_final (polished response)

5. Store response_final in state for log_interaction_node

## Inputs from agent state
- predicted_label: Category from run_inference_node (e.g., "REFUND")
- route_decision: Routing tier from confidence_router (e.g., "AUTO_REPLY")
- confidence_score: For context logging (not used in response generation)
- raw_message: Original customer message (for logging context)

## Outputs to agent state
- response_final: Final customer-facing response message (string)

## Output format
```python
{
  "response_final": str  # e.g., "We've initiated your refund. You should see it within 3-5 business days."
}
```

## Template Structure
```python
RESPONSE_TEMPLATES = {
    "ACCOUNT": {
        "AUTO_REPLY": "We've sent password reset instructions to your email. If you don't see it, check your spam folder.",
        "CLARIFY": "To secure your account, I need to verify a few details. Can you confirm the email associated with your account?",
        "ESCALATE": "For security reasons, account issues require manual verification. An agent will contact you within 24 hours."
    },
    "REFUND": {...},
    ...
}
```

## Notes
- **Template coverage**: Pre-written templates ensure response consistency and speed. Rather than generating from scratch (slow, variable quality), use templates as a base.
- **LLM polishing**: GPT-4o-mini is used only for tone adjustment and brevity verification, NOT for content generation. This keeps responses safe and aligned with company policy.
- **Tone adaptation**: AUTO_REPLY is direct and decisive ("We've..."). CLARIFY is collaborative ("Can you...?"). ESCALATE is apologetic and reassuring ("...will contact you within...").
- **Length constraints**: Shorter responses (2-3 sentences) perform better in customer satisfaction studies. Longer responses overwhelm customers.
- **No prompting about policy**: Templates already encode company policy (refund timeline, contact info, etc.). LLM is forbidden from inventing policy.
- **Example flow**:
  1. Predicted: REFUND, Confidence: 0.92
  2. Route: AUTO_REPLY
  3. Template: "We've initiated your refund. You should see it within [timeline]."
  4. LLM polish: (minor tweaks to wording)
  5. Final: "We've processed your refund. You should see it within 3-5 business days."
