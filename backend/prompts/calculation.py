"""Calculation prompt — when a formula-based computation is detected."""

CALCULATION_PROMPT = """## Calculation Required
{calculation_info}

## Financial Domain Knowledge
{knowledge_context}

## Retrieved Context
{context}

## Question
{query}

Extract the required components from the context, apply the formula, and show your calculation step by step.
If any component is missing from the context, state which values are unavailable."""
