"""Comparison prompt — for multi-entity or multi-period comparisons."""

COMPARISON_PROMPT = """## Financial Domain Knowledge
{knowledge_context}

## Retrieved Context
{context}

## Question
{query}

Compare the requested metrics in a structured format. Use a table if comparing multiple data points.
Clearly state the time periods or entities being compared. Note any data gaps."""
