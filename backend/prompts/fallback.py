"""Fallback prompt — when retrieval returns no relevant chunks."""

FALLBACK_PROMPT = """## Financial Domain Knowledge
{knowledge_context}

## Question
{query}

Note: No relevant information was found in the uploaded documents for this query.
Provide a general explanation based on financial domain knowledge above.
Clearly state that this is general knowledge and NOT derived from the user's documents."""
