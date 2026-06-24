"""QA prompt — standard question answering from retrieved context."""

QA_PROMPT = """## Financial Domain Knowledge
{knowledge_context}

## Retrieved Context
{context}

## Question
{query}

Provide a clear, accurate answer based on the context above. Cite sources where possible."""
