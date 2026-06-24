"""
Context Assembler — builds the final LLM prompt from reranked chunks + knowledge + query.

ARCHITECTURE (where this fits):
    Retriever → Reranker → Context Assembler (this file) → LLM Generation

WHAT THIS FILE DOES:
    1. Takes reranked chunks from the Reranker
    2. Formats them into a structured context string with source metadata
    3. Injects relevant financial domain knowledge (formulas, terms, sector context)
    4. Selects the right prompt template based on query type
    5. Assembles everything into a final (system_message, user_message) pair
       ready to send to the LLM

WHY THIS IS A SEPARATE STEP:
    The retriever finds chunks. The reranker scores them. But neither knows
    HOW to present them to the LLM. The assembler handles:
    - Formatting (numbered chunks with metadata)
    - Knowledge injection (formulas when calculating, fallback when no chunks)
    - Prompt selection (QA vs calculation vs comparison vs fallback)
    - Token budgeting (don't exceed LLM context window)
"""

from typing import List, Optional, Tuple

from src.retrieval.reranker import RankedChunk
from src.retrieval.queryprocessor import QueryEnhancer
from src.augmentation.prompt import (
    SYSTEM_PROMPT,
    QA_PROMPT,
    CALCULATION_PROMPT,
    COMPARISON_PROMPT,
    FALLBACK_PROMPT,
)
from data.knowledge_base.financial_knowledge import build_knowledge_context, FINANCIAL_FORMULAS
from utils.logging import set_logger

logger = set_logger(__name__)


class ContextAssembler:
    """
    Assembles reranked chunks, financial knowledge, and query into LLM-ready messages.

    Usage:
        assembler = ContextAssembler()
        system_msg, user_msg = assembler.assemble(
            query="What is IHCL's ROA?",
            chunks=reranked_chunks,
            query_type="calculation",
            metric_name="ROA",
        )
        # Send system_msg and user_msg to LLM
    """

    def __init__(self, max_context_tokens: int = 6000):
        """
        PARAMETERS:
            max_context_tokens: rough token budget for the context section.

            WHY TOKEN BUDGETING?
                LLMs have a finite context window (e.g., 128k for GPT-4).
                But more context ≠ better answers. Research shows LLMs
                perform WORSE with too much context ("lost in the middle" problem).
                Keeping context focused improves answer quality and reduces cost.

            WHY 6000 DEFAULT?
                ~6000 tokens ≈ ~4500 words ≈ ~5-8 chunks of 1200 chars each.
                Leaves room for system prompt, knowledge context, and the
                LLM's response within a reasonable token budget.
        """
        self._max_context_tokens = max_context_tokens
        self._query_enhancer = QueryEnhancer()

    # -------------------------------------------------------------------
    # FORMAT CHUNKS: Turn RankedChunks into a readable context string
    # -------------------------------------------------------------------
    def _format_chunks(self, chunks: List[RankedChunk]) -> str:
        """
        Format reranked chunks into a structured, numbered context string.

        WHY NUMBERED WITH METADATA?
            - Numbers let the LLM reference specific chunks: "According to [Chunk 3]..."
            - Source metadata (file, page) enables citation in the answer
            - Relevance score helps the LLM weigh conflicting information
              (higher-scored chunk is more trustworthy)

        EXAMPLE OUTPUT:
            [Chunk 1] (Source: ihcl.pdf | Page: 45 | Relevance: 0.92)
            IHCL reported total revenue of ₹6,000 Cr for FY2023, representing
            a growth of 12% over the previous year...

            [Chunk 2] (Source: ihcl.pdf | Page: 52 | Relevance: 0.87)
            The company's total assets stood at ₹15,200 Cr...
        """
        formatted_parts = []

        for i, chunk in enumerate(chunks, 1):
            metadata = chunk.document.metadata or {}

            source = metadata.get("file_id", "Unknown")
            page = metadata.get("page", metadata.get("page_number", "N/A"))

            header = f"[Chunk {i}] (Source: {source} | Page: {page} | Relevance: {chunk.rerank_score:.2f})"
            content = chunk.document.page_content.strip()

            formatted_parts.append(f"{header}\n{content}")

        return "\n\n---\n\n".join(formatted_parts)

    # -------------------------------------------------------------------
    # DETECT QUERY TYPE: Determine which prompt template to use
    # -------------------------------------------------------------------
    def _detect_query_type(
        self, query: str, chunks: List[RankedChunk]
    ) -> Tuple[str, dict]:
        """
        Determine the query type and gather extra context for prompt selection.

        RETURNS:
            (query_type, extra_info) where:
            - query_type: "calculation" | "comparison" | "fallback" | "qa"
            - extra_info: dict with additional context (metric_name, etc.)

        PRIORITY ORDER:
            1. No chunks → "fallback" (nothing was retrieved)
            2. Calculation detected → "calculation" (need formula + steps)
            3. Comparison detected → "comparison" (need structured comparison)
            4. Default → "qa" (standard question answering)
        """
        if not chunks:
            return "fallback", {}

        is_calc, metric_name, required_inputs = (
            self._query_enhancer.detect_calculation_request(query)
        )
        if is_calc:
            return "calculation", {
                "metric_name": metric_name,
                "required_inputs": required_inputs,
            }

        sub_queries = self._query_enhancer.decompose_complex_query(query)
        if len(sub_queries) > 1:
            return "comparison", {}

        return "qa", {}

    # -------------------------------------------------------------------
    # BUILD CALCULATION INFO: Format formula details for calculation prompts
    # -------------------------------------------------------------------
    @staticmethod
    def _build_calculation_info(metric_name: str, required_inputs: List[str]) -> str:
        """
        Build a calculation instruction string from the financial knowledge base.

        EXAMPLE OUTPUT:
            Metric: ROA (Return on Assets)
            Formula: PAT / Total Assets * 100
            Required components: PAT, Total Assets
            Unit: %
            Interpretation: Higher ROA indicates more efficient use of assets
        """
        formula_data = FINANCIAL_FORMULAS.get(metric_name, {})

        if not formula_data:
            return f"Metric: {metric_name}\nRequired components: {', '.join(required_inputs)}"

        parts = [
            f"Metric: {metric_name}",
            f"Formula: {formula_data['formula']}",
            f"Required components: {', '.join(formula_data['components'])}",
            f"Unit: {formula_data['unit']}",
            f"Interpretation: {formula_data['interpretation']}",
        ]
        return "\n".join(parts)

    # -------------------------------------------------------------------
    # SELECT PROMPT: Choose the right template based on query type
    # -------------------------------------------------------------------
    def _select_and_fill_prompt(
        self,
        query: str,
        context: str,
        knowledge_context: str,
        query_type: str,
        extra_info: dict,
    ) -> str:
        """
        Select the appropriate prompt template and fill in the variables.

        WHY DIFFERENT TEMPLATES?
            Each query type needs different instructions:
            - QA: "answer from context, cite sources"
            - Calculation: "extract components, show formula, compute step-by-step"
            - Comparison: "use a table, note gaps"
            - Fallback: "use general knowledge, disclaim it's not from documents"

            Using one generic prompt for all types leads to worse answers.
        """
        template_map = {
            "qa": QA_PROMPT,
            "calculation": CALCULATION_PROMPT,
            "comparison": COMPARISON_PROMPT,
            "fallback": FALLBACK_PROMPT,
        }

        template = template_map.get(query_type, QA_PROMPT)

        fill_values = {
            "context": context,
            "query": query,
            "knowledge_context": knowledge_context or "No additional domain knowledge applicable.",
        }

        if query_type == "calculation":
            metric_name = extra_info.get("metric_name", "")
            required_inputs = extra_info.get("required_inputs", [])
            fill_values["calculation_info"] = self._build_calculation_info(
                metric_name, required_inputs
            )

        return template.format(**fill_values)

    def assemble(
        self,
        query: str,
        chunks: List[RankedChunk],
        metric_name: Optional[str] = None,
        sector: Optional[str] = None,
    ) -> Tuple[str, str]:
        """
        Assemble the final LLM prompt from all components.

        FLOW:
            1. Detect query type (QA / calculation / comparison / fallback)
            2. Format chunks into numbered context string
            3. Build financial knowledge context (formulas, terms, sector info)
            4. Select prompt template and fill variables
            5. Return (system_message, user_message) tuple

        PARAMETERS:
            query:       original user question
            chunks:      List[RankedChunk] from the reranker (can be empty)
            metric_name: optional — override metric detection for knowledge injection
            sector:      optional — inject sector-specific context

        RETURNS:
            Tuple[str, str]:
                - system_message: persona + rules (SYSTEM_PROMPT)
                - user_message: context + knowledge + query (filled template)

        WHY RETURN A TUPLE, NOT A SINGLE STRING?
            LLM APIs (OpenAI, etc.) expect separate system and user messages:
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ]
            The system message sets behavior, the user message contains the task.
            Mixing them into one message reduces LLM instruction-following quality.
        """
        # Step 1: Detect query type
        query_type, extra_info = self._detect_query_type(query, chunks)
        logger.info("Query type detected: %s", query_type)

        # Step 2: Format chunks
        context = self._format_chunks(chunks) if chunks else ""

        # Step 3: Build knowledge context
        knowledge_metric = metric_name or extra_info.get("metric_name")
        knowledge_context = build_knowledge_context(
            metric_name=knowledge_metric,
            sector=sector,
        )

        # Step 4: Select and fill prompt template
        user_message = self._select_and_fill_prompt(
            query=query,
            context=context,
            knowledge_context=knowledge_context,
            query_type=query_type,
            extra_info=extra_info,
        )

        logger.info(
            "Context assembled: %d chunks, query_type=%s, knowledge=%s",
            len(chunks), query_type, bool(knowledge_context),
        )

        return SYSTEM_PROMPT, user_message
