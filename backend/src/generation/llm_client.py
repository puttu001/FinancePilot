"""
LLM Client — sends assembled prompts to the LLM and validates responses.

ARCHITECTURE (where this fits):
    Retriever → Reranker → Context Assembler → LLM Client (this file)

WHAT THIS FILE DOES:
    1. Takes (system_message, user_message) from the ContextAssembler
    2. Sends them to the LLM (OpenAI API via LangChain)
    3. Validates the response (format, emptiness, truncation)
    4. Returns a structured GenerationResult with response + metadata

WHAT IT DOES NOT DO:
    - Prompt construction: that's the ContextAssembler's job
    - Quality evaluation (faithfulness, hallucination detection): that's evaluation/'s job
    - This file only does lightweight sanity checks on the raw LLM output
"""

import time
from typing import Optional
from dataclasses import dataclass, field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from utils.logging import set_logger
from utils.config import get_config

logger = set_logger(__name__)


# ---------------------------------------------------------------------------
# Data class for the generation result
# ---------------------------------------------------------------------------
# WHY a dataclass?
#   The LLM returns just a string. But downstream code (API, evaluation, logging)
#   needs metadata too: which model was used, how long it took, token usage,
#   whether validation passed. Bundling it all together avoids passing
#   5 separate variables around.
# ---------------------------------------------------------------------------
@dataclass
class GenerationResult:
    answer: str                          # the LLM's response text
    model: str = ""                      # which model generated this
    latency_ms: float = 0.0             # how long the LLM call took
    input_tokens: int = 0                # tokens in the prompt
    output_tokens: int = 0               # tokens in the response
    is_valid: bool = True                # did it pass validation?
    validation_errors: list = field(default_factory=list)  # what failed


# ---------------------------------------------------------------------------
# LLM Client class
# ---------------------------------------------------------------------------
class LLMClient:
    """
    Sends prompts to the LLM and returns validated responses.

    Usage:
        client = LLMClient()
        result = await client.generate(system_msg, user_msg)
        print(result.answer)

    WHY WRAP THE LLM CALL?
        1. Consistent interface — rest of the codebase doesn't need to know
           if you're using OpenAI, Anthropic, or a local model
        2. Validation — every response passes through sanity checks
        3. Metadata — latency, token usage, model info tracked automatically
        4. Error handling — API failures don't crash the pipeline
    """

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        """
        Initialize the LLM client.

        PARAMETERS:
            model:       LLM model name (default from config)
            temperature: controls randomness (0 = deterministic, 1 = creative)
                         For financial analysis, low temperature is better —
                         we want precise, consistent answers, not creative ones.
            max_tokens:  max response length. Prevents runaway responses
                         that waste tokens and money.

        WHY CONFIGURABLE?
            Different use cases need different settings:
            - QA: gpt-4o-mini, temp=0, fast and cheap
            - Complex analysis: gpt-4o, temp=0, more capable
            - Evaluation (LLM-as-judge): separate model to avoid self-bias
        """
        generation_config = get_config("generation", {})
        self._model_name = model or generation_config.get("model", "gpt-4o-mini")
        self._temperature = temperature if temperature is not None else generation_config.get("temperature", 0)
        self._max_tokens = max_tokens or generation_config.get("max_tokens", 2048)

        self._llm = ChatOpenAI(
            model=self._model_name,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

        logger.info(
            "LLM client initialized: model=%s, temp=%s, max_tokens=%s",
            self._model_name, self._temperature, self._max_tokens,
        )

    # -------------------------------------------------------------------
    # VALIDATION: Lightweight sanity checks on the LLM response
    # -------------------------------------------------------------------
    def _validate_response(self, response_text: str) -> tuple[bool, list[str]]:
        """
        Run quick sanity checks on the LLM's raw response.

        WHAT WE CHECK:
            1. Empty response — LLM returned nothing (API issue or token limit hit)
            2. Truncation — response cuts off mid-sentence (max_tokens too low)
            3. Refusal without reason — LLM refused but didn't explain why

        WHAT WE DON'T CHECK HERE (that's evaluation/'s job):
            - Factual correctness (is the answer right?)
            - Faithfulness (is it grounded in the context?)
            - Hallucination (did it make up numbers?)

        RETURNS:
            (is_valid, list_of_errors)
        """
        errors = []

        if not response_text or not response_text.strip():
            errors.append("Empty response from LLM")
            return False, errors

        stripped = response_text.strip()

        if not stripped[-1] in ".!?:;)]\"’”" and len(stripped) > 100:
            errors.append("Response may be truncated (doesn't end with terminal punctuation)")

        refusal_phrases = [
            "i cannot",
            "i'm unable to",
            "i am unable to",
            "as an ai",
            "i don't have access",
        ]
        lower = stripped.lower()
        if any(phrase in lower for phrase in refusal_phrases):
            if "based on the available documents" not in lower:
                errors.append("LLM refused to answer without using the expected fallback phrasing")

        is_valid = len(errors) == 0
        return is_valid, errors

    # -------------------------------------------------------------------
    # CORE: Send prompt to LLM
    # -------------------------------------------------------------------
    async def generate(
        self,
        system_message: str,
        user_message: str,
    ) -> GenerationResult:
        """
        Send the assembled prompt to the LLM and return a validated result.

        HOW IT WORKS:
            1. Build LangChain message objects (SystemMessage + HumanMessage)
            2. Call the LLM (async for non-blocking in FastAPI)
            3. Extract response text and token usage metadata
            4. Run validation checks
            5. Return GenerationResult with everything bundled

        PARAMETERS:
            system_message: from ContextAssembler — LLM persona and rules
            user_message:   from ContextAssembler — context + knowledge + query

        RETURNS:
            GenerationResult with answer, metadata, and validation status

        WHY ASYNC?
            The LLM API call is I/O-bound (network request to OpenAI).
            Using async lets FastAPI handle other requests while waiting
            for the LLM response (typically 1-5 seconds).
        """
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=user_message),
        ]

        start_time = time.perf_counter()

        try:
            response = await self._llm.ainvoke(messages)
            latency_ms = (time.perf_counter() - start_time) * 1000

            answer = response.content
            usage = response.usage_metadata or {}

            is_valid, errors = self._validate_response(answer)
            if errors:
                logger.warning("Validation issues: %s", errors)

            result = GenerationResult(
                answer=answer,
                model=self._model_name,
                latency_ms=round(latency_ms, 2),
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                is_valid=is_valid,
                validation_errors=errors,
            )

            logger.info(
                "LLM response: model=%s, latency=%.0fms, tokens=%d→%d, valid=%s",
                self._model_name, latency_ms,
                result.input_tokens, result.output_tokens, is_valid,
            )

            return result

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error("LLM call failed: %s", str(e), exc_info=True)

            return GenerationResult(
                answer="An error occurred while generating the response. Please try again.",
                model=self._model_name,
                latency_ms=round(latency_ms, 2),
                is_valid=False,
                validation_errors=[f"LLM API error: {str(e)}"],
            )
