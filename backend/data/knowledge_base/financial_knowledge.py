"""
Static financial domain knowledge — always available to the LLM regardless of retrieval results.

WHY THIS EXISTS:
    1. FALLBACK: When retrieval returns no chunks or low-confidence chunks,
       the LLM still has domain knowledge to give meaningful responses
    2. GROUNDING: Even with good chunks, the LLM needs to know that
       ROA = PAT / Total Assets, not guess the formula
    3. TERM RESOLUTION: Financial documents use interchangeable terms
       (PAT = Net Profit = Net Income). This mapping prevents confusion

THIS IS NOT:
    - A replacement for retrieved context (chunks from actual reports are always better)
    - An exhaustive financial textbook (only what's needed for the RAG pipeline)
    - Dynamic data (no company-specific numbers here)
"""

# ---------------------------------------------------------------------------
# Financial term mappings — synonyms and abbreviations
# ---------------------------------------------------------------------------
# WHY?
#   Financial documents are inconsistent with terminology.
#   One report says "PAT", another says "Net Profit", another says "Net Income".
#   The LLM needs to know these are the same thing so it doesn't treat them
#   as different metrics when assembling answers from multiple chunks.
# ---------------------------------------------------------------------------
TERM_MAPPINGS = {
    "PAT": ["Profit After Tax", "Net Profit", "Net Income", "Bottom Line"],
    "PBT": ["Profit Before Tax", "Pre-Tax Profit"],
    "EBITDA": ["Earnings Before Interest Tax Depreciation and Amortization", "Operating Profit (before D&A)"],
    "EBIT": ["Earnings Before Interest and Tax", "Operating Profit"],
    "Revenue": ["Total Revenue", "Total Income", "Sales", "Turnover", "Top Line"],
    "Total Assets": ["Gross Assets", "Asset Base"],
    "Total Equity": ["Shareholders' Equity", "Net Worth", "Shareholders' Funds"],
    "Total Debt": ["Total Borrowings", "Gross Debt", "Interest-Bearing Liabilities"],
    "GNPA": ["Gross Non-Performing Assets", "Gross NPA", "Gross Bad Loans"],
    "NNPA": ["Net Non-Performing Assets", "Net NPA", "Net Bad Loans"],
    "NIM": ["Net Interest Margin", "Interest Spread"],
    "CRAR": ["Capital to Risk-Weighted Assets Ratio", "Capital Adequacy Ratio", "CAR"],
    "AUM": ["Assets Under Management"],
    "NAV": ["Net Asset Value"],
    "EPS": ["Earnings Per Share"],
    "BVPS": ["Book Value Per Share"],
    "DPS": ["Dividend Per Share"],
    "PE": ["Price to Earnings Ratio", "P/E Ratio"],
    "PB": ["Price to Book Ratio", "P/B Ratio"],
    "ROA": ["Return on Assets"],
    "ROE": ["Return on Equity"],
    "ROCE": ["Return on Capital Employed"],
}


# ---------------------------------------------------------------------------
# Financial formulas — how to calculate key ratios
# ---------------------------------------------------------------------------
# WHY?
#   When a user asks "Calculate ROA" and retrieval finds PAT and Total Assets
#   in separate chunks, the LLM needs to know the formula to compute it.
#   Without this, it might hallucinate a wrong formula.
# ---------------------------------------------------------------------------
FINANCIAL_FORMULAS = {
    "ROA": {
        "formula": "PAT / Total Assets * 100",
        "components": ["PAT", "Total Assets"],
        "unit": "%",
        "interpretation": "Higher ROA indicates more efficient use of assets to generate profit",
    },
    "ROE": {
        "formula": "PAT / Total Equity * 100",
        "components": ["PAT", "Total Equity"],
        "unit": "%",
        "interpretation": "Measures profitability relative to shareholders' equity",
    },
    "Debt to Equity": {
        "formula": "Total Debt / Total Equity",
        "components": ["Total Debt", "Total Equity"],
        "unit": "x",
        "interpretation": "Lower ratio indicates less leverage. Below 1x is generally conservative",
    },
    "Current Ratio": {
        "formula": "Current Assets / Current Liabilities",
        "components": ["Current Assets", "Current Liabilities"],
        "unit": "x",
        "interpretation": "Above 1x means the company can cover short-term obligations. Below 1x is a liquidity concern",
    },
    "Quick Ratio": {
        "formula": "(Current Assets - Inventory) / Current Liabilities",
        "components": ["Current Assets", "Inventory", "Current Liabilities"],
        "unit": "x",
        "interpretation": "Stricter liquidity test than current ratio — excludes inventory",
    },
    "Net Profit Margin": {
        "formula": "PAT / Revenue * 100",
        "components": ["PAT", "Revenue"],
        "unit": "%",
        "interpretation": "Percentage of revenue retained as profit after all expenses",
    },
    "Operating Profit Margin": {
        "formula": "EBIT / Revenue * 100",
        "components": ["EBIT", "Revenue"],
        "unit": "%",
        "interpretation": "Profitability from core operations before interest and tax",
    },
    "EBITDA Margin": {
        "formula": "EBITDA / Revenue * 100",
        "components": ["EBITDA", "Revenue"],
        "unit": "%",
        "interpretation": "Operating profitability before non-cash charges",
    },
    "Interest Coverage Ratio": {
        "formula": "EBIT / Interest Expense",
        "components": ["EBIT", "Interest Expense"],
        "unit": "x",
        "interpretation": "Ability to pay interest. Below 1.5x is risky",
    },
    "EPS": {
        "formula": "PAT / Total Shares Outstanding",
        "components": ["PAT", "Total Shares Outstanding"],
        "unit": "INR",
        "interpretation": "Profit allocated to each outstanding share",
    },
    "BVPS": {
        "formula": "Total Equity / Total Shares Outstanding",
        "components": ["Total Equity", "Total Shares Outstanding"],
        "unit": "INR",
        "interpretation": "Net asset value per share",
    },
    "NIM": {
        "formula": "(Interest Income - Interest Expense) / Average Interest-Earning Assets * 100",
        "components": ["Interest Income", "Interest Expense", "Average Interest-Earning Assets"],
        "unit": "%",
        "interpretation": "Banking metric — spread between lending and borrowing rates",
    },
    "GNPA Ratio": {
        "formula": "Gross NPAs / Gross Advances * 100",
        "components": ["Gross NPAs", "Gross Advances"],
        "unit": "%",
        "interpretation": "Asset quality indicator for banks. Lower is better",
    },
    "NNPA Ratio": {
        "formula": "Net NPAs / Net Advances * 100",
        "components": ["Net NPAs", "Net Advances"],
        "unit": "%",
        "interpretation": "Asset quality after provisions. Lower is better",
    },
    "Provision Coverage Ratio": {
        "formula": "Total Provisions / Gross NPAs * 100",
        "components": ["Total Provisions", "Gross NPAs"],
        "unit": "%",
        "interpretation": "Higher PCR means better provisioning against bad loans",
    },
}


# ---------------------------------------------------------------------------
# Sector-specific context — which metrics matter for which industry
# ---------------------------------------------------------------------------
# WHY?
#   Asking about NIM for an FMCG company makes no sense.
#   This helps the LLM understand which metrics are relevant for which sector,
#   and provide appropriate caveats when a metric doesn't apply.
# ---------------------------------------------------------------------------
SECTOR_METRICS = {
    "Banking & NBFC": {
        "key_metrics": ["NIM", "GNPA Ratio", "NNPA Ratio", "CRAR", "Provision Coverage Ratio", "ROA", "ROE", "Cost to Income Ratio"],
        "context": "For banks and NBFCs, asset quality (NPA ratios) and capital adequacy (CRAR) are critical regulatory metrics. NIM reflects core lending profitability.",
    },
    "IT & Technology": {
        "key_metrics": ["Revenue Growth", "EBITDA Margin", "PAT Margin", "ROE", "Employee Count", "Attrition Rate", "Revenue per Employee"],
        "context": "IT companies are asset-light. Focus on margins, growth rates, and workforce metrics. Debt levels are usually minimal.",
    },
    "Manufacturing & Industrial": {
        "key_metrics": ["EBITDA Margin", "ROA", "ROCE", "Debt to Equity", "Current Ratio", "Capacity Utilization", "Capex"],
        "context": "Capital-intensive sector. Asset efficiency (ROA, ROCE) and leverage (D/E) are critical. Watch for capex cycles.",
    },
    "FMCG & Consumer": {
        "key_metrics": ["Revenue Growth", "Gross Margin", "EBITDA Margin", "ROE", "Inventory Turnover", "Distribution Reach"],
        "context": "Typically low debt, high margins, and strong cash flows. Volume growth and market share are key drivers.",
    },
    "Pharma & Healthcare": {
        "key_metrics": ["R&D Spend", "EBITDA Margin", "ROE", "ANDA Pipeline", "Revenue Mix (Domestic vs Export)"],
        "context": "R&D spend drives future growth. Regulatory approvals (ANDA for US market) are key catalysts.",
    },
    "Real Estate & Hospitality": {
        "key_metrics": ["Revenue", "EBITDA Margin", "Debt to Equity", "Occupancy Rate", "ARR", "RevPAR", "NAV"],
        "context": "Highly leveraged sector. Occupancy rates and average room rates (ARR) matter for hospitality. Pre-sales and project pipeline for real estate.",
    },
    "Energy & Utilities": {
        "key_metrics": ["EBITDA", "Debt to Equity", "ROCE", "Plant Load Factor", "Capacity Addition", "Tariff Rates"],
        "context": "Regulated sector with high capex. Focus on leverage, capacity utilization, and regulatory tariff changes.",
    },
}


# ---------------------------------------------------------------------------
# Helper functions to retrieve knowledge
# ---------------------------------------------------------------------------
def get_formula(metric_name: str) -> dict | None:
    """Look up a formula by metric name (case-insensitive, checks aliases too)."""
    upper = metric_name.upper()
    for key, value in FINANCIAL_FORMULAS.items():
        if key.upper() == upper:
            return {key: value}
    for abbr, aliases in TERM_MAPPINGS.items():
        if upper == abbr.upper() or any(upper == a.upper() for a in aliases):
            if abbr in FINANCIAL_FORMULAS:
                return {abbr: FINANCIAL_FORMULAS[abbr]}
    return None


def get_term_aliases(term: str) -> list[str]:
    """Get all known aliases for a financial term."""
    upper = term.upper()
    for key, aliases in TERM_MAPPINGS.items():
        if key.upper() == upper or any(upper == a.upper() for a in aliases):
            return [key] + aliases
    return [term]


def get_sector_context(sector: str) -> dict | None:
    """Get sector-specific metric context."""
    for key, value in SECTOR_METRICS.items():
        if sector.lower() in key.lower():
            return {key: value}
    return None


def build_knowledge_context(metric_name: str = None, sector: str = None) -> str:
    """
    Build a knowledge context string to inject into the LLM prompt.

    Used by the context assembler when:
    - Retrieval returns weak/no results (fallback)
    - A calculation is detected (inject relevant formula)
    - Sector context helps ground the answer
    """
    parts = []

    if metric_name:
        formula = get_formula(metric_name)
        if formula:
            for name, details in formula.items():
                parts.append(
                    f"Formula — {name}: {details['formula']} "
                    f"(Unit: {details['unit']}). {details['interpretation']}"
                )
        aliases = get_term_aliases(metric_name)
        if len(aliases) > 1:
            parts.append(f"Note: {aliases[0]} is also known as: {', '.join(aliases[1:])}")

    if sector:
        ctx = get_sector_context(sector)
        if ctx:
            for name, details in ctx.items():
                parts.append(f"Sector context ({name}): {details['context']}")
                parts.append(f"Key metrics for {name}: {', '.join(details['key_metrics'])}")

    return "\n".join(parts)
