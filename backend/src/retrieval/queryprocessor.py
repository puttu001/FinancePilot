import re
from typing import List, Dict, Tuple

class QueryEnhancer:
    """Enhance queries for better retrieval and reduce ambiguity"""

    def __init__(self):
        self.financial_context = {
            "profit": ["PAT", "profit after tax", "net profit", "net income"],
            "revenue": ["total revenue", "total income", "sales", "turnover"],
            "debt": ["total debt", "borrowings", "total borrowings"],
            "npa": ["non-performing assets", "GNPA", "NNPA", "bad loans"],
            "margin": ["net margin", "profit margin", "operating margin"],
        }

    def enhance_query(self, query: str) -> str:
        """
        Enhance query with financial context and expand abbreviations
        """
        enhanced = query

        # Expand common abbreviations
        for short, expansions in self.financial_context.items():
            if short in query.lower():
                enhanced += f" {' '.join(expansions)}"

        # Add context for ratio queries
        if 'ratio' in query.lower() or 'calculate' in query.lower():
            enhanced += " financial metrics calculation formula components"

        return enhanced

    def decompose_complex_query(self, query: str) -> List[str]:
        """
        Break down complex queries into simpler sub-queries
        """
        # Check for multi-part questions
        if ' and ' in query.lower() or ' also ' in query.lower():
            parts = re.split(r'\s+and\s+|\s+also\s+', query, flags=re.IGNORECASE)
            return [part.strip() + '?' if not part.endswith('?') else part.strip() for part in parts]

        # Check for comparison queries
        if 'compare' in query.lower() or 'vs' in query.lower() or 'versus' in query.lower():
            entities = re.findall(r'between\s+(\w+)\s+and\s+(\w+)', query, re.IGNORECASE)
            if entities:
                entity1, entity2 = entities[0]
                base_q = query.replace(f'between {entity1} and {entity2}', '').strip()
                return [
                    f"{base_q} for {entity1}",
                    f"{base_q} for {entity2}"
                ]

        return [query]

    def detect_calculation_request(self, query: str) -> Tuple[bool, str, List[str]]:
        """
        Detect if query requires calculation

        Returns:
            (is_calculation, metric_name, required_inputs)
        """
        calc_keywords = ['calculate', 'compute', 'find', 'what is the']

        if not any(kw in query.lower() for kw in calc_keywords):
            return False, None, []

        # Detect metric type
        metric_patterns = {
            'ROA': ['roa', 'return on assets'],
            'ROE': ['roe', 'return on equity'],
            'D/E': ['debt to equity', 'd/e ratio', 'debt equity'],
            'Net Margin': ['net margin', 'profit margin', 'net profit margin'],
            'Current Ratio': ['current ratio', 'liquidity ratio'],
        }

        for metric, keywords in metric_patterns.items():
            if any(kw in query.lower() for kw in keywords):
                # Define required inputs for each metric
                required_inputs = {
                    'ROA': ['PAT', 'Total Assets'],
                    'ROE': ['PAT', 'Total Equity'],
                    'D/E': ['Total Debt', 'Total Equity'],
                    'Net Margin': ['PAT', 'Revenue'],
                    'Current Ratio': ['Current Assets', 'Current Liabilities'],
                }
                return True, metric, required_inputs.get(metric, [])

        return False, None, []

