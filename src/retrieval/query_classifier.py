"""
Query classifier for adaptive hybrid retrieval.

Per report section 4.3.10, the relative weight of sparse (BM25) vs dense
(INSTRUCTOR) retrieval should depend on query type:

  - Code/syntax queries        -> favor BM25 (lexical precision matters)
                                  alpha = 0.6
  - Conceptual queries         -> favor dense (semantic matters more)
                                  alpha = 0.4
  - Definitional ("what is X") -> balanced
                                  alpha = 0.5

This is a simple keyword-based router. It runs in microseconds and gives
the retriever a single number that controls how the two scores are fused.
"""

import re

CODE_KEYWORDS = {
    "code", "syntax", "function", "method", "import", "print", "return",
    "class", "variable", "loop", "for loop", "while loop", "if statement",
    "regex", "regular expression", "error", "exception", "compile",
    "library", "package", "module", "api", "argument", "parameter",
    "example", "snippet", "implementation",
}

CONCEPTUAL_KEYWORDS = {
    "why", "how", "explain", "explanation", "describe", "compare",
    "difference", "versus", "vs", "vs.", "purpose", "reason", "concept",
    "understand", "intuition", "advantage", "disadvantage", "benefit",
    "pros", "cons", "tradeoff", "trade-off",
}

DEFINITION_KEYWORDS = {
    "what", "define", "definition", "meaning", "is a", "is an",
}


# Regex hints that strongly imply code intent regardless of keywords:
#   - presence of backticks/code delimiters
#   - parens with identifiers, e.g. word_tokenize()
#   - dotted access like nltk.tokenize
_CODE_REGEX = re.compile(r"`[^`]+`|\w+\([^)]*\)|\w+\.\w+")


def classify_query(query: str) -> dict:
    """
    Return a dict with the inferred query type and the sparse weight alpha.
        {"type": "code" | "conceptual" | "definitional", "alpha": float}
    """
    q = query.lower().strip()

    if _CODE_REGEX.search(q):
        return {"type": "code", "alpha": 0.6}

    words = set(re.findall(r"[a-zA-Z][a-zA-Z\-']+", q))

    # Check conceptual signals FIRST. Phrases like "explain how X" or
    # "compare X and Y" are explanatory regardless of whether the topic
    # is technical (e.g. "explain how regex works" is conceptual, not code).
    if words & CONCEPTUAL_KEYWORDS:
        return {"type": "conceptual", "alpha": 0.4}

    if words & CODE_KEYWORDS:
        return {"type": "code", "alpha": 0.6}

    if words & DEFINITION_KEYWORDS:
        return {"type": "definitional", "alpha": 0.5}

    # Default: balanced
    return {"type": "definitional", "alpha": 0.5}
