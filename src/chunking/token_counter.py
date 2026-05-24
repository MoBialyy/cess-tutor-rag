"""
Token counter using INSTRUCTOR-XL's tokenizer.

We load only the tokenizer (~5MB), not the full model (~5GB), so this is
fast and cheap. The whole point is to count tokens *the way the embedding
model will see them later*, so chunks fit cleanly into its 512-token window.
"""

from transformers import AutoTokenizer


_TOKENIZER = None
_MODEL_ID = "hkunlp/instructor-xl"


def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(_MODEL_ID)
    return _TOKENIZER


def count_tokens(text: str) -> int:
    """Return the number of tokens INSTRUCTOR-XL will see for this string."""
    if not text:
        return 0
    # add_special_tokens=False -> just the raw token count for the content
    return len(_get_tokenizer().encode(text, add_special_tokens=False))
