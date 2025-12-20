from __future__ import annotations
from typing import Dict
import tiktoken

_DEFAULT_ENCODING = "cl100k_base"

def count_tokens(text: str, *, encoding: str = _DEFAULT_ENCODING) -> int:
    if not text:
        return 0
    enc = tiktoken.get_encoding(encoding)
    return len(enc.encode(text))

def count_turn_tokens(
    *,
    user_text: str,
    assistant_text: str,
    encoding: str = _DEFAULT_ENCODING,
) -> Dict[str, int]:
    user_tokens = count_tokens(user_text, encoding=encoding)
    assistant_tokens = count_tokens(assistant_text, encoding=encoding)

    return {
        "input_total": user_tokens,
        "output_total": assistant_tokens,
        "total": user_tokens + assistant_tokens,
        "encoding": encoding,
    }
