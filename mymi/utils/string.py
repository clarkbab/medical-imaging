from typing import *

def truncate_str(s: str, max_chars: Optional[int] = None) -> str:
    if max_chars is None:
        return s
    elif len(s) > max_chars:
        return s[:max_chars]

    return s
