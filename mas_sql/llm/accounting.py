"""Explicit policies for preflight estimates and authoritative server usage."""
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class TokenAccounting:
    """Separate strict preflight validation from usage-based accounting."""

    mode: Literal["strict", "usage"] = "strict"
    prompt_margin_tokens: int = 0

    def __post_init__(self) -> None:
        """Reject unknown modes and invalid estimation margins."""
        if self.mode not in ("strict", "usage"):
            raise ValueError("token_accounting.mode must be strict or usage")
        if (isinstance(self.prompt_margin_tokens, bool)
                or not isinstance(self.prompt_margin_tokens, int)
                or self.prompt_margin_tokens < 0):
            raise ValueError("prompt_margin_tokens must be a nonnegative integer")

    def reserve(self, preflight_tokens: int) -> int:
        """Reserve estimated input tokens without charging them to usage."""
        return preflight_tokens + self.prompt_margin_tokens

    @property
    def strict_preflight(self) -> bool:
        """Return whether every prompt count must match server usage."""
        return self.mode == "strict"
