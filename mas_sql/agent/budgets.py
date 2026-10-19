"""Shared episode limits independent of token accounting policy."""
import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Budget:
    """Bound model calls, SQL calls, tokens, context, and elapsed time."""

    max_llm_calls: int = 8
    max_tool_calls: int = 8
    max_total_tokens: int = 32_000
    max_output_tokens: int = 4096
    max_context_tokens: int = 32_768
    max_episode_s: float = 300

    def __post_init__(self) -> None:
        """Reject nonpositive, nonfinite, or nonintegral discrete limits."""
        for key, value in asdict(self).items():
            valid_type: bool = isinstance(value, (int, float)) if key == "max_episode_s" else isinstance(value, int)
            if isinstance(value, bool) or not valid_type or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Budget {key} must be positive and finite (integer for call/token caps)")
