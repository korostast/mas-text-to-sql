"""Gold-free downstream token reservations for the sequential state machine."""
from dataclasses import dataclass
from typing import Any, Mapping
from mas_sql.agent.sequential.actions import Role

CLOSURE_VERSION: str = "closure-reserve-v2"
MIN_COMPLETION_TOKENS: int = 1024
SNAPSHOT_GROWTH_TOKENS: int = 1536


@dataclass(frozen=True)
class ClosurePlan:
    """Reserve future work without charging estimates as actual usage."""

    holdback_tokens: int
    min_output_tokens: int
    allowed: dict[str, bool]
    future_costs: dict[str, int]
    prompt_reserves: dict[Role, int]
    growth_tokens: int
    repair_tokens: int
    repair_turns: int

    def payload(self) -> dict[str, Any]:
        """Expose the allocation and its non-guarantee in packets and traces."""
        return {"version": CLOSURE_VERSION, "holdback_tokens": self.holdback_tokens,
                "min_output_tokens": self.min_output_tokens, "future_costs": self.future_costs,
                "prompt_reserves": self.prompt_reserves, "snapshot_growth_tokens": self.growth_tokens,
                "finalization_repair_tokens": self.repair_tokens, "finalization_repair_turns": self.repair_turns,
                "semantics": "Estimated future prompts plus reserved completion floors; not prepaid usage or guaranteed semantic/transport success"}


class ClosurePlanner:
    """Prefer an explicit finish over an optional unaffordable new branch."""

    @staticmethod
    def candidate_cost(prompt_reserves: Mapping[Role, int], floor: int, growth: int) -> int:
        """Reserve a verdict, explicit submit and one bounded finalization repair."""
        return (prompt_reserves["critic"] + growth + floor
                + prompt_reserves["author"] + 2 * growth + floor
                + prompt_reserves["author"] + 3 * growth + floor)

    @staticmethod
    def plan(role: Role, initial: bool, remaining_tokens: int, max_output_tokens: int,
             prompt_reserves: Mapping[Role, int], count_allowed: Mapping[str, Any],
             *, repairing: bool = False) -> ClosurePlan:
        """Budget each optional branch through its mandatory return to Author."""
        floor: int = min(MIN_COMPLETION_TOKENS, max_output_tokens)
        growth: int = SNAPSHOT_GROWTH_TOKENS
        author: int = prompt_reserves["author"]
        # Each path includes ONE terminal repair, not a recursive repair reserve.
        candidate: int = ClosurePlanner.candidate_cost(prompt_reserves, floor, growth)
        author_diagnostic: int = author + growth + floor + author + 2 * growth + floor
        critic_diagnostic: int = candidate
        repair: int = 0 if repairing else author + growth + floor
        review: int = author_diagnostic
        mandatory: int = candidate if initial else review if role == "critic" else repair
        costs: dict[str, int] = {"candidate": candidate,
                                "diagnostic": author_diagnostic if role == "author" else critic_diagnostic,
                                "review_sql": review, "submit_sql": repair}
        current_minimum: int = prompt_reserves[role] + floor
        allowed: dict[str, bool] = {
            key: bool(count_allowed.get(key)) and (not repairing or key == "submit_sql")
                 and remaining_tokens >= current_minimum + cost
            for key, cost in costs.items()}
        # Hold the largest affordable branch advertised to this call, not all branches' sum.
        holdback: int = max([mandatory] + [costs[key] for key, value in allowed.items() if value])
        return ClosurePlan(holdback, floor, allowed, costs, dict(prompt_reserves), growth, repair, 0 if repairing else 1)