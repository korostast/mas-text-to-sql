"""Summaries of observable checks and offline outcomes, without causal claims."""
from dataclasses import asdict, dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class BehaviorSnapshot:
    """Represent the countable subset of one episode's investigation trace."""

    diagnostic_calls: int
    successful_diagnostic_calls: int
    revision_count: int
    revisions_citing_diagnostics: int
    submission_attempts: int
    failed_submissions: int
    recovered_submission_error: bool

    @classmethod
    def read(cls, episode: dict[str, Any]) -> "BehaviorSnapshot | None":
        """Return None for old artifacts rather than treating missing data as zero."""
        raw: Any = episode.get("investigation")
        if raw is None:
            return None
        if not isinstance(raw, dict) or raw.get("format_version") != 1:
            raise ValueError("Unsupported investigation telemetry")
        values: dict[str, Any] = {}
        for name in cls.__dataclass_fields__:
            value: Any = raw.get(name)
            valid: bool = isinstance(value, bool) if name == "recovered_submission_error" else (
                isinstance(value, int) and not isinstance(value, bool) and value >= 0)
            if not valid:
                raise ValueError(f"Invalid investigation field: {name}")
            values[name] = value
        return cls(**values)


def investigation_summary(episodes: Sequence[dict[str, Any]], scored: bool = False) -> dict[str, Any]:
    """Count behavior and paired EX outcomes without asserting diagnostic utility."""
    available: list[tuple[dict[str, Any], BehaviorSnapshot]] = []
    for episode in episodes:
        snapshot: BehaviorSnapshot | None = BehaviorSnapshot.read(episode)
        if snapshot is not None:
            available.append((episode, snapshot))
    summary: dict[str, Any] = {"coverage_episodes": len(available), "total_episodes": len(episodes),
                               "interpretation": "Observed associations, not proof that a diagnostic caused a repair."}
    names: tuple[str, ...] = tuple(BehaviorSnapshot.__dataclass_fields__)
    for name in names:
        summary[name if name != "recovered_submission_error" else "episodes_recovering_submission_error"] = (
            sum(asdict(snapshot)[name] for _, snapshot in available) if available else None)
    summary["episodes_with_diagnostics"] = sum(s.diagnostic_calls > 0 for _, s in available) if available else None
    if scored:
        diagnostics: list[dict[str, Any]] = [e for e, s in available if s.diagnostic_calls > 0]
        summary["repairs_in_episodes_with_diagnostics"] = sum(e["EX_first"] == 0 and e["EX_final"] == 1 for e in diagnostics) if available else None
        summary["regressions_in_episodes_with_diagnostics"] = sum(e["EX_first"] == 1 and e["EX_final"] == 0 for e in diagnostics) if available else None
        summary["unrepaired_failures_with_diagnostics"] = sum(e["EX_first"] == 0 and e["EX_final"] == 0 for e in diagnostics) if available else None
    return summary
