"""Fingerprint and record the one-shot workflow answer without gold access."""
import hashlib
from dataclasses import dataclass
from typing import Any
from mas_sql.agent.actions import Action


def sql_fingerprint(sql: str) -> str:
    """Identify exact SQL text, ignoring outer whitespace only."""
    return hashlib.sha256(sql.strip().encode("utf-8")).hexdigest()


@dataclass
class InvestigationTracker:
    """Record one baseline answer and its submission outcome."""

    evidence_available: bool
    first_sql: str | None = None
    version: int = 0
    submission_attempts: int = 0
    failed_submissions: int = 0

    def validate(self, action: Action) -> None:
        """Allow only a direct workflow submission."""
        if action.name != "submit_sql":
            raise ValueError("Workflow accepts only submit_sql")

    def capture_first(self, action: Action) -> bool:
        """Freeze the generated answer before execution feedback."""
        if self.first_sql is None:
            self.first_sql = action.sql
            return True
        return False

    def record(self, action: Action, result: dict[str, Any], observation_id: int) -> None:
        """Count the single baseline submission and execution status."""
        self.version = 1
        self.submission_attempts += 1
        self.failed_submissions += result["status"] != "ok"

    def summary(self) -> dict[str, Any]:
        """Return baseline telemetry compatible with the offline evaluator."""
        return {"format_version": 1, "candidate_versions": self.version,
                "diagnostic_calls": 0, "successful_diagnostic_calls": 0,
                "revision_count": 0, "revisions_citing_diagnostics": 0,
                "submission_attempts": self.submission_attempts,
                "failed_submissions": self.failed_submissions,
                "recovered_submission_error": False, "diagnostics": [], "revisions": [],
                "grounding_semantics": "No diagnostic or revision actions in one-shot workflow."}
