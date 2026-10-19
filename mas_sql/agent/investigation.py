"""Track candidate versions, diagnostic checks, and verifiable source references."""
import hashlib
from dataclasses import dataclass, field
from typing import Any
from mas_sql.agent.actions import Action


def sql_fingerprint(sql: str) -> str:
    """Identify exact SQL text, ignoring outer whitespace only."""
    return hashlib.sha256(sql.strip().encode("utf-8")).hexdigest()


@dataclass
class InvestigationTracker:
    """Record observable research behavior without acting as a gold oracle."""

    evidence_available: bool
    first_sql: str | None = None
    current_sql: str | None = None
    version: int = 0
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    revisions: list[dict[str, Any]] = field(default_factory=list)
    observations: dict[int, dict[str, Any]] = field(default_factory=dict)
    submission_attempts: int = 0
    failed_submissions: int = 0
    accepted_submission: bool = False

    def _resolve(self, source: str) -> int | None:
        """Accept only task/schema sources or already executed observation IDs."""
        if source in ("question", "schema"):
            return None
        if source == "evidence" and self.evidence_available:
            return None
        if source.startswith("observation:"):
            suffix: str = source.removeprefix("observation:")
            if suffix.isascii() and suffix.isdecimal() and str(int(suffix)) == suffix:
                number: int = int(suffix)
                if number in self.observations:
                    return number
        raise ValueError(f"Unavailable revision source: {source}; use question, evidence, schema or observation:N")

    def validate(self, action: Action) -> None:
        """Require an initial candidate and explicit grounds for every text change."""
        if action.is_diagnostic:
            if self.first_sql is None:
                raise ValueError("First action must propose an answer candidate before SQL feedback")
            return
        if self.current_sql is not None and sql_fingerprint(action.sql) != sql_fingerprint(self.current_sql):
            if action.revision is None:
                raise ValueError("Changed candidate requires revision with a short reason and sources")
        if action.revision is not None:
            for source in action.revision.sources:
                self._resolve(source)

    def capture_first(self, action: Action) -> bool:
        """Freeze the first valid answer candidate before execution feedback."""
        if not action.is_diagnostic and self.first_sql is None:
            self.first_sql = action.sql
            return True
        return False

    def record(self, action: Action, result: dict[str, Any], observation_id: int) -> dict[str, Any] | None:
        """Associate one executed SQL with its version, check, and references."""
        revision_record: dict[str, Any] | None = None
        if action.is_diagnostic:
            self.diagnostics.append({"observation_id": observation_id, "check": action.check,
                                     "status": result["status"], "referenced_by_revisions": []})
        else:
            changed: bool = self.current_sql is not None and sql_fingerprint(action.sql) != sql_fingerprint(self.current_sql)
            if self.current_sql is None or changed:
                old_version: int = self.version
                self.version += 1
                if changed:
                    assert action.revision is not None  # Validated before execution.
                    refs: list[int] = [n for s in action.revision.sources if (n := self._resolve(s)) is not None]
                    diagnostic_refs: list[int] = [n for n in refs if self.observations[n]["purpose"] == "diagnostic"]
                    revision_record = {"from_version": old_version, "version": self.version,
                                       "observation_id": observation_id, **action.revision.payload(),
                                       "cited_observations": refs, "cited_diagnostics": diagnostic_refs}
                    self.revisions.append(revision_record)
                    for diagnostic in self.diagnostics:
                        if diagnostic["observation_id"] in diagnostic_refs:
                            diagnostic["referenced_by_revisions"].append(self.version)
            self.current_sql = action.sql
        self.observations[observation_id] = {"purpose": action.purpose or "submission", "status": result["status"]}
        if action.name == "submit_sql":
            self.submission_attempts += 1
            self.accepted_submission = result["status"] == "ok"
            self.failed_submissions += result["status"] != "ok"
        return revision_record

    def summary(self) -> dict[str, Any]:
        """Return behavior telemetry; reference validity is not semantic proof."""
        return {"format_version": 1, "candidate_versions": self.version,
                "diagnostic_calls": len(self.diagnostics),
                "successful_diagnostic_calls": sum(d["status"] == "ok" for d in self.diagnostics),
                "revision_count": len(self.revisions),
                "revisions_citing_diagnostics": sum(bool(r["cited_diagnostics"]) for r in self.revisions),
                "submission_attempts": self.submission_attempts,
                "failed_submissions": self.failed_submissions,
                "recovered_submission_error": self.failed_submissions > 0 and self.accepted_submission,
                "diagnostics": self.diagnostics, "revisions": self.revisions,
                "grounding_semantics": "self-reported reason; source existence checked, truth not verified"}
