"""Pure versioned state and transition validation with no model or SQL access."""
from dataclasses import dataclass, field
from typing import Any, Literal
from mas_sql.agent.actions import Revision
from mas_sql.agent.investigation import sql_fingerprint
from mas_sql.agent.sequential.actions import Issue, Role, TeamAction
from mas_sql.agent.sequential.grounding import validate_grounds

Phase = Literal["AUTHOR_INITIAL", "CRITIC_REVIEW", "AUTHOR_DECIDE"]


@dataclass(frozen=True)
class Candidate:
    """Bind exact candidate text and its execution to an immutable version."""

    version: int
    sql: str
    fingerprint: str
    observation_id: int
    revision: Revision | None

    def payload(self) -> dict[str, Any]:
        """Expose only public candidate metadata."""
        return {"version": self.version, "sql": self.sql, "sql_fingerprint": self.fingerprint,
                "execution_observation_id": self.observation_id,
                "revision": self.revision.payload() if self.revision is not None else None}


@dataclass(frozen=True)
class Review:
    """Bind a public verdict to one exact candidate version."""

    id: str
    candidate_version: int
    fingerprint: str
    verdict: str
    issues: tuple[Issue, ...]

    def payload(self) -> dict[str, Any]:
        """Serialize an immutable report for both role contexts."""
        return {"id": self.id, "candidate_version": self.candidate_version,
                "sql_fingerprint": self.fingerprint, "verdict": self.verdict,
                "issues": [issue.payload() for issue in self.issues]}


@dataclass
class EpisodeState:
    """Own public state; source existence never proves semantic correctness."""

    evidence_available: bool
    source_texts: dict[str, str] = field(default_factory=dict)
    phase: Phase = "AUTHOR_INITIAL"
    first_sql: str | None = None
    candidates: list[Candidate] = field(default_factory=list)
    observations: dict[int, dict[str, Any]] = field(default_factory=dict)
    reviews: list[Review] = field(default_factory=list)
    current_review: Review | None = None
    diagnostics_used: dict[Role, bool] = field(default_factory=lambda: {"author": False, "critic": False})
    revisions: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    submissions: list[dict[str, Any]] = field(default_factory=list)

    @property
    def actor(self) -> Role:
        """Return the fixed next role without dynamic task routing."""
        return "critic" if self.phase == "CRITIC_REVIEW" else "author"

    @property
    def current(self) -> Candidate | None:
        """Return the current candidate, never a diagnostic SQL."""
        return self.candidates[-1] if self.candidates else None

    def available_sources(self) -> list[str]:
        """List every public source in the same order used by context snapshots."""
        available: list[str] = ["question", "schema"]
        if self.evidence_available:
            available.append("evidence")
        available.extend(f"observation:{number}" for number in self.observations)
        available.extend(review.id for review in self.reviews)
        return available

    def _sources(self, sources: tuple[str, ...]) -> None:
        """Resolve only existing public references already available to both roles."""
        available: set[str] = set(self.available_sources())
        if not set(sources) <= available:
            raise ValueError("Unavailable sources: " + ", ".join(sorted(set(sources) - available)))

    def _target(self, action: TeamAction) -> Candidate:
        """Reject stale versions and altered text identities."""
        current: Candidate | None = self.current
        if (current is None or action.candidate_version != current.version
                or action.sql_fingerprint != current.fingerprint):
            raise ValueError("Action must target the current candidate version and fingerprint")
        return current

    def validate(self, action: TeamAction, role: Role) -> None:
        """Check all state-dependent preconditions without modifying state."""
        if role != self.actor:
            raise ValueError("Wrong active role")
        current: Candidate | None = self.current
        if action.purpose == "candidate":
            if role != "author":
                raise ValueError("Only Author may change a candidate")
            assert action.sql is not None
            if current is None:
                if action.revision is not None:
                    raise ValueError("First candidate has no prior version to revise")
            else:
                if self.current_review is None:
                    raise ValueError("Current candidate must be reviewed before revision")
                if sql_fingerprint(action.sql) == current.fingerprint:
                    raise ValueError("Unchanged candidate: submit the reviewed version instead")
                if action.revision is None:
                    raise ValueError("Changed candidate requires revision reason and sources")
                self._sources(action.revision.sources)
                validate_grounds(action.revision.basis, action.revision.grounds, action.revision.sources,
                                 self.source_texts, self.observations, current.version)
            return
        if action.purpose == "diagnostic":
            if current is None or (role == "author" and self.current_review is None):
                raise ValueError("First action must propose an answer candidate before SQL feedback")
            if self.diagnostics_used[role]:
                raise ValueError("Only one diagnostic is allowed in the current role turn")
            if role == "critic":
                self._target(action)
            return
        if action.name == "review_sql":
            if role != "critic":
                raise ValueError("Only Critic may issue a review")
            current = self._target(action)
            for issue in action.issues:
                self._sources(issue.sources)
                validate_grounds(issue.basis, issue.grounds, issue.sources, self.source_texts,
                                 self.observations, current.version)
                if issue.category == "execution" and issue.basis != "execution":
                    raise ValueError("Execution issues require the actual failed execution as grounds")
            if action.verdict == "approve" and self.observations[current.observation_id]["result"]["status"] != "ok":
                raise ValueError("A candidate with failed execution cannot be approved; report the execution issue")
            return
        if action.name == "submit_sql":
            if role != "author":
                raise ValueError("Only Author may submit")
            self._target(action)
            review: Review | None = self.current_review
            if review is None:
                raise ValueError("Submission requires a review of the current version")
            if review.verdict == "request_changes":
                response = action.response_to_review
                if (response is None or response.review_id != review.id
                        or set(response.issue_ids) != {issue.id for issue in review.issues}):
                    raise ValueError("Submission with disagreement must respond to every issue in the current review")
                self._sources(response.sources)
                assert current is not None
                validate_grounds(response.basis, response.grounds, response.sources, self.source_texts,
                                 self.observations, current.version)
            elif action.response_to_review is not None:
                raise ValueError("response_to_review is only used for a request_changes verdict")
            return
        raise ValueError("Unsupported action in this phase")

    def propose(self, action: TeamAction, observation_id: int) -> Candidate:
        """Freeze an answer before execution and invalidate prior approval."""
        assert action.sql is not None
        old: Candidate | None = self.current
        candidate: Candidate = Candidate(len(self.candidates) + 1, action.sql, sql_fingerprint(action.sql),
                                         observation_id, action.revision)
        if self.first_sql is None:
            self.first_sql = action.sql
        if old is not None:
            assert action.revision is not None
            refs: list[int] = [int(source.removeprefix("observation:")) for source in action.revision.sources
                               if source.startswith("observation:")]
            diagnostic_refs: list[int] = [number for number in refs if self.observations[number]["purpose"] == "diagnostic"]
            self.revisions.append({"from_version": old.version, "version": candidate.version,
                                   "observation_id": observation_id, **action.revision.payload(),
                                   "cited_observations": refs, "cited_diagnostics": diagnostic_refs})
            for diagnostic in self.diagnostics:
                if diagnostic["observation_id"] in diagnostic_refs:
                    diagnostic["referenced_by_revisions"].append(candidate.version)
        self.candidates.append(candidate)
        self.current_review = None
        self.diagnostics_used["critic"] = False
        return candidate

    def observe(self, role: Role, action: TeamAction, result: dict[str, Any], observation_id: int) -> None:
        """Publish a bounded tool observation and advance only candidate execution."""
        assert self.current is not None
        self.observations[observation_id] = {"observation_id": observation_id, "role": role,
            "purpose": action.purpose or "submission", "candidate_version": self.current.version,
            "sql": action.sql if action.sql is not None else self.current.sql,
            "sql_fingerprint": sql_fingerprint(action.sql if action.sql is not None else self.current.sql),
            "candidate_sql_fingerprint": self.current.fingerprint,
            "check": action.check, "result": result}
        if action.purpose == "candidate":
            self.phase = "CRITIC_REVIEW"
        elif action.purpose == "diagnostic":
            self.diagnostics_used[role] = True
            self.diagnostics.append({"observation_id": observation_id, "role": role, "check": action.check,
                                     "status": result["status"], "referenced_by_revisions": []})

    def accept_review(self, action: TeamAction) -> Review:
        """Publish a validated verdict and return control to Author."""
        assert self.current is not None and action.verdict is not None
        review: Review = Review(f"review:{len(self.reviews) + 1}", self.current.version,
                                self.current.fingerprint, action.verdict, action.issues)
        self.reviews.append(review)
        self.current_review = review
        self.diagnostics_used["author"] = False
        self.phase = "AUTHOR_DECIDE"
        return review

    def investigation(self) -> dict[str, Any]:
        """Expose baseline-compatible behavior counters and explicit limitations."""
        accepted: bool = any(submission["accepted"] for submission in self.submissions)
        failed: int = sum(not submission["accepted"] for submission in self.submissions)
        return {"format_version": 1, "candidate_versions": len(self.candidates),
                "diagnostic_calls": len(self.diagnostics),
                "successful_diagnostic_calls": sum(item["status"] == "ok" for item in self.diagnostics),
                "revision_count": len(self.revisions),
                "revisions_citing_diagnostics": sum(bool(item["cited_diagnostics"]) for item in self.revisions),
                "submission_attempts": len(self.submissions), "failed_submissions": failed,
                "recovered_submission_error": failed > 0 and accepted,
                "diagnostics": self.diagnostics, "revisions": self.revisions,
                "grounding_semantics": "Verbatim direct anchors, basis/provenance/freshness checked; entailment and SQL correctness not verified"}
