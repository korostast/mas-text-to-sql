"""Strict public action contracts for the sequential Author-Critic protocol."""
import json
import re
from dataclasses import dataclass
from typing import Any, Literal
from mas_sql.agent.actions import ActionParser, Revision
from mas_sql.agent.sequential.grounding import Ground, parse_basis, parse_grounds

Role = Literal["author", "critic"]
CATEGORIES: frozenset[str] = frozenset({"projection", "filter", "join", "aggregation", "ties", "execution", "other"})


@dataclass(frozen=True)
class GroundedRevision(Revision):
    """Require an Author-owned direct premise, not only a peer citation."""

    basis: str
    grounds: tuple[Ground, ...]

    def payload(self) -> dict[str, Any]:
        """Preserve baseline revision fields and add inspectable evidence."""
        return {**super().payload(), "basis": self.basis, "grounds": [g.payload() for g in self.grounds]}


@dataclass(frozen=True)
class Issue:
    """Represent a concise public objection, not a reasoning transcript."""

    id: str
    category: str
    message: str
    sources: tuple[str, ...]
    basis: str
    grounds: tuple[Ground, ...]

    def payload(self) -> dict[str, Any]:
        """Serialize a public issue for exchange and audit."""
        return {"id": self.id, "category": self.category, "message": self.message, "sources": list(self.sources),
                "basis": self.basis, "grounds": [g.payload() for g in self.grounds]}


@dataclass(frozen=True)
class ReviewResponse:
    """Record an author's explicit response to all current objections."""

    review_id: str
    issue_ids: tuple[str, ...]
    reason: str
    sources: tuple[str, ...]
    basis: str
    grounds: tuple[Ground, ...]

    def payload(self) -> dict[str, Any]:
        """Serialize the stated basis without claiming it is correct."""
        return {"review_id": self.review_id, "issue_ids": list(self.issue_ids),
                "reason": self.reason, "sources": list(self.sources),
                "basis": self.basis, "grounds": [g.payload() for g in self.grounds]}


@dataclass(frozen=True)
class TeamAction:
    """Represent one validated role action before state-dependent validation."""

    name: str
    sql: str | None = None
    purpose: str | None = None
    check: str | None = None
    revision: GroundedRevision | None = None
    candidate_version: int | None = None
    sql_fingerprint: str | None = None
    verdict: str | None = None
    issues: tuple[Issue, ...] = ()
    response_to_review: ReviewResponse | None = None

    def payload(self) -> dict[str, Any]:
        """Return the exact accepted public action shape."""
        value: dict[str, Any] = {"action": self.name}
        for key in ("sql", "purpose", "check", "candidate_version", "sql_fingerprint", "verdict"):
            item: Any = getattr(self, key)
            if item is not None:
                value[key] = item
        if self.revision is not None:
            value["revision"] = self.revision.payload()
        if self.verdict is not None:
            value["issues"] = [issue.payload() for issue in self.issues]
        if self.response_to_review is not None:
            value["response_to_review"] = self.response_to_review.payload()
        return value


class TeamActionParser:
    """Reject unknown fields, duplicate keys, wrong roles and oversized text."""

    @staticmethod
    def _keys(value: dict[str, Any], required: set[str], optional: set[str] | None = None) -> None:
        """Require all mandatory keys and reject any unrecognized key."""
        if not required <= set(value) or set(value) - required - (optional or set()):
            raise ValueError(f"Expected fields {sorted(required)}; optional {sorted(optional or set())}")

    @staticmethod
    def _sources(value: object) -> tuple[str, ...]:
        """Validate a bounded list; state validation checks actual availability."""
        return ActionParser._revision({"reason": "Public source reference", "sources": value}).sources

    @staticmethod
    def _target(value: dict[str, Any]) -> tuple[int, str]:
        """Validate a positive version and an exact SHA256 target."""
        version: Any = value.get("candidate_version")
        fingerprint: Any = value.get("sql_fingerprint")
        if isinstance(version, bool) or not isinstance(version, int) or version < 1:
            raise ValueError("candidate_version must be a positive integer")
        if not isinstance(fingerprint, str) or re.fullmatch(r"[0-9a-f]{64}", fingerprint) is None:
            raise ValueError("sql_fingerprint must be the runtime-provided SHA256")
        return version, fingerprint

    @classmethod
    def _issue(cls, value: object) -> Issue:
        """Validate one stable local issue identifier and public statement."""
        if not isinstance(value, dict):
            raise ValueError("Each issue must be an object")
        cls._keys(value, {"id", "category", "message", "sources", "basis", "grounds"})
        ident: Any = value["id"]
        if not isinstance(ident, str) or re.fullmatch(r"I[1-9][0-9]{0,2}", ident) is None:
            raise ValueError("Issue id must be I1..I999")
        if not isinstance(value["category"], str) or value["category"] not in CATEGORIES:
            raise ValueError("Unsupported issue category")
        return Issue(ident, value["category"], ActionParser._text(value["message"], "issue.message", 600),
                     cls._sources(value["sources"]), parse_basis(value["basis"]), parse_grounds(value["grounds"]))

    @classmethod
    def _revision(cls, value: object) -> GroundedRevision:
        """Parse a sequential-only revision without changing baseline contracts."""
        if not isinstance(value, dict):
            raise ValueError("revision must be an object")
        cls._keys(value, {"reason", "sources", "basis", "grounds"})
        revision: Revision = ActionParser._revision({key: value[key] for key in ("reason", "sources")})
        return GroundedRevision(revision.reason, revision.sources, parse_basis(value["basis"]), parse_grounds(value["grounds"]))

    @classmethod
    def _response(cls, value: object) -> ReviewResponse:
        """Validate an author's concise response to a specific review."""
        if not isinstance(value, dict):
            raise ValueError("response_to_review must be an object")
        cls._keys(value, {"review_id", "issue_ids", "reason", "sources", "basis", "grounds"})
        ident: str = ActionParser._text(value["review_id"], "review_id", 80)
        ids: Any = value["issue_ids"]
        if (not isinstance(ids, list) or not 1 <= len(ids) <= 8
                or not all(isinstance(i, str) and re.fullmatch(r"I[1-9][0-9]{0,2}", i) for i in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError("issue_ids must contain 1..8 unique issue IDs")
        return ReviewResponse(ident, tuple(ids), ActionParser._text(value["reason"], "response reason", 600),
                              cls._sources(value["sources"]), parse_basis(value["basis"]), parse_grounds(value["grounds"]))

    @classmethod
    def parse(cls, content: str, role: Role) -> TeamAction:
        """Parse exactly one role-specific action without repairs or inference."""
        value: Any = json.loads(content, object_pairs_hook=ActionParser._unique)
        if not isinstance(value, dict):
            raise ValueError("Expected one JSON action object")
        if role not in ("author", "critic"):
            raise ValueError("Unknown role")
        name: Any = value.get("action")
        if name == "execute_sql":
            purpose: Any = value.get("purpose")
            sql: str = ActionParser._text(value.get("sql"), "sql", 100_000)
            if role == "author" and purpose == "candidate":
                cls._keys(value, {"action", "sql", "purpose"}, {"revision"})
                revision: GroundedRevision | None = cls._revision(value["revision"]) if "revision" in value else None
                return TeamAction(name, sql=sql, purpose=purpose, revision=revision)
            if purpose == "diagnostic":
                required: set[str] = {"action", "sql", "purpose", "check"}
                if role == "critic":
                    required |= {"candidate_version", "sql_fingerprint"}
                cls._keys(value, required)
                target: tuple[int | None, str | None] = cls._target(value) if role == "critic" else (None, None)
                return TeamAction(name, sql=sql, purpose=purpose,
                                  check=ActionParser._text(value["check"], "check", 600),
                                  candidate_version=target[0], sql_fingerprint=target[1])
        elif name == "submit_sql" and role == "author":
            cls._keys(value, {"action", "candidate_version", "sql_fingerprint"}, {"response_to_review"})
            version, fingerprint = cls._target(value)
            response: ReviewResponse | None = cls._response(value["response_to_review"]) if "response_to_review" in value else None
            return TeamAction(name, candidate_version=version, sql_fingerprint=fingerprint, response_to_review=response)
        elif name == "review_sql" and role == "critic":
            cls._keys(value, {"action", "candidate_version", "sql_fingerprint", "verdict", "issues"})
            version, fingerprint = cls._target(value)
            verdict: Any = value["verdict"]
            raw: Any = value["issues"]
            if verdict not in ("approve", "request_changes") or not isinstance(raw, list) or len(raw) > 8:
                raise ValueError("Review needs verdict approve/request_changes and at most 8 issues")
            issues: tuple[Issue, ...] = tuple(cls._issue(item) for item in raw)
            if len({issue.id for issue in issues}) != len(issues):
                raise ValueError("Issue IDs must be unique within a review")
            if (verdict == "approve") != (not issues):
                raise ValueError("approve requires no issues; request_changes requires at least one")
            return TeamAction(name, candidate_version=version, sql_fingerprint=fingerprint, verdict=verdict, issues=issues)
        raise ValueError(f"Action is not allowed for {role}")
