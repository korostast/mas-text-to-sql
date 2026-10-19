"""Validate SQL actions and concise, externally inspectable revision grounds."""
import json
from dataclasses import dataclass
from typing import Any, Literal

ActionName = Literal["execute_sql", "submit_sql"]
Purpose = Literal["candidate", "diagnostic"]


@dataclass(frozen=True)
class Revision:
    """Identify the stated reason and sources for a candidate change."""

    reason: str
    sources: tuple[str, ...]

    def payload(self) -> dict[str, Any]:
        """Serialize revision metadata without claiming semantic correctness."""
        return {"reason": self.reason, "sources": list(self.sources)}


@dataclass(frozen=True)
class Action:
    """Represent one validated SQL action, not a hidden reasoning trace."""

    name: ActionName
    sql: str
    purpose: Purpose | None = None
    check: str | None = None
    revision: Revision | None = None

    @property
    def is_diagnostic(self) -> bool:
        """Return whether this SQL investigates a hypothesis rather than answers."""
        return self.purpose == "diagnostic"

    def payload(self) -> dict[str, Any]:
        """Return the public JSON action shape used in event logs."""
        result: dict[str, Any] = {"action": self.name, "sql": self.sql}
        if self.purpose is not None:
            result["purpose"] = self.purpose
        if self.check is not None:
            result["check"] = self.check
        if self.revision is not None:
            result["revision"] = self.revision.payload()
        return result


class ActionParser:
    """Enforce action structure before any SQL reaches the dispatcher."""

    @staticmethod
    def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        """Reject duplicate JSON keys instead of silently overriding values."""
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    @staticmethod
    def _text(value: object, label: str, maximum: int) -> str:
        """Require bounded nonempty text at protocol boundaries."""
        if not isinstance(value, str) or not value.strip() or len(value) > maximum:
            raise ValueError(f"{label} must be nonempty text of at most {maximum} characters")
        return value

    @classmethod
    def _revision(cls, value: object) -> Revision:
        """Validate a short explanation and explicit source references."""
        if not isinstance(value, dict) or set(value) != {"reason", "sources"}:
            raise ValueError("revision requires exactly reason and sources")
        reason: str = cls._text(value["reason"], "revision.reason", 600)
        sources: object = value["sources"]
        if (not isinstance(sources, list) or not 1 <= len(sources) <= 8
                or not all(isinstance(s, str) and 0 < len(s) <= 80 for s in sources)):
            raise ValueError("revision.sources must contain 1..8 source references")
        if len(set(sources)) != len(sources):
            raise ValueError("revision.sources must be unique")
        return Revision(reason, tuple(sources))

    @classmethod
    def parse(cls, content: str, architecture: str) -> Action:
        """Parse one action without repairing JSON or evaluating model text."""
        value: object = json.loads(content, object_pairs_hook=cls._unique)
        if not isinstance(value, dict):
            raise ValueError("Expected a JSON action object")
        if set(value) & {"tool", "result", "rows", "observation_id"}:
            raise ValueError("Only runtime returns tool observations; request action=execute_sql or submit_sql with sql")
        sql: str = cls._text(value.get("sql"), "sql", 100_000)
        name: object = value.get("action")
        if architecture == "workflow":
            if name != "submit_sql" or set(value) != {"action", "sql"}:
                raise ValueError("Workflow requires exactly action=submit_sql and sql")
            return Action("submit_sql", sql)
        if architecture != "single":
            raise ValueError("Unknown architecture")
        revision: Revision | None = cls._revision(value["revision"]) if "revision" in value else None
        if name == "submit_sql" and not set(value) - {"action", "sql", "revision"}:
            return Action("submit_sql", sql, revision=revision)
        if name == "execute_sql" and not set(value) - {"action", "sql", "purpose", "check", "revision"}:
            purpose: object = value.get("purpose")
            if purpose == "candidate" and "check" not in value:
                return Action("execute_sql", sql, "candidate", revision=revision)
            if purpose == "diagnostic" and revision is None:
                check: str = cls._text(value.get("check"), "diagnostic check", 600)
                return Action("execute_sql", sql, "diagnostic", check=check)
        raise ValueError("Unknown action, purpose or unexpected fields")
