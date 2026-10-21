"""Check public evidence anchors without pretending to prove SQL semantics."""
import json
from dataclasses import dataclass
from typing import Any, Mapping
from mas_sql.agent.actions import ActionParser

BASES: frozenset[str] = frozenset({"requirement", "schema", "observed_data", "execution"})


@dataclass(frozen=True)
class Ground:
    """Point to a verbatim bounded excerpt of a direct public source."""

    source: str
    quote: str

    def payload(self) -> dict[str, str]:
        """Serialize an inspectable excerpt, not a private reasoning trace."""
        return {"source": self.source, "quote": self.quote}


def parse_grounds(value: object) -> tuple[Ground, ...]:
    """Reject absent, unbounded, repeated or indirect evidence anchors."""
    if not isinstance(value, list) or not 1 <= len(value) <= 3:
        raise ValueError("grounds requires 1..3 direct source/quote objects")
    result: list[Ground] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"source", "quote"}:
            raise ValueError("Each ground requires exactly source and quote")
        source: str = ActionParser._text(item["source"], "ground.source", 80)
        if source not in ("question", "evidence", "schema") and not source.startswith("observation:"):
            raise ValueError("A review is not direct evidence; ground it in task, schema or an observation")
        result.append(Ground(source, ActionParser._text(item["quote"], "ground.quote", 240)))
    if len(set(result)) != len(result):
        raise ValueError("grounds must be unique")
    return tuple(result)


def parse_basis(value: object) -> str:
    """Require one declared type of independently inspectable premise."""
    if not isinstance(value, str) or value not in BASES:
        raise ValueError("basis must be requirement, schema, observed_data or execution")
    return value


def validate_grounds(basis: str, grounds: tuple[Ground, ...], sources: tuple[str, ...],
                     texts: Mapping[str, str], observations: Mapping[int, dict[str, Any]],
                     current_version: int) -> None:
    """Verify quote existence, provenance, freshness and declared premise type."""
    if not {ground.source for ground in grounds} <= set(sources):
        raise ValueError("Every ground.source must also appear in sources")
    supported: bool = False
    for ground in grounds:
        observation: dict[str, Any] | None = None
        if ground.source.startswith("observation:"):
            suffix: str = ground.source.removeprefix("observation:")
            if not suffix.isdigit() or int(suffix) not in observations:
                raise ValueError("Ground refers to an unavailable observation")
            observation = observations[int(suffix)]
            if observation["candidate_version"] != current_version:
                raise ValueError("Ground observation must target the current candidate version")
            result: dict[str, Any] = observation["result"]
            if result.get("preview_truncated"):
                raise ValueError("A truncated preview cannot ground a blocking change; use a focused diagnostic")
            text: str = json.dumps(result, ensure_ascii=False)
        else:
            text = texts.get(ground.source, "")
        if not text or ground.quote not in text:
            raise ValueError("Ground quote must be a verbatim excerpt of its nonempty public source")
        if basis == "requirement" and ground.source in ("question", "evidence"):
            supported = True
        elif basis == "schema" and ground.source == "schema":
            supported = True
        elif basis == "observed_data" and observation is not None:
            supported |= observation["purpose"] == "diagnostic" and observation["result"]["status"] == "ok"
        elif basis == "execution" and observation is not None:
            supported |= observation["purpose"] in ("candidate", "submission") and observation["result"]["status"] != "ok"
    if not supported:
        raise ValueError("Declared basis lacks its required direct source; data hypotheses need a successful diagnostic")