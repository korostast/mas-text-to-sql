"""Behavior regressions for bounded SQL investigation without extra tools."""
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from typing import Any
from mas_sql.agent.actions import ActionParser
from mas_sql.evaluation.behavior import investigation_summary
from mas_sql.llm.client import ClientError
from mas_sql.agent.runtime import Budget, solve
from mas_sql.llm.types import ChatResponse, Message


def answer(sql: str, sources: list[str] | None = None) -> dict[str, Any]:
    """Build a submission with an optional short, public correction basis."""
    value: dict[str, Any] = {"action": "submit_sql", "sql": sql}
    if sources is not None:
        value["revision"] = {"reason": "Use the verified identifier or requested expression", "sources": sources}
    return value


def candidate(sql: str) -> dict[str, Any]:
    """Build one answer candidate test."""
    return {"action": "execute_sql", "purpose": "candidate", "sql": sql}


def diagnostic(sql: str, check: str = "Check available identifier names") -> dict[str, Any]:
    """Build a diagnostic with an externally observable test objective."""
    return {"action": "execute_sql", "purpose": "diagnostic", "sql": sql, "check": check}


class ScriptedModel:
    """Return predetermined actions and capture exactly what the model sees."""

    def __init__(self, actions: list[dict[str, Any]]) -> None:
        """Store the script without any server or benchmark access."""
        self.actions = iter(actions)
        self.calls: list[list[Message]] = []

    def count(self, messages: list[Message], timeout_s: float) -> int:
        """Return a stable prompt count for budget tests."""
        return 10

    def complete(self, messages: list[Message], max_tokens: int, timeout_s: float) -> ChatResponse:
        """Return one scripted action and explicit usage, without retrying."""
        self.calls.append([m.copy() for m in messages])
        try:
            action: dict[str, Any] = next(self.actions)
        except StopIteration as exc:
            raise ClientError("Test action script exhausted") from exc
        return {"content": json.dumps(action), "reasoning": None, "finish_reason": "stop",
                "usage": {"prompt_tokens": 10, "completion_tokens": 20}, "response_id": "script-only"}


class InvestigationTests(unittest.TestCase):
    """Exercise real guarded SQLite and protocol transitions on synthetic data."""

    def setUp(self) -> None:
        """Create a static toy table and an ordinary task with no gold input."""
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "toy.sqlite"
        con = sqlite3.connect(self.path)
        con.executescript("CREATE TABLE items(name TEXT); INSERT INTO items VALUES('Alice'),('Bob');")
        con.close()
        self.task = {"id": "one", "db_id": "toy", "question": "List item names", "evidence": "name is the item label"}

    def tearDown(self) -> None:
        """Remove toy database files."""
        self.tmp.cleanup()

    def run_script(self, actions: list[dict[str, Any]], budget: Budget | None = None,
                   architecture: str = "single") -> tuple[dict[str, Any], ScriptedModel]:
        """Solve the toy task with explicit actions and no remote requests."""
        model = ScriptedModel(actions)
        return solve(self.task, self.path, architecture, model, budget), model

    def test_failed_submit_can_be_repaired(self) -> None:
        """A failed submission remains an observation, not an accepted final."""
        bad: str = "SELECT unknown FROM items"
        good: str = "SELECT name FROM items"
        result, model = self.run_script([answer(bad), answer(good, ["schema", "observation:1"])])
        self.assertEqual(result["stop_reason"], "submitted")
        self.assertEqual(result["first_sql"], bad)
        self.assertEqual(result["final_sql"], good)
        self.assertEqual((result["llm_calls"], result["tool_calls"]), (2, 2))
        self.assertEqual(result["usage"]["total_tokens"], 60)
        feedback = json.loads(model.calls[1][-1]["content"])
        self.assertFalse(feedback["accepted"])
        self.assertEqual(feedback["observation_id"], 1)
        self.assertTrue(result["investigation"]["recovered_submission_error"])
        self.assertEqual(result["investigation"]["candidate_versions"], 2)

    def test_failure_then_diagnostic_then_repair(self) -> None:
        """A model may choose research before repairing, within the same budget."""
        result, _ = self.run_script([
            answer("SELECT missing FROM items"), diagnostic("SELECT * FROM items LIMIT 0"),
            answer("SELECT name FROM items", ["observation:1", "observation:2", "schema"]),
        ])
        behavior = result["investigation"]
        self.assertEqual(result["stop_reason"], "submitted")
        self.assertEqual(behavior["diagnostic_calls"], 1)
        self.assertEqual(behavior["successful_diagnostic_calls"], 1)
        self.assertEqual(behavior["revisions_citing_diagnostics"], 1)
        self.assertEqual(behavior["diagnostics"][0]["referenced_by_revisions"], [2])
        self.assertEqual(behavior["revisions"][0]["cited_diagnostics"], [2])

    def test_last_failed_submit_has_no_final_fallback(self) -> None:
        """Tool exhaustion preserves the failed artifact but never invents final SQL."""
        result, _ = self.run_script([answer("SELECT missing FROM items")], Budget(max_tool_calls=1))
        self.assertEqual(result["stop_reason"], "tool_call_budget")
        self.assertIsNone(result["final_sql"])
        self.assertFalse(result["submitted"])
        self.assertEqual(result["last_submission"]["result"]["status"], "error")
        self.assertEqual(result["llm_calls"], 1)

    def test_llm_exhaustion_after_failed_submit(self) -> None:
        """Recovery does not grant a hidden extra model call."""
        result, model = self.run_script([answer("SELECT missing FROM items")], Budget(max_llm_calls=1))
        self.assertEqual(result["stop_reason"], "llm_call_budget")
        self.assertIsNone(result["final_sql"])
        self.assertEqual(len(model.calls), 1)

    def test_workflow_still_has_one_call_and_no_recovery(self) -> None:
        """Preserve the one-shot workflow contract even on execution failure."""
        bad: str = "SELECT missing FROM items"
        result, model = self.run_script([answer(bad), answer("SELECT 1")], architecture="workflow")
        self.assertEqual(result["stop_reason"], "submitted_execution_failed")
        self.assertEqual(result["final_sql"], bad)
        self.assertFalse(result["submission_accepted"])
        self.assertEqual(len(model.calls), 1)

    def test_easy_task_does_not_force_diagnostics(self) -> None:
        """Allow direct submission when no investigation is needed."""
        result, _ = self.run_script([answer("SELECT name FROM items")])
        self.assertEqual(result["stop_reason"], "submitted")
        self.assertEqual(result["tool_calls"], 1)
        self.assertEqual(result["investigation"]["diagnostic_calls"], 0)

    def test_unchanged_candidate_needs_no_revision(self) -> None:
        """Ignore outer whitespace while retaining the exact first SQL artifact."""
        result, _ = self.run_script([candidate(" SELECT 1 "), answer("SELECT 1")])
        self.assertEqual(result["stop_reason"], "submitted")
        self.assertEqual(result["investigation"]["candidate_versions"], 1)
        self.assertEqual(result["investigation"]["revision_count"], 0)

    def test_unsupported_change_gets_protocol_feedback(self) -> None:
        """Require grounds before executing a changed answer, without rewriting it."""
        result, _ = self.run_script([candidate("SELECT 1"), answer("SELECT 2"), answer("SELECT 2", ["question"])])
        self.assertEqual(result["stop_reason"], "submitted")
        self.assertEqual(result["tool_calls"], 2)
        self.assertEqual(result["llm_calls"], 3)
        self.assertTrue(any(e["type"] == "protocol_error" for e in result["events"]))

    def test_forged_future_reference_is_rejected(self) -> None:
        """Reject references to observations that were never executed."""
        result, _ = self.run_script([candidate("SELECT 1"), answer("SELECT 2", ["observation:999"]),
                                     answer("SELECT 2", ["question"])])
        self.assertEqual(result["tool_calls"], 2)
        self.assertEqual(result["stop_reason"], "submitted")
        self.assertEqual(result["investigation"]["revisions"][0]["sources"], ["question"])

    def test_check_is_required_for_diagnostic(self) -> None:
        """Keep diagnostic intent inspectable rather than inferring it from SQL."""
        incomplete: dict[str, Any] = {"action": "execute_sql", "purpose": "diagnostic", "sql": "SELECT 2"}
        result, _ = self.run_script([candidate("SELECT 1"), incomplete, diagnostic("SELECT 2"), answer("SELECT 1")])
        self.assertEqual((result["llm_calls"], result["tool_calls"]), (4, 3))
        self.assertEqual(result["investigation"]["diagnostic_calls"], 1)

    def test_no_diagnostics_before_initial_candidate(self) -> None:
        """Preserve cold-start EX_first semantics despite richer prompts."""
        result, _ = self.run_script([diagnostic("SELECT 2"), answer("SELECT 1")])
        self.assertEqual(result["first_sql"], "SELECT 1")
        self.assertEqual(result["tool_calls"], 1)
        self.assertEqual(result["investigation"]["diagnostic_calls"], 0)

    def test_diagnostic_does_not_replace_answer_version(self) -> None:
        """Track diagnostic results separately from the current answer candidate."""
        result, _ = self.run_script([candidate("SELECT 1"), diagnostic("SELECT 999"), answer("SELECT 1")])
        self.assertEqual(result["first_sql"], result["final_sql"])
        self.assertEqual(result["investigation"]["candidate_versions"], 1)

    def test_empty_evidence_cannot_be_cited(self) -> None:
        """Do not accept an evidence reference when the task contains none."""
        self.task["evidence"] = ""
        result, _ = self.run_script([candidate("SELECT 1"), answer("SELECT 2", ["evidence"]), answer("SELECT 1")])
        self.assertEqual(result["final_sql"], "SELECT 1")
        self.assertEqual(result["tool_calls"], 2)

    def test_whitespace_inside_literal_is_a_real_change(self) -> None:
        """Do not erase SQL literal differences with naive whitespace normalization."""
        result, _ = self.run_script([candidate("SELECT 'a  b'"), answer("SELECT 'a b'"), answer("SELECT 'a  b'")])
        self.assertEqual(result["final_sql"], "SELECT 'a  b'")
        self.assertEqual(result["tool_calls"], 2)

    def test_initial_event_is_an_immutable_snapshot(self) -> None:
        """Do not let later messages mutate the stored initial conversation."""
        result, _ = self.run_script([candidate("SELECT 1"), answer("SELECT 1")])
        self.assertEqual(len(result["events"][0]["messages"]), 2)
        requests = [e for e in result["events"] if e["type"] == "llm_request"]
        self.assertGreater(len(requests[1]["messages"]), len(requests[0]["messages"]))

    def test_grounding_is_not_a_semantic_oracle(self) -> None:
        """A valid reference proves source availability, not truth of an answer."""
        result, _ = self.run_script([candidate("SELECT 1"), answer("SELECT 999", ["question"])])
        self.assertEqual(result["final_sql"], "SELECT 999")
        self.assertNotIn("EX_final", result)

    def test_missing_grounding_or_invalid_sources_cannot_bypass_budget(self) -> None:
        """Rejected actions consume model calls but never execute hidden SQL."""
        result, _ = self.run_script([candidate("SELECT 1"), answer("SELECT 2", ["gold"])], Budget(max_llm_calls=2))
        self.assertEqual(result["stop_reason"], "llm_call_budget")
        self.assertEqual(result["tool_calls"], 1)
        self.assertIsNone(result["final_sql"])

    def test_quality_summary_does_not_assume_causality(self) -> None:
        """Join paired EX outcomes to behavior without marking every check useful."""
        result, _ = self.run_script([candidate("SELECT 1"), diagnostic("SELECT 2"), answer("SELECT 3", ["observation:2"])])
        rows = [{**result, "EX_first": 0, "EX_final": 1},
                {**result, "EX_first": 1, "EX_final": 0}]
        summary = investigation_summary(rows, scored=True)
        self.assertEqual(summary["repairs_in_episodes_with_diagnostics"], 1)
        self.assertEqual(summary["regressions_in_episodes_with_diagnostics"], 1)
        self.assertEqual(summary["coverage_episodes"], 2)
        old = investigation_summary([{"EX_first": 0, "EX_final": 1}], scored=True)
        self.assertEqual(old["coverage_episodes"], 0)
        self.assertIsNone(old["diagnostic_calls"])

    def test_denied_write_submission_can_recover_without_modification(self) -> None:
        """Recovery preserves read-only enforcement instead of weakening the guard."""
        before: bytes = self.path.read_bytes()
        result, _ = self.run_script([answer("DELETE FROM items"), answer("SELECT name FROM items", ["observation:1", "schema"])])
        self.assertEqual(result["stop_reason"], "submitted")
        self.assertEqual(self.path.read_bytes(), before)
        self.assertTrue(result["investigation"]["recovered_submission_error"])

    def test_repeated_failed_submissions_stay_within_tool_budget(self) -> None:
        """Two failed submissions consume both slots without a hidden final fallback."""
        result, model = self.run_script([answer("SELECT missing FROM items"), answer("SELECT other_missing FROM items", ["schema"])], Budget(max_tool_calls=2))
        self.assertEqual(result["stop_reason"], "tool_call_budget")
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(result["investigation"]["failed_submissions"], 2)
        self.assertFalse(result["investigation"]["recovered_submission_error"])
        self.assertIsNone(result["final_sql"])

    def test_transport_loss_keeps_details_as_known_partials_only(self) -> None:
        """Unknown later usage prevents partial detail counts looking complete."""
        from mas_sql.agent.generation import TokenLedger
        ledger = TokenLedger()
        ledger.add({"prompt_tokens": 10, "completion_tokens": 20,
                    "completion_tokens_details": {"reasoning_tokens": 12}})
        ledger.complete = False
        self.assertIsNone(ledger.snapshot()["reasoning_tokens"])
        self.assertEqual(ledger.snapshot(known_only=True)["reasoning_tokens"], 12)
        self.assertIsNone(ledger.snapshot()["total_tokens"])
        self.assertEqual(ledger.snapshot(known_only=True)["total_tokens"], 30)

    def test_parser_rejects_extra_fields_and_duplicate_keys(self) -> None:
        """Keep structured metadata bounded and strict at the transport boundary."""
        for value in [
            '{"action":"submit_sql","sql":"SELECT 1","sql":"SELECT 2"}',
            json.dumps({**answer("SELECT 1"), "unexpected": True}),
            json.dumps(answer("SELECT 1", [])),
            json.dumps(diagnostic("SELECT 1", "x" * 601)),
        ]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                ActionParser.parse(value, "single")
