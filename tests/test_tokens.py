"""Regression tests for mismatched system-prompt preflight counts."""
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from typing import Any
from mas_sql.llm.probe import ProbeRunner
from mas_sql.agent.runtime import Budget, solve
from mas_sql.llm.accounting import TokenAccounting
from mas_sql.llm.types import ChatResponse, Message


class MismatchedClient:
    """Emulate a server whose system-message rendering differs in generation."""

    def __init__(self, actions: list[dict[str, str]] | None = None,
                 input_tokens: int = 17, output_tokens: int = 20,
                 missing_usage: bool = False) -> None:
        """Configure reported usage without changing the preflight endpoint."""
        self.actions = iter(actions or [{"action": "submit_sql", "sql": "SELECT 1"}])
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.missing_usage = missing_usage
        self.caps: list[int] = []
        self.probe = False

    def count(self, messages: list[Message], timeout_s: float) -> int:
        """Always return the unadjusted tokenizer estimate."""
        return 10

    def complete(self, messages: list[Message], max_tokens: int, timeout_s: float) -> ChatResponse:
        """Return actual usage and preserve the requested cap for assertions."""
        self.caps.append(max_tokens)
        has_system: bool = any(m["role"] == "system" for m in messages)
        actual: int = self.input_tokens if has_system else 10
        action: dict[str, Any] = {"ok": True} if self.probe else next(self.actions)
        return {"content": json.dumps(action), "reasoning": None, "finish_reason": "stop",
                "usage": {} if self.missing_usage else {"prompt_tokens": actual, "completion_tokens": self.output_tokens},
                "response_id": "synthetic-only"}


class AccountingTests(unittest.TestCase):
    """Check that usage mode remains accountable rather than ignoring budgets."""

    def setUp(self) -> None:
        """Create a small static database with no benchmark data."""
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "toy.sqlite"
        connection = sqlite3.connect(self.path)
        connection.execute("CREATE TABLE toy(n INTEGER)")
        connection.close()
        self.task = {"id": "one", "db_id": "toy", "question": "Return one", "evidence": "", "difficulty": "simple"}

    def tearDown(self) -> None:
        """Remove synthetic database artifacts."""
        self.tmp.cleanup()

    def test_usage_mode_submits_despite_mismatch(self) -> None:
        """Keep valid SQL and charge actual input rather than the preflight count."""
        result = solve(self.task, self.path, "workflow", MismatchedClient(), accounting=TokenAccounting("usage", 5))
        self.assertEqual(result["stop_reason"], "submitted")
        self.assertEqual(result["usage"]["total_tokens"], 37)
        self.assertEqual(result["tokenizer_mismatch_count"], 1)
        warnings = [e for e in result["events"] if e["type"] == "tokenizer_mismatch"]
        self.assertEqual(warnings[0]["severity"], "warning")
        self.assertEqual(warnings[0]["delta_tokens"], 7)
        self.assertTrue(warnings[0]["margin_exceeded"])

    def test_next_call_uses_actual_cumulative_usage(self) -> None:
        """Subtract actual charged tokens and reserve the margin on every turn."""
        client = MismatchedClient([{"action": "execute_sql", "purpose": "candidate", "sql": "SELECT 1"},
                                   {"action": "submit_sql", "sql": "SELECT 1"}])
        result = solve(self.task, self.path, "single", client, Budget(max_total_tokens=110),
                       accounting=TokenAccounting("usage", 5))
        self.assertEqual(client.caps, [95, 58])
        self.assertEqual(result["usage"]["total_tokens"], 74)
        self.assertEqual(result["tokenizer_mismatch_count"], 2)
        self.assertEqual(result["stop_reason"], "submitted")

    def test_actual_overrun_still_stops(self) -> None:
        """Do not turn compatibility mode into unlimited-token execution."""
        result = solve(self.task, self.path, "single", MismatchedClient(input_tokens=30, output_tokens=1),
                       Budget(max_total_tokens=25), accounting=TokenAccounting("usage"))
        self.assertEqual(result["stop_reason"], "token_budget_overrun")
        self.assertEqual(result["usage"]["total_tokens"], 31)
        self.assertEqual(result["tool_calls"], 0)

    def test_actual_context_overrun_stops(self) -> None:
        """Report a context violation independently of the total episode budget."""
        result = solve(self.task, self.path, "single", MismatchedClient(input_tokens=40, output_tokens=5),
                       Budget(max_context_tokens=35), accounting=TokenAccounting("usage"))
        self.assertEqual(result["stop_reason"], "context_budget_overrun")
        self.assertEqual(result["tool_calls"], 0)

    def test_missing_usage_still_stops(self) -> None:
        """Never substitute preflight estimates for missing actual usage."""
        result = solve(self.task, self.path, "single", MismatchedClient(missing_usage=True),
                       accounting=TokenAccounting("usage", 5))
        self.assertEqual(result["stop_reason"], "usage_missing")
        self.assertIsNone(result["usage"]["total_tokens"])

    def test_reserve_exhausts_budget_before_generation(self) -> None:
        """Include the margin when deciding whether another call fits."""
        client = MismatchedClient()
        result = solve(self.task, self.path, "single", client, Budget(max_total_tokens=20),
                       accounting=TokenAccounting("usage", 11))
        self.assertEqual(result["stop_reason"], "token_budget")
        self.assertEqual(client.caps, [])

    def test_smaller_actual_prompt_is_also_only_a_warning(self) -> None:
        """Handle either sign of renderer drift without an assumed offset."""
        result = solve(self.task, self.path, "workflow", MismatchedClient(input_tokens=9),
                       accounting=TokenAccounting("usage"))
        self.assertEqual(result["stop_reason"], "submitted")
        self.assertEqual(result["usage"]["total_tokens"], 29)

    def test_invalid_policy(self) -> None:
        """Reject misspelled modes and invalid safety margins."""
        for kwargs in [{"mode": "ignore"}, {"prompt_margin_tokens": -1}, {"prompt_margin_tokens": True}]:
            with self.assertRaises(ValueError):
                TokenAccounting(**kwargs)

    def test_probe_covers_system_and_history(self) -> None:
        """Detect the user-only blind spot while permitting explicit usage mode."""
        for mode in ("strict", "usage"):
            client = MismatchedClient()
            client.probe = True
            snapshots: list[dict[str, Any]] = []
            report = ProbeRunner(client, TokenAccounting(mode, 5), 100, 1000, 30).run(
                lambda value: snapshots.append(json.loads(json.dumps(value))))
            self.assertEqual([c["delta_tokens"] for c in report["cases"]], [0, 7, 7])
            self.assertEqual(report["compatible"], mode == "usage")
            self.assertTrue(report["cases"][0]["passed"])
            self.assertFalse(report["tokenizer_matches_usage"])
            self.assertTrue(report["complete"])
            self.assertGreaterEqual(len(snapshots), 3)
