"""Deterministic protocol, accounting and isolation tests; no real model calls."""
import json
import sqlite3
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch
from mas_sql.agent.budgets import Budget
from mas_sql.agent.investigation import sql_fingerprint
from mas_sql.agent.runtime import solve
from mas_sql.agent.sequential.actions import TeamActionParser
from mas_sql.database.executor import SQLLimits, execute
from mas_sql.evaluation.behavior import team_summary
from mas_sql.evaluation.evaluator import score_team_reviews
from mas_sql.llm.accounting import TokenAccounting
from mas_sql.llm.errors import ClientError, ResponseFormatError
from mas_sql.llm.types import ChatResponse, Message
from mas_sql.llm.responses import ChatResponseParser

SQL: str = "SELECT name FROM items"
Script = dict[str, Any] | str | Callable[[dict[str, Any]], dict[str, Any]] | Exception


def candidate(sql: str = SQL, sources: list[str] | None = None) -> dict[str, Any]:
    """Create a public candidate with optional revision grounds."""
    result: dict[str, Any] = {"action": "execute_sql", "purpose": "candidate", "sql": sql}
    if sources is not None:
        result["revision"] = {"reason": "Use requested output from verified sources",
                              "sources": list(dict.fromkeys([*sources, "question"])),
                              "basis": "requirement", "grounds": [{"source": "question", "quote": "item names"}]}
    return result


def issue(ident: str = "I1", sources: list[str] | None = None) -> dict[str, Any]:
    """Create a bounded reviewer objection."""
    refs: list[str] = sources if sources is not None else ["question"]
    source: str = refs[0]
    basis: str = "observed_data" if source == "observation:2" else "execution" if source == "observation:1" else "requirement"
    quote: str = "[[2]]" if basis == "observed_data" else '"error"' if basis == "execution" else "item names"
    return {"id": ident, "category": "execution" if basis == "execution" else "projection",
            "message": "Check requested output columns", "sources": refs,
            "basis": basis, "grounds": [{"source": source, "quote": quote}]}


def review(sql: str = SQL, version: int = 1, issues: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Bind a reviewer verdict to exact candidate text."""
    return {"action": "review_sql", "candidate_version": version, "sql_fingerprint": sql_fingerprint(sql),
            "verdict": "request_changes" if issues else "approve", "issues": issues or []}


def submit(sql: str = SQL, version: int = 1, disagreement: bool = False) -> dict[str, Any]:
    """Send a version reference, never SQL text, to the final execution gate."""
    result: dict[str, Any] = {"action": "submit_sql", "candidate_version": version, "sql_fingerprint": sql_fingerprint(sql)}
    if disagreement:
        result["response_to_review"] = {"review_id": "review:1", "issue_ids": ["I1"],
                                        "reason": "The question requests exactly this output", "sources": ["question", "review:1"],
                                        "basis": "requirement", "grounds": [{"source": "question", "quote": "item names"}]}
    return result


def diagnostic(critic: bool = False, sql: str = SQL, version: int = 1) -> dict[str, Any]:
    """Create a diagnostic that never replaces the answer candidate."""
    result: dict[str, Any] = {"action": "execute_sql", "purpose": "diagnostic",
                              "sql": "SELECT COUNT(*) FROM items", "check": "Check row count"}
    if critic:
        result.update(candidate_version=version, sql_fingerprint=sql_fingerprint(sql))
    return result


class TeamModel:
    """Serve a scripted sequence and capture only actually requested histories."""

    def __init__(self, actions: list[Script], usage: object = None, finish: str = "stop") -> None:
        """Initialize deterministic actions and explicit token accounting."""
        self.actions = iter(actions)
        self.calls: list[list[Message]] = []
        self.caps: list[int] = []
        self.usage = {"prompt_tokens": 10, "completion_tokens": 20} if usage is None else usage
        self.finish = finish

    def count(self, messages: list[Message], timeout_s: float) -> int:
        """Return a small synthetic preflight count, not real tokenization."""
        return 10

    def complete(self, messages: list[Message], max_tokens: int, timeout_s: float) -> ChatResponse:
        """Return one action without network access or hidden retry."""
        self.calls.append(deepcopy(messages))
        self.caps.append(max_tokens)
        try:
            action: Script = next(self.actions)
        except StopIteration as exc:
            raise ClientError("Test script exhausted") from exc
        if isinstance(action, Exception):
            raise action
        if callable(action):
            action = action(json.loads(messages[-1]["content"]))
        return {"content": action if isinstance(action, str) else json.dumps(action), "reasoning": "PRIVATE_REASONING_SENTINEL",
                "finish_reason": self.finish, "usage": self.usage, "response_id": "team-mock"}


class SequentialTests(unittest.TestCase):
    """Exercise real guarded SQLite behind deterministic agent actions."""

    def setUp(self) -> None:
        """Create an isolated toy database and an allowlisted task input."""
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "toy.sqlite"
        con = sqlite3.connect(self.path)
        con.executescript("CREATE TABLE items(name TEXT); INSERT INTO items VALUES('Alice'),('Bob');")
        con.close()
        self.task = {"id": "one", "db_id": "toy", "question": "List item names", "evidence": "name is the label",
                     "sql": "HIDDEN_GOLD_SENTINEL", "EX": "HIDDEN_SCORE_SENTINEL"}

    def tearDown(self) -> None:
        """Remove the local toy database."""
        self.tmp.cleanup()

    def run_script(self, script: list[Script], budget: Budget | None = None,
                   accounting: TokenAccounting | None = None) -> tuple[dict[str, Any], TeamModel]:
        """Run one episode with only a scripted local model."""
        client: TeamModel = TeamModel(script)
        result: dict[str, Any] = solve(self.task, self.path, "sequential", client, budget, accounting=accounting)
        return result, client

    def test_approve_submit_three_calls(self) -> None:
        """The minimum path uses three generations and two SQL executions."""
        result, model = self.run_script([candidate(), review(), submit()])
        self.assertEqual(result["stop_reason"], "submitted")
        self.assertEqual((result["llm_calls"], result["tool_calls"]), (3, 2))
        self.assertEqual(result["first_sql"], SQL)
        self.assertEqual(result["final_sql"], SQL)
        self.assertEqual(result["usage"]["total_tokens"], 90)
        self.assertEqual(result["team"]["role_calls"], {"author": 2, "critic": 1})
        self.assertFalse(result["team"]["submitted_with_disagreement"])
        self.assertEqual(result["protocol_version"], "sequential-author-critic-v2")
        kinds: list[str] = [event["type"] for event in result["events"]]
        self.assertLess(kinds.index("first_candidate"), kinds.index("tool_result"))
        first_packet: dict[str, Any] = json.loads(model.calls[0][-1]["content"])
        self.assertEqual(first_packet["observations"], [])
        self.assertIsNone(first_packet["current_candidate"])
        self.assertEqual(len([e for e in result["events"] if e["type"] == "critic_review"]), 1)

    def test_revision_must_be_reviewed_again(self) -> None:
        """A repair produces a new version and an independent second report."""
        bad: str = "SELECT name, length(name) FROM items"
        result, model = self.run_script([candidate(bad), review(bad, issues=[issue()]),
            candidate(SQL, ["question", "review:1"]), review(SQL, 2), submit(SQL, 2)])
        self.assertEqual(result["final_sql"], SQL)
        self.assertEqual(result["first_sql"], bad)
        self.assertEqual((result["llm_calls"], result["tool_calls"]), (5, 3))
        self.assertEqual(result["investigation"]["revision_count"], 1)
        packet: dict[str, Any] = json.loads(model.calls[3][-1]["content"])
        self.assertIsNone(packet["current_review"])
        self.assertEqual(packet["current_candidate"]["version"], 2)
        self.assertEqual(result["team"]["final_review"]["candidate_version"], 2)

    def test_disagreement_is_explicit_and_not_approval(self) -> None:
        """The answer owner may retain SQL while preserving objections in telemetry."""
        result, _ = self.run_script([candidate(), review(issues=[issue()]), submit(disagreement=True)])
        self.assertEqual(result["stop_reason"], "submitted")
        self.assertTrue(result["team"]["submitted_with_disagreement"])
        self.assertEqual(result["team"]["final_review"]["verdict"], "request_changes")
        self.assertEqual(result["last_submission"]["response_to_review"]["review_id"], "review:1")

    def test_disagreement_must_cover_every_issue(self) -> None:
        """Partial responses do not bypass a two-issue report."""
        good: dict[str, Any] = submit(disagreement=True)
        good["response_to_review"]["issue_ids"] = ["I1", "I2"]
        result, _ = self.run_script([candidate(), review(issues=[issue(), issue("I2")]), submit(disagreement=True), good])
        self.assertEqual(result["stop_reason"], "submitted")
        self.assertEqual(result["llm_calls"], 4)
        self.assertEqual(result["tool_calls"], 2)

    def test_submission_without_required_disagreement_rejected(self) -> None:
        """A request_changes report cannot be silently treated as approval."""
        result, _ = self.run_script([candidate(), review(issues=[issue()]), submit(), submit(disagreement=True)])
        self.assertEqual(result["stop_reason"], "submitted")
        self.assertEqual(result["llm_calls"], 4)

    def test_both_roles_may_diagnose_with_one_global_budget(self) -> None:
        """Diagnostic observations are public but never replace the candidate."""
        result, model = self.run_script([candidate(), diagnostic(True), review(), diagnostic(), submit()])
        self.assertEqual(result["stop_reason"], "submitted")
        self.assertEqual((result["llm_calls"], result["tool_calls"]), (5, 4))
        self.assertEqual(result["team"]["diagnostic_calls_by_role"], {"author": 1, "critic": 1})
        self.assertEqual(result["final_sql"], SQL)
        packet: dict[str, Any] = json.loads(model.calls[-1][-1]["content"])
        self.assertEqual([o["role"] for o in packet["observations"]], ["author", "critic", "author"])

    def test_second_critic_diagnostic_rejected(self) -> None:
        """Only one diagnostic is permitted per candidate review."""
        result, _ = self.run_script([candidate(), diagnostic(True), diagnostic(True), review(), submit()])
        self.assertEqual(result["stop_reason"], "submitted")
        self.assertEqual(result["tool_calls"], 3)
        self.assertEqual(result["llm_calls"], 5)

    def test_second_author_diagnostic_rejected(self) -> None:
        """Only one diagnostic is permitted per Author decision turn."""
        result, _ = self.run_script([candidate(), review(), diagnostic(), diagnostic(), submit()])
        self.assertEqual(result["stop_reason"], "submitted")
        self.assertEqual(result["tool_calls"], 3)

    def test_diagnostic_before_first_candidate_rejected(self) -> None:
        """No SQL feedback can precede the frozen first candidate."""
        result, model = self.run_script([diagnostic(), candidate(), review(), submit()])
        self.assertEqual(result["stop_reason"], "submitted")
        self.assertEqual(result["first_sql"], SQL)
        self.assertEqual(result["tool_calls"], 2)
        self.assertEqual(json.loads(model.calls[1][-1]["content"])["observations"], [])

    def test_first_submit_rejected(self) -> None:
        """A direct submission cannot skip independent review."""
        result, _ = self.run_script([submit(), candidate(), review(), submit()])
        self.assertEqual(result["stop_reason"], "submitted")
        self.assertEqual(result["llm_calls"], 4)

    def test_critic_cannot_replace_or_submit_candidate(self) -> None:
        """Invalid Critic actions execute no SQL and stay in its private history."""
        private: dict[str, Any] = candidate("SELECT 'CRITIC_INVALID_SENTINEL'")
        result, model = self.run_script([candidate(), private, submit(), review(), submit()])
        self.assertEqual(result["stop_reason"], "submitted")
        self.assertEqual(result["tool_calls"], 2)
        self.assertNotIn("CRITIC_INVALID_SENTINEL", json.dumps(model.calls[-1]))

    def test_author_cannot_review(self) -> None:
        """Author cannot forge its own approval."""
        result, _ = self.run_script([review(), candidate(), review(), submit()])
        self.assertEqual(result["stop_reason"], "submitted")
        self.assertEqual(result["llm_calls"], 4)

    def test_changed_sql_in_submit_is_rejected(self) -> None:
        """The submit contract accepts identity only, not replacement SQL."""
        changed: dict[str, Any] = {**submit(), "sql": "SELECT 'replacement'"}
        result, _ = self.run_script([candidate(), review(), changed, submit()])
        self.assertEqual(result["final_sql"], SQL)
        self.assertEqual(result["tool_calls"], 2)

    def test_stale_review_and_submission_rejected(self) -> None:
        """Neither an old review nor an old target can approve a revised candidate."""
        new: str = SQL + " WHERE name IS NOT NULL"
        result, _ = self.run_script([candidate(), review(), candidate(new, ["schema"]),
            review(), review(new, 2), submit(), submit(new, 2)])
        self.assertEqual(result["stop_reason"], "submitted")
        self.assertEqual(result["llm_calls"], 7)
        self.assertEqual(result["tool_calls"], 3)
        self.assertEqual(result["team"]["final_review"]["candidate_version"], 2)

    def test_unchanged_candidate_retest_rejected(self) -> None:
        """No-progress candidate execution cannot consume an unbounded tool loop."""
        result, _ = self.run_script([candidate(), review(), candidate(" " + SQL + " "), submit()])
        self.assertEqual(result["stop_reason"], "submitted")
        self.assertEqual(result["investigation"]["candidate_versions"], 1)
        self.assertEqual(result["tool_calls"], 2)

    def test_revision_requires_available_sources(self) -> None:
        """A future report or observation cannot justify a change."""
        new: str = SQL + " WHERE 1=1"
        result, _ = self.run_script([candidate(), review(), candidate(new), candidate(new, ["review:2"]),
                                    candidate(new, ["observation:999"]), submit()])
        self.assertEqual(result["stop_reason"], "submitted")
        self.assertEqual(result["investigation"]["revision_count"], 0)
        self.assertEqual(result["tool_calls"], 2)

    def test_empty_evidence_cannot_be_cited(self) -> None:
        """Empty evidence is not an available grounding source."""
        self.task["evidence"] = ""
        result, _ = self.run_script([candidate(), review(issues=[issue(sources=["evidence"])]), review(), submit()])
        self.assertEqual(result["stop_reason"], "submitted")
        self.assertEqual(len(result["team"]["reviews"]), 1)

    def test_failed_candidate_cannot_be_approved(self) -> None:
        """Execution failure requires a public issue before Author can respond."""
        bad: str = "SELECT missing FROM items"
        result, _ = self.run_script([candidate(bad), review(bad), review(bad, issues=[issue(sources=["observation:1"])]),
            candidate(SQL, ["schema", "observation:1", "review:1"]), review(SQL, 2), submit(SQL, 2)])
        self.assertEqual(result["stop_reason"], "submitted")
        self.assertEqual(result["first_sql"], bad)
        self.assertEqual(result["investigation"]["revision_count"], 1)

    def test_failed_final_can_be_repaired_with_new_review(self) -> None:
        """A transient final failure never becomes an accepted answer."""
        call_count: int = 0
        def failing_second(path: str | Path, sql: str, limits: SQLLimits) -> dict[str, Any]:
            """Simulate one final execution failure, not a real model error."""
            nonlocal call_count
            call_count += 1
            return {"status": "error", "error": "synthetic final failure", "wall_s": 0.0} if call_count == 2 else execute(path, sql, limits)
        new: str = SQL + " WHERE 1=1"
        with patch("mas_sql.agent.sequential.runtime.execute", side_effect=failing_second):
            result, _ = self.run_script([candidate(), review(), submit(), candidate(new, ["observation:2", "schema"]),
                                        review(new, 2), submit(new, 2)])
        self.assertEqual(result["stop_reason"], "submitted")
        self.assertEqual((result["llm_calls"], result["tool_calls"]), (6, 4))
        self.assertTrue(result["investigation"]["recovered_submission_error"])
        self.assertEqual(result["investigation"]["failed_submissions"], 1)

    def test_failed_final_and_exhaustion_never_promote_candidate(self) -> None:
        """An approved candidate is still not a final answer if final execution fails."""
        original: Callable[..., dict[str, Any]] = execute
        def fail_final(path: str | Path, sql: str, limits: SQLLimits) -> dict[str, Any]:
            """Return one real candidate result followed by a synthetic error."""
            if not hasattr(fail_final, "called"):
                setattr(fail_final, "called", True)
                return original(path, sql, limits)
            return {"status": "error", "error": "final failed", "wall_s": 0.0}
        with patch("mas_sql.agent.sequential.runtime.execute", side_effect=fail_final):
            result, _ = self.run_script([candidate(), review(), submit()], Budget(max_tool_calls=2))
        self.assertEqual(result["stop_reason"], "tool_call_budget")
        self.assertIsNone(result["final_sql"])
        self.assertIsNotNone(result["last_submission"])
        self.assertFalse(result["submission_accepted"])

    def test_final_sql_slot_is_reserved(self) -> None:
        """A forbidden diagnostic consumes an LLM call but no reserved SQL slot."""
        result, _ = self.run_script([candidate(), diagnostic(True), review(), submit()], Budget(max_tool_calls=2))
        self.assertEqual(result["stop_reason"], "submitted")
        self.assertEqual(result["tool_calls"], 2)
        self.assertEqual(result["investigation"]["diagnostic_calls"], 0)

    def test_call_reservation_blocks_unreviewable_revision(self) -> None:
        """A revision cannot use the last call and bypass a new reviewer."""
        result, _ = self.run_script([candidate(), review(), candidate(SQL + " WHERE 1=1", ["schema"]), "not JSON"],
                                    Budget(max_llm_calls=4))
        self.assertEqual(result["stop_reason"], "finalization_repair_exhausted")
        self.assertIsNone(result["final_sql"])
        self.assertEqual(result["investigation"]["candidate_versions"], 1)
        self.assertEqual(result["tool_calls"], 1)

    def test_low_budget_fails_without_hidden_extra_calls(self) -> None:
        """An impossible minimum path is reported before contacting the model."""
        for budget in (Budget(max_llm_calls=2), Budget(max_llm_calls=3), Budget(max_tool_calls=1)):
            with self.subTest(budget=budget):
                result, model = self.run_script([], budget)
                self.assertEqual(result["stop_reason"], "insufficient_review_budget")
                self.assertEqual(result["llm_calls"], 0)
                self.assertEqual(model.calls, [])

    def test_no_automatic_submit_after_approval(self) -> None:
        """Critic approval is not an executable final action."""
        result, _ = self.run_script([candidate(), review(), "not JSON", "not JSON"], Budget(max_llm_calls=4))
        self.assertEqual(result["stop_reason"], "finalization_repair_exhausted")
        self.assertIsNone(result["final_sql"])
        self.assertEqual(result["tool_calls"], 1)

    def test_critic_diagnostic_reserves_verdict_and_author_calls(self) -> None:
        """Optional Critic research cannot take the calls needed for a verdict and submit."""
        result, _ = self.run_script([candidate(), diagnostic(True), review()], Budget(max_llm_calls=4))
        self.assertEqual(result["stop_reason"], "insufficient_review_budget")
        self.assertEqual(result["tool_calls"], 1)
        self.assertEqual(result["investigation"]["diagnostic_calls"], 0)

    def test_token_budget_shared_between_roles(self) -> None:
        """An impossible token closure stops without charging projected usage."""
        result, model = self.run_script([candidate(), review()], Budget(max_total_tokens=65))
        self.assertEqual(result["stop_reason"], "insufficient_closure_budget")
        self.assertEqual(result["usage"]["total_tokens"], 0)
        self.assertEqual(result["llm_calls"], 0)
        self.assertEqual(model.caps, [])
        self.assertIsNone(result["final_sql"])

    def test_global_deadline_applies_before_first_model_call(self) -> None:
        """A spent deadline grants no implicit first call."""
        result, model = self.run_script([], Budget(max_episode_s=0.001))
        self.assertEqual(result["stop_reason"], "episode_time_budget")
        self.assertEqual(model.calls, [])

    def test_role_histories_do_not_contain_reasoning_or_gold(self) -> None:
        """Task allowlisting and public-only packets exclude hidden content."""
        result, model = self.run_script([candidate(), review(), submit()])
        text: str = json.dumps(model.calls)
        self.assertNotIn("HIDDEN_GOLD_SENTINEL", text)
        self.assertNotIn("HIDDEN_SCORE_SENTINEL", text)
        self.assertNotIn("PRIVATE_REASONING_SENTINEL", text)
        critic_assistants: list[str] = [m["content"] for m in model.calls[1] if m["role"] == "assistant"]
        self.assertEqual(critic_assistants, [])
        author_assistants: list[str] = [m["content"] for m in model.calls[-1] if m["role"] == "assistant"]
        self.assertEqual(author_assistants, [])
        self.assertEqual(json.loads(model.calls[-1][-1]["content"])["current_candidate"]["sql"], SQL)
        self.assertEqual(result["team"]["role_known_usage"]["critic"]["total_tokens"], 30)

    def test_invalid_author_output_not_shared_with_critic(self) -> None:
        """Protocol repair history is private to the emitting role."""
        result, model = self.run_script(["AUTHOR_INVALID_SENTINEL", candidate(), review(), submit()])
        self.assertEqual(result["stop_reason"], "submitted")
        self.assertNotIn("AUTHOR_INVALID_SENTINEL", json.dumps(model.calls[2]))

    def test_fresh_histories_between_tasks(self) -> None:
        """The same transport does not share conversations across solve calls."""
        model: TeamModel = TeamModel([candidate(), review(), submit(), candidate(), review(), submit()])
        for _ in range(2):
            result: dict[str, Any] = solve(self.task, self.path, "sequential", model)
            self.assertEqual(result["stop_reason"], "submitted")
        self.assertEqual(model.calls[0][:2], model.calls[3][:2])
        for index in (0, 3):
            packet: dict[str, Any] = json.loads(model.calls[index][-1]["content"])
            self.assertIsNone(packet["current_candidate"])
            self.assertEqual(packet["observations"], [])
            self.assertEqual(len(model.calls[index]), 3)

    def test_generation_events_are_role_attributed(self) -> None:
        """Preflight, requests and responses identify the active role and version."""
        result, _ = self.run_script([candidate(), review(), submit()])
        requests: list[dict[str, Any]] = [e for e in result["events"] if e["type"] == "llm_request"]
        self.assertEqual([e["role"] for e in requests], ["author", "critic", "author"])
        self.assertEqual([e["candidate_version"] for e in requests], [None, 1, 1])
        for kind in ("preflight", "llm_response"):
            self.assertEqual([e["role"] for e in result["events"] if e["type"] == kind], ["author", "critic", "author"])
        self.assertNotIn("role", result["events"][-1])

    def test_diagnostic_grounding_is_tracked(self) -> None:
        """A cited Critic diagnostic can support an Author revision publicly."""
        new: str = SQL + " WHERE name IS NOT NULL"
        result, _ = self.run_script([candidate(), diagnostic(True), review(issues=[issue(sources=["observation:2"])]),
            candidate(new, ["observation:2", "review:1"]), review(new, 2), submit(new, 2)])
        self.assertEqual(result["stop_reason"], "submitted")
        self.assertEqual(result["investigation"]["revisions_citing_diagnostics"], 1)
        self.assertEqual(result["investigation"]["diagnostics"][0]["referenced_by_revisions"], [2])

    def test_offline_review_outcomes_and_summary(self) -> None:
        """Offline scoring labels approval errors without sending truth to agents."""
        result, model = self.run_script([candidate("SELECT COUNT(*) FROM items"), review("SELECT COUNT(*) FROM items"),
                                         submit("SELECT COUNT(*) FROM items")])
        outcomes: list[dict[str, Any]] = score_team_reviews(result["team"], self.path, [["Alice"], ["Bob"]], SQLLimits())
        self.assertEqual(outcomes[0]["EX"], 0)
        scored: dict[str, Any] = {**result, "review_outcomes": outcomes, "EX_first": 0, "EX_final": 0}
        self.assertEqual(team_summary([scored], scored=True)["approvals_of_incorrect_candidates"], 1)
        self.assertNotIn("review_outcomes", json.dumps(model.calls))

    def test_offline_detects_review_identity_mismatch(self) -> None:
        """An inconsistent audit artifact cannot silently score another SQL."""
        result, _ = self.run_script([candidate(), review(), submit()])
        result["team"]["reviews"][0]["sql_fingerprint"] = "0" * 64
        with self.assertRaises(ValueError):
            score_team_reviews(result["team"], self.path, [], SQLLimits())

    def test_usage_drift_uses_existing_usage_policy(self) -> None:
        """Usage mode preserves actual totals rather than preflight estimates."""
        model: TeamModel = TeamModel([candidate(), review(), submit()], usage={"prompt_tokens": 17, "completion_tokens": 20})
        result: dict[str, Any] = solve(self.task, self.path, "sequential", model,
            accounting=TokenAccounting(mode="usage", prompt_margin_tokens=512))
        self.assertEqual(result["stop_reason"], "submitted")
        self.assertEqual(result["usage"]["total_tokens"], 111)
        self.assertEqual(result["tokenizer_mismatch_count"], 3)
        self.assertEqual(sum(int(v["total_tokens"]) for v in result["team"]["role_known_usage"].values()), 111)

    def test_unknown_usage_stops_before_sql_execution(self) -> None:
        """Unknown accounting cannot be bypassed by a valid action body."""
        model: TeamModel = TeamModel([candidate()], usage={})
        result: dict[str, Any] = solve(self.task, self.path, "sequential", model)
        self.assertEqual(result["stop_reason"], "usage_missing")
        self.assertEqual(result["tool_calls"], 0)
        self.assertFalse(result["usage_complete"])

    def test_malformed_envelope_charges_known_usage(self) -> None:
        """Malformed content follows the existing accounting-aware fatal path."""
        try:
            ChatResponseParser.parse({"choices": [{"message": {"content": None}, "finish_reason": "stop"}],
                                      "usage": {"prompt_tokens": 10, "completion_tokens": 20}}, 200)
        except ResponseFormatError as caught:
            error: ResponseFormatError = caught
        result, _ = self.run_script([error])
        self.assertEqual(result["stop_reason"], "response_format_error")
        self.assertEqual(result["usage"]["total_tokens"], 30)
        self.assertEqual(result["team"]["role_known_usage"]["author"]["total_tokens"], 30)
        self.assertEqual(result["tool_calls"], 0)

    def test_malformed_critic_response_charges_critic_usage(self) -> None:
        """The second role uses the same fatal accounting path as the first."""
        try:
            ChatResponseParser.parse({"choices": [{"message": {"content": None}, "finish_reason": "stop"}],
                                      "usage": {"prompt_tokens": 10, "completion_tokens": 20}}, 200)
        except ResponseFormatError as error:
            result, _ = self.run_script([candidate(), error])
        self.assertEqual(result["stop_reason"], "response_format_error")
        self.assertEqual(result["usage"]["total_tokens"], 60)
        self.assertEqual(result["team"]["role_calls"], {"author": 1, "critic": 1})
        self.assertEqual(result["team"]["role_known_usage"]["critic"]["total_tokens"], 30)
        self.assertEqual(result["tool_calls"], 1)
        self.assertIsNone(result["final_sql"])

    def test_malformed_partial_usage_stops_without_estimating_missing_tokens(self) -> None:
        """Known partial usage is retained and unknown totals stay unknown."""
        try:
            ChatResponseParser.parse({"choices": [], "usage": {"prompt_tokens": 10}}, 200)
        except ResponseFormatError as error:
            result, _ = self.run_script([candidate(), error])
        self.assertEqual(result["stop_reason"], "response_format_error_usage_unknown")
        self.assertFalse(result["usage_complete"])
        self.assertIsNone(result["usage"]["total_tokens"])
        self.assertEqual(result["usage_known_totals"]["total_tokens"], 40)
        self.assertEqual(result["team"]["role_known_usage"]["critic"]["input_tokens"], 10)
        self.assertFalse(result["team"]["role_known_usage"]["critic"]["complete"])

    def test_actual_token_overrun_prevents_tool_dispatch(self) -> None:
        """Compatibility accounting is not permission to exceed the token cap."""
        model: TeamModel = TeamModel([candidate()], usage={"prompt_tokens": 32000, "completion_tokens": 20})
        result: dict[str, Any] = solve(self.task, self.path, "sequential", model, Budget(max_total_tokens=32000),
            accounting=TokenAccounting(mode="usage", prompt_margin_tokens=0))
        self.assertEqual(result["stop_reason"], "token_budget_overrun")
        self.assertEqual(result["tool_calls"], 0)
        self.assertTrue(result["token_budget_overrun"])

    def test_context_and_output_caps_are_enforced(self) -> None:
        """Per-response caps remain enforced over the team ledger."""
        for budget, reason in ((Budget(max_context_tokens=25), "insufficient_closure_budget"),
                               (Budget(max_output_tokens=5), "server_output_cap_violation")):
            with self.subTest(reason=reason):
                result, _ = self.run_script([candidate()], budget)
                self.assertEqual(result["stop_reason"], reason)
                self.assertEqual(result["tool_calls"], 0)

    def test_truncated_action_is_not_executed_and_repair_is_charged(self) -> None:
        """Even parseable JSON is not accepted after a truncated completion."""
        model: TeamModel = TeamModel([candidate("SELECT 'TRUNCATED'"), candidate(), review(), submit()])
        original: Callable[..., ChatResponse] = model.complete
        def complete(messages: list[Message], max_tokens: int, timeout_s: float) -> ChatResponse:
            """Mark only the first scripted response as truncated."""
            response: ChatResponse = original(messages, max_tokens, timeout_s)
            if len(model.calls) == 1:
                response["finish_reason"] = "length"
            return response
        with patch.object(model, "complete", side_effect=complete):
            result: dict[str, Any] = solve(self.task, self.path, "sequential", model)
        self.assertEqual(result["stop_reason"], "submitted")
        self.assertEqual(result["first_sql"], SQL)
        self.assertEqual(result["llm_calls"], 4)
        self.assertEqual(result["usage"]["total_tokens"], 120)

    def test_observations_distinguish_diagnostic_and_candidate_fingerprints(self) -> None:
        """A diagnostic trace identifies both executed SQL and its answer target."""
        result, _ = self.run_script([candidate(), diagnostic(True), review(), submit()])
        observation: dict[str, Any] = next(event for event in result["events"]
                                          if event["type"] == "tool_result" and event["purpose"] == "diagnostic")
        self.assertEqual(observation["sql_fingerprint"], sql_fingerprint("SELECT COUNT(*) FROM items"))
        self.assertEqual(observation["candidate_sql_fingerprint"], sql_fingerprint(SQL))

    def test_preview_truncation_is_preserved_for_reviewer(self) -> None:
        """The team does not silently turn a bounded preview into a full result."""
        sql: str = "SELECT name FROM items UNION ALL SELECT name FROM items"
        model: TeamModel = TeamModel([candidate(sql), review(sql), submit(sql)])
        result: dict[str, Any] = solve(self.task, self.path, "sequential", model, limits=SQLLimits(preview_rows=1))
        packet: dict[str, Any] = json.loads(model.calls[1][-1]["content"])
        observation: dict[str, Any] = packet["observations"][0]["result"]
        self.assertTrue(observation["preview_truncated"])
        self.assertEqual(len(observation["rows"]), 1)
        self.assertEqual(result["final_result"]["row_count"], 4)
        self.assertEqual(result["final_sql"], sql)

    def test_transport_failure_stops_without_retry(self) -> None:
        """A failed network envelope never triggers an invisible second request."""
        result, model = self.run_script([ClientError("synthetic transport error")])
        self.assertEqual(result["stop_reason"], "transport_error_usage_unknown")
        self.assertEqual(len(model.calls), 1)
        self.assertIsNone(result["final_sql"])

    def test_duplicate_json_keys_and_unknown_fields_rejected(self) -> None:
        """No parser repair can hide a conflicting action or fabricated tool result."""
        invalid: list[str] = ['{"action":"submit_sql","action":"execute_sql"}',
            json.dumps({**candidate(), "rows": []}), json.dumps({**review(), "issues": [issue()]}),
            json.dumps({**review(), "candidate_version": True}), json.dumps({**review(), "sql_fingerprint": "bad"})]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                TeamActionParser.parse(value, "critic")

    def test_sql_write_attempt_is_rejected_by_shared_executor(self) -> None:
        """The new role protocol does not weaken SQLite write guards."""
        bad: str = "DELETE FROM items"
        result, _ = self.run_script([candidate(bad), review(bad, issues=[issue(sources=["observation:1"])]),
                                    candidate(SQL, ["schema", "observation:1"]), review(SQL, 2), submit(SQL, 2)])
        self.assertEqual(result["stop_reason"], "submitted")
        self.assertEqual(result["final_result"]["row_count"], 2)
        observations: list[dict[str, Any]] = [e for e in result["events"] if e["type"] == "tool_result"]
        self.assertEqual(observations[0]["observation"]["status"], "rejected")


if __name__ == "__main__":
    unittest.main()
