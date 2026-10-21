"""Exercise future-token holdbacks and explicit finalization without real LLMs."""
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from typing import Any
from mas_sql.agent.budgets import Budget
from mas_sql.agent.events import EventLog
from mas_sql.agent.generation import GenerationController
from mas_sql.agent.runtime import solve
from mas_sql.agent.sequential.closure import ClosurePlanner
from mas_sql.llm.accounting import TokenAccounting
from mas_sql.llm.errors import ClientError
from mas_sql.llm.types import ChatResponse, Message
from test_sequential import TeamModel, candidate, diagnostic, issue, review, submit

COUNTS: dict[str, bool] = {"candidate": True, "diagnostic": True, "review_sql": True, "submit_sql": True}


class SpendingModel(TeamModel):
    """Consume every requested output allowance to stress downstream reservation."""

    def count(self, messages: list[Message], timeout_s: float) -> int:
        """Return a stable synthetic prompt count."""
        return 1000

    def complete(self, messages: list[Message], max_tokens: int, timeout_s: float) -> ChatResponse:
        """Report cap-sized usage without making any real model call."""
        response = super().complete(messages, max_tokens, timeout_s)
        response["usage"] = {"prompt_tokens": 1000, "completion_tokens": max_tokens}
        return response


class RepairSpendingModel(SpendingModel):
    """Account for a larger bounded repair prompt while spending every output cap."""

    def count(self, messages: list[Message], timeout_s: float) -> int:
        """Add repair feedback growth without relying on the rejected response size."""
        return 1700 if any('"finalization_only": true' in m["content"] for m in messages) else 1000

    def complete(self, messages: list[Message], max_tokens: int, timeout_s: float) -> ChatResponse:
        """Report the actual synthetic prompt and completion allocation."""
        response = super().complete(messages, max_tokens, timeout_s)
        response["usage"]["prompt_tokens"] = self.count(messages, timeout_s)
        return response


class ClosureTests(unittest.TestCase):
    """Test allocation arithmetic and its actual generation enforcement."""

    def test_author_turn_disables_unaffordable_optional_work(self) -> None:
        """A closing Author call should not advertise a new branch."""
        plan = ClosurePlanner.plan("author", False, 7000, 4096, {"author": 1000, "critic": 1000}, COUNTS)
        self.assertTrue(plan.allowed["submit_sql"])
        self.assertFalse(plan.allowed["candidate"])
        self.assertFalse(plan.allowed["diagnostic"])
        self.assertEqual(plan.holdback_tokens, 3560)
        self.assertEqual(plan.repair_turns, 1)

    def test_critic_holds_author_finalization(self) -> None:
        """The reviewer cannot spend the known closing allocation."""
        plan = ClosurePlanner.plan("critic", False, 11000, 4096, {"author": 1000, "critic": 1000}, COUNTS)
        self.assertFalse(plan.allowed["diagnostic"])
        self.assertTrue(plan.allowed["review_sql"])
        self.assertEqual(plan.holdback_tokens, 8656)

    def test_initial_candidate_reserves_both_roles(self) -> None:
        """The first query also needs an affordable review/submit chain."""
        plan = ClosurePlanner.plan("author", True, 2000, 4096, {"author": 1000, "critic": 1000}, COUNTS)
        self.assertFalse(plan.allowed["candidate"])
        self.assertGreater(plan.holdback_tokens, 2000)

    def test_optional_work_cannot_bypass_call_mask(self) -> None:
        """Tokens do not replace unavailable calls or final SQL slots."""
        plan = ClosurePlanner.plan("author", False, 32000, 4096, {"author": 10, "critic": 10},
                                   {**COUNTS, "candidate": False, "diagnostic": False})
        self.assertFalse(plan.allowed["candidate"])
        self.assertFalse(plan.allowed["diagnostic"])

    def test_holdback_uses_largest_branch_not_sum(self) -> None:
        """Mutually exclusive actions do not each acquire a separate ledger."""
        plan = ClosurePlanner.plan("author", False, 32000, 4096, {"author": 1000, "critic": 1000}, COUNTS)
        self.assertEqual(plan.holdback_tokens, max(plan.future_costs.values()))

    def test_generation_subtracts_holdback_but_charges_only_usage(self) -> None:
        """Reservations lower output caps without being spent tokens."""
        model = TeamModel([candidate()], usage={"prompt_tokens": 10, "completion_tokens": 20})
        generation = GenerationController(model, Budget(max_total_tokens=4000), TokenAccounting(), EventLog(), 8)
        response = generation.request([], lambda: 60.0, holdback_tokens=2000, min_output_tokens=1024)
        self.assertIsNotNone(response)
        self.assertEqual(model.caps, [1990])
        self.assertEqual(generation.ledger.total_tokens, 30)

    def test_tiny_tail_output_does_not_call_model(self) -> None:
        """Do not request hundreds of tokens when the completion floor cannot fit."""
        for tail in (247, 390):
            model = TeamModel([])
            generation = GenerationController(model, Budget(max_total_tokens=10 + tail), TokenAccounting(), EventLog(), 8)
            self.assertIsNone(generation.request([], lambda: 60.0, min_output_tokens=1024))
            self.assertEqual(generation.stop_reason, "insufficient_closure_budget")
            self.assertEqual(generation.calls, 0)
            self.assertEqual(generation.ledger.total_tokens, 0)

    def test_invalid_reservations_rejected(self) -> None:
        """A caller cannot inject a negative reserve or a boolean token cap."""
        generation = GenerationController(TeamModel([]), Budget(), TokenAccounting(), EventLog(), 8)
        for kwargs in ({"holdback_tokens": -1}, {"holdback_tokens": True}, {"min_output_tokens": 0}, {"min_output_tokens": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                generation.request([], lambda: 60.0, **kwargs)

    def test_default_generation_path_is_unchanged(self) -> None:
        """Workflow/single callers do not opt into sequential reservation floors."""
        model = TeamModel([candidate()], usage={"prompt_tokens": 10, "completion_tokens": 20})
        generation = GenerationController(model, Budget(max_total_tokens=65), TokenAccounting(), EventLog(), 8)
        generation.request([], lambda: 60.0)
        self.assertEqual(model.caps, [55])
        self.assertNotIn("min_output_tokens", next(e for e in generation.log.events if e["type"] == "preflight"))

    def test_projection_is_not_a_generation_or_usage(self) -> None:
        """Tokenizer work has timing/events, not fabricated LLM charges."""
        generation = GenerationController(TeamModel([]), Budget(), TokenAccounting(mode="usage", prompt_margin_tokens=512), EventLog(), 8)
        self.assertEqual(generation.projected_reserve([], lambda: 60.0, "author"), 522)
        self.assertEqual(generation.calls, 0)
        self.assertEqual(generation.ledger.responses, 0)
        self.assertEqual(generation.ledger.total_tokens, 0)

    def test_projection_respects_deadline(self) -> None:
        """No planning request is granted after the cooperative deadline."""
        generation = GenerationController(TeamModel([]), Budget(), TokenAccounting(), EventLog(), 8)
        self.assertIsNone(generation.projected_reserve([], lambda: 0.0, "critic"))
        self.assertEqual(generation.stop_reason, "episode_time_budget")

    def test_spending_every_cap_still_leaves_explicit_closing_call(self) -> None:
        """A supported three-call path survives worst-case allocated output usage."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "toy.sqlite"
            con = sqlite3.connect(path)
            con.executescript("CREATE TABLE items(name TEXT); INSERT INTO items VALUES('x');")
            con.close()
            model = SpendingModel([candidate(), review(), submit()])
            result = solve({"id": "closure-toy", "db_id": "toy", "question": "List item names"}, path, "sequential", model,
                           Budget(max_total_tokens=18500))
            self.assertEqual(result["stop_reason"], "submitted")
            self.assertLess(result["usage"]["total_tokens"], 18500)
            self.assertGreaterEqual(18500 - result["usage"]["total_tokens"], 3560)
            self.assertEqual((result["llm_calls"], result["tool_calls"]), (3, 2))
            packet = json.loads(model.calls[-1][-1]["content"])
            self.assertFalse(packet["allowed_actions"]["candidate"])
            self.assertFalse(packet["allowed_actions"]["diagnostic"])
            self.assertTrue(packet["allowed_actions"]["submit_sql"])
            self.assertTrue(all(cap >= 1024 for cap in model.caps))
            self.assertGreater(sum(e["type"] == "budget_projection" for e in result["events"]), 3)
            self.assertEqual(sum(e["type"] == "llm_request" for e in result["events"]), 3)
            self.assertNotIn("finalization_repair_started", {e["type"] for e in result["events"]})

    def test_projection_failure_is_fatal_without_model_retry(self) -> None:
        """Tokenizer failure cannot silently disable budget protection."""
        class BrokenCounter(TeamModel):
            """Fail the planning endpoint deterministically."""

            def count(self, messages: list[Message], timeout_s: float) -> int:
                """Raise before any completion is sent."""
                raise ClientError("synthetic tokenizer failure")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "toy.sqlite"
            con = sqlite3.connect(path)
            con.execute("CREATE TABLE items(name TEXT)")
            con.close()
            model = BrokenCounter([])
            result = solve({"id": "counter-toy", "db_id": "toy", "question": "List names"}, path, "sequential", model)
            self.assertEqual(result["stop_reason"], "tokenize_error")
            self.assertEqual(model.calls, [])
            self.assertEqual(result["tool_calls"], 0)
            self.assertEqual(result["usage"]["total_tokens"], 0)

    def test_grown_candidate_context_is_checked_before_sql_dispatch(self) -> None:
        """A known large candidate cannot bypass the earlier growth estimate."""
        class SizeCounter(TeamModel):
            """Expose synthetic prompt growth from a candidate marker."""

            def count(self, messages: list[Message], timeout_s: float) -> int:
                """Make the new candidate unaffordable without querying SQL."""
                current = json.loads(messages[-1]["content"]).get("current_candidate")
                return 20000 if current and "OVERSIZED_CONTEXT_MARKER" in current["sql"] else 10

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "toy.sqlite"
            con = sqlite3.connect(path)
            con.execute("CREATE TABLE items(name TEXT)")
            con.close()
            model = SizeCounter([candidate("SELECT name FROM items -- OVERSIZED_CONTEXT_MARKER"),
                                 candidate(), review(), submit()])
            result = solve({"id": "size-toy", "db_id": "toy", "question": "List item names"}, path, "sequential", model)
            self.assertEqual(result["stop_reason"], "submitted")
            self.assertEqual(result["tool_calls"], 2)
            self.assertNotIn("OVERSIZED_CONTEXT_MARKER", result["first_sql"])
            check = next(e for e in result["events"] if e["type"] == "candidate_closure_check")
            self.assertFalse(check["feasible"])

    def test_disabled_revision_is_not_executed_or_promoted(self) -> None:
        """An action outside the closing mask does not become fallback SQL."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "toy.sqlite"
            con = sqlite3.connect(path)
            con.execute("CREATE TABLE items(name TEXT)")
            con.close()
            new = "SELECT name FROM items WHERE 1=1"
            model = SpendingModel([candidate(), review(), candidate(new, ["question"]), "not JSON"])
            result = solve({"id": "mask-toy", "db_id": "toy", "question": "List item names"}, path, "sequential", model,
                           Budget(max_total_tokens=18500, max_llm_calls=4))
            self.assertIsNone(result["final_sql"])
            self.assertEqual(result["tool_calls"], 1)
            self.assertEqual(len(result["team"]["candidates"]), 1)
            self.assertEqual(result["stop_reason"], "finalization_repair_exhausted")

    def test_repair_does_not_reserve_another_repair(self) -> None:
        """The one reserved terminal attempt is released, not recursively renewed."""
        plan = ClosurePlanner.plan("author", False, 3000, 4096, {"author": 1700, "critic": 1000},
                                   COUNTS, repairing=True)
        self.assertTrue(plan.allowed["submit_sql"])
        self.assertFalse(plan.allowed["candidate"])
        self.assertFalse(plan.allowed["diagnostic"])
        self.assertEqual((plan.holdback_tokens, plan.repair_tokens, plan.repair_turns), (0, 0, 0))

    def test_exact_floor_boundary_for_first_closing_call(self) -> None:
        """One fewer token refuses the first attempt rather than borrowing its repair."""
        for tokens, expected in ((5583, False), (5584, True)):
            plan = ClosurePlanner.plan("author", False, tokens, 4096, {"author": 1000, "critic": 1000}, COUNTS)
            self.assertEqual(plan.allowed["submit_sql"], expected)

    def test_candidate_cost_includes_only_one_terminal_repair(self) -> None:
        """Projected candidate size and branch planner use identical downstream math."""
        plan = ClosurePlanner.plan("author", True, 32000, 4096, {"author": 1000, "critic": 1000}, COUNTS)
        self.assertEqual(plan.future_costs["candidate"], 15288)
        self.assertEqual(plan.future_costs["candidate"], ClosurePlanner.candidate_cost(plan.prompt_reserves, 1024, 1536))

    def test_closing_violation_has_room_for_one_explicit_repair(self) -> None:
        """Disabled revision/diagnostic and invalid JSON cannot consume the repair hold."""
        bad_actions = [candidate("SELECT name FROM items WHERE 1=1", ["question"]), diagnostic(),
                       "not JSON", {"action": "submit_sql", "candidate_version": 99, "sql_fingerprint": "x" * 64}]
        for bad in bad_actions:
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "toy.sqlite"
                con = sqlite3.connect(path)
                con.executescript("CREATE TABLE items(name TEXT); INSERT INTO items VALUES('x');")
                con.close()
                model = RepairSpendingModel([candidate(), review(), bad, submit()])
                result = solve({"id": "terminal-repair-toy", "db_id": "toy", "question": "List item names"},
                               path, "sequential", model, Budget(max_total_tokens=18500))
                self.assertEqual(result["stop_reason"], "submitted")
                self.assertEqual((result["llm_calls"], result["tool_calls"]), (4, 2))
                self.assertEqual(result["usage"]["total_tokens"], 18500)
                self.assertFalse(result["token_budget_overrun"])
                self.assertEqual(result["first_sql"], result["final_sql"])
                self.assertEqual(len(result["team"]["candidates"]), 1)
                self.assertEqual(result["finalization_mode"], "repair")
                first, repair_packet = [json.loads(messages[-1]["content"]) for messages in model.calls[-2:]]
                for packet in (first, repair_packet):
                    self.assertFalse(packet["allowed_actions"]["candidate"])
                    self.assertFalse(packet["allowed_actions"]["diagnostic"])
                    self.assertTrue(packet["allowed_actions"]["submit_sql"])
                self.assertEqual(first["allowed_actions"]["closure"]["finalization_repair_turns"], 1)
                self.assertEqual(repair_packet["allowed_actions"]["closure"]["finalization_repair_turns"], 0)
                self.assertEqual(len(model.calls[-1]), 4)
                self.assertNotIn("assistant", [m["role"] for m in model.calls[-1]])
                self.assertTrue(all(cap >= 1024 for cap in model.caps))

    def test_finalization_repair_never_renews_after_second_violation(self) -> None:
        """Even spare tokens and calls cannot start a third terminal generation."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "toy.sqlite"
            con = sqlite3.connect(path)
            con.execute("CREATE TABLE items(name TEXT)")
            con.close()
            # Two tool slots force finalization while plenty of tokens/calls remain.
            model = TeamModel([candidate(), review(), "not JSON", "still not JSON", submit()])
            result = solve({"id": "no-renew-toy", "db_id": "toy", "question": "List item names"},
                           path, "sequential", model, Budget(max_tool_calls=2))
            self.assertEqual(result["stop_reason"], "finalization_repair_exhausted")
            self.assertEqual((result["llm_calls"], result["tool_calls"]), (4, 1))
            self.assertIsNone(result["final_sql"])
            self.assertEqual(sum(e["type"] == "finalization_repair_started" for e in result["events"]), 1)
            self.assertEqual(sum(e["type"] == "finalization_repair_exhausted" for e in result["events"]), 1)

    def test_terminal_disagreement_still_requires_grounds(self) -> None:
        """A bounded repair cannot bypass exact-version review or disagreement checks."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "toy.sqlite"
            con = sqlite3.connect(path)
            con.execute("CREATE TABLE items(name TEXT)")
            con.close()
            model = RepairSpendingModel([candidate(), review(issues=[issue()]), submit(), submit(disagreement=True)])
            result = solve({"id": "disagreement-repair-toy", "db_id": "toy", "question": "List item names"},
                           path, "sequential", model, Budget(max_total_tokens=18500, max_llm_calls=4))
            self.assertEqual(result["stop_reason"], "submitted")
            self.assertTrue(result["team"]["submitted_with_disagreement"])
            self.assertEqual((result["llm_calls"], result["tool_calls"]), (4, 2))

    def test_terminal_feedback_does_not_echo_unbounded_response(self) -> None:
        """The audit keeps the response, but the bounded repair context does not."""
        from mas_sql.agent.sequential.roles import AuthorAgent
        agent = AuthorAgent({"question": "List item names"}, "CREATE TABLE items(name TEXT)")
        marker = "REJECTED_LONG_RESPONSE_" * 10000
        agent.last_response = marker
        agent.protocol_error("Bad action: " + marker, finalization=True)
        self.assertEqual(len(agent.repair_messages), 1)
        payload = json.loads(agent.repair_messages[0]["content"])
        self.assertEqual(len(payload["protocol_error"]), 240)
        self.assertNotIn(marker, agent.repair_messages[0]["content"])

    def test_closing_repair_keeps_usage_margin_and_call_caps(self) -> None:
        """The deployed usage accounting policy also protects an explicit repair."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "toy.sqlite"
            con = sqlite3.connect(path)
            con.execute("CREATE TABLE items(name TEXT)")
            con.close()
            model = RepairSpendingModel([candidate(), review(), "not JSON", submit()])
            result = solve({"id": "margin-repair-toy", "db_id": "toy", "question": "List item names"},
                           path, "sequential", model, Budget(max_total_tokens=22000, max_llm_calls=4),
                           accounting=TokenAccounting(mode="usage", prompt_margin_tokens=512))
            self.assertEqual(result["stop_reason"], "submitted")
            self.assertEqual((result["llm_calls"], result["tool_calls"]), (4, 2))
            self.assertLessEqual(result["usage"]["total_tokens"], 22000)
            self.assertTrue(result["usage_complete"])
            self.assertFalse(result["token_budget_overrun"])
            repair_preflight = [e for e in result["events"] if e["type"] == "preflight"][-1]
            self.assertEqual(repair_preflight["reserved_prompt_tokens"], 2212)
            self.assertEqual(repair_preflight["holdback_tokens"], 0)

    def test_completion_floor_respects_a_lower_configured_cap(self) -> None:
        """Repair allocation never silently raises a user's smaller output ceiling."""
        plan = ClosurePlanner.plan("author", False, 5000, 128, {"author": 1000, "critic": 1000}, COUNTS)
        self.assertEqual(plan.min_output_tokens, 128)
        self.assertEqual(plan.repair_tokens, 2664)
        self.assertTrue(plan.allowed["submit_sql"])


if __name__ == "__main__":
    unittest.main()