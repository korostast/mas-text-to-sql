"""Regressions for source-preserving snapshots, private repair and trace IDs."""
import json
import sqlite3
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import patch
from mas_sql.agent.runtime import solve
from mas_sql.agent.sequential.actions import TeamActionParser
from mas_sql.agent.sequential.prompts import CONTEXT_VERSION, PROMPT_VERSION, AUTHOR, CRITIC, turn_packet
from mas_sql.agent.sequential.state import EpisodeState
from mas_sql.database.executor import SQLLimits, execute
from test_sequential import TeamModel, SQL, candidate, diagnostic, issue, review, submit


class SnapshotContextTests(unittest.TestCase):
    """Exercise generic toy tasks without reusing pilot-specific corrections."""

    def setUp(self) -> None:
        """Create a local table and a gold-free task."""
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "toy.sqlite"
        con = sqlite3.connect(self.path)
        con.executescript("CREATE TABLE items(name TEXT NOT NULL PRIMARY KEY); INSERT INTO items VALUES('Alice'),('Bob');")
        con.close()
        self.task: dict[str, str] = {"id": "task-opaque", "db_id": "toy", "question": "List item names", "evidence": "Use item names"}

    def tearDown(self) -> None:
        """Delete the isolated database."""
        self.tmp.cleanup()

    def repeated_revision(self) -> tuple[dict[str, Any], TeamModel]:
        """Run a generic three-version sequence inside unchanged global caps."""
        second: str = SQL + " WHERE name IS NOT NULL"
        third: str = SQL + " WHERE 1=1"
        model: TeamModel = TeamModel([candidate(), review(issues=[issue()]),
            candidate(second, ["observation:1", "review:1"]), review(second, 2, [issue()]),
            candidate(third, ["observation:1", "review:1", "review:2"]), review(third, 3), submit(third, 3)])
        return solve(self.task, self.path, "sequential", model), model

    def test_one_snapshot_and_no_conversation_replay(self) -> None:
        """Every valid call uses base messages plus exactly one current snapshot."""
        result, model = self.repeated_revision()
        self.assertEqual(result["stop_reason"], "submitted")
        self.assertEqual(result["llm_calls"], 7)
        for messages in model.calls:
            self.assertEqual(len(messages), 3)
            self.assertEqual([message["role"] for message in messages], ["system", "user", "user"])
            self.assertEqual(set(json.loads(messages[1]["content"])), {"question", "evidence", "schema"})
            self.assertEqual(json.loads(messages[-1]["content"])["context_version"], CONTEXT_VERSION)

    def test_all_candidate_sql_versions_are_preserved_once(self) -> None:
        """Old SQL remains available without copies in execution observations."""
        result, model = self.repeated_revision()
        packet: dict[str, Any] = json.loads(model.calls[-1][-1]["content"])
        versions: list[dict[str, Any]] = packet["candidate_history"] + [packet["current_candidate"]]
        self.assertEqual(versions, result["team"]["candidates"])
        self.assertEqual(len({version["version"] for version in versions}), 3)
        self.assertEqual([observation["candidate_version"] for observation in packet["observations"]], [1, 2, 3])
        for observation in packet["observations"]:
            self.assertNotIn("sql", observation)
            self.assertNotIn("sql_fingerprint", observation)
            self.assertEqual(observation["result"]["status"], "ok")

    def test_current_review_is_not_duplicated_in_review_history(self) -> None:
        """Current and historical reports together contain each report once."""
        result, model = self.repeated_revision()
        packet: dict[str, Any] = json.loads(model.calls[-1][-1]["content"])
        self.assertEqual(packet["current_review"]["id"], "review:3")
        self.assertEqual([item["id"] for item in packet["reviews"]], ["review:1", "review:2"])
        self.assertEqual(packet["reviews"] + [packet["current_review"]], result["team"]["reviews"])

    def test_old_sources_remain_visible_and_citable_after_revisions(self) -> None:
        """No valid source is silently dropped to save prompt space."""
        result, model = self.repeated_revision()
        self.assertEqual(result["stop_reason"], "submitted")
        for messages in model.calls:
            packet: dict[str, Any] = json.loads(messages[-1]["content"])
            expected: list[str] = ["question", "schema", "evidence"]
            expected.extend(f"observation:{item['observation_id']}" for item in packet["observations"])
            review_ids: list[str] = [item["id"] for item in packet["reviews"]]
            if packet["current_review"] is not None:
                review_ids.append(packet["current_review"]["id"])
            expected.extend(review_ids)
            self.assertEqual(packet["available_sources"], expected)
        self.assertIn("review:1", result["investigation"]["revisions"][-1]["sources"])

    def test_diagnostic_sql_and_check_are_not_lost(self) -> None:
        """Compact tool references do not hide diagnostic query meaning."""
        model: TeamModel = TeamModel([candidate(), diagnostic(True), review(), diagnostic(), submit()])
        result: dict[str, Any] = solve(self.task, self.path, "sequential", model)
        self.assertEqual(result["stop_reason"], "submitted")
        packet: dict[str, Any] = json.loads(model.calls[-1][-1]["content"])
        diagnostics: list[dict[str, Any]] = [item for item in packet["observations"] if item["purpose"] == "diagnostic"]
        self.assertEqual([item["role"] for item in diagnostics], ["critic", "author"])
        for observation in diagnostics:
            self.assertEqual(observation["sql"], "SELECT COUNT(*) FROM items")
            self.assertEqual(observation["check"], "Check row count")
            self.assertIn("sql_fingerprint", observation)

    def test_only_latest_private_error_exchange_is_replayed(self) -> None:
        """Repeated invalid responses do not build a second growing history."""
        model: TeamModel = TeamModel(["INVALID_OLD_SENTINEL", "INVALID_NEW_SENTINEL", candidate(), review(), submit()])
        result: dict[str, Any] = solve(self.task, self.path, "sequential", model)
        self.assertEqual(result["stop_reason"], "submitted")
        self.assertEqual(len(model.calls[1]), 5)
        self.assertEqual(len(model.calls[2]), 5)
        self.assertIn("INVALID_OLD_SENTINEL", json.dumps(model.calls[1]))
        self.assertNotIn("INVALID_OLD_SENTINEL", json.dumps(model.calls[2]))
        self.assertIn("INVALID_NEW_SENTINEL", json.dumps(model.calls[2]))
        for messages in model.calls[3:]:
            self.assertEqual(len(messages), 3)
            self.assertNotIn("INVALID_NEW_SENTINEL", json.dumps(messages))

    def test_critic_private_error_is_not_transferred_to_author(self) -> None:
        """A peer sees validated public reports, not invalid reviewer prose."""
        model: TeamModel = TeamModel([candidate(), "CRITIC_PRIVATE_BAD_SENTINEL", review(), submit()])
        result: dict[str, Any] = solve(self.task, self.path, "sequential", model)
        self.assertEqual(result["stop_reason"], "submitted")
        self.assertIn("CRITIC_PRIVATE_BAD_SENTINEL", json.dumps(model.calls[2]))
        self.assertNotIn("CRITIC_PRIVATE_BAD_SENTINEL", json.dumps(model.calls[3]))
        self.assertEqual(len(model.calls[3]), 3)

    def test_expired_approval_still_requires_new_review(self) -> None:
        """Snapshot rebuilding cannot restore approval for a different version."""
        result, model = self.repeated_revision()
        for index in (3, 5):
            packet: dict[str, Any] = json.loads(model.calls[index][-1]["content"])
            self.assertIsNone(packet["current_review"])
            self.assertGreater(len(packet["reviews"]), 0)
        self.assertEqual(result["team"]["final_review"]["candidate_version"], 3)

    def test_final_failure_feedback_is_part_of_next_snapshot(self) -> None:
        """Removing raw history does not hide a rejected final execution."""
        calls: int = 0
        def fail_second(path: str | Path, sql: str, limits: SQLLimits) -> dict[str, Any]:
            """Fail the first explicit final execution only."""
            nonlocal calls
            calls += 1
            return {"status": "error", "error": "FINAL_FAILURE_SENTINEL", "wall_s": 0.0} if calls == 2 else execute(path, sql, limits)
        new: str = SQL + " WHERE 1=1"
        model: TeamModel = TeamModel([candidate(), review(), submit(), candidate(new, ["observation:2"]),
                                      review(new, 2), submit(new, 2)])
        with patch("mas_sql.agent.sequential.runtime.execute", side_effect=fail_second):
            result: dict[str, Any] = solve(self.task, self.path, "sequential", model)
        self.assertEqual(result["stop_reason"], "submitted")
        packet: dict[str, Any] = json.loads(model.calls[3][-1]["content"])
        self.assertFalse(packet["submission_feedback"]["accepted"])
        self.assertEqual(packet["submission_feedback"]["observation_id"], 2)
        self.assertEqual(packet["observations"][1]["result"]["error"], "FINAL_FAILURE_SENTINEL")
        self.assertEqual(len(model.calls[3]), 3)

    def test_disagreement_still_requires_explicit_author_action(self) -> None:
        """A compact snapshot preserves objections and the original rebuttal gate."""
        model: TeamModel = TeamModel([candidate(), review(issues=[issue()]), submit(disagreement=True)])
        result: dict[str, Any] = solve(self.task, self.path, "sequential", model)
        packet: dict[str, Any] = json.loads(model.calls[-1][-1]["content"])
        self.assertEqual(packet["current_review"]["verdict"], "request_changes")
        self.assertTrue(result["team"]["submitted_with_disagreement"])
        self.assertEqual(result["llm_calls"], 3)

    def test_review_event_uses_review_id_not_task_id(self) -> None:
        """Raw review events leave the reserved task-ID field untouched."""
        model: TeamModel = TeamModel([candidate(), review(), submit()])
        emitted: list[dict[str, Any]] = []
        result: dict[str, Any] = solve(self.task, self.path, "sequential", model, emit=emitted.append)
        event: dict[str, Any] = next(item for item in emitted if item["type"] == "critic_review")
        self.assertNotIn("id", event)
        self.assertEqual(event["review_id"], "review:1")
        self.assertEqual(result["team"]["reviews"][0]["id"], "review:1")
        self.assertEqual(result["last_submission"]["review_id"], "review:1")

    def test_context_version_is_recorded_in_results_and_call_events(self) -> None:
        """Manifest-facing context identity is separate from action protocol identity."""
        model: TeamModel = TeamModel([candidate(), review(), submit()])
        result: dict[str, Any] = solve(self.task, self.path, "sequential", model)
        self.assertEqual(result["context_version"], "public-snapshot-v2")
        self.assertEqual(result["prompt_version"], "sequential-action-contract-v3")
        self.assertEqual(result["protocol_version"], "sequential-author-critic-v2")
        self.assertEqual(result["events"][0]["context_version"], CONTEXT_VERSION)
        for event in result["events"]:
            if event["type"] in ("llm_request", "llm_response", "preflight"):
                self.assertEqual(event["context_version"], CONTEXT_VERSION)

    def test_snapshot_does_not_mutate_or_alias_public_state(self) -> None:
        """Editing a serialized packet cannot change authoritative observations."""
        state: EpisodeState = EpisodeState(True)
        action = TeamActionParser.parse(json.dumps(candidate()), "author")
        state.propose(action, 1)
        state.observe("author", action, {"status": "ok", "rows": [["ORIGINAL"]]}, 1)
        state.accept_review(TeamActionParser.parse(json.dumps(review(issues=[issue()])), "critic"))
        before: EpisodeState = deepcopy(state)
        packet: dict[str, Any] = turn_packet(state, {"llm_calls": 4}, {"submit_sql": True})
        self.assertEqual(state, before)
        packet["observations"][0]["result"]["rows"][0][0] = "MUTATED"
        packet["current_review"]["issues"][0]["sources"].append("future")
        self.assertEqual(state, before)

    def test_empty_evidence_is_not_in_available_sources(self) -> None:
        """Snapshot source indexing matches the state's actual reference validator."""
        self.task["evidence"] = ""
        model: TeamModel = TeamModel([candidate(), review(), submit()])
        result: dict[str, Any] = solve(self.task, self.path, "sequential", model)
        self.assertEqual(result["stop_reason"], "submitted")
        for messages in model.calls:
            self.assertNotIn("evidence", json.loads(messages[-1]["content"])["available_sources"])

    def test_grounding_prompt_is_generic_and_requires_direct_anchors(self) -> None:
        """Policy instructions require direct anchors, not hardcoded SQL repairs."""
        self.assertEqual(PROMPT_VERSION, "sequential-action-contract-v3")
        for phrase in ("PRIMARY KEY", "UNIQUE", "nullability", "EVERY explicit", "unit of counting", "hypotheses"):
            self.assertIn(phrase, CRITIC)
        self.assertIn("reviewer's premise", AUTHOR)
        self.assertIn("no mandatory diagnostic", CRITIC)
        self.assertIn("only the brief", CRITIC)
        for name in ("frpm", "CDSCode", "Student_Club", "Coldsnap"):
            self.assertNotIn(name, AUTHOR + CRITIC)


if __name__ == "__main__":
    unittest.main()
