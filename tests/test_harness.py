import ast
import dataclasses
import json
import random
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from typing import Any
from mas_sql.llm.types import ChatResponse, Message

from mas_sql.llm.client import ClientError, usage_counts
from mas_sql.data.io import digest, read_json, write_json
from mas_sql.data.dataset import validate_tasks
from mas_sql.data.dataset import choose, prepare
from mas_sql.evaluation.evaluator import evaluate
from mas_sql.data.manifest import verify_manifest
from mas_sql.evaluation.metrics import ex_bird, filtered_mean, rv_reward, soft_f1_bird
from mas_sql.agent.runtime import Budget, parse_action, solve
from mas_sql.database.executor import SQLLimits, execute, preview, schema


class FakeClient:
    def __init__(self, actions: list[Any], usage: dict[str, Any] | None = None,
                 finish_reason: str = "stop") -> None:
        """Prepare scripted actions and synthetic reported usage."""
        self.actions = iter(actions)
        self.messages = []
        self.request_caps = []
        self.usage = {"prompt_tokens": 10, "completion_tokens": 20,
                      "completion_tokens_details": {"reasoning_tokens": 12}} if usage is None else usage
        self.finish_reason = finish_reason

    def count(self, messages: list[Message], timeout_s: float) -> int:
        """Return a fixed count for deterministic budget tests."""
        return 10

    def complete(self, messages: list[Message], max_tokens: int, timeout_s: float) -> ChatResponse:
        """Return the next scripted completion or transport failure."""
        self.messages.append([m.copy() for m in messages])
        self.request_caps.append(max_tokens)
        action = next(self.actions)
        if isinstance(action, Exception):
            raise action
        return {"content": action if isinstance(action, str) else json.dumps(action),
                "reasoning": "private test reasoning", "usage": self.usage, "finish_reason": self.finish_reason}


def submit(sql: str, revision: dict[str, Any] | None = None) -> dict[str, Any]:
    """Create a submission with optional public revision grounds."""
    action: dict[str, Any] = {"action": "submit_sql", "sql": sql}
    if revision is not None:
        action["revision"] = revision
    return action


def candidate(sql: str, revision: dict[str, Any] | None = None) -> dict[str, Any]:
    """Create a candidate test with optional public revision grounds."""
    action: dict[str, Any] = {"action": "execute_sql", "purpose": "candidate", "sql": sql}
    if revision is not None:
        action["revision"] = revision
    return action


class DatabaseCase(unittest.TestCase):
    def setUp(self) -> None:
        """Create an isolated toy database and task."""
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "toy").mkdir()
        self.path = self.root / "toy" / "toy.sqlite"
        con = sqlite3.connect(self.path)
        con.executescript('CREATE TABLE "odd table"(id INTEGER, name TEXT, b BLOB); INSERT INTO "odd table" VALUES(1,\'Alice\',X\'01FF\'),(2,\'Bob\',NULL);')
        con.close()
        self.task = {"id": "42", "db_id": "toy", "question": "List names", "evidence": "name is person name", "difficulty": "simple"}

    def tearDown(self) -> None:
        """Remove temporary database artifacts."""
        self.tmp.cleanup()


class SQLTests(DatabaseCase):
    def test_select_and_schema(self) -> None:
        """Verify select and schema."""
        result = execute(self.path, 'SELECT name FROM "odd table" ORDER BY id')
        self.assertEqual(result["rows"], [["Alice"], ["Bob"]])
        self.assertIn('CREATE TABLE "odd table"', schema(self.path))
        self.assertNotIn("Alice", schema(self.path))

    def test_blob(self) -> None:
        """Verify blob."""
        self.assertEqual(execute(self.path, 'SELECT b FROM "odd table" WHERE id=1')["rows"], [[{"$blob_base64": "Af8="}]])

    def test_readonly_adversarial(self) -> None:
        """Verify readonly adversarial."""
        before = digest(self.path)
        for sql in ["DELETE FROM 'odd table'", "DROP TABLE 'odd table'", "ATTACH DATABASE '/tmp/bad.sqlite' AS bad", "PRAGMA user_version=5", "SELECT 1; DELETE FROM 'odd table'", "WITH x AS (SELECT 1) DELETE FROM 'odd table'", "SELECT load_extension('/tmp/no.so')", "SELECT writefile('/tmp/bad','data')"]:
            with self.subTest(sql=sql):
                self.assertNotEqual(execute(self.path, sql)["status"], "ok")
        self.assertEqual(digest(self.path), before)

    def test_comments_and_cte(self) -> None:
        """Verify comments and cte."""
        self.assertEqual(execute(self.path, '/* hi */ --comment\nWITH x AS (SELECT 1 AS n) SELECT n FROM x')["rows"], [[1]])

    def test_no_silent_row_limit(self) -> None:
        """Verify no silent row limit."""
        result = execute(self.path, 'SELECT id FROM "odd table"', SQLLimits(max_rows=1))
        self.assertEqual(result["status"], "resource_limit")
        self.assertNotIn("rows", result)

    def test_byte_limit(self) -> None:
        """Verify byte limit."""
        result = execute(self.path, "SELECT printf('%01000d',1)", SQLLimits(max_bytes=100))
        self.assertNotEqual(result["status"], "ok")

    def test_timeout(self) -> None:
        """Verify timeout."""
        result = execute(self.path, 'WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n) SELECT sum(x) FROM n', SQLLimits(timeout_s=0.05))
        self.assertNotEqual(result["status"], "ok")
        self.assertLess(result["wall_s"], 2)

    def test_preview_and_long_columns(self) -> None:
        """Verify preview and long columns."""
        result = execute(self.path, 'SELECT id FROM "odd table" ORDER BY id')
        p = preview(result, SQLLimits(preview_rows=1))
        self.assertTrue(p["preview_truncated"])
        self.assertEqual(p["row_count"], 2)
        self.assertEqual(len(result["rows"]), 2)
        p = preview({**result, "columns": ["a" * 10000]}, SQLLimits(preview_chars=256))
        self.assertLessEqual(len(json.dumps(p)), 256)


class RuntimeTests(DatabaseCase):
    def test_workflow_one_call_no_feedback(self) -> None:
        """Verify workflow one call no feedback."""
        client = FakeClient([submit('SELECT name FROM "odd table"')])
        result = solve(self.task, self.path, "workflow", client)
        self.assertEqual(result["llm_calls"], 1)
        self.assertEqual(result["tool_calls"], 1)
        self.assertEqual(result["stop_reason"], "submitted")
        self.assertEqual(result["first_sql"], result["final_sql"])
        self.assertEqual(result["usage"]["total_tokens"], 30)  # reasoning is a subset, not +12
        self.assertEqual(result["usage"]["reasoning_tokens"], 12)
        self.assertNotIn("Alice", json.dumps(client.messages))
        self.assertNotIn("reasoning", json.dumps(client.messages))

    def test_single_repairs_and_preserves_first(self) -> None:
        """Verify single repairs and preserves first."""
        client = FakeClient([candidate("SELECT unknown FROM missing"), candidate('SELECT name FROM "odd table"', {"reason": "Fix identifiers using the error and DDL", "sources": ["schema", "observation:1"]}), submit('SELECT name FROM "odd table"')])
        result = solve(self.task, self.path, "single", client)
        self.assertEqual(result["stop_reason"], "submitted")
        self.assertEqual(result["first_sql"], "SELECT unknown FROM missing")
        self.assertEqual(result["tool_calls"], 3)
        self.assertIn("no such table", json.dumps(client.messages[1]))
        self.assertEqual(result["usage"]["total_tokens"], 90)

    def test_budget_unfinished_no_last_candidate_fallback(self) -> None:
        """Verify budget unfinished no last candidate fallback."""
        result = solve(self.task, self.path, "single", FakeClient([candidate("SELECT 1")]), Budget(max_llm_calls=1))
        self.assertIsNone(result["final_sql"])
        self.assertEqual(result["first_sql"], "SELECT 1")
        self.assertEqual(result["stop_reason"], "llm_call_budget")

    def test_final_is_checked_not_previous_candidate(self) -> None:
        """Keep failed final attempts out of accepted single-agent answers."""
        client = FakeClient([candidate("SELECT 1"), submit("SELECT missing FROM absent", {
            "reason": "Use the requested relation", "sources": ["question"]})])
        result = solve(self.task, self.path, "single", client, Budget(max_llm_calls=2))
        self.assertEqual(result["stop_reason"], "llm_call_budget")
        self.assertIsNone(result["final_sql"])
        self.assertNotEqual(result["last_submission"]["result"]["status"], "ok")
        self.assertFalse(result["submission_accepted"])

    def test_diagnostic_not_first_candidate(self) -> None:
        """Verify diagnostic not first candidate."""
        diag = {"action": "execute_sql", "purpose": "diagnostic", "check": "Check the literal two", "sql": "SELECT 2"}
        result = solve(self.task, self.path, "single", FakeClient([diag, candidate("SELECT 1"), diag, submit("SELECT 1")]))
        self.assertEqual(result["first_sql"], "SELECT 1")
        self.assertEqual(result["tool_calls"], 3)
        self.assertEqual(result["llm_calls"], 4)

    def test_last_tool_slot_reserved(self) -> None:
        """Verify last tool slot reserved."""
        result = solve(self.task, self.path, "single", FakeClient([candidate("SELECT 1"), candidate("SELECT 2", {"reason": "Use requested constant", "sources": ["question"]}), submit("SELECT 2", {"reason": "Use requested constant", "sources": ["question"]})]), Budget(max_tool_calls=2))
        self.assertEqual(result["tool_calls"], 2)
        self.assertEqual(result["stop_reason"], "submitted")
        self.assertTrue(any(e["type"] == "tool_rejected" for e in result["events"]))

    def test_invalid_json_recoverable_but_workflow_not_retried(self) -> None:
        """Verify invalid json recoverable but workflow not retried."""
        result = solve(self.task, self.path, "single", FakeClient(["oops", submit("SELECT 1")]))
        self.assertEqual(result["stop_reason"], "submitted")
        result = solve(self.task, self.path, "workflow", FakeClient(["oops"]))
        self.assertEqual(result["stop_reason"], "protocol_error")
        self.assertEqual(result["llm_calls"], 1)

    def test_unknown_usage_halts_not_zero(self) -> None:
        """Verify unknown usage halts not zero."""
        result = solve(self.task, self.path, "single", FakeClient([submit("SELECT 1")], usage={}))
        self.assertEqual(result["stop_reason"], "usage_missing")
        self.assertIsNone(result["usage"]["total_tokens"])
        self.assertEqual(result["tool_calls"], 0)

    def test_transport_unknown_no_retry(self) -> None:
        """Verify transport unknown no retry."""
        result = solve(self.task, self.path, "single", FakeClient([ClientError("lost")]))
        self.assertEqual(result["llm_calls"], 1)
        self.assertFalse(result["usage_complete"])
        self.assertEqual(result["stop_reason"], "transport_error_usage_unknown")

    def test_token_budget_preflight(self) -> None:
        """Verify token budget preflight."""
        result = solve(self.task, self.path, "single", FakeClient([]), Budget(max_total_tokens=10))
        self.assertEqual(result["llm_calls"], 0)
        self.assertEqual(result["stop_reason"], "token_budget")
        client = FakeClient([submit("SELECT 1")], usage={"prompt_tokens": 10, "completion_tokens": 15})
        result = solve(self.task, self.path, "single", client, Budget(max_total_tokens=25))
        self.assertEqual(client.request_caps, [15])
        self.assertEqual(result["usage"]["total_tokens"], 25)

    def test_token_mismatch_halts(self) -> None:
        """Verify token mismatch halts."""
        result = solve(self.task, self.path, "single", FakeClient([submit("SELECT 1")], usage={"prompt_tokens": 11, "completion_tokens": 10}))
        self.assertEqual(result["stop_reason"], "tokenizer_mismatch")
        self.assertEqual(result["tool_calls"], 0)

    def test_token_overrun_reported(self) -> None:
        """Verify token overrun reported."""
        result = solve(self.task, self.path, "single", FakeClient([submit("SELECT 1")]), Budget(max_total_tokens=25))
        self.assertTrue(result["token_budget_overrun"])
        self.assertEqual(result["stop_reason"], "token_budget_overrun")

    def test_truncated_content_not_submitted(self) -> None:
        """Verify truncated content not submitted."""
        result = solve(self.task, self.path, "workflow", FakeClient([submit("SELECT 1")], finish_reason="length"))
        self.assertEqual(result["stop_reason"], "completion_incomplete")
        self.assertIsNone(result["final_sql"])

    def test_gold_forbidden_in_tasks(self) -> None:
        """Verify gold forbidden in tasks."""
        with self.assertRaises(ValueError):
            validate_tasks([{**self.task, "SQL": "SELECT secret"}])

    def test_protocol_strict(self) -> None:
        """Verify protocol strict."""
        for value in ['{"action":"submit_sql","sql":"SELECT 1","sql":"SELECT 2"}', '```json\n{}\n```', 'null', '{"action":"shell","sql":"SELECT 1"}']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_action(value, "single")


class MetricTests(unittest.TestCase):
    def test_ex_set_quirks(self) -> None:
        """Verify ex set quirks."""
        self.assertEqual(ex_bird([[1], [2], [1]], [[2], [1]]), 1)
        self.assertEqual(ex_bird([[1, 2]], [[2, 1]]), 0)
        self.assertEqual(ex_bird([[1.0]], [[1]]), 1)

    def test_softf1_quirks(self) -> None:
        """Verify softf1 quirks."""
        self.assertEqual(soft_f1_bird([[1], [2]], [[2], [1]]), 0)
        self.assertEqual(soft_f1_bird([[1, 1]], [[1]]), 1)
        self.assertEqual(soft_f1_bird([[2, 1]], [[1, 2]]), 1)
        self.assertEqual(soft_f1_bird([], []), 1)

    def test_original_metric_regression(self) -> None:
        """Verify original metric regression."""
        # Compare many random cases against the untouched upstream functions.
        upstream = Path(__file__).resolve().parents[1] / "evaluation" / "evaluation_f1.py"
        tree = ast.parse(upstream.read_text())
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {"calculate_row_match", "calculate_f1_score"}]
        scope = {}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(upstream), "exec"), scope)
        rng = random.Random(71)
        for _ in range(300):
            pred = [tuple(rng.choices([None, 0, 1, 2, "a"], k=rng.randrange(1, 4))) for _ in range(rng.randrange(5))]
            gold = [tuple(rng.choices([None, 0, 1, 2, "a"], k=rng.randrange(1, 4))) for _ in range(rng.randrange(5))]
            self.assertAlmostEqual(soft_f1_bird(pred, gold), scope["calculate_f1_score"](pred, gold))

    def test_rves_aggregation_safe(self) -> None:
        """Verify rves aggregation safe."""
        self.assertEqual(filtered_mean([1, 1, 1]), 1)
        self.assertEqual(filtered_mean([2]), 2)
        self.assertEqual([rv_reward(x) for x in [2, 1, .5, .25, .01]], [1.25, 1, .75, .5, .25])


class DataEvalTests(DatabaseCase):
    def make_data(self) -> Path:
        """Create a stratifiable synthetic dataset with explicit IDs."""
        dataset = self.root / "dataset.json"
        write_json(dataset, [{"question_id": 100 + i, "db_id": "toy", "question": f"List names {i}", "evidence": "known", "SQL": 'SELECT name FROM "odd table"', "difficulty": ["simple", "moderate", "challenging"][i % 3]} for i in range(90)])
        return dataset

    def test_splits_reproducible_nonoverlap_gold_isolated(self) -> None:
        """Verify splits reproducible nonoverlap gold isolated."""
        dataset = self.make_data()
        first = self.root / "a"
        second = self.root / "b"
        a = prepare(dataset, self.root, first)
        b = prepare(dataset, self.root, second)
        self.assertEqual(a, b)
        self.assertEqual(len(read_json(first / "eval.tasks.json")), 50)
        self.assertEqual(len(read_json(first / "pilot.tasks.json")), 10)
        self.assertFalse(set(a["splits"]["eval"]["ids"]) & set(a["splits"]["pilot"]["ids"]))
        self.assertNotIn('"SQL"', (first / "eval.tasks.json").read_text())
        validate_tasks(read_json(first / "eval.tasks.json"))
        with self.assertRaises(ValueError):
            prepare(dataset, self.root, first)
        verify_manifest(first / "eval.tasks.json", self.root, first / "eval.gold.json")
        with open(self.path, "ab") as f:
            f.write(b"tamper")
        with self.assertRaises(ValueError):
            verify_manifest(first / "eval.tasks.json", self.root)

    def test_cover_all_dbs_matching(self) -> None:
        """Verify cover all dbs matching."""
        rows = [{"id": f"{db}-{d}-{i}", "db_id": str(db), "difficulty": d} for db in range(11) for d in ["simple", "moderate", "challenging"] for i in range(4)]
        result = choose(rows, {"simple": 15, "moderate": 25, "challenging": 10}, random.Random(42), True)
        self.assertEqual(len(result), 50)
        self.assertEqual(len({r["db_id"] for r in result}), 11)
        self.assertEqual(len({r["id"] for r in result}), 50)

    def test_missing_difficulty_fails(self) -> None:
        """Verify missing difficulty fails."""
        dataset = self.make_data()
        items = read_json(dataset)
        del items[0]["difficulty"]
        write_json(dataset, items)
        with self.assertRaises(ValueError):
            prepare(dataset, self.root, self.root / "out")

    def test_offline_eval_id_join_and_incorrect_rves_zero(self) -> None:
        """Verify offline eval id join and incorrect rves zero."""
        split = self.root / "split"
        prepare(self.make_data(), self.root, split, eval_quotas=(1, 1, 1), pilot_quotas=(1, 1, 1))
        tasks = read_json(split / "eval.tasks.json")
        out = self.root / "run"
        write_json(out / "run.json", {"tasks_sha256": digest(split / "eval.tasks.json"), "manifest_sha256": digest(split / "manifest.json")})
        predictions = [{"id": t["id"], "db_id": t["db_id"], "first_sql": 'SELECT name FROM "odd table"', "final_sql": "SELECT 99", "stop_reason": "submitted"} for t in reversed(tasks)]
        write_json(out / "results.json", predictions)
        report = evaluate(split / "eval.tasks.json", split / "eval.gold.json", out, self.root, self.root / "report.json", rves_repeats=2)
        self.assertEqual(report["summary"]["EX_first"], 1)
        self.assertEqual(report["summary"]["EX_final"], 0)
        self.assertEqual(report["summary"]["R_VES_safe_v1"], 0)
        self.assertEqual(report["summary"]["regressions"], 3)
        predictions[0]["id"] = "unknown"
        write_json(out / "results.json", predictions)
        with self.assertRaises(ValueError):
            evaluate(split / "eval.tasks.json", split / "eval.gold.json", out, self.root, self.root / "fail.json")


class MoreRegressionTests(DatabaseCase):
    def test_partial_usage_retains_known_not_zero_total(self) -> None:
        """Verify partial usage retains known not zero total."""
        result = solve(self.task, self.path, "workflow", FakeClient([submit("SELECT 1")], usage={"prompt_tokens": 10}))
        self.assertIsNone(result["usage"]["total_tokens"])
        self.assertEqual(result["usage_known_totals"]["input_tokens"], 10)
        self.assertEqual(result["usage_known_totals"]["total_tokens"], 10)

    def test_long_error_preview_bounded(self) -> None:
        """Verify long error preview bounded."""
        result = preview({"status": "error", "error": "\\\"\n" * 50000}, SQLLimits(preview_chars=256))
        self.assertLessEqual(len(json.dumps(result, ensure_ascii=False)), 256)

    def test_server_output_cap_violation(self) -> None:
        """Verify server output cap violation."""
        result = solve(self.task, self.path, "single", FakeClient([submit("SELECT 1")]), Budget(max_output_tokens=1))
        self.assertEqual(result["stop_reason"], "server_output_cap_violation")
        self.assertEqual(result["tool_calls"], 0)

    def test_invalid_budget_types(self) -> None:
        """Verify invalid budget types."""
        for kw in [{"max_llm_calls": 1.5}, {"max_total_tokens": True}, {"max_episode_s": float("inf")}]:
            with self.assertRaises(ValueError):
                Budget(**kw)
        for kw in [{"max_rows": 1.5}, {"timeout_s": float("nan")}, {"preview_chars": 5}]:
            with self.assertRaises(ValueError):
                SQLLimits(**kw)


if __name__ == "__main__":
    unittest.main()
