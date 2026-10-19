"""Regression tests for malformed envelopes, preserved usage, and safe diagnostics."""
import json
import os
import sqlite3
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import Mock, patch

from mas_sql.agent.actions import ActionParser
from mas_sql.agent.budgets import Budget
from mas_sql.agent.events import EventLog
from mas_sql.agent.generation import GenerationController
from mas_sql.agent.prompts import PROMPT_VERSION, SINGLE
from mas_sql.agent.runtime import solve
from mas_sql.llm.accounting import TokenAccounting
from mas_sql.llm.client import ChatClient
from mas_sql.llm.errors import ClientError, ResponseFormatError
from mas_sql.llm.probe import ProbeRunner
from mas_sql.llm.responses import ChatResponseParser
from mas_sql.llm.types import Message


def envelope(content: Any = None, usage: Any = None, finish: Any = "stop") -> dict[str, Any]:
    """Build a synthetic server envelope without benchmark-specific content."""
    return {"id": "mock-response", "choices": [{"message": {"content": content,
             "reasoning_content": "PRIVATE_REASONING"}, "finish_reason": finish}], "usage": usage}


class ResponseParserTests(unittest.TestCase):
    """Validate envelopes independently of the model's action JSON."""

    def test_valid_response_has_safe_diagnostics(self) -> None:
        """Keep normal content and attach a bounded structural summary."""
        result = ChatResponseParser.parse(envelope('{"action":"submit_sql","sql":"SELECT 1"}',
                                                  {"prompt_tokens": 10, "completion_tokens": 4}), 200)
        self.assertEqual(result["diagnostics"]["http_status"], 200)
        self.assertEqual(result["diagnostics"]["content_type"], "string")
        self.assertEqual(result["diagnostics"]["choices_count"], 1)
        self.assertNotIn("PRIVATE_REASONING", json.dumps(result["diagnostics"]))

    def test_malformed_shape_matrix_preserves_usage(self) -> None:
        """Classify malformed shapes without losing independent token counts."""
        usage: dict[str, int] = {"prompt_tokens": 10, "completion_tokens": 4}
        cases: list[tuple[object, str]] = [
            ([], "response_not_object"),
            ({"usage": usage}, "choices_not_array"),
            ({"choices": {}, "usage": usage}, "choices_not_array"),
            ({"choices": [], "usage": usage}, "choices_empty"),
            ({"choices": [1], "usage": usage}, "choice_not_object"),
            ({"choices": [{"message": []}], "usage": usage}, "message_not_object"),
            ({"choices": [{"message": {}}], "usage": usage}, "content_missing"),
            (envelope(None, usage), "content_not_string"),
            (envelope([{"text": "BODY_SECRET"}], usage), "content_not_string"),
            (envelope({}, usage), "content_not_string"),
            (envelope("text", usage, 1), "finish_reason_not_string"),
            ({"choices": [{"message": {"content": "a"}}, {"message": {"content": "b"}}],
              "usage": usage}, "unexpected_choice_count"),
        ]
        for body, code in cases:
            with self.subTest(code=code, body_type=type(body).__name__):
                with self.assertRaises(ResponseFormatError) as caught:
                    ChatResponseParser.parse(body, 200)
                error = caught.exception
                self.assertEqual(error.code, code)
                self.assertEqual(error.usage, usage if isinstance(body, dict) else None)
                self.assertEqual(error.kind, "response_format")
                self.assertNotIn("BODY_SECRET", str(error) + json.dumps(error.diagnostics))
                self.assertNotIn("PRIVATE_REASONING", str(error) + json.dumps(error.diagnostics))

    def test_empty_string_is_an_action_protocol_problem(self) -> None:
        """Leave empty string validation to the existing action parser."""
        self.assertEqual(ChatResponseParser.parse(envelope(""))["content"], "")

    def test_null_content_never_falls_back_to_reasoning(self) -> None:
        """Reject reasoning-only or tool-only responses rather than executing them."""
        body = envelope(None, {"prompt_tokens": 10, "completion_tokens": 4}, "tool_calls")
        body["choices"][0]["message"]["reasoning_content"] = '{"action":"submit_sql","sql":"SELECT 1"}'
        with self.assertRaises(ResponseFormatError) as caught:
            ChatResponseParser.parse(body)
        self.assertEqual(caught.exception.diagnostics["finish_reason"], "tool_calls")

    def test_nonstandard_metadata_is_not_dumped(self) -> None:
        """Drop arbitrary fields and bound recognized text metadata."""
        body = envelope(None)
        body.update(id="BAD\nID", unexpected="BODY_SECRET")
        body["choices"][0]["finish_reason"] = {"secret": "BODY_SECRET"}
        with self.assertRaises(ResponseFormatError) as caught:
            ChatResponseParser.parse(body)
        diagnostic = caught.exception.diagnostics
        self.assertIsNone(diagnostic["response_id"])
        self.assertEqual(diagnostic["finish_reason"], "other")
        self.assertNotIn("BODY_SECRET", json.dumps(diagnostic))


class GenerationResponseTests(unittest.TestCase):
    """Charge reported usage once and stop safely on a malformed envelope."""

    def request(self, body: object, budget: Budget | None = None,
                accounting: TokenAccounting | None = None) -> tuple[GenerationController, EventLog]:
        """Make one synthetic request through the actual response parser."""
        client = ChatClient({"model": "mock"})
        log = EventLog()
        controller = GenerationController(client, budget or Budget(), accounting or TokenAccounting("usage", 0), log, 8)
        client.last_http_status = 200
        with patch.object(client, "count", return_value=10), patch.object(client, "post", return_value=body) as post:
            self.assertIsNone(controller.request([{"role": "user", "content": "test"}], lambda: 30.0))
            self.assertEqual(post.call_count, 1)
        return controller, log

    def test_malformed_complete_usage_is_charged(self) -> None:
        """Keep exact known usage despite the absence of action text."""
        controller, log = self.request(envelope(None, {"prompt_tokens": 12, "completion_tokens": 7}))
        self.assertEqual(controller.stop_reason, "response_format_error")
        self.assertTrue(controller.ledger.complete)
        self.assertEqual(controller.ledger.total_tokens, 19)
        self.assertEqual(controller.ledger.responses, 1)
        self.assertEqual(controller.mismatch_count, 1)
        event = next(e for e in log.events if e["type"] == "llm_error")
        self.assertFalse(event["usage_unknown"])
        self.assertEqual(event["diagnostics"]["http_status"], 200)
        self.assertFalse(any(e["type"] == "llm_response" for e in log.events))

    def test_partial_usage_stays_incomplete(self) -> None:
        """Retain known input without treating missing output as zero usage."""
        controller, _ = self.request(envelope(None, {"prompt_tokens": 12}))
        self.assertEqual(controller.stop_reason, "response_format_error_usage_unknown")
        self.assertFalse(controller.ledger.complete)
        self.assertIsNone(controller.ledger.snapshot()["total_tokens"])
        self.assertEqual(controller.ledger.snapshot(True)["input_tokens"], 12)

    def test_missing_usage_is_not_invented(self) -> None:
        """Keep absent usage unknown for an otherwise parsed response body."""
        controller, _ = self.request(envelope())
        self.assertEqual(controller.stop_reason, "response_format_error_usage_unknown")
        self.assertIsNone(controller.ledger.snapshot()["input_tokens"])

    def test_accounting_violations_keep_priority(self) -> None:
        """Enforce actual limits even when response text is unusable."""
        cases: list[tuple[Budget, dict[str, int], TokenAccounting, str]] = [
            (Budget(max_total_tokens=50), {"prompt_tokens": 10, "completion_tokens": 90},
             TokenAccounting("usage", 0), "token_budget_overrun"),
            (Budget(max_context_tokens=40), {"prompt_tokens": 50, "completion_tokens": 1},
             TokenAccounting("usage", 0), "context_budget_overrun"),
            (Budget(max_output_tokens=5), {"prompt_tokens": 10, "completion_tokens": 6},
             TokenAccounting("usage", 0), "server_output_cap_violation"),
            (Budget(), {"prompt_tokens": 12, "completion_tokens": 7},
             TokenAccounting(), "tokenizer_mismatch"),
        ]
        for budget, usage, accounting, expected in cases:
            with self.subTest(expected=expected):
                controller, _ = self.request(envelope(None, usage), budget, accounting)
                self.assertEqual(controller.stop_reason, expected)
                self.assertEqual(controller.ledger.total_tokens, usage["prompt_tokens"] + usage["completion_tokens"])

    def test_failure_log_does_not_dump_response_body(self) -> None:
        """Log normalized counters rather than arbitrary usage/body fields."""
        body = envelope(["BODY_SECRET"], {"prompt_tokens": 10, "completion_tokens": 4, "extra": "BODY_SECRET"})
        body["error"] = "BODY_SECRET"
        _, log = self.request(body)
        self.assertNotIn("BODY_SECRET", json.dumps(log.events))
        self.assertNotIn("PRIVATE_REASONING", json.dumps(log.events))

    def test_episode_keeps_prior_usage_and_never_accepts_reasoning(self) -> None:
        """Stop a multi-turn episode without promoting its last tested candidate."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "toy.sqlite"
            connection = sqlite3.connect(path)
            connection.execute("CREATE TABLE toy(n INTEGER)")
            connection.close()
            client = ChatClient({"model": "mock"})
            first = envelope('{"action":"execute_sql","purpose":"candidate","sql":"SELECT 1"}',
                             {"prompt_tokens": 10, "completion_tokens": 5})
            second = envelope(None, {"prompt_tokens": 10, "completion_tokens": 7})
            second["choices"][0]["message"]["reasoning_content"] = '{"action":"submit_sql","sql":"SELECT 1"}'
            task = {"id": "toy", "db_id": "toy", "question": "Return one", "evidence": ""}
            with patch.object(client, "count", return_value=10), patch.object(client, "post", side_effect=[first, second]) as post:
                result = solve(task, path, "single", client)
            self.assertEqual(post.call_count, 2)
            self.assertEqual(result["stop_reason"], "response_format_error")
            self.assertEqual(result["usage"]["total_tokens"], 32)
            self.assertTrue(result["usage_complete"])
            self.assertEqual(result["tool_calls"], 1)
            self.assertEqual(result["first_sql"], "SELECT 1")
            self.assertIsNone(result["final_sql"])
            self.assertFalse(result["submission_accepted"])

    def test_probe_preserves_usage_but_remains_incompatible(self) -> None:
        """Do not let valid accounting turn an invalid probe into a pass."""
        client = ChatClient({"model": "mock"})
        runner = ProbeRunner(client, TokenAccounting("usage", 0), 100, 200, 30)
        saved = Mock()
        with patch.object(client, "count", return_value=10), patch.object(client, "post", return_value=envelope(None, {"prompt_tokens": 12, "completion_tokens": 7})) as post:
            report = runner.run(saved)
        self.assertEqual(post.call_count, 1)
        self.assertFalse(report["compatible"])
        self.assertFalse(report["complete"])
        self.assertEqual(report["cases"][0]["usage"]["total_tokens"], 19)
        self.assertTrue(report["cases"][0]["usage_complete"])
        self.assertEqual(report["cases"][0]["error_kind"], "response_format")
        self.assertNotIn("PRIVATE_REASONING", json.dumps(report))


class ResponseHTTP(BaseHTTPRequestHandler):
    """Serve controlled local responses for transport-boundary regression tests."""

    status: ClassVar[int] = 200
    body: ClassVar[bytes] = b'{}'
    calls: ClassVar[int] = 0

    def log_message(self, format: str, *args: object) -> None:
        """Suppress local HTTP server noise."""
        pass

    def do_POST(self) -> None:
        """Return the configured status and body exactly once per request."""
        self.rfile.read(int(self.headers["Content-Length"]))
        type(self).calls += 1
        self.send_response(self.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(self.body)))
        self.end_headers()
        self.wfile.write(self.body)


class ResponseHTTPTests(unittest.TestCase):
    """Exercise real local HTTP without contacting any model endpoint."""

    def test_status_classification_no_retry_and_no_body_leak(self) -> None:
        """Distinguish malformed envelopes, invalid JSON, and HTTP failures."""
        server = ThreadingHTTPServer(("127.0.0.1", 0), ResponseHTTP)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        clean_env = {k:v for k,v in os.environ.items() if not k.startswith("MAS_SQL_")}
        try:
            with patch.dict(os.environ, clean_env, clear=True):
                client = ChatClient({"model": "mock", "base_url": f"http://127.0.0.1:{server.server_port}/v1"})
            cases: list[tuple[int, bytes, str]] = [
                (200, json.dumps(envelope(None, {"prompt_tokens": 10, "completion_tokens": 4})).encode(), "response_format"),
                (200, b'[]', "response_format"),
                (200, b'INVALID_JSON_BODY_SECRET', "response_json"),
                (429, b'HTTP_BODY_SECRET', "http"),
                (503, b'HTTP_BODY_SECRET', "http"),
            ]
            for status, body, kind in cases:
                with self.subTest(status=status, kind=kind):
                    ResponseHTTP.status, ResponseHTTP.body, ResponseHTTP.calls = status, body, 0
                    with self.assertRaises(ClientError) as caught:
                        client.complete([{"role": "user", "content": "test"}], 100, 30)
                    self.assertEqual(caught.exception.kind, kind)
                    self.assertEqual(caught.exception.http_status, status)
                    self.assertEqual(ResponseHTTP.calls, 1)
                    self.assertNotIn("BODY_SECRET", str(caught.exception))
                    if isinstance(caught.exception, ResponseFormatError):
                        self.assertEqual(caught.exception.diagnostics["http_status"], status)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


class PromptContractTests(unittest.TestCase):
    """Keep complete action examples without task-specific benchmark hints."""

    def test_observation_shaped_actions_get_explicit_feedback(self) -> None:
        """Reject fabricated tool results with a boundary-specific explanation."""
        content = json.dumps({"tool": "execute_sql", "observation_id": 3,
                              "result": {"status": "ok", "rows": [[0]]}})
        with self.assertRaisesRegex(ValueError, "Only runtime returns tool observations"):
            ActionParser.parse(content, "single")

    def test_examples_are_valid_actions_and_revision_is_attached(self) -> None:
        """Validate syntax-only examples through the production action parser."""
        examples = [line for line in SINGLE.splitlines() if line.startswith('{"action":') and line.endswith('}')]
        self.assertGreaterEqual(len(examples), 4)
        for line in examples:
            with self.subTest(action=line):
                ActionParser.parse(line, "single")
        revisions = [json.loads(line) for line in examples if '"revision"' in line]
        self.assertTrue(any(action["action"] == "submit_sql" for action in revisions))
        self.assertIn('Only the runtime returns tool results', SINGLE)
        self.assertIn('no mandatory diagnostic or repair sequence', SINGLE)
        self.assertEqual(PROMPT_VERSION, "action-contract-v2")
