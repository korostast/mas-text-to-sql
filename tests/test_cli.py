"""End-to-end CLI against a local mock HTTP server, NOT against a real LLM."""
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, ClassVar
from mas_sql.data.io import read_json, write_json


class MockHTTP(BaseHTTPRequestHandler):
    """Emulate chat rendering drift only when a system message is present."""

    requests: ClassVar[list[tuple[str, dict[str, Any]]]] = []
    system_delta: ClassVar[int] = 0
    simulate_recovery: ClassVar[bool] = False
    malformed_content: ClassVar[bool] = False
    simulate_sequential_repair: ClassVar[bool] = False

    def log_message(self, format: str, *args: object) -> None:
        """Suppress HTTP noise during the test run."""
        pass

    def do_POST(self) -> None:
        """Serve deterministic token counts and synthetic action responses."""
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.requests.append((self.path, body))
        if self.path == "/tokenize":
            value = {"count": 10}
        elif self.path == "/v1/chat/completions":
            action: dict[str, Any]
            if body["messages"][-1]["content"].startswith("Return exactly this JSON"):
                action = {"ok": True}
            elif any("You are SQL Author." in m["content"] or "You are SQL Critic." in m["content"]
                     for m in body["messages"] if m["role"] == "system"):
                packet: dict[str, Any] = json.loads(body["messages"][-1]["content"])
                current: dict[str, Any] | None = packet["current_candidate"]
                if current is None:
                    action = {"action": "execute_sql", "purpose": "candidate",
                              "sql": "SELECT missing_column FROM people" if self.simulate_sequential_repair else "SELECT name FROM people"}
                elif packet["active_role"] == "critic":
                    bad: bool = "missing_column" in current["sql"]
                    action = {"action": "review_sql", "candidate_version": current["version"],
                              "sql_fingerprint": current["sql_fingerprint"], "verdict": "request_changes" if bad else "approve",
                              "issues": [{"id": "I1", "category": "execution", "message": "Missing column; check DDL",
                                          "sources": ["schema", "observation:1"], "basis": "execution",
                                          "grounds": [{"source": "observation:1", "quote": "missing_column"}]}] if bad else []}
                elif "missing_column" in current["sql"]:
                    action = {"action": "execute_sql", "purpose": "candidate", "sql": "SELECT name FROM people",
                              "revision": {"reason": "Use the actual column from schema and error",
                                           "sources": ["schema", "observation:1", "review:1"], "basis": "execution",
                                           "grounds": [{"source": "observation:1", "quote": "missing_column"}]}}
                else:
                    action = {"action": "submit_sql", "candidate_version": current["version"],
                              "sql_fingerprint": current["sql_fingerprint"]}
            elif self.simulate_recovery and len(body["messages"]) == 2:
                action = {"action": "submit_sql", "sql": "SELECT missing_column FROM people"}
            elif self.simulate_recovery:
                action = {"action": "submit_sql", "sql": "SELECT name FROM people",
                          "revision": {"reason": "Fix the missing column using DDL and the error",
                                       "sources": ["schema", "observation:1"]}}
            else:
                action = {"action": "submit_sql", "sql": "SELECT name FROM people"}
            value = {"id": "mock-only", "choices": [{"message": {"content": json.dumps(action), "reasoning_content": "mock reasoning"}, "finish_reason": "stop"}],
                     "usage": {"prompt_tokens": 10 + (self.system_delta if any(m["role"] == "system" for m in body["messages"]) else 0), "completion_tokens": 20, "completion_tokens_details": {"reasoning_tokens": 12}}}
            if self.malformed_content:
                value["choices"][0]["message"]["content"] = None
        else:
            self.send_error(404)
            return
        payload = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class CLITest(unittest.TestCase):
    """Exercise strict and usage policies through the real command line."""

    def test_prepare_probe_run_both_and_evaluate(self) -> None:
        """Run both architectures without leaking gold or rejecting usage drift."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "toy").mkdir()
            con = sqlite3.connect(root / "toy" / "toy.sqlite")
            con.executescript("CREATE TABLE people(name TEXT); INSERT INTO people VALUES('Alice'),('Bob');")
            con.close()
            # Same result as model prediction, but a distinguishable hidden gold string.
            write_json(root / "dataset.json", [{"question_id": 1000 + i, "db_id": "toy", "question": f"List names {i}", "evidence": "person name", "SQL": "SELECT name FROM people /* HIDDEN_GOLD */", "difficulty": ["simple", "moderate", "challenging"][i % 3]} for i in range(90)])
            server = ThreadingHTTPServer(("127.0.0.1", 0), MockHTTP)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            env = {k: v for k, v in os.environ.items() if not k.startswith("MAS_SQL_")}
            MockHTTP.requests = []
            try:
                write_json(root / "config.json", {"model": "mock-not-real-llm", "base_url": f"http://127.0.0.1:{server.server_port}/v1"})
                def cli(*args: str | Path) -> None:
                    """Execute one command and require a successful exit."""
                    proc = subprocess.run([sys.executable, "-m", "mas_sql", *map(str, args)], capture_output=True, text=True, env=env, timeout=60)
                    self.assertEqual(proc.returncode, 0, proc.stderr)
                cli("prepare", "--dataset", root / "dataset.json", "--db-root", root, "--out", root / "split")
                cli("probe", "--config", root / "config.json", "--out", root / "probe.json")
                for arch in ["workflow", "single", "sequential"]:
                    cli("run", "--architecture", arch, "--config", root / "config.json", "--tasks", root / "split/pilot.tasks.json", "--db-root", root, "--out", root / arch)
                    cli("evaluate", "--tasks", root / "split/pilot.tasks.json", "--gold", root / "split/pilot.gold.json", "--run", root / arch, "--db-root", root, "--out", root / f"{arch}.report.json", "--rves-repeats", "2")
                    self.assertEqual(read_json(root / f"{arch}.report.json")["summary"]["EX_final"], 1)
                    self.assertEqual(len(read_json(root / arch / "results.json")), 10)
                    metadata = read_json(root / arch / "run.json")
                    if arch == "sequential":
                        self.assertEqual(metadata["context_version"], "public-snapshot-v2")
                        self.assertEqual(metadata["closure_version"], "closure-reserve-v2")
                        self.assertEqual(read_json(root / arch / "resource_summary.json")["closure_version"], "closure-reserve-v2")
                        self.assertEqual(read_json(root / f"{arch}.report.json")["closure_version"], "closure-reserve-v2")
                        self.assertEqual(read_json(root / arch / "resource_summary.json")["context_version"], "public-snapshot-v2")
                        self.assertEqual(read_json(root / f"{arch}.report.json")["context_version"], "public-snapshot-v2")
                        review_events: int = 0
                        for episode_file in (root / arch / "episodes").glob("*.json"):
                            episode = read_json(episode_file)
                            self.assertEqual(episode["context_version"], "public-snapshot-v2")
                            self.assertEqual(episode["closure_version"], "closure-reserve-v2")
                            for line in episode_file.with_suffix(".events.jsonl").read_text().splitlines():
                                event = json.loads(line)
                                self.assertEqual(event["id"], episode["id"])
                                if event["type"] == "critic_review":
                                    review_events += 1
                                    self.assertEqual(event["review_id"], "review:1")
                        self.assertEqual(review_events, 10)
                    else:
                        self.assertNotIn("context_version", metadata)
                    self.assertEqual(read_json(root / arch / "resource_summary.json")["known_tokens"], 900 if arch == "sequential" else 300)
                # Same HTTP endpoints, but generation now renders system messages differently.
                MockHTTP.system_delta = 7
                failed = subprocess.run([sys.executable, "-m", "mas_sql", "probe", "--config", str(root / "config.json"),
                                         "--out", str(root / "strict_mismatch_probe.json")],
                                        capture_output=True, text=True, env=env, timeout=60)
                self.assertEqual(failed.returncode, 2)
                self.assertFalse(read_json(root / "strict_mismatch_probe.json")["compatible"])
                config = read_json(root / "config.json")
                config["token_accounting"] = {"mode": "usage", "prompt_margin_tokens": 512}
                write_json(root / "usage_config.json", config)
                cli("probe", "--config", root / "usage_config.json", "--out", root / "usage_probe.json")
                for arch in ["workflow", "single", "sequential"]:
                    output = root / (arch + "_usage")
                    cli("run", "--architecture", arch, "--config", root / "usage_config.json", "--tasks", root / "split/pilot.tasks.json", "--db-root", root, "--out", output)
                    self.assertTrue(all(r["stop_reason"] == "submitted" for r in read_json(output / "results.json")))
                    summary = read_json(output / "resource_summary.json")
                    self.assertEqual(summary["known_tokens"], 1110 if arch == "sequential" else 370)
                    self.assertEqual(summary["tokenizer_mismatch_count"], 30 if arch == "sequential" else 10)
                MockHTTP.simulate_recovery = True
                recovery = root / "single_recovery"
                cli("run", "--architecture", "single", "--config", root / "usage_config.json", "--tasks", root / "split/pilot.tasks.json", "--db-root", root, "--out", recovery)
                cli("evaluate", "--tasks", root / "split/pilot.tasks.json", "--gold", root / "split/pilot.gold.json", "--run", recovery, "--db-root", root, "--out", root / "recovery.report.json")
                report = read_json(root / "recovery.report.json")
                self.assertEqual(report["summary"]["EX_first"], 0)
                self.assertEqual(report["summary"]["EX_final"], 1)
                self.assertEqual(report["summary"]["repairs"], 10)
                quality = report["investigation_quality"]
                self.assertEqual(quality["coverage_episodes"], 10)
                self.assertEqual(quality["episodes_recovering_submission_error"], 10)
                self.assertEqual(quality["repairs_in_episodes_with_diagnostics"], 0)
                resources = read_json(recovery / "resource_summary.json")
                self.assertEqual(resources["known_tokens"], 740)
                self.assertEqual(resources["investigation"]["failed_submissions"], 10)
                MockHTTP.simulate_recovery = False
                MockHTTP.simulate_sequential_repair = True
                team_recovery = root / "sequential_recovery"
                cli("run", "--architecture", "sequential", "--config", root / "usage_config.json", "--tasks", root / "split/pilot.tasks.json", "--db-root", root, "--out", team_recovery)
                cli("evaluate", "--tasks", root / "split/pilot.tasks.json", "--gold", root / "split/pilot.gold.json", "--run", team_recovery, "--db-root", root, "--out", team_recovery / "evaluation.json")
                team_report = read_json(team_recovery / "evaluation.json")
                self.assertEqual(team_report["summary"]["EX_first"], 0)
                self.assertEqual(team_report["summary"]["EX_final"], 1)
                self.assertEqual(team_report["summary"]["repairs"], 10)
                self.assertEqual(team_report["team_quality"]["scored_reviews"], 20)
                resources = read_json(team_recovery / "resource_summary.json")
                self.assertEqual(resources["known_tokens"], 1850)
                self.assertEqual(resources["team"]["llm_calls_by_role"], {"author": 30, "critic": 20})
                self.assertEqual(resources["protocol_version"], "sequential-author-critic-v2")
                self.assertEqual(read_json(team_recovery / "run.json")["prompt_version"], "sequential-action-contract-v3")
                MockHTTP.simulate_sequential_repair = False
                # A malformed envelope still carries real usage through the whole CLI.
                MockHTTP.malformed_content = True
                malformed_probe = root / "malformed_probe.json"
                failed = subprocess.run([sys.executable, "-m", "mas_sql", "probe", "--config", str(root / "usage_config.json"),
                                         "--out", str(malformed_probe)], capture_output=True, text=True, env=env, timeout=60)
                self.assertEqual(failed.returncode, 2)
                case = read_json(malformed_probe)["cases"][0]
                self.assertEqual(case["error_kind"], "response_format")
                self.assertEqual(case["diagnostics"]["http_status"], 200)
                self.assertTrue(case["usage_complete"])
                for arch in ["workflow", "single", "sequential"]:
                    malformed = root / (arch + "_malformed")
                    cli("run", "--architecture", arch, "--config", root / "usage_config.json", "--tasks", root / "split/pilot.tasks.json",
                        "--db-root", root, "--out", malformed)
                    cli("evaluate", "--tasks", root / "split/pilot.tasks.json", "--gold", root / "split/pilot.gold.json",
                        "--run", malformed, "--db-root", root, "--out", malformed / "evaluation.json")
                    results = read_json(malformed / "results.json")
                    self.assertTrue(all(r["stop_reason"] == "response_format_error" for r in results))
                    self.assertTrue(all(r["final_sql"] is None and r["tool_calls"] == 0 for r in results))
                    summary = read_json(malformed / "resource_summary.json")
                    self.assertEqual(summary["known_tokens"], 370)
                    self.assertTrue(summary["usage_complete"])
                    self.assertEqual(summary["unknown_usage_episodes"], 0)
                    self.assertEqual(summary["llm_calls"], 10)
                    self.assertEqual(summary["prompt_version"], "sequential-action-contract-v3" if arch == "sequential" else "action-contract-v2")
                    self.assertEqual(read_json(malformed / "evaluation.json")["summary"]["EX_final"], 0)
                self.assertNotIn("HIDDEN_GOLD", json.dumps(MockHTTP.requests))
                self.assertTrue(all("max_tokens" in b for p, b in MockHTTP.requests if p.endswith("chat/completions")))
            finally:
                MockHTTP.system_delta = 0
                MockHTTP.simulate_recovery = False
                MockHTTP.malformed_content = False
                MockHTTP.simulate_sequential_repair = False
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)
