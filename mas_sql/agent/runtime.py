"""Bounded SQL episodes with recoverable submissions and observable investigation."""
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping
from mas_sql.agent.actions import Action, ActionParser
from mas_sql.agent.budgets import Budget
from mas_sql.agent.events import EventLog, EventSink
from mas_sql.agent.generation import GenerationController
from mas_sql.agent.investigation import InvestigationTracker, sql_fingerprint
from mas_sql.agent.prompts import PROMPT_VERSION, initial_messages
from mas_sql.agent.sequential.runtime import SequentialEpisodeRunner
from mas_sql.database.executor import SQLLimits, execute, preview, schema
from mas_sql.llm.accounting import TokenAccounting
from mas_sql.llm.types import ChatResponse, Message, ModelClient

PROTOCOL_VERSION = "sql-investigation-v1"


def parse_action(content: str, architecture: str) -> dict[str, Any]:
    """Expose the validated JSON action shape for existing API callers."""
    return ActionParser.parse(content, architecture).payload()


class EpisodeRunner:
    """Orchestrate SQL decisions while delegating accounting and trace state."""

    def __init__(self, task: Mapping[str, str], path: str | Path, architecture: str,
                 client: ModelClient, budget: Budget, limits: SQLLimits,
                 accounting: TokenAccounting, emit: EventSink | None = None) -> None:
        """Set up an isolated task with shared schema and no gold access."""
        if architecture not in ("workflow", "single"):
            raise ValueError("Architecture must be workflow or single")
        self.task = task
        self.path = path
        self.architecture = architecture
        self.budget = budget
        self.limits = limits
        self.accounting = accounting
        self.log = EventLog(emit)
        self.started: float = time.perf_counter()
        self.deadline: float = self.started + budget.max_episode_s
        self.generation = GenerationController(client, budget, accounting, self.log,
                                               1 if architecture == "workflow" else budget.max_llm_calls)
        self.investigation = InvestigationTracker(bool(task.get("evidence", "")))
        self.messages: list[Message] = initial_messages(task, schema(path), architecture)
        self.tool_calls: int = 0
        self.tool_s: float = 0.0
        self.final_sql: str | None = None
        self.final_result: dict[str, Any] | None = None
        self.last_submission: dict[str, Any] | None = None
        self.stop_reason: str | None = None
        self.log.add("initial", messages=self.messages, budget=asdict(budget), sql_limits=asdict(limits),
                     token_accounting=asdict(accounting), protocol_version=PROTOCOL_VERSION, prompt_version=PROMPT_VERSION)

    def remaining_s(self) -> float:
        """Return the remaining cooperative episode deadline."""
        return max(0.0, self.deadline - time.perf_counter())

    def _protocol_error(self, error: str) -> None:
        """Return validation feedback without executing or accepting model SQL."""
        self.log.add("protocol_error", error=error)
        if self.generation.stop_reason or self.architecture == "workflow":
            self.stop_reason = self.generation.stop_reason or "protocol_error"
        else:
            self.messages.append({"role": "user", "content": f"Protocol error: {error}. Return one valid action JSON object."})

    def _read_action(self, response: ChatResponse) -> Action | None:
        """Validate a complete action and freeze the unassisted first candidate."""
        self.messages.append({"role": "assistant", "content": response["content"]})
        if response.get("finish_reason") != "stop":
            self._protocol_error("Completion was truncated/not stopped; return a complete action")
            if self.architecture == "workflow" and self.generation.stop_reason is None:
                self.stop_reason = "completion_incomplete"
            return None
        try:
            action: Action = ActionParser.parse(response["content"], self.architecture)
            self.investigation.validate(action)
        except (ValueError, TypeError) as exc:
            self._protocol_error(str(exc))
            return None
        if self.investigation.capture_first(action):
            self.log.add("first_candidate", sql=action.sql, sql_fingerprint=sql_fingerprint(action.sql))
        if self.generation.stop_reason is not None:
            self.stop_reason = self.generation.stop_reason
            return None
        return action

    def _remaining_budget(self) -> dict[str, int | float]:
        """Report remaining resources without representing estimates as usage."""
        return {"llm_calls": self.generation.call_cap - self.generation.calls,
                "tool_calls": self.budget.max_tool_calls - self.tool_calls,
                "total_tokens": self.budget.max_total_tokens - self.generation.ledger.total_tokens,
                "episode_s": self.remaining_s()}

    def _feedback(self, action: Action, observation: dict[str, Any], accepted: bool = False) -> None:
        """Associate visible feedback with the exact action and observation ID."""
        payload: dict[str, Any] = {"tool": action.name, "observation_id": self.tool_calls,
                                   "result": observation, "remaining": self._remaining_budget()}
        if action.name == "submit_sql":
            payload.update(accepted=accepted, instruction="Submission failed execution and was NOT accepted. Use the error to repair or diagnose within the remaining budget; submit again when ready.")
        self.messages.append({"role": "user", "content": json.dumps(payload, ensure_ascii=False)})

    def _execute(self, action: Action) -> None:
        """Execute one guarded SQL action and track its candidate/check provenance."""
        available_s: float = self.remaining_s() - 1.0
        if available_s <= 0:
            self.stop_reason = "episode_time_budget"
            return
        self.tool_calls += 1
        call_limits = SQLLimits(**{**asdict(self.limits), "timeout_s": min(self.limits.timeout_s, available_s)})
        result: dict[str, Any] = execute(self.path, action.sql, call_limits)
        self.tool_s += result["wall_s"]
        revision: dict[str, Any] | None = self.investigation.record(action, result, self.tool_calls)
        if revision is not None:
            self.log.add("candidate_revision", **revision, sql_fingerprint=sql_fingerprint(action.sql))
        observation: dict[str, Any] = preview(result, self.limits)
        self.log.add("tool_result", action=action.payload(), observation=observation, tool_index=self.tool_calls,
                     observation_id=self.tool_calls, candidate_version=None if action.is_diagnostic else self.investigation.version,
                     sql_fingerprint=sql_fingerprint(action.sql))
        if action.name == "submit_sql":
            self._submission(action, result, observation)
        else:
            self._feedback(action, observation)

    def _submission(self, action: Action, result: dict[str, Any], observation: dict[str, Any]) -> None:
        """Accept executable SQL or give single-agent failures another bounded turn."""
        self.last_submission = {"sql": action.sql, "observation_id": self.tool_calls,
                                "result": observation if result["status"] == "ok" else result}
        if result["status"] == "ok" or self.architecture == "workflow":
            self.final_sql, self.final_result = action.sql, result
            self.stop_reason = "submitted" if result["status"] == "ok" else "submitted_execution_failed"
            return
        self.log.add("submission_rejected", sql=action.sql, observation_id=self.tool_calls,
                     status=result["status"], reason="execution_failed", remaining=self._remaining_budget())
        self._feedback(action, observation)
        if self.tool_calls >= self.budget.max_tool_calls:
            self.stop_reason = "tool_call_budget"

    def _dispatch(self, action: Action) -> None:
        """Preserve a final execution slot without forcing diagnostic work."""
        if self.remaining_s() <= 1.0:
            self.stop_reason = "episode_time_budget"
            return
        if self.tool_calls >= self.budget.max_tool_calls:
            self.stop_reason = "tool_call_budget"
            return
        if action.name == "execute_sql" and self.tool_calls >= self.budget.max_tool_calls - 1:
            self.log.add("tool_rejected", action=action.payload(), reason="Reserve last slot for submit_sql")
            self.messages.append({"role": "user", "content": "Tool budget: only one call remains, reserved for submit_sql. Submit your final answer now."})
            return
        self._execute(action)

    def _result(self) -> dict[str, Any]:
        """Build a compatible episode artifact with additive research telemetry."""
        self.log.add("stop", reason=self.stop_reason)
        return {"id": self.task["id"], "db_id": self.task["db_id"], "architecture": self.architecture,
                "protocol_version": PROTOCOL_VERSION, "prompt_version": PROMPT_VERSION,
                "first_sql": self.investigation.first_sql, "final_sql": self.final_sql,
                "final_result": self.final_result, "last_submission": self.last_submission,
                "stop_reason": self.stop_reason, "submitted": self.final_sql is not None,
                "submission_accepted": self.final_result is not None and self.final_result["status"] == "ok",
                "llm_calls": self.generation.calls, "tool_calls": self.tool_calls,
                "usage": self.generation.ledger.snapshot(), "usage_known_totals": self.generation.ledger.snapshot(True),
                "token_accounting": asdict(self.accounting), "tokenizer_mismatch_count": self.generation.mismatch_count,
                "preflight_is_estimate": not self.accounting.strict_preflight,
                "usage_complete": self.generation.ledger.complete,
                "token_budget_overrun": self.generation.ledger.total_tokens > self.budget.max_total_tokens,
                "investigation": self.investigation.summary(),
                "timing": {"solve_s": time.perf_counter() - self.started, "llm_s": self.generation.llm_s,
                           "tool_s": self.tool_s, "tokenize_s": self.generation.tokenize_s}, "events": self.log.events}

    def run(self) -> dict[str, Any]:
        """Run the adaptive loop until acceptance, a fatal error, or a budget stop."""
        while self.stop_reason is None and self.generation.calls < self.generation.call_cap:
            response: ChatResponse | None = self.generation.request(self.messages, self.remaining_s)
            if response is None:
                self.stop_reason = self.generation.stop_reason
                break
            action: Action | None = self._read_action(response)
            if action is not None:
                self._dispatch(action)
        self.stop_reason = self.stop_reason or "llm_call_budget"
        return self._result()


def solve(task: Mapping[str, str], path: str | Path, architecture: str,
          client: ModelClient, budget: Budget | None = None, limits: SQLLimits | None = None,
          emit: EventSink | None = None, accounting: TokenAccounting | None = None) -> dict[str, Any]:
    """Solve one task with a fresh runner and no cross-task model history."""
    if architecture == "sequential":
        return SequentialEpisodeRunner(task, path, client, budget or Budget(), limits or SQLLimits(),
                                       accounting or TokenAccounting(), emit).run()
    return EpisodeRunner(task, path, architecture, client, budget or Budget(), limits or SQLLimits(),
                         accounting or TokenAccounting(), emit).run()
