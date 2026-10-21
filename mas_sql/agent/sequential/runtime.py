"""Fixed-order Author-Critic orchestration over one shared resource ledger."""
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal, Mapping
from mas_sql.agent.budgets import Budget
from mas_sql.agent.events import EventLog, EventSink
from mas_sql.agent.generation import GenerationController
from mas_sql.agent.investigation import sql_fingerprint
from mas_sql.agent.sequential.actions import Role, TeamAction, TeamActionParser
from mas_sql.agent.sequential.closure import CLOSURE_VERSION, ClosurePlan, ClosurePlanner
from mas_sql.agent.sequential.prompts import CONTEXT_VERSION, PROTOCOL_VERSION, PROMPT_VERSION, turn_packet
from mas_sql.agent.sequential.roles import AuthorAgent, CriticAgent, RoleAgent
from mas_sql.agent.sequential.state import Candidate, EpisodeState, Review
from mas_sql.database.executor import SQLLimits, execute, preview, schema
from mas_sql.llm.accounting import TokenAccounting
from mas_sql.llm.types import ChatResponse, ModelClient


class SequentialEpisodeRunner:
    """Own tools, transitions and termination without interpreting semantic truth."""

    def __init__(self, task: Mapping[str, str], path: str | Path, client: ModelClient,
                 budget: Budget, limits: SQLLimits, accounting: TokenAccounting,
                 emit: EventSink | None = None) -> None:
        """Create fresh role contexts and a single episode-level generation ledger."""
        self.task = task
        self.path = path
        self.budget = budget
        self.limits = limits
        self.accounting = accounting
        self.started: float = time.perf_counter()
        self.deadline: float = self.started + budget.max_episode_s
        self.log = EventLog(emit)
        self.generation = GenerationController(client, budget, accounting, self.log, budget.max_llm_calls)
        ddl: str = schema(path)
        self.state = EpisodeState(bool(task.get("evidence", "")),
                                  source_texts={"question": task["question"], "evidence": task.get("evidence", ""), "schema": ddl})
        self.plan: ClosurePlan | None = None
        self.finalization_mode: Literal["open", "first", "repair"] = "open"
        self.agents: dict[Role, RoleAgent] = {"author": AuthorAgent(task, ddl), "critic": CriticAgent(task, ddl)}
        self.tool_calls: int = 0
        self.tool_s: float = 0.0
        self.final_sql: str | None = None
        self.final_result: dict[str, Any] | None = None
        self.last_submission: dict[str, Any] | None = None
        self.stop_reason: str | None = None
        self.role_calls: dict[Role, int] = {"author": 0, "critic": 0}
        self.role_usage: dict[Role, dict[str, int | bool]] = {
            role: {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "complete": True}
            for role in self.agents}
        self.log.add("initial", architecture="sequential", budget=asdict(budget), sql_limits=asdict(limits),
                     token_accounting=asdict(accounting), protocol_version=PROTOCOL_VERSION,
                     prompt_version=PROMPT_VERSION, context_version=CONTEXT_VERSION,
                     closure_version=CLOSURE_VERSION, role_messages={role: agent.messages for role, agent in self.agents.items()})

    def remaining_s(self) -> float:
        """Return the remaining cooperative wall-time budget."""
        return max(0.0, self.deadline - time.perf_counter())

    def _remaining(self) -> dict[str, int | float]:
        """Report one shared budget, not independent per-agent allowances."""
        return {"llm_calls": self.budget.max_llm_calls - self.generation.calls,
                "tool_calls": self.budget.max_tool_calls - self.tool_calls,
                "total_tokens": self.budget.max_total_tokens - self.generation.ledger.total_tokens,
                "episode_s": self.remaining_s()}

    def _count_allowed(self, future_calls: int) -> dict[str, Any]:
        """Preserve downstream call and SQL-slot reservations."""
        tools: int = self.budget.max_tool_calls - self.tool_calls
        role: Role = self.state.actor
        author: bool = role == "author"
        current: Candidate | None = self.state.current
        return {
            "candidate": author and self.finalization_mode == "open" and future_calls >= 3 and tools >= 2,
            "diagnostic": current is not None and not self.state.diagnostics_used[role] and tools >= 2
                and self.finalization_mode == "open" and future_calls >= (2 if author else 3),
            "review_sql": not author and future_calls >= 2,
            "submit_sql": author and self.state.current_review is not None and tools >= 1
                and future_calls >= (0 if self.finalization_mode == "repair" else 1),
            "resource_note": "Optional work reserves verdict, explicit submit and ONE finalization repair call. "
                "Future token growth is estimated, not guaranteed."}

    def _allowed(self, future_calls: int) -> dict[str, Any]:
        """Intersect call/tool allowances with this request's token allocation."""
        result: dict[str, Any] = self._count_allowed(future_calls)
        if self.plan is not None:
            for key, permitted in self.plan.allowed.items():
                result[key] = bool(result[key]) and permitted
            result["closure"] = self.plan.payload()
        return result

    def _plan(self, future_calls: int) -> ClosurePlan | None:
        """Tokenize both known role snapshots without charging phantom model usage."""
        allowed: dict[str, Any] = self._count_allowed(future_calls)
        packet: dict[str, Any] = turn_packet(self.state, self._remaining(), allowed)
        reserves: dict[Role, int] = {}
        for role, agent in self.agents.items():
            reserve: int | None = self.generation.projected_reserve(agent.context(packet), self.remaining_s, role)
            if reserve is None:
                return None
            reserves[role] = reserve
        plan: ClosurePlan = ClosurePlanner.plan(self.state.actor, self.state.current is None,
            self.budget.max_total_tokens - self.generation.ledger.total_tokens,
            self.budget.max_output_tokens, reserves, allowed, repairing=self.finalization_mode == "repair")
        self.log.add("closure_plan", role=self.state.actor, phase=self.state.phase, **plan.payload(),
                     allowed_actions=plan.allowed, finalization_mode=self.finalization_mode)
        return plan

    def _validate_resources(self, action: TeamAction) -> None:
        """Reject work that would consume a reserved downstream call or SQL slot."""
        key: str = action.purpose if action.name == "execute_sql" else action.name
        allowed: dict[str, Any] = self._allowed(self.budget.max_llm_calls - self.generation.calls)
        if not allowed.get(key, False):
            raise ValueError(f"Resource reservation disallows {key}; use another allowed action")
        if self.plan is not None and self._remaining()["total_tokens"] < self.plan.future_costs[key]:
            raise ValueError(f"Actual usage leaves insufficient closure reserve for {key}; retain the reviewed version")
        if action.purpose == "candidate":
            self._project_candidate(action)

    def _project_candidate(self, action: TeamAction) -> None:
        """Recheck known new SQL size before dispatch; no projected SQL is executed."""
        assert self.plan is not None
        shadow: EpisodeState = deepcopy(self.state)
        next_id: int = self.tool_calls + 1
        shadow.propose(action, next_id)
        shadow.observe("author", action, {"status": "pending", "rows": [], "preview_truncated": False}, next_id)
        packet: dict[str, Any] = turn_packet(shadow, self._remaining(), self._count_allowed(3))
        reserves: dict[Role, int] = {}
        for role, agent in self.agents.items():
            reserve: int | None = self.generation.projected_reserve(agent.context(packet), self.remaining_s, role)
            if reserve is None:
                raise ValueError("Cannot project candidate closure after tokenizer failure")
            reserves[role] = reserve
        minimum: int = ClosurePlanner.candidate_cost(reserves, self.plan.min_output_tokens, self.plan.growth_tokens)
        self.log.add("candidate_closure_check", required_tokens=minimum,
                     remaining_tokens=self._remaining()["total_tokens"],
                     feasible=self._remaining()["total_tokens"] >= minimum)
        if self._remaining()["total_tokens"] < minimum:
            raise ValueError("New candidate context exceeds closure allocation; retain the current reviewed version")

    def _error(self, role: Role, error: str, action: TeamAction | None = None) -> None:
        """Charge the failed model turn but share no invalid message with its peer."""
        self.log.add("protocol_error", role=role, phase=self.state.phase, error=error,
                     action=action.payload() if action is not None else None)
        if role == "author" and self.finalization_mode != "open":
            if self.finalization_mode == "repair":
                self.stop_reason = "finalization_repair_exhausted"
                self.log.add("finalization_repair_exhausted", role=role, remaining=self._remaining())
                return
            self.finalization_mode = "repair"
            self.agents[role].protocol_error(error, finalization=True)
            self.log.add("finalization_repair_started", role=role, remaining=self._remaining())
        else:
            self.agents[role].protocol_error(error)

    def _request(self, role: Role) -> ChatResponse | None:
        """Attribute actual shared usage even when a malformed response stops work."""
        before: dict[str, int | None] = self.generation.ledger.snapshot(True)
        calls_before: int = self.generation.calls
        future_calls: int = self.budget.max_llm_calls - self.generation.calls - 1
        count_allowed: dict[str, Any] = self._count_allowed(future_calls)
        if not any(count_allowed[key] for key in
                   ("candidate", "diagnostic", "review_sql", "submit_sql")):
            self.generation.stop_reason = "insufficient_review_budget"
            return None
        self.plan = self._plan(future_calls)
        if self.plan is None:
            return None
        if not any(self.plan.allowed.values()):
            self.generation.stop_reason = "insufficient_closure_budget"
            return None
        if (role == "author" and self.finalization_mode == "open" and self.plan.allowed["submit_sql"]
                and not self.plan.allowed["candidate"] and not self.plan.allowed["diagnostic"]):
            self.finalization_mode = "first"
            self.log.add("finalization_started", role=role, repair_tokens=self.plan.repair_tokens,
                         repair_calls=1, remaining=self._remaining())
        response: ChatResponse | None = self.agents[role].request(
            turn_packet(self.state, self._remaining(), self._allowed(future_calls)), self.generation,
            self.remaining_s, self.plan)
        self.role_calls[role] += self.generation.calls - calls_before
        after: dict[str, int | None] = self.generation.ledger.snapshot(True)
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            self.role_usage[role][key] += int(after[key] or 0) - int(before[key] or 0)
        if not self.generation.ledger.complete:
            self.role_usage[role]["complete"] = False
        return response

    def _execute(self, role: Role, action: TeamAction) -> None:
        """Execute one validated action with the same guarded SQLite limits."""
        available_s: float = self.remaining_s() - 1.0
        if available_s <= 0:
            self.stop_reason = "episode_time_budget"
            return
        next_id: int = self.tool_calls + 1
        if action.purpose == "candidate":
            candidate: Candidate = self.state.propose(action, next_id)
            if candidate.version == 1:
                self.log.add("first_candidate", role=role, sql=candidate.sql, sql_fingerprint=candidate.fingerprint,
                             candidate_version=candidate.version)
            else:
                self.log.add("candidate_revision", role=role, **self.state.revisions[-1], sql_fingerprint=candidate.fingerprint)
        assert self.state.current is not None
        sql: str = action.sql if action.sql is not None else self.state.current.sql
        self.tool_calls = next_id
        call_limits: SQLLimits = SQLLimits(**{**asdict(self.limits), "timeout_s": min(self.limits.timeout_s, available_s)})
        result: dict[str, Any] = execute(self.path, sql, call_limits)
        self.tool_s += result["wall_s"]
        observation: dict[str, Any] = preview(result, self.limits)
        self.state.observe(role, action, observation, next_id)
        self.log.add("tool_result", role=role, action=action.payload(), observation=observation,
                     observation_id=next_id, tool_index=next_id, candidate_version=self.state.current.version,
                     sql_fingerprint=sql_fingerprint(sql), candidate_sql_fingerprint=self.state.current.fingerprint,
                     purpose=action.purpose or "submission")
        if action.name == "submit_sql":
            self._submission(action, result, observation, next_id)

    def _submission(self, action: TeamAction, result: dict[str, Any], observation: dict[str, Any],
                    observation_id: int) -> None:
        """Accept only an explicit exact-version submission with successful execution."""
        current: Candidate | None = self.state.current
        review: Review | None = self.state.current_review
        assert current is not None and review is not None
        accepted: bool = result["status"] == "ok"
        record: dict[str, Any] = {"candidate_version": current.version, "sql_fingerprint": current.fingerprint,
            "review_id": review.id, "critic_verdict": review.verdict, "accepted": accepted,
            "observation_id": observation_id, "submitted_with_disagreement": review.verdict == "request_changes",
            "response_to_review": action.response_to_review.payload() if action.response_to_review is not None else None}
        self.state.submissions.append(record)
        self.last_submission = {"sql": current.sql, "observation_id": observation_id,
                                "result": observation if accepted else result, **record}
        self.log.add("submission_accepted" if accepted else "submission_rejected", role="author", **record)
        if accepted:
            self.final_sql, self.final_result = current.sql, result
            self.stop_reason = "submitted"
        else:
            if self.tool_calls >= self.budget.max_tool_calls:
                self.stop_reason = "tool_call_budget"

    def _result(self) -> dict[str, Any]:
        """Build baseline-compatible outputs with additive team telemetry."""
        self.log.add("stop", reason=self.stop_reason, phase=self.state.phase)
        final_submission: dict[str, Any] | None = next((item for item in reversed(self.state.submissions) if item["accepted"]), None)
        return {"id": self.task["id"], "db_id": self.task["db_id"], "architecture": "sequential",
            "protocol_version": PROTOCOL_VERSION, "prompt_version": PROMPT_VERSION, "context_version": CONTEXT_VERSION,
            "closure_version": CLOSURE_VERSION, "finalization_mode": self.finalization_mode,
            "first_sql": self.state.first_sql, "final_sql": self.final_sql, "final_result": self.final_result,
            "last_submission": self.last_submission, "stop_reason": self.stop_reason,
            "submitted": self.final_sql is not None, "submission_accepted": self.final_sql is not None,
            "llm_calls": self.generation.calls, "tool_calls": self.tool_calls,
            "usage": self.generation.ledger.snapshot(), "usage_known_totals": self.generation.ledger.snapshot(True),
            "token_accounting": asdict(self.accounting), "tokenizer_mismatch_count": self.generation.mismatch_count,
            "preflight_is_estimate": not self.accounting.strict_preflight,
            "usage_complete": self.generation.ledger.complete,
            "token_budget_overrun": self.generation.ledger.total_tokens > self.budget.max_total_tokens,
            "investigation": self.state.investigation(),
            "team": {"format_version": 1, "phase": self.state.phase, "role_calls": self.role_calls,
                     "role_known_usage": self.role_usage,
                     "diagnostic_calls_by_role": {role: sum(item["role"] == role for item in self.state.diagnostics) for role in self.agents},
                     "candidates": [candidate.payload() for candidate in self.state.candidates],
                     "reviews": [review.payload() for review in self.state.reviews],
                     "submissions": self.state.submissions,
                     "final_review": self.state.current_review.payload() if final_submission is not None and self.state.current_review else None,
                     "submitted_with_disagreement": final_submission["submitted_with_disagreement"] if final_submission else None},
            "timing": {"solve_s": time.perf_counter() - self.started, "llm_s": self.generation.llm_s,
                       "tool_s": self.tool_s, "tokenize_s": self.generation.tokenize_s}, "events": self.log.events}

    def run(self) -> dict[str, Any]:
        """Run a bounded fixed-order loop without retries outside its global caps."""
        while self.stop_reason is None:
            if self.remaining_s() <= 1.0:
                self.stop_reason = "episode_time_budget"
                break
            if self.generation.calls >= self.budget.max_llm_calls:
                self.stop_reason = "llm_call_budget"
                break
            if self.tool_calls >= self.budget.max_tool_calls:
                self.stop_reason = "tool_call_budget"
                break
            if self.state.current is None and (self.budget.max_llm_calls - self.generation.calls < 4
                                               or self.budget.max_tool_calls - self.tool_calls < 2):
                self.stop_reason = "insufficient_review_budget"
                break
            role: Role = self.state.actor
            phase: str = self.state.phase
            response: ChatResponse | None = self._request(role)
            if response is None or self.generation.stop_reason is not None:
                self.stop_reason = self.generation.stop_reason or "llm_call_budget"
                break
            action: TeamAction | None = None
            try:
                if response.get("finish_reason") != "stop":
                    raise ValueError("Completion was truncated/not stopped; return a complete action")
                action = TeamActionParser.parse(response["content"], role)
                self.state.validate(action, role)
                self._validate_resources(action)
            except (ValueError, TypeError) as exc:
                if self.generation.stop_reason is not None:
                    self.stop_reason = self.generation.stop_reason
                    break
                self._error(role, str(exc), action)
                continue
            self.agents[role].action_accepted()
            self.log.add("action_accepted", role=role, phase=phase, action=action.payload())
            if action.name == "review_sql":
                review: Review = self.state.accept_review(action)
                review_event: dict[str, Any] = review.payload()
                review_event["review_id"] = review_event.pop("id")
                self.log.add("critic_review", role=role, **review_event)
            else:
                self._execute(role, action)
            if self.state.phase != phase:
                self.log.add("phase_transition", from_phase=phase, to_phase=self.state.phase,
                             candidate_version=self.state.current.version if self.state.current else None)
        return self._result()
