"""Transport, actual-usage accounting, and model-call budget enforcement."""
import time
from dataclasses import dataclass, field
from typing import Any, Callable
from mas_sql.agent.budgets import Budget
from mas_sql.llm.client import usage_counts
from mas_sql.llm.errors import ClientError, ResponseFormatError
from mas_sql.agent.events import EventLog
from mas_sql.llm.accounting import TokenAccounting
from mas_sql.llm.types import ChatResponse, Message, ModelClient, UsageCounts


@dataclass
class TokenLedger:
    """Accumulate only reported usage while retaining partial known totals."""

    input_tokens: int = 0
    output_tokens: int = 0
    complete: bool = True
    responses: int = 0
    details: dict[str, int] = field(default_factory=lambda: {"reasoning_tokens": 0, "cached_input_tokens": 0})
    detail_reports: dict[str, int] = field(default_factory=lambda: {"reasoning_tokens": 0, "cached_input_tokens": 0})

    @property
    def total_tokens(self) -> int:
        """Return known input plus output tokens without double-counting details."""
        return self.input_tokens + self.output_tokens

    def add(self, raw: object) -> UsageCounts:
        """Charge all available counts and mark incomplete usage explicitly."""
        counts: UsageCounts = usage_counts(raw)
        self.responses += 1
        if counts["input_tokens"] is not None:
            self.input_tokens += counts["input_tokens"]
        if counts["output_tokens"] is not None:
            self.output_tokens += counts["output_tokens"]
        self.complete = self.complete and counts["total_tokens"] is not None
        for key in self.details:
            value: Any = counts[key]
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                self.details[key] += value
                self.detail_reports[key] += 1
        return counts

    def snapshot(self, known_only: bool = False) -> dict[str, int | None]:
        """Expose unknown totals as None, with known partials available separately."""
        values: dict[str, int | None] = {
            "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }
        if not known_only and not self.complete:
            values = {key: None for key in values}
        for key in self.details:
            available: bool = (self.detail_reports[key] > 0 if known_only else
                               self.complete and self.responses > 0 and self.detail_reports[key] == self.responses)
            values[key] = self.details[key] if available else None
        return values


class GenerationController:
    """Apply common model-call limits without knowing SQL or gold answers."""

    def __init__(self, client: ModelClient, budget: Budget, accounting: TokenAccounting,
                 log: EventLog, call_cap: int) -> None:
        """Initialize one episode-level usage ledger and transport controller."""
        self.client = client
        self.budget = budget
        self.accounting = accounting
        self.log = log
        self.call_cap = call_cap
        self.ledger = TokenLedger()
        self.calls: int = 0
        self.llm_s: float = 0.0
        self.tokenize_s: float = 0.0
        self.mismatch_count: int = 0
        self.stop_reason: str | None = None

    def _count(self, messages: list[Message], remaining: Callable[[], float]) -> int | None:
        """Read preflight tokens and retain tokenizer failure as a stop reason."""
        tick: float = time.perf_counter()
        try:
            return self.client.count(messages, remaining())
        except ClientError as exc:
            self.log.add("tokenize_error", error=str(exc))
            self.stop_reason = "tokenize_error"
            return None
        finally:
            self.tokenize_s += time.perf_counter() - tick

    def _account(self, raw_usage: object, preflight: int, reserve: int,
                 output_cap: int) -> tuple[UsageCounts, bool]:
        """Charge actual usage and validate the selected accounting policy."""
        counts: UsageCounts = self.ledger.add(raw_usage)
        actual: int | None = counts["input_tokens"]
        output: int | None = counts["output_tokens"]
        mismatch: bool = actual is not None and actual != preflight
        if mismatch:
            assert actual is not None
            self.mismatch_count += 1
            self.log.add("tokenizer_mismatch", severity="error" if self.accounting.strict_preflight else "warning",
                         preflight_tokens=preflight, actual_input_tokens=actual, delta_tokens=actual - preflight,
                         margin_exceeded=actual > reserve, accounting_mode=self.accounting.mode)
        if self.ledger.total_tokens > self.budget.max_total_tokens:
            self.stop_reason = "token_budget_overrun"
        elif not self.ledger.complete:
            self.stop_reason = "usage_missing"
        elif mismatch and self.accounting.strict_preflight:
            self.stop_reason = "tokenizer_mismatch"
        elif actual is not None and output is not None and actual + output > self.budget.max_context_tokens:
            self.stop_reason = "context_budget_overrun"
        elif output is not None and output > output_cap:
            self.stop_reason = "server_output_cap_violation"
        return counts, mismatch

    def request(self, messages: list[Message], remaining: Callable[[], float],
                context: dict[str, Any] | None = None, *, holdback_tokens: int = 0,
                min_output_tokens: int = 1) -> ChatResponse | None:
        """Tag all events of one shared call while preserving accounting semantics."""
        if (isinstance(holdback_tokens, bool) or not isinstance(holdback_tokens, int) or holdback_tokens < 0
                or isinstance(min_output_tokens, bool) or not isinstance(min_output_tokens, int) or min_output_tokens < 1):
            raise ValueError("Reservations need nonnegative integer holdback and positive integer output floor")
        with self.log.scoped(context):
            return self._request(messages, remaining, holdback_tokens, min_output_tokens)

    def projected_reserve(self, messages: list[Message], remaining: Callable[[], float], role: str) -> int | None:
        """Measure a planning snapshot, never a generation or a ledger charge."""
        if remaining() <= 0:
            self.stop_reason = "episode_time_budget"
            return None
        count: int | None = self._count(messages, remaining)
        if count is None:
            return None
        reserve: int = self.accounting.reserve(count)
        self.log.add("budget_projection", projected_role=role, prompt_tokens=count, reserved_prompt_tokens=reserve)
        return reserve

    def _request(self, messages: list[Message], remaining: Callable[[], float],
                 holdback_tokens: int = 0, min_output_tokens: int = 1) -> ChatResponse | None:
        """Request one completion without retries or approximate usage charges."""
        if self.calls >= self.call_cap:
            self.stop_reason = "llm_call_budget"
            return None
        if remaining() <= 0:
            self.stop_reason = "episode_time_budget"
            return None
        preflight: int | None = self._count(messages, remaining)
        if preflight is None:
            return None
        reserve: int = self.accounting.reserve(preflight)
        output_cap: int = min(self.budget.max_output_tokens,
                              self.budget.max_total_tokens - self.ledger.total_tokens - reserve - holdback_tokens,
                              self.budget.max_context_tokens - reserve)
        self.log.add("preflight", prompt_tokens=preflight, reserved_prompt_tokens=reserve,
                     output_cap=output_cap, accounting_mode=self.accounting.mode,
                     **({"holdback_tokens": holdback_tokens, "min_output_tokens": min_output_tokens}
                        if holdback_tokens or min_output_tokens > 1 else {}))
        if (holdback_tokens or min_output_tokens > 1) and output_cap < min_output_tokens:
            self.stop_reason = "insufficient_closure_budget"
            return None
        if output_cap <= 0:
            self.stop_reason = "token_budget" if self.budget.max_total_tokens - self.ledger.total_tokens <= reserve else "context_budget"
            return None
        if remaining() <= 0:
            self.stop_reason = "episode_time_budget"
            return None
        self.calls += 1
        self.log.add("llm_request", messages=messages.copy(), max_tokens=output_cap)
        tick: float = time.perf_counter()
        try:
            response: ChatResponse = self.client.complete(messages, output_cap, remaining())
        except ResponseFormatError as exc:
            counts, mismatch = self._account(exc.usage, preflight, reserve, output_cap)
            self.log.add("llm_error", error=str(exc), error_kind=exc.kind, error_code=exc.code,
                         diagnostics=exc.diagnostics, normalized_usage=counts,
                         tokenizer_mismatch=mismatch, usage_unknown=counts["total_tokens"] is None)
            if self.stop_reason in (None, "usage_missing"):
                self.stop_reason = ("response_format_error" if self.ledger.complete else
                                    "response_format_error_usage_unknown")
            return None
        except ClientError as exc:
            self.ledger.complete = False
            self.log.add("llm_error", error=str(exc), error_kind=exc.kind,
                         http_status=exc.http_status, usage_unknown=True)
            self.stop_reason = "transport_error_usage_unknown"
            return None
        finally:
            self.llm_s += time.perf_counter() - tick
        counts, mismatch = self._account(response.get("usage"), preflight, reserve, output_cap)
        self.log.add("llm_response", **response, normalized_usage=counts, tokenizer_mismatch=mismatch)
        return response
