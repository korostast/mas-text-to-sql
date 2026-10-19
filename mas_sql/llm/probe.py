"""Probe accounting across representative message-role patterns."""
import json
from dataclasses import asdict
from typing import Any, Callable
from mas_sql.llm.client import usage_counts
from mas_sql.llm.errors import ClientError, ResponseFormatError
from mas_sql.llm.accounting import TokenAccounting
from mas_sql.llm.types import Message, ModelClient


class ProbeRunner:
    """Check protocol and accounting without using benchmark questions."""

    def __init__(self, client: ModelClient, accounting: TokenAccounting,
                 output_cap: int, context_cap: int, timeout_s: float) -> None:
        """Store the same transport and accounting settings used by episodes."""
        self.client = client
        self.accounting = accounting
        self.output_cap = output_cap
        self.context_cap = context_cap
        self.timeout_s = timeout_s

    @staticmethod
    def cases() -> list[tuple[str, list[Message]]]:
        """Cover user-only, system-plus-user, and multi-turn conversations."""
        request: Message = {"role": "user", "content": 'Return exactly this JSON object in content: {"ok":true}'}
        system: Message = {"role": "system", "content": "Return only the JSON requested by the user, without Markdown."}
        return [
            ("user_only", [request]),
            ("system_user", [system, request]),
            ("system_multiturn", [system, request, {"role": "assistant", "content": '{"ok":true}'}, request]),
        ]

    def run(self, save: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
        """Persist every case and report compatibility with the selected mode."""
        results: list[dict[str, Any]] = []
        report: dict[str, Any] = {"token_accounting": asdict(self.accounting), "cases": results,
                                  "compatible": False, "complete": False}
        for name, messages in self.cases():
            case: dict[str, Any] = {"name": name, "messages": messages, "passed": False}
            try:
                count: int = self.client.count(messages, self.timeout_s)
                cap: int = min(self.output_cap, self.context_cap - self.accounting.reserve(count))
                if cap <= 0:
                    raise ClientError("Probe prompt reserve exhausts the configured context budget")
                response = self.client.complete(messages, cap, self.timeout_s)
                usage = usage_counts(response.get("usage"))
                actual: int | None = usage["input_tokens"]
                try:
                    protocol_ok: bool = json.loads(response["content"]) == {"ok": True} and response["finish_reason"] == "stop"
                except (ValueError, TypeError):
                    protocol_ok = False
                matches: bool = actual == count if actual is not None else False
                context_ok: bool = usage["total_tokens"] is not None and usage["total_tokens"] <= self.context_cap
                output_ok: bool = usage["output_tokens"] is not None and usage["output_tokens"] <= cap
                passed: bool = (protocol_ok and context_ok and output_ok and
                                (matches or not self.accounting.strict_preflight))
                case.update(prompt_count=count, reserved_prompt_tokens=self.accounting.reserve(count),
                            actual_input_tokens=actual, delta_tokens=actual - count if actual is not None else None,
                            tokenizer_matches_usage=matches, protocol_ok=protocol_ok,
                            response=response, usage=usage, output_cap=cap, passed=passed)
            except ResponseFormatError as exc:
                usage = usage_counts(exc.usage)
                actual = usage["input_tokens"]
                case.update(error=str(exc), error_kind=exc.kind, error_code=exc.code,
                            diagnostics=exc.diagnostics, usage=usage, usage_complete=usage["total_tokens"] is not None,
                            prompt_count=count, reserved_prompt_tokens=self.accounting.reserve(count),
                            actual_input_tokens=actual, delta_tokens=actual - count if actual is not None else None,
                            tokenizer_matches_usage=actual == count if actual is not None else False,
                            output_cap=cap, protocol_ok=False)
                results.append(case)
                save(report)
                break
            except ClientError as exc:
                case.update(error=str(exc), error_kind=exc.kind, http_status=exc.http_status)
                results.append(case)
                save(report)
                break
            results.append(case)
            save(report)
        report["complete"] = len(results) == len(self.cases()) and all("error" not in c for c in results)
        report["compatible"] = report["complete"] and all(c["passed"] for c in results)
        report["tokenizer_matches_usage"] = report["complete"] and all(c.get("tokenizer_matches_usage", False) for c in results)
        save(report)
        return report
