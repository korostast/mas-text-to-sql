"""Typed client failures with safe metadata and separately recoverable usage."""
from typing import Any
from mas_sql.llm.types import ResponseDiagnostics


class ClientError(RuntimeError):
    """Describe a failed request without exposing credentials or response bodies."""

    def __init__(self, message: str, *, kind: str = "transport",
                 http_status: int | None = None) -> None:
        """Attach a failure category and an optional HTTP status."""
        super().__init__(message)
        self.kind: str = kind
        self.http_status: int | None = http_status


class ResponseFormatError(ClientError):
    """Preserve reported usage when a chat envelope cannot supply an action."""

    def __init__(self, code: str, usage: Any, diagnostics: ResponseDiagnostics) -> None:
        """Keep structural diagnostics without copying content or reasoning."""
        super().__init__(f"Malformed Chat Completions response: {code}",
                         kind="response_format", http_status=diagnostics["http_status"])
        self.code: str = code
        self.usage: Any = usage
        self.diagnostics: ResponseDiagnostics = diagnostics
