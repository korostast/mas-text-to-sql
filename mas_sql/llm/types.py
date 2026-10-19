"""Typed boundaries shared by transports and the agent runtime."""
from typing import Any, NotRequired, Protocol, TypedDict


class Message(TypedDict):
    """Represent one text-only Chat Completions message."""

    role: str
    content: str


class ResponseDiagnostics(TypedDict):
    """Describe envelope shape without recording response body text."""

    http_status: int | None
    response_type: str
    choices_type: str
    choices_count: int | None
    choice_type: str
    message_type: str
    content_present: bool
    content_type: str
    content_length: int | None
    finish_reason: str | None
    response_id: str | None
    usage_present: bool
    usage_type: str


class ChatResponse(TypedDict):
    """Preserve response text and raw accounting without parsing reasoning."""

    content: str
    reasoning: Any
    finish_reason: str | None
    usage: Any
    response_id: str | None
    diagnostics: NotRequired[ResponseDiagnostics]


class UsageCounts(TypedDict):
    """Represent reported token counts; None means unknown, never zero."""

    input_tokens: int | None
    output_tokens: int | None
    reasoning_tokens: int | None
    cached_input_tokens: int | None
    total_tokens: int | None


class ModelClient(Protocol):
    """Define the transport boundary used by real and test clients."""

    def count(self, messages: list[Message], timeout_s: float) -> int:
        """Return the server's preflight prompt count."""
        ...

    def complete(self, messages: list[Message], max_tokens: int, timeout_s: float) -> ChatResponse:
        """Request one non-streaming completion without implicit retries."""
        ...
