"""Parse chat envelopes independently of action JSON and token accounting."""
from typing import Any
from mas_sql.llm.errors import ResponseFormatError
from mas_sql.llm.types import ChatResponse, ResponseDiagnostics


def json_type(value: object) -> str:
    """Name a JSON value's type without including its contents."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if isinstance(value, (int, float)):
        return "number"
    return "other"


class ChatResponseParser:
    """Validate a chat response while keeping usage independent of content."""

    @staticmethod
    def diagnostics(response: object, http_status: int | None) -> ResponseDiagnostics:
        """Summarize expected envelope fields, never body or reasoning text."""
        envelope: dict[str, Any] = response if isinstance(response, dict) else {}
        choices: Any = envelope.get("choices")
        choice: Any = choices[0] if isinstance(choices, list) and choices else None
        message: Any = choice.get("message") if isinstance(choice, dict) else None
        content: Any = message.get("content") if isinstance(message, dict) else None
        finish: Any = choice.get("finish_reason") if isinstance(choice, dict) else None
        identifier: Any = envelope.get("id")
        safe_id: str | None = (identifier if isinstance(identifier, str) and len(identifier) <= 256
                               and all(character.isprintable() for character in identifier) else None)
        safe_finish: str | None = (finish if finish in ("stop", "length", "tool_calls", "function_call", "content_filter")
                                  else None if finish is None else "other")
        return {"http_status": http_status, "response_type": json_type(response),
                "choices_type": json_type(choices), "choices_count": len(choices) if isinstance(choices, list) else None,
                "choice_type": json_type(choice), "message_type": json_type(message),
                "content_present": isinstance(message, dict) and "content" in message,
                "content_type": json_type(content), "content_length": len(content) if isinstance(content, str) else None,
                "finish_reason": safe_finish, "response_id": safe_id,
                "usage_present": "usage" in envelope, "usage_type": json_type(envelope.get("usage"))}

    @classmethod
    def parse(cls, response: object, http_status: int | None = None) -> ChatResponse:
        """Return text content or raise a typed error retaining independent usage."""
        diagnostics: ResponseDiagnostics = cls.diagnostics(response, http_status)
        envelope: dict[str, Any] = response if isinstance(response, dict) else {}
        choices: Any = envelope.get("choices")
        choice: Any = choices[0] if isinstance(choices, list) and choices else None
        message: Any = choice.get("message") if isinstance(choice, dict) else None
        code: str | None = None
        if not isinstance(response, dict):
            code = "response_not_object"
        elif not isinstance(choices, list):
            code = "choices_not_array"
        elif not choices:
            code = "choices_empty"
        elif len(choices) != 1:
            code = "unexpected_choice_count"
        elif not isinstance(choice, dict):
            code = "choice_not_object"
        elif not isinstance(message, dict):
            code = "message_not_object"
        elif "content" not in message:
            code = "content_missing"
        elif not isinstance(message["content"], str):
            code = "content_not_string"
        elif choice.get("finish_reason") is not None and not isinstance(choice["finish_reason"], str):
            code = "finish_reason_not_string"
        if code is not None:
            raise ResponseFormatError(code, envelope.get("usage"), diagnostics)
        assert isinstance(message, dict) and isinstance(choice, dict)
        return {"content": message["content"], "reasoning": message.get("reasoning_content", message.get("reasoning")),
                "finish_reason": choice.get("finish_reason"), "usage": envelope.get("usage"),
                "response_id": diagnostics["response_id"], "diagnostics": diagnostics}
