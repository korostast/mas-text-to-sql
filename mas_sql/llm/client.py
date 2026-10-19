"""OpenAI-compatible non-streaming transport, with vLLM preflight token counting."""
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, NoReturn
from mas_sql.llm.errors import ClientError
from mas_sql.llm.responses import ChatResponseParser
from mas_sql.llm.types import ChatResponse, Message, UsageCounts


def usage_counts(usage: object) -> UsageCounts:
    """Normalize reported usage while preserving unknown fields as None."""
    usage = usage if isinstance(usage, dict) else {}
    def count(*keys: str) -> int | None:
        """Read the first valid nonnegative count among known aliases."""
        for key in keys:
            value = usage.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                return value
        return None
    input_tokens = count("prompt_tokens", "input_tokens")
    output_tokens = count("completion_tokens", "output_tokens")
    details = usage.get("completion_tokens_details") or usage.get("output_tokens_details") or {}
    prompt_details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    details = details if isinstance(details, dict) else {}
    prompt_details = prompt_details if isinstance(prompt_details, dict) else {}
    return {"input_tokens": input_tokens, "output_tokens": output_tokens,
            "reasoning_tokens": details.get("reasoning_tokens"),
            "cached_input_tokens": prompt_details.get("cached_tokens"),
            "total_tokens": input_tokens + output_tokens if input_tokens is not None and output_tokens is not None else None}


class ChatClient:
    """Call compatible chat and tokenizer endpoints without hidden retries."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Resolve endpoint settings and protect the Authorization boundary."""
        self.base_url = os.environ.get("MAS_SQL_BASE_URL", config.get("base_url", "http://localhost:8000/v1")).rstrip("/")
        self.model = os.environ.get("MAS_SQL_MODEL", config["model"])
        self.api_key = os.environ.get("MAS_SQL_API_KEY", "")
        parsed = urllib.parse.urlsplit(self.base_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("base_url must be an HTTP(S) URL without embedded credentials/query")
        # vLLM /tokenize is normally outside /v1. Override for reverse proxies.
        default_tokenize = self.base_url.removesuffix("/v1") + "/tokenize"
        self.tokenize_url = os.environ.get("MAS_SQL_TOKENIZE_URL", config.get("tokenize_url") or default_tokenize)
        target = urllib.parse.urlsplit(self.tokenize_url)
        if (target.scheme, target.netloc) != (parsed.scheme, parsed.netloc):
            raise ValueError("Tokenize endpoint must use the same origin (Authorization protection)")
        self.timeout_s = config.get("http_timeout_s", 180)
        self.temperature = config.get("temperature", 0)
        self.seed = config.get("seed", 42)
        self.extra_body = config.get("extra_body", {})
        if set(self.extra_body) & {"model", "messages", "max_tokens", "max_completion_tokens", "stream", "n", "tools", "tool_choice", "temperature", "seed"}:
            raise ValueError("extra_body cannot override core generation/protocol fields")
        self.last_http_status: int | None = None
        self._opener = urllib.request.build_opener(NoRedirect())

    def post(self, url: str, payload: dict[str, Any], timeout_s: float) -> Any:
        """Post once and retain JSON separately from its envelope validation."""
        self.last_http_status = None
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        request = urllib.request.Request(url, json.dumps(payload).encode("utf-8"), headers)
        try:
            with self._opener.open(request, timeout=max(0.001, min(timeout_s, self.timeout_s))) as response:
                self.last_http_status = response.status
                return json.load(response)
        except urllib.error.HTTPError as exc:
            raise ClientError(f"HTTP {exc.code}; error body not logged; request not retried",
                              kind="http", http_status=exc.code) from exc
        except ValueError as exc:
            raise ClientError("Response JSON could not be decoded; body not logged; request not retried",
                              kind="response_json", http_status=self.last_http_status) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            # Do not log server error bodies / credentials. No automatic retries:
            # a lost response may still have consumed generation tokens remotely.
            raise ClientError(f"{type(exc).__name__}; request not retried",
                              http_status=self.last_http_status) from exc

    def count(self, messages: list[Message], timeout_s: float) -> int:
        """Fetch a preflight count, which may differ from generation usage."""
        template = {k: v for k, v in self.extra_body.items() if k in {"chat_template", "chat_template_kwargs", "add_special_tokens", "documents"}}
        value = self.post(self.tokenize_url, {**template, "model": self.model, "messages": messages,
                          "add_generation_prompt": True}, timeout_s)
        if not isinstance(value, dict):
            raise ClientError("/tokenize returned a non-object JSON response")
        count = value.get("count")
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ClientError("/tokenize did not return a positive integer count")
        return count

    def complete(self, messages: list[Message], max_tokens: int, timeout_s: float) -> ChatResponse:
        """Generate one response and preserve the authoritative server usage."""
        response = self.post(self.base_url + "/chat/completions", {
            **self.extra_body, "model": self.model, "messages": messages,
            "temperature": self.temperature, "seed": self.seed,
            "max_tokens": max_tokens, "stream": False, "n": 1}, timeout_s)
        return ChatResponseParser.parse(response, self.last_http_status)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """Prevent redirects from forwarding bearer credentials."""

    def redirect_request(self, req: urllib.request.Request, fp: Any, code: int,
                         msg: str, headers: Any, newurl: str) -> NoReturn:
        """Reject every redirect rather than reusing Authorization."""
        raise ClientError("HTTP redirects disabled to avoid leaking Authorization", kind="redirect", http_status=code)
