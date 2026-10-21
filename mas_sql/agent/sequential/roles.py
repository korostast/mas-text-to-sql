"""Isolated role contexts assembled from one source-preserving public snapshot."""
import json
from typing import Any, Callable, Mapping
from mas_sql.agent.generation import GenerationController
from mas_sql.agent.sequential.actions import Role
from mas_sql.agent.sequential.closure import ClosurePlan
from mas_sql.agent.sequential.prompts import CONTEXT_VERSION, initial_messages
from mas_sql.llm.types import ChatResponse, Message


class RoleAgent:
    """Own role instructions and bounded private repair feedback, not a budget."""

    def __init__(self, role: Role, task: Mapping[str, str], ddl: str) -> None:
        """Create a task-local base context without another role's private history."""
        self.role: Role = role
        self.base_messages: list[Message] = initial_messages(task, ddl, role)
        self.messages: list[Message] = [message.copy() for message in self.base_messages]
        self.last_response: str | None = None
        self.repair_messages: list[Message] = []

    def context(self, packet: dict[str, Any]) -> list[Message]:
        """Render a planning/request context without changing private state."""
        return ([message.copy() for message in self.base_messages]
                + [message.copy() for message in self.repair_messages]
                + [{"role": "user", "content": json.dumps(packet, ensure_ascii=False)}])

    def request(self, packet: dict[str, Any], generation: GenerationController,
                remaining: Callable[[], float], plan: ClosurePlan) -> ChatResponse | None:
        """Rebuild one snapshot and enforce downstream token holdback."""
        self.messages = self.context(packet)
        self.last_response = None
        response: ChatResponse | None = generation.request(self.messages, remaining,
            context={"role": self.role, "phase": packet["phase"], "context_version": CONTEXT_VERSION,
                     "candidate_version": packet["current_candidate"]["version"] if packet["current_candidate"] else None},
            holdback_tokens=plan.holdback_tokens, min_output_tokens=plan.min_output_tokens)
        if response is not None:
            self.last_response = response["content"]
        return response

    def protocol_error(self, error: str, *, finalization: bool = False) -> None:
        """Replay only the latest invalid response and feedback to its own role."""
        self.repair_messages = []
        if finalization:
            self.repair_messages.append({"role": "user", "content": json.dumps({
                "protocol_error": error[:240], "finalization_only": True,
                "instruction": "Return only an explicit submit_sql for the current reviewed version; "
                    "include grounded response_to_review if required. No revision or diagnostic is allowed. "
                    "One repair attempt remains; no SQL was executed."}, ensure_ascii=False)})
            return
        if self.last_response is not None:
            self.repair_messages.append({"role": "assistant", "content": self.last_response})
        self.repair_messages.append({"role": "user", "content": json.dumps({"protocol_error": error,
            "instruction": "Return one allowed action JSON object; no SQL was executed."}, ensure_ascii=False)})

    def action_accepted(self) -> None:
        """Discard repair exchange once its role has produced a valid action."""
        self.repair_messages = []
        self.last_response = None


class AuthorAgent(RoleAgent):
    """Represent the answer owner without a separate resource ledger."""

    def __init__(self, task: Mapping[str, str], ddl: str) -> None:
        """Create the Author context."""
        super().__init__("author", task, ddl)


class CriticAgent(RoleAgent):
    """Represent the independent advisory reviewer, not an answer generator."""

    def __init__(self, task: Mapping[str, str], ddl: str) -> None:
        """Create the Critic context."""
        super().__init__("critic", task, ddl)
