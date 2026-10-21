"""Ordered runtime event recording with optional durable delivery."""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

Event = dict[str, Any]
EventSink = Callable[[Event], None]


@dataclass
class EventLog:
    """Keep one ordered event stream for transport, SQL, and policy decisions."""

    sink: EventSink | None = None
    events: list[Event] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)

    @contextmanager
    def scoped(self, context: dict[str, Any] | None) -> Iterator[None]:
        """Attach temporary call metadata without leaking it into later events."""
        previous: dict[str, Any] = self.context
        self.context = {**previous, **(context or {})}
        try:
            yield
        finally:
            self.context = previous

    def add(self, kind: str, **data: Any) -> None:
        """Record one event and forward it to the configured trace sink."""
        event: Event = {"sequence": len(self.events), "type": kind, **deepcopy(self.context), **deepcopy(data)}
        self.events.append(event)
        if self.sink is not None:
            self.sink(event)
