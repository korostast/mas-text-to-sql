"""Ordered runtime event recording with optional durable delivery."""
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable

Event = dict[str, Any]
EventSink = Callable[[Event], None]


@dataclass
class EventLog:
    """Keep one ordered event stream for transport, SQL, and policy decisions."""

    sink: EventSink | None = None
    events: list[Event] = field(default_factory=list)

    def add(self, kind: str, **data: Any) -> None:
        """Record one event and forward it to the configured trace sink."""
        event: Event = {"sequence": len(self.events), "type": kind, **deepcopy(data)}
        self.events.append(event)
        if self.sink is not None:
            self.sink(event)
