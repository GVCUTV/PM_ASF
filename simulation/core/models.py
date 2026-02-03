from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class Stage(str, Enum):
    DEV = "DEV"
    REVIEW = "REVIEW"
    TESTING = "TESTING"


class EventType(str, Enum):
    ARRIVAL = "ARRIVAL"
    SERVICE_COMPLETION = "SERVICE_COMPLETION"


@dataclass
class Ticket:
    ticket_id: int
    arrival_time: float
    current_stage: Optional[Stage] = None
    queue_entry_time: Optional[float] = None
    service_start_time: Optional[float] = None
    completion_time: Optional[float] = None
    stage_waits: Dict[Stage, float] = field(default_factory=dict)
    stage_services: Dict[Stage, float] = field(default_factory=dict)
    stage_cycles: Dict[Stage, int] = field(default_factory=dict)
    stage_history: List[Stage] = field(default_factory=list)

    def record_wait(self, stage: Stage, wait_time: float) -> None:
        self.stage_waits[stage] = self.stage_waits.get(stage, 0.0) + wait_time

    def record_service(self, stage: Stage, service_time: float) -> None:
        self.stage_services[stage] = self.stage_services.get(stage, 0.0) + service_time
        self.stage_cycles[stage] = self.stage_cycles.get(stage, 0) + 1
        self.stage_history.append(stage)


@dataclass
class Event:
    time: float
    event_type: EventType
    ticket_id: Optional[int] = None
    stage: Optional[Stage] = None


@dataclass
class SystemState:
    queues: Dict[Stage, List[int]] = field(
        default_factory=lambda: {Stage.DEV: [], Stage.REVIEW: [], Stage.TESTING: []}
    )
    active_services: Dict[Stage, List[int]] = field(
        default_factory=lambda: {Stage.DEV: [], Stage.REVIEW: [], Stage.TESTING: []}
    )
    tickets: Dict[int, Ticket] = field(default_factory=dict)
    closed_tickets: List[int] = field(default_factory=list)
