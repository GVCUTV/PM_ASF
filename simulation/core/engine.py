import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .developer_pool import DeveloperPool
from .metrics import MetricsCollector
from .models import Event, EventType, Stage, SystemState, Ticket
from .rng import RNGStreams, exponential, lognormal, weibull


@dataclass
class SimulationConfig:
    arrival_rate: float
    feedback_review: float
    feedback_testing: float
    service_params: Dict[Stage, Dict[str, Dict[str, float]]]
    horizon: float
    service_time_scale: float = 1.0
    service_time_caps: Optional[Dict[Stage, float]] = None


class SimulationEngine:
    def __init__(
        self,
        config: SimulationConfig,
        developer_pool: DeveloperPool,
        rngs: RNGStreams,
        metrics: MetricsCollector,
    ) -> None:
        self.config = config
        self.developer_pool = developer_pool
        self.rngs = rngs
        self.metrics = metrics
        self.state = SystemState()
        self.event_queue: List[Tuple[float, int, Event]] = []
        self.event_counter = 0
        self.current_time = 0.0
        self.ticket_counter = 0

    def schedule_event(self, event: Event) -> None:
        heapq.heappush(self.event_queue, (event.time, self.event_counter, event))
        self.event_counter += 1

    def initialize(self) -> None:
        self.schedule_event(Event(time=0.0, event_type=EventType.ARRIVAL))

    def run(self) -> None:
        self.initialize()
        while self.event_queue:
            next_time, _, event = self.event_queue[0]
            if next_time > self.config.horizon:
                break
            self._advance_time(next_time)
            heapq.heappop(self.event_queue)
            if event.event_type == EventType.ARRIVAL:
                self._handle_arrival(event)
            elif event.event_type == EventType.SERVICE_COMPLETION:
                self._handle_service_completion(event)

    def _advance_time(self, target_time: float) -> None:
        while self.current_time < target_time:
            next_transition = self.developer_pool.next_transition_time()
            segment_end = min(target_time, next_transition)
            capacity_counts = self.developer_pool.capacity_counts()
            busy_counts = {stage: len(self.state.active_services[stage]) for stage in Stage}
            queue_lengths = {stage: len(self.state.queues[stage]) for stage in Stage}
            self.metrics.update_time(segment_end, capacity_counts, busy_counts, queue_lengths)
            self.current_time = segment_end
            if self.current_time == next_transition:
                self.developer_pool.update(self.current_time)

    def _handle_arrival(self, event: Event) -> None:
        self.ticket_counter += 1
        ticket = Ticket(ticket_id=self.ticket_counter, arrival_time=event.time)
        ticket.current_stage = Stage.DEV
        ticket.queue_entry_time = event.time
        self.state.tickets[ticket.ticket_id] = ticket
        self.state.queues[Stage.DEV].append(ticket.ticket_id)
        self.metrics.record_arrival()

        next_arrival_time = event.time + exponential(self.rngs.arrivals, self.config.arrival_rate)
        if next_arrival_time <= self.config.horizon:
            self.schedule_event(Event(time=next_arrival_time, event_type=EventType.ARRIVAL))

        self._try_start_service(Stage.DEV)

    def _try_start_service(self, stage: Stage) -> None:
        capacity_counts = self.developer_pool.capacity_counts()
        available = capacity_counts[stage] - len(self.state.active_services[stage])
        while available > 0 and self.state.queues[stage]:
            ticket_id = self.state.queues[stage].pop(0)
            ticket = self.state.tickets[ticket_id]
            ticket.current_stage = stage
            if ticket.queue_entry_time is not None:
                wait_time = self.current_time - ticket.queue_entry_time
                ticket.record_wait(stage, wait_time)
            ticket.service_start_time = self.current_time
            service_time = self._sample_service_time(stage)
            ticket.record_service(stage, service_time)
            self.state.active_services[stage].append(ticket_id)
            self.schedule_event(
                Event(
                    time=self.current_time + service_time,
                    event_type=EventType.SERVICE_COMPLETION,
                    ticket_id=ticket_id,
                    stage=stage,
                )
            )
            available -= 1

    def _sample_service_time(self, stage: Stage) -> float:
        params = self.config.service_params[stage]
        distribution = params["distribution"].lower()
        parameters = params["parameters"]
        if distribution == "lognormal":
            service_time = lognormal(self.rngs.services, parameters["mu"], parameters["sigma"])
        elif distribution == "weibull":
            service_time = weibull(self.rngs.services, parameters["shape"], parameters["scale"])
        elif distribution == "exponential":
            rate = parameters.get("rate") or parameters.get("lambda")
            if rate is None:
                raise ValueError("Exponential parameters missing rate")
            service_time = exponential(self.rngs.services, float(rate))
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")
        service_time *= self.config.service_time_scale
        if self.config.service_time_caps and stage in self.config.service_time_caps:
            cap = self.config.service_time_caps[stage]
            if cap is not None:
                service_time = min(service_time, cap)
        return service_time

    def _handle_service_completion(self, event: Event) -> None:
        if event.ticket_id is None or event.stage is None:
            return
        stage = event.stage
        if event.ticket_id in self.state.active_services[stage]:
            self.state.active_services[stage].remove(event.ticket_id)
        ticket = self.state.tickets[event.ticket_id]
        self.metrics.record_completion(ticket, stage, event.time)

        if stage == Stage.DEV:
            self._enqueue_ticket(ticket, Stage.REVIEW, event.time)
        elif stage == Stage.REVIEW:
            if self.rngs.routing.random() < self.config.feedback_review:
                self.metrics.record_feedback(Stage.REVIEW)
                self._enqueue_ticket(ticket, Stage.DEV, event.time)
            else:
                self._enqueue_ticket(ticket, Stage.TESTING, event.time)
        elif stage == Stage.TESTING:
            if self.rngs.routing.random() < self.config.feedback_testing:
                self.metrics.record_feedback(Stage.TESTING)
                self._enqueue_ticket(ticket, Stage.DEV, event.time)
            else:
                ticket.completion_time = event.time
                self.state.closed_tickets.append(ticket.ticket_id)

        self._try_start_service(stage)
        if stage != Stage.DEV:
            self._try_start_service(Stage.DEV)
        if stage != Stage.REVIEW:
            self._try_start_service(Stage.REVIEW)
        if stage != Stage.TESTING:
            self._try_start_service(Stage.TESTING)

    def _enqueue_ticket(self, ticket: Ticket, stage: Stage, time: float) -> None:
        ticket.current_stage = stage
        ticket.queue_entry_time = time
        self.state.queues[stage].append(ticket.ticket_id)
