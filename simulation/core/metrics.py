from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .models import Stage, Ticket


@dataclass
class BatchStats:
    capacity_time: Dict[Stage, float] = field(
        default_factory=lambda: {Stage.DEV: 0.0, Stage.REVIEW: 0.0, Stage.TESTING: 0.0}
    )
    busy_time: Dict[Stage, float] = field(
        default_factory=lambda: {Stage.DEV: 0.0, Stage.REVIEW: 0.0, Stage.TESTING: 0.0}
    )
    queue_time: Dict[Stage, float] = field(
        default_factory=lambda: {Stage.DEV: 0.0, Stage.REVIEW: 0.0, Stage.TESTING: 0.0}
    )
    completions: Dict[Stage, int] = field(
        default_factory=lambda: {Stage.DEV: 0, Stage.REVIEW: 0, Stage.TESTING: 0}
    )
    closed_tickets: int = 0
    ticket_times: List[float] = field(default_factory=list)
    feedback_review: int = 0
    feedback_test: int = 0


class MetricsCollector:
    def __init__(
        self,
        horizon: float,
        batch_size: Optional[float] = None,
        batch_count: Optional[int] = None,
    ) -> None:
        self.horizon = horizon
        self.batch_size = batch_size
        self.batch_count = batch_count
        self.last_time = 0.0
        self.arrivals = 0
        self.closed_tickets = 0
        self.feedback_review = 0
        self.feedback_test = 0
        self.capacity_time = {Stage.DEV: 0.0, Stage.REVIEW: 0.0, Stage.TESTING: 0.0}
        self.busy_time = {Stage.DEV: 0.0, Stage.REVIEW: 0.0, Stage.TESTING: 0.0}
        self.queue_time = {Stage.DEV: 0.0, Stage.REVIEW: 0.0, Stage.TESTING: 0.0}
        self.completions = {Stage.DEV: 0, Stage.REVIEW: 0, Stage.TESTING: 0}
        self.ticket_times: List[float] = []
        self.batch_stats: List[BatchStats] = []
        if self.batch_size and self.batch_count:
            self.batch_stats = [BatchStats() for _ in range(self.batch_count)]

    def record_arrival(self) -> None:
        self.arrivals += 1

    def update_time(
        self,
        new_time: float,
        capacity_counts: Dict[Stage, int],
        busy_counts: Dict[Stage, int],
        queue_lengths: Dict[Stage, int],
    ) -> None:
        if new_time < self.last_time:
            return
        if self.batch_size and self.batch_count:
            self._update_time_batches(new_time, capacity_counts, busy_counts, queue_lengths)
        else:
            delta = new_time - self.last_time
            for stage in Stage:
                self.capacity_time[stage] += capacity_counts[stage] * delta
                self.busy_time[stage] += busy_counts[stage] * delta
                self.queue_time[stage] += queue_lengths[stage] * delta
        self.last_time = new_time

    def _update_time_batches(
        self,
        new_time: float,
        capacity_counts: Dict[Stage, int],
        busy_counts: Dict[Stage, int],
        queue_lengths: Dict[Stage, int],
    ) -> None:
        current = self.last_time
        while current < new_time:
            batch_index = int(current // self.batch_size)
            if batch_index >= self.batch_count:
                break
            batch_end = (batch_index + 1) * self.batch_size
            segment_end = min(new_time, batch_end)
            delta = segment_end - current
            batch = self.batch_stats[batch_index]
            for stage in Stage:
                batch.capacity_time[stage] += capacity_counts[stage] * delta
                batch.busy_time[stage] += busy_counts[stage] * delta
                batch.queue_time[stage] += queue_lengths[stage] * delta
            current = segment_end

    def record_completion(self, ticket: Ticket, stage: Stage, time: float) -> None:
        self.completions[stage] += 1
        if stage == Stage.TESTING:
            self.closed_tickets += 1
            self.ticket_times.append(time - ticket.arrival_time)
        if self.batch_stats and self.batch_size:
            batch_index = int(time // self.batch_size)
            if batch_index < len(self.batch_stats):
                batch = self.batch_stats[batch_index]
                batch.completions[stage] += 1
                if stage == Stage.TESTING:
                    batch.closed_tickets += 1
                    batch.ticket_times.append(time - ticket.arrival_time)

    def record_feedback(self, stage: Stage) -> None:
        if stage == Stage.REVIEW:
            self.feedback_review += 1
        elif stage == Stage.TESTING:
            self.feedback_test += 1
        if self.batch_stats and self.batch_size:
            batch_index = int(self.last_time // self.batch_size)
            if batch_index < len(self.batch_stats):
                batch = self.batch_stats[batch_index]
                if stage == Stage.REVIEW:
                    batch.feedback_review += 1
                elif stage == Stage.TESTING:
                    batch.feedback_test += 1

    def summary(self) -> Dict[str, float]:
        summary: Dict[str, float] = {}
        for stage in Stage:
            throughput = self.completions[stage] / self.horizon if self.horizon > 0 else 0.0
            capacity = self.capacity_time[stage]
            utilization = self.busy_time[stage] / capacity if capacity > 0 else 0.0
            avg_queue = self.queue_time[stage] / self.horizon if self.horizon > 0 else 0.0
            summary[f"throughput_{stage.value.lower()}"] = throughput
            summary[f"utilization_{stage.value.lower()}"] = utilization
            summary[f"avg_queue_{stage.value.lower()}"] = avg_queue
        total_review = self.completions[Stage.REVIEW]
        total_test = self.completions[Stage.TESTING]
        summary["feedback_rate_review"] = (
            self.feedback_review / total_review if total_review > 0 else 0.0
        )
        summary["feedback_rate_testing"] = (
            self.feedback_test / total_test if total_test > 0 else 0.0
        )
        summary["closed_tickets"] = float(self.closed_tickets)
        summary["mean_time_in_system"] = (
            sum(self.ticket_times) / len(self.ticket_times) if self.ticket_times else 0.0
        )
        return summary
