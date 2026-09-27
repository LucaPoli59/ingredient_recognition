"""Exact physical-batch and Lightning accumulation arithmetic."""

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class BatchPlan:
    requested_batch_size: int
    physical_batch_size: int
    accumulate_grad_batches: int

    def to_dict(self):
        return asdict(self)


def batch_candidates(requested_batch_size: int) -> list[int]:
    if requested_batch_size < 1:
        raise ValueError("requested batch size must be positive")
    return [size for size in range(requested_batch_size, 0, -1)
            if requested_batch_size % size == 0]


def resolve_batch_plan(requested_batch_size: int, max_allowed_batch_size: int) -> BatchPlan:
    if max_allowed_batch_size < 1:
        raise ValueError("maximum physical batch size must be positive")
    physical = next(size for size in batch_candidates(requested_batch_size)
                    if size <= max_allowed_batch_size)
    return BatchPlan(requested_batch_size, physical, requested_batch_size // physical)


def accumulation_loss_factor(
        batch_index: int, actual_batch_size: int, train_records: int, plan: BatchPlan) -> float:
    """Correct Lightning's fixed 1/K scaling for the incomplete final group."""
    group_start = (batch_index // plan.accumulate_grad_batches) * plan.requested_batch_size
    group_records = min(plan.requested_batch_size, train_records - group_start)
    if not 0 < actual_batch_size <= plan.physical_batch_size or group_records <= 0:
        raise ValueError("invalid batch index/record count for accumulation")
    return plan.accumulate_grad_batches * actual_batch_size / group_records


def planned_optimizer_steps(train_records: int, plan: BatchPlan) -> int:
    return math.ceil(train_records / plan.requested_batch_size)
