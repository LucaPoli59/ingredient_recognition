"""Opt-in exact batching arithmetic for the Phase 5 experimental models.

These helpers do not change BaseLGNM or the frozen selector. Sample-mean
normalization assumes one finite, fixed-size, single-device loader with
``drop_last=False`` and no custom sampler. A runtime adapter must pass the
records actually consumed after applying its batch limit, not blindly the
whole dataset size. Gradient accumulation does not reproduce large-batch
BatchNorm statistics.
"""

from dataclasses import dataclass
import math
from typing import ClassVar


def _integer(value: int, name: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, not a bool")
    return value


@dataclass(frozen=True)
class ExactBatchPlan:
    requested_batch_size: int
    physical_batch_size: int
    accumulate_grad_batches: int
    max_physical_batch_size: int | None = None

    SCHEMA_VERSION: ClassVar[int] = 1

    def __post_init__(self) -> None:
        _integer(self.requested_batch_size, "requested_batch_size")
        _integer(self.physical_batch_size, "physical_batch_size")
        _integer(self.accumulate_grad_batches, "accumulate_grad_batches")
        if self.max_physical_batch_size is not None:
            _integer(self.max_physical_batch_size, "max_physical_batch_size")
            if self.physical_batch_size > self.max_physical_batch_size:
                raise ValueError("physical batch exceeds the declared cap")
        if self.actual_effective_batch_size != self.requested_batch_size:
            raise ValueError("physical batch times accumulation must equal requested batch")

    @classmethod
    def resolve(
            cls, requested_batch_size: int,
            max_physical_batch_size: int | None = None,
            physical_batch_size: int | None = None,
    ) -> "ExactBatchPlan":
        requested = _integer(requested_batch_size, "requested_batch_size")
        cap = requested if max_physical_batch_size is None else _integer(
            max_physical_batch_size, "max_physical_batch_size")
        if physical_batch_size is None:
            physical = requested if cap >= requested else 1
            if cap < requested:
                for divisor in range(1, math.isqrt(requested) + 1):
                    if requested % divisor == 0:
                        for candidate in (divisor, requested // divisor):
                            if physical < candidate <= cap:
                                physical = candidate
        else:
            physical = _integer(physical_batch_size, "physical_batch_size")
            if requested % physical != 0:
                raise ValueError("explicit physical batch must divide requested batch exactly")
            if physical > cap:
                raise ValueError("physical batch exceeds the declared cap")
        return cls(requested, physical, requested // physical, max_physical_batch_size)

    @property
    def actual_effective_batch_size(self) -> int:
        """Size of a complete accumulation group; the final group can be smaller."""
        return self.physical_batch_size * self.accumulate_grad_batches

    def to_config(self) -> dict:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "requested_batch_size": self.requested_batch_size,
            "max_physical_batch_size": self.max_physical_batch_size,
            "physical_batch_size": self.physical_batch_size,
            "accumulate_grad_batches": self.accumulate_grad_batches,
        }

    @classmethod
    def from_config(cls, config: dict) -> "ExactBatchPlan":
        expected = {
            "schema_version", "requested_batch_size", "max_physical_batch_size",
            "physical_batch_size", "accumulate_grad_batches",
        }
        if not isinstance(config, dict) or set(config) != expected:
            raise ValueError("exact batch configuration fields are missing or unsupported")
        version = _integer(config["schema_version"], "schema_version")
        if version != cls.SCHEMA_VERSION:
            raise ValueError("unsupported exact batch schema version")
        # Restore the saved plan verbatim; never re-resolve against a current cap.
        return cls(**{key: value for key, value in config.items() if key != "schema_version"})


def finite_loader_consumed_records(
        dataset_records: int, plan: ExactBatchPlan,
        limit_train_batches: int | float = 1.0,
) -> int:
    """Count records for a conventional non-dropping loader and batch limit.

An integer limit is a number of microbatches; a float in [0, 1] is a fraction
of loader batches, truncated down. Thus ``1`` and ``1.0`` differ. A positive
fraction yielding no batches is rejected. Runtime integration must verify
that the actual Lightning horizon/loader follows these assumptions, and
cannot reuse this count for distributed, sampled or prematurely stopped
execution.
    """
    records = _integer(dataset_records, "dataset_records", minimum=0)
    batches = (records + plan.physical_batch_size - 1) // plan.physical_batch_size
    if type(limit_train_batches) is int:
        limit = _integer(limit_train_batches, "limit_train_batches", minimum=0)
        consumed_batches = min(batches, limit)
    elif type(limit_train_batches) is float:
        if not math.isfinite(limit_train_batches) or not 0 <= limit_train_batches <= 1:
            raise ValueError("fractional limit_train_batches must be finite and in [0, 1]")
        consumed_batches = int(batches * limit_train_batches)
        if records and limit_train_batches > 0 and consumed_batches == 0:
            raise ValueError("positive fractional limit_train_batches yields no batches")
    else:
        raise ValueError("limit_train_batches must be an int or float, not a bool")
    return min(records, consumed_batches * plan.physical_batch_size)


def planned_optimizer_updates(consumed_records: int, plan: ExactBatchPlan) -> int:
    """Include the final partial accumulation group as one optimizer update."""
    records = _integer(consumed_records, "consumed_records", minimum=0)
    return (records + plan.requested_batch_size - 1) // plan.requested_batch_size


def accumulation_loss_scale(
        batch_index: int, batch_records: int, consumed_records: int,
        plan: ExactBatchPlan,
) -> float:
    """Scale a mean-reduced loss before Lightning's fixed division by K.

Multiply only the loss returned for backward, not the loss logged as an
unmodified per-sample mean. K*n_i/N_group corrects both a short microbatch
and a short final accumulation group without dropping any records.
    """
    index = _integer(batch_index, "batch_index", minimum=0)
    actual = _integer(batch_records, "batch_records")
    records = _integer(consumed_records, "consumed_records")
    remaining = records - index * plan.physical_batch_size
    expected = min(plan.physical_batch_size, remaining)
    if expected <= 0 or actual != expected:
        raise ValueError("microbatch size/index differs from the consumed-record contract")
    group_start = (index // plan.accumulate_grad_batches) * plan.requested_batch_size
    group_records = min(plan.requested_batch_size, records - group_start)
    return plan.accumulate_grad_batches * actual / group_records
