"""Independent within-trial plateau state and full-horizon winner eligibility."""

from dataclasses import dataclass
import math

from optuna.trial import TrialState


def eligible_complete_trials(study):
    return [trial for trial in study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
            if trial.user_attrs.get("completed_epochs") == 40
            and trial.user_attrs.get("optimizer_updates") == 15000
            and trial.value is not None and math.isfinite(trial.value)]


@dataclass
class PlateauState:
    patience: int
    min_delta: float
    best_observed: float = -math.inf
    patience_best: float = -math.inf
    bad_checks: int = 0

    def __post_init__(self):
        if (type(self.patience) is not int or self.patience < 1
                or not math.isfinite(self.min_delta) or self.min_delta < 0):
            raise ValueError("invalid plateau patience/tolerance")

    def observe(self, value):
        if not math.isfinite(value):
            raise ValueError("nonfinite HPO objective")
        self.best_observed = max(self.best_observed, value)
        # Strict checkpoint best and tolerance-based patience are independent.
        if value > self.patience_best + self.min_delta:
            self.patience_best, self.bad_checks = value, 0
        else:
            self.bad_checks += 1

    def should_prune(self, epoch, rule, study):
        if epoch < rule["min_completed_epochs"] or self.bad_checks < self.patience:
            return False, None
        eligible = eligible_complete_trials(study)
        incumbent = max((trial.value for trial in eligible), default=None)
        return incumbent is not None and self.best_observed <= incumbent, incumbent
