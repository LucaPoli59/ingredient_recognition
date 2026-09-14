import json
import math
import tempfile
import unittest
from pathlib import Path

from scripts.analise_exp.compare_experiments.analysis.aggregation import aggregate_experiment
from scripts.analise_exp.compare_experiments.analysis.curves import summarize_curve
from scripts.analise_exp.compare_experiments.cli import build_report, main
from scripts.analise_exp.compare_experiments.normalization import select_metric_series
from scripts.analise_exp.compare_experiments.readers.config import decode_data
from scripts.analise_exp.compare_experiments.readers.wandb_local import (
    canonical_tensor_key,
    histogram_statistics,
    normalized_cdf_distance,
    read_trials as read_wandb_trials,
    reconcile_sessions,
)


class ExperimentComparisonTests(unittest.TestCase):
    def test_data_only_config_decoder_keeps_callable_as_text(self):
        encoded = {
            "model": ["class", "<class 'package.DoesNotExist'>"],
            "values": ["list", "[1, 2, 3]"],
            "nested": ["config", {"enabled": ["bool", True]}],
        }

        decoded = decode_data(encoded)

        self.assertEqual(decoded["model"], "<class 'package.DoesNotExist'>")
        self.assertEqual(decoded["values"], [1, 2, 3])
        self.assertTrue(decoded["nested"]["enabled"])

    def test_curve_summary_uses_observed_window(self):
        summary = summarize_curve([
            {"epoch": 0, "value": 4.0},
            {"epoch": 1, "value": 2.0},
            {"epoch": 2, "value": 1.0},
        ], "min", target=1.5)

        self.assertEqual(summary["best"], {"value": 1.0, "coordinate": 2.0, "observation_index": 2})
        self.assertAlmostEqual(summary["normalized_auc"], 2.25)
        self.assertEqual(summary["last"]["value"], 1.0)
        self.assertEqual(summary["target_crossing"]["coordinate"], 2.0)
        censored = summarize_curve([{"epoch": 0, "value": 4.0}], "min", target=1.5)
        self.assertEqual(censored["target_crossing"]["status"], "censored")

    def test_reconciliation_fills_resumed_csv_history_from_tensorboard(self):
        selected = select_metric_series(
            {"series": {"val_loss": [
                {"step": 20.0, "epoch": 2.0, "value": 0.3, "source": "metrics.csv"},
            ]}},
            {"val_loss": [
                {"step": 10, "wall_time": 1.0, "value": 0.5, "source": "events-a"},
                {"step": 20, "wall_time": 2.0, "value": 0.3, "source": "events-b"},
            ]},
            "val_loss",
        )

        self.assertEqual(selected["analysis_axis"], "optimizer_step")
        self.assertEqual([point["analysis_coordinate"] for point in selected["points"]], [10.0, 20.0])
        self.assertEqual(selected["points"][1]["source"], "metrics.csv")

    def test_wandb_session_conflicts_are_preserved(self):
        def session(source, rms):
            return {
                "status": "available",
                "source": source,
                "series": {
                    "parameters/model.weight": {
                        "trajectory": [{
                            "history_record_clocks": {
                                "epoch": 1,
                                "trainer/global_step": 10,
                                "_step": 2,
                                "_timestamp": 3,
                                "_runtime": 4,
                            },
                            "rms_midpoint_estimate": rms,
                        }]
                    }
                },
            }

        reconciliation = reconcile_sessions([session("first", 1.0), session("second", 2.0)])

        self.assertEqual(reconciliation["cross_session_conflicts"], 1)
        self.assertEqual(reconciliation["policy"], "Sessions remain separate; no conflicting observations are averaged.")

    def test_corrupt_optional_wandb_session_is_reported_without_aborting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment = root / "family"
            experiment.mkdir()
            wandb_root = root / "wandb"
            wandb_root.mkdir()
            (wandb_root / "run-family-trial_0.wandb").write_bytes(b"not-a-wandb-stream")

            result = read_wandb_trials(wandb_root, experiment, [0])

        session = result["trials"]["0"]["sessions"][0]
        self.assertEqual(session["status"], "unavailable")
        self.assertIn("reason", session)

    def test_histogram_statistics_are_labeled_midpoint_estimates(self):
        summary = histogram_statistics([0.0, 1.0, 2.0], [1, 3])

        self.assertEqual(summary["recorded_count"], 4)
        self.assertAlmostEqual(summary["mean_midpoint_estimate"], 1.25)
        self.assertAlmostEqual(summary["rms_midpoint_estimate"], math.sqrt(1.75))
        self.assertEqual(normalized_cdf_distance(
            ([0.0, 1.0, 2.0], [1, 3]),
            ([0.0, 1.0, 2.0], [1, 3]),
        ), 0.0)
        self.assertEqual(
            canonical_tensor_key("parameters/graph_72model.layer.weight"),
            "parameters/model.layer.weight",
        )

    def test_mixed_objectives_have_cohort_bests_without_one_comparable_best(self):
        def trial(number, objective, weighted):
            return {
                "number": number,
                "objective": objective,
                "objective_source": "fixture",
                "state": "COMPLETE",
                "comparability_signature": {
                    "metadata_filename": "targets.json",
                    "feature_label": "ingredients",
                    "label_order_sha256": "same",
                    "loss_fn": "BCE",
                    "weighted_loss": weighted,
                },
                "curve": {
                    "status": "available",
                    "points": [{"coordinate": 1.0, "value": objective}],
                },
            }

        summary = aggregate_experiment([trial(0, 0.2, False), trial(1, 0.1, True)], "min")

        self.assertIsNone(summary["best_trial"])
        self.assertEqual(summary["study_selected_trial"]["number"], 1)
        self.assertEqual(len(summary["cohorts"]), 2)

    @staticmethod
    def _make_experiment(root: Path, name: str, offset: float) -> Path:
        experiment = root / name
        classes = ["salt", "pepper"]
        for number in (0, 1):
            trial = experiment / f"trial_{number}"
            trial.mkdir(parents=True)
            config = {
                "hyper_parameters": {
                    "lr": 0.01 * (number + 1),
                    "loss_fn": "torch.nn.BCEWithLogitsLoss",
                    "weighted_loss": False,
                    "log_per_ingredient_metrics": True,
                    "torch_model": {"type": f"models.{name}"},
                },
                "datamodule_hyper_parameters": {
                    "metadata_filename": "targets.json",
                    "feature_label": "ingredients_target",
                    "category": None,
                    "label_encoder": {"classes": classes},
                },
                "trainer_hyper_parameters": {"max_epochs": 2},
            }
            (trial / "trial_config.json").write_text(json.dumps(config), encoding="utf-8")
            rows = [
                "epoch,step,val_loss,val_per_ingredient/f1/0,val_per_ingredient/f1/1",
                f"0,10,{0.5 + offset + number * 0.1},0.4,0.6",
                f"1,20,{0.3 + offset + number * 0.1},0.5,0.7",
            ]
            (trial / "metrics.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
        alias = experiment / "trial_best"
        alias.mkdir()
        (alias / "metrics.csv").write_bytes((experiment / "trial_0" / "metrics.csv").read_bytes())
        return experiment

    def test_end_to_end_json_and_html_share_one_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left = self._make_experiment(root, "family_a", 0.0)
            right = self._make_experiment(root, "family_b", 0.05)
            report = build_report(
                [left, right],
                wandb_scope="none",
                include_tensorboard=False,
            )

            self.assertEqual(len(report["experiments"]), 2)
            self.assertEqual(report["experiments"][0]["summary"]["best_trial"]["number"], 0)
            ingredient = report["experiments"][0]["ingredient_analysis"]["series"][0]
            self.assertEqual(ingredient["label"], "salt")
            self.assertEqual(ingredient["trajectory"][-1]["contributors"], 2)
            aggregate_curves = report["experiments"][0]["summary"]["aggregate_curves"]
            self.assertEqual(next(iter(aggregate_curves.values()))["points"][-1]["contributors"], 2)
            self.assertEqual(report["inter_experiment"]["pairs"][0]["shared_objective_cohorts"], 1)

            output = root / "report"
            exit_code = main([
                "--experiments", str(left), str(right),
                "--output", str(output),
                "--wandb-scope", "none",
                "--no-tensorboard",
            ])
            self.assertEqual(exit_code, 0)
            saved = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
            html = (output / "comparison.html").read_text(encoding="utf-8")
            self.assertEqual(saved["schema_version"], report["schema_version"])
            self.assertIn('id="report-data"', html)
            self.assertIn("family_a", html)


if __name__ == "__main__":
    unittest.main()
