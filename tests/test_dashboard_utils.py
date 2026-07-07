import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.dashboard_utils import (
    get_project_paths,
    list_class_names,
    model_comparison_records,
    report_accuracy,
    resize_cam,
    safe_read_csv,
    sample_images_for_class,
    top_probabilities,
)


class DashboardUtilsTest(unittest.TestCase):
    def test_get_project_paths_builds_expected_directories(self):
        root = Path("C:/example/project")

        paths = get_project_paths(root)

        self.assertEqual(paths.results_dir, root / "outputs" / "results")
        self.assertEqual(paths.figures_dir, root / "outputs" / "figures")
        self.assertEqual(paths.models_dir, root / "outputs" / "models")
        self.assertEqual(paths.validation_crops_dir, root / "crops" / "validation")

    def test_safe_read_csv_returns_empty_dataframe_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.csv"

            df = safe_read_csv(missing)

        self.assertTrue(df.empty)

    def test_report_accuracy_reads_accuracy_row_precision_column(self):
        df = pd.DataFrame(
            {
                "precision": [0.91, 0.82],
                "recall": [0.91, 0.82],
                "f1-score": [0.91, 0.82],
                "support": [120, 120],
            },
            index=["accuracy", "macro avg"],
        )

        self.assertEqual(report_accuracy(df), 0.91)

    def test_model_comparison_records_normalizes_columns(self):
        df = pd.DataFrame(
            {
                "model": ["Baseline CNN", "ResNet18"],
                "accuracy": [0.76, 0.89],
                "macro_f1": [0.74, 0.88],
            }
        )

        records = model_comparison_records(df)

        self.assertEqual(records[0]["Model"], "Baseline CNN")
        self.assertEqual(records[1]["Accuracy"], 0.89)
        self.assertEqual(records[1]["Macro F1"], 0.88)

    def test_model_comparison_records_allows_accuracy_only_file(self):
        df = pd.DataFrame(
            {
                "model": ["Baseline CNN"],
                "accuracy": [0.893443],
                "report_path": ["outputs/results/baseline_classification_report.csv"],
            }
        )

        records = model_comparison_records(df)

        self.assertEqual(records, [{"Model": "Baseline CNN", "Accuracy": 0.893443}])

    def test_list_and_sample_images_for_class(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "crazing").mkdir()
            (root / "scratches").mkdir()
            (root / "crazing" / "b.jpg").write_bytes(b"image")
            (root / "crazing" / "a.png").write_bytes(b"image")
            (root / "crazing" / "note.txt").write_text("skip", encoding="utf-8")

            self.assertEqual(list_class_names(root), ["crazing", "scratches"])
            samples = sample_images_for_class(root, "crazing", limit=1)

        self.assertEqual([sample.name for sample in samples], ["a.png"])

    def test_top_probabilities_sorts_descending_and_limits(self):
        records = top_probabilities(
            ["crazing", "scratches", "patches"],
            [0.1, 0.7, 0.2],
            limit=2,
        )

        self.assertEqual(records, [("scratches", 0.7), ("patches", 0.2)])

    def test_resize_cam_matches_image_width_and_height(self):
        cam = [[0.0, 1.0], [0.5, 0.25]]

        resized = resize_cam(cam, width=224, height=128)

        self.assertEqual(resized.shape, (128, 224))
        self.assertGreaterEqual(float(resized.min()), 0.0)
        self.assertLessEqual(float(resized.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
