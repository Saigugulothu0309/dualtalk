import csv
import os
import tempfile
import unittest

from src.testing.gesture_validation import (
    GestureValidationSession,
    format_test_prediction_label,
    get_app_gesture_for_test_label,
)


class GestureValidationTests(unittest.TestCase):
    def test_prediction_labels_cover_launch_gesture_names(self):
        self.assertEqual(format_test_prediction_label("call"), "CALL")
        self.assertEqual(format_test_prediction_label("thanks"), "THANKS")
        self.assertEqual(format_test_prediction_label("ok/super"), "OK / SUPER")
        self.assertEqual(
            format_test_prediction_label("bathroom/restrom/toilet"),
            "BATHROOM / RESTROOM / TOILET",
        )

    def test_app_gesture_mapping_reuses_sender_pipeline(self):
        self.assertEqual(get_app_gesture_for_test_label("WAIT"), "HOLD")
        self.assertEqual(get_app_gesture_for_test_label("CALL"), "COME HERE")
        self.assertEqual(get_app_gesture_for_test_label("THANKS"), "THANK YOU")
        self.assertEqual(get_app_gesture_for_test_label("DANGER"), "STOP")
        self.assertIsNone(get_app_gesture_for_test_label("OK / SUPER"))

    def test_session_writes_csv_and_reports_accuracy_and_confusions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = os.path.join(temp_dir, "gesture_test.csv")
            session = GestureValidationSession(
                log_path=log_path,
                start_gesture="wait",
            )

            session.record_prediction("wait", 0.91)
            session.record_prediction("hold", 0.84)
            session.advance(1)
            session.record_prediction("call", 0.88)

            with open(log_path, newline="", encoding="utf-8") as file_obj:
                rows = list(csv.DictReader(file_obj))

            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["expected_gesture"], "WAIT")
            self.assertEqual(rows[0]["predicted_gesture"], "WAIT")
            self.assertEqual(rows[0]["correct"], "true")
            self.assertEqual(rows[1]["correct"], "false")

            summary_lines = session.build_accuracy_summary()
            self.assertIn("WAIT: 1/2", summary_lines)
            self.assertIn("HELP: 0/1", summary_lines)

            confusion_lines = session.build_confusion_summary()
            self.assertIn("WAIT -> HOLD: 1", confusion_lines)
            self.assertIn("HELP -> CALL: 1", confusion_lines)


if __name__ == "__main__":
    unittest.main()
