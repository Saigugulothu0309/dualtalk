import unittest

from src.gestures.label_policy import canonicalize_prediction_label


class PredictionLabelPolicyTests(unittest.TestCase):
    def test_supported_model_labels_map_to_app_gesture_ids(self):
        self.assertEqual(canonicalize_prediction_label("help"), "HELP")
        self.assertEqual(canonicalize_prediction_label("thanks"), "THANK YOU")
        self.assertEqual(canonicalize_prediction_label("danger"), "STOP")
        self.assertEqual(canonicalize_prediction_label("call"), "COME HERE")
        self.assertEqual(canonicalize_prediction_label("wait"), "HOLD")

    def test_unsupported_model_labels_do_not_enter_pipeline(self):
        self.assertIsNone(canonicalize_prediction_label("ok/super"))


if __name__ == "__main__":
    unittest.main()
