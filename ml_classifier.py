import os
import numpy as np

try:
    import joblib
    from train_model import MODEL_PATH, extract_feature_vector
    HAS_ML = True
except ImportError:
    HAS_ML = False
    MODEL_PATH = os.path.join('model', 'gesture_classifier.joblib')


LABEL_DISPLAY_MAP = {
    "OPEN_PALM": ("HELLO", "Hello"),
    "FIST": ("YES", "Yes"),
    "THUMBS_UP": ("GOOD", "Good / Agree"),
    "THUMBS_DOWN": ("BAD", "Bad / Disagree"),
    "PEACE": ("PEACE", "Peace / Victory"),
    "I_LOVE_YOU": ("I_LOVE_YOU", "I Love You"),
    "ROCK_ON": ("ROCK", "Rock / Party"),
    "OK_SIGN": ("OK", "OK / Perfect"),
    "POINTING": ("YOU", "You / Pointing"),
    "SIGN_L": ("LETTER_L", "ASL Letter L"),
    "CALL_ME": ("CALL_ME", "Call Me"),
    "THREE_FINGERS": ("NUMBER_3", "ASL Number 3"),
    "FOUR_FINGERS": ("NUMBER_4", "ASL Number 4"),
    "PALM_FLAT": ("STOP", "Stop / Hold")
}


class MLGestureClassifier:
    def __init__(self, model_path=MODEL_PATH):
        self.model = None
        self.load_model(model_path)

    def load_model(self, model_path):
        if os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
            except Exception as e:
                print(f"Error loading ML model: {e}")
                self.model = None

    def predict(self, lm3d):
        """
        Predicts gesture label and confidence from (21, 3) landmarks using trained ML model.
        Returns: (internal_label, display_text, confidence_score)
        """
        if not HAS_ML or self.model is None:
            return None, None, 0.0


        try:
            features = extract_feature_vector(lm3d).reshape(1, -1)
            probs = self.model.predict_proba(features)[0]
            max_idx = np.argmax(probs)
            confidence = float(probs[max_idx])
            raw_label = self.model.classes_[max_idx]

            if confidence >= 0.35:
                label_tuple = LABEL_DISPLAY_MAP.get(raw_label, (raw_label, raw_label.replace("_", " ").title()))
                return label_tuple[0], label_tuple[1], confidence
        except Exception as err:
            print(f"ML Prediction Error: {err}")

        return None, None, 0.0


ml_classifier_instance = MLGestureClassifier()


def classify_with_ml(lm3d):
    return ml_classifier_instance.predict(lm3d)
