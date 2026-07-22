import pytest
import numpy as np
from gestures import classify_static, classify_single_hand
from ml_classifier import classify_with_ml, MLGestureClassifier
from train_model import generate_synthetic_landmarks, extract_feature_vector


def test_feature_vector_extraction():
    lm = np.zeros((21, 3), dtype=np.float32)
    features = extract_feature_vector(lm)
    assert features.shape == (75,)  # 63 coords + 12 extra features


def test_ml_classifier_instance():
    classifier = MLGestureClassifier()
    assert classifier.model is not None


def test_ml_prediction_open_palm():
    samples = generate_synthetic_landmarks("OPEN_PALM", samples=5)
    for feat in samples:
        # Reconstruct landmark array from flat coords
        lm3d = feat[:63].reshape(21, 3)
        label, text, conf = classify_with_ml(lm3d)
        assert label is not None
        assert conf > 0.35


def test_classify_static_single_hand():
    lm3d = np.zeros((21, 3), dtype=np.float32)
    lm3d[8] = [0.15, -0.5, 0.0]
    lm3d[12] = [0.0, -0.55, 0.0]
    lm3d[16] = [-0.15, -0.5, 0.0]
    lm3d[20] = [-0.25, -0.45, 0.0]
    lm3d[4] = [0.3, -0.3, 0.0]

    label, text = classify_static([lm3d])
    assert label is not None
    assert isinstance(text, str)
