import numpy as np
import pytest

from utils import TemporalSmoother, angle_3d, euclidean_dist, get_finger_extension_ratio


def test_euclidean_dist():
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([3.0, 4.0, 0.0])
    assert pytest.approx(euclidean_dist(p1, p2)) == 5.0


def test_angle_3d():
    p1 = np.array([1.0, 0.0, 0.0])
    p_center = np.array([0.0, 0.0, 0.0])
    p2 = np.array([0.0, 1.0, 0.0])
    # 90 degrees in radians is ~1.5708
    angle = angle_3d(p1, p_center, p2)
    assert pytest.approx(angle, 0.01) == 1.57


def test_finger_extension_ratio():
    lm = np.zeros((21, 3))
    lm[0] = [0.0, 0.0, 0.0]  # Wrist
    lm[5] = [0.0, -0.2, 0.0] # Index MCP
    lm[8] = [0.0, -0.5, 0.0] # Index Tip
    ratio = get_finger_extension_ratio(lm, 8, 5, 0)
    assert pytest.approx(ratio, abs=1e-3) == 2.5



def test_temporal_smoother():
    smoother = TemporalSmoother(maxlen=4)
    assert smoother.get() == (None, 0.0)

    smoother.add("Hello")
    smoother.add("Hello")
    smoother.add("Hello")
    label, conf = smoother.get()
    assert label == "Hello"
    assert conf == 1.0

    smoother.reset()
    assert smoother.get() == (None, 0.0)
