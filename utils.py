import numpy as np
from collections import deque


class TemporalSmoother:
    """
    Advanced Temporal Smoother with majority voting and adaptive confidence thresholding.
    """

    def __init__(self, maxlen=5, min_confidence=0.4):
        self.buffer = deque(maxlen=maxlen)
        self.scores = deque(maxlen=maxlen)
        self.min_confidence = min_confidence
        self.sentence = []
        self.last_added_gesture = None
        self.stable_counter = 0

    def add(self, label, confidence=1.0):
        if label:
            self.buffer.append(label)
            self.scores.append(confidence)

    def get(self):
        if not self.buffer:
            return None, 0.0

        labels = list(self.buffer)
        unique_labels, counts = np.unique(labels, return_counts=True)
        max_idx = np.argmax(counts)
        dominant_label = unique_labels[max_idx]
        confidence = counts[max_idx] / len(labels)

        if confidence >= self.min_confidence:
            return dominant_label, float(confidence)
        return dominant_label, float(confidence)

    def update_sentence(self, label, min_stable_frames=2):
        if label == self.last_added_gesture:
            self.stable_counter += 1
        else:
            self.last_added_gesture = label
            self.stable_counter = 1

        if self.stable_counter == min_stable_frames:
            if label not in ["UNKNOWN", "Detecting...", None]:
                self.sentence.append(label)
                return True
        return False

    def clear(self):
        self.buffer.clear()
        self.scores.clear()
        self.sentence.clear()
        self.last_added_gesture = None
        self.stable_counter = 0

    def get_sentence_text(self):
        return " ".join(self.sentence)


def get_finger_extension_ratio(lm, tip_idx, mcp_idx, wrist_idx=0):
    """
    Calculates scale-and-rotation invariant finger extension ratio.
    Ratio > 1.45 indicates extended finger.
    Ratio < 1.15 indicates curled finger.
    """
    d_tip_wrist = np.linalg.norm(lm[tip_idx] - lm[wrist_idx])
    d_mcp_wrist = np.linalg.norm(lm[mcp_idx] - lm[wrist_idx]) + 1e-6
    return float(d_tip_wrist / d_mcp_wrist)


def angle_3d(a, b, c):
    """Computes 3D angle (in radians) at vertex b."""
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-8 or norm_bc < 1e-8:
        return 0.0
    cosang = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosang = np.clip(cosang, -1.0, 1.0)
    return float(np.arccos(cosang))


def euclidean_dist(p1, p2):
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def hand_distance_and_center(lm_left, lm_right):
    left_wrist = lm_left[0]
    right_wrist = lm_right[0]
    dist = float(np.linalg.norm(left_wrist - right_wrist))
    center = (left_wrist + right_wrist) / 2.0
    return dist, center


