import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from utils import get_finger_extension_ratio, angle_3d, euclidean_dist

MODEL_DIR = 'model'
MODEL_PATH = os.path.join(MODEL_DIR, 'gesture_classifier.joblib')

WRIST = 0
THUMB_TIP, THUMB_MCP = 4, 2
INDEX_TIP, INDEX_MCP = 8, 5
MIDDLE_TIP, MIDDLE_MCP = 12, 9
RING_TIP, RING_MCP = 16, 13
PINKY_TIP, PINKY_MCP = 20, 17


def extract_feature_vector(lm3d):
    """
    Extracts a 75-dimensional normalized feature vector from a (21, 3) 3D landmark array.
    Features include normalized 3D coordinates, extension ratios, and joint angles.
    """
    lm3d = np.array(lm3d, dtype=np.float32)
    wrist = lm3d[WRIST]
    scale_ref = np.linalg.norm(lm3d[MIDDLE_MCP] - wrist) + 1e-6
    lm = (lm3d - wrist) / scale_ref

    # 1. Flattened normalized 3D coordinates (21 * 3 = 63 features)
    coords_flat = lm.flatten()

    # 2. Extension ratios (5 features)
    r_idx = get_finger_extension_ratio(lm, INDEX_TIP, INDEX_MCP)
    r_mid = get_finger_extension_ratio(lm, MIDDLE_TIP, MIDDLE_MCP)
    r_rng = get_finger_extension_ratio(lm, RING_TIP, RING_MCP)
    r_pky = get_finger_extension_ratio(lm, PINKY_TIP, PINKY_MCP)
    r_thm = euclidean_dist(lm[THUMB_TIP], lm[INDEX_MCP])

    # 3. Inter-fingertip distances (5 features)
    d_thm_idx = euclidean_dist(lm[THUMB_TIP], lm[INDEX_TIP])
    d_thm_mid = euclidean_dist(lm[THUMB_TIP], lm[MIDDLE_TIP])
    d_idx_mid = euclidean_dist(lm[INDEX_TIP], lm[MIDDLE_TIP])
    d_mid_rng = euclidean_dist(lm[MIDDLE_TIP], lm[RING_TIP])
    d_rng_pky = euclidean_dist(lm[RING_TIP], lm[PINKY_TIP])

    # 4. Key 3D joint angles (2 features)
    thumb_ang = angle_3d(lm[THUMB_TIP], lm[THUMB_MCP], lm[INDEX_MCP])
    index_ang = angle_3d(lm[INDEX_TIP], lm[INDEX_MCP], lm[WRIST])

    extra_features = np.array([
        r_idx, r_mid, r_rng, r_pky, r_thm,
        d_thm_idx, d_thm_mid, d_idx_mid, d_mid_rng, d_rng_pky,
        thumb_ang, index_ang
    ], dtype=np.float32)

    return np.hstack([coords_flat, extra_features])


def generate_synthetic_landmarks(gesture_type, samples=200):
    """
    Generates synthetic landmark samples for a gesture class with natural Gaussian noise.
    """
    data = []
    for _ in range(samples):
        lm = np.zeros((21, 3), dtype=np.float32)
        lm[0] = [0.0, 0.0, 0.0]  # Wrist
        lm[5] = [0.1, -0.2, 0.0]  # Index MCP
        lm[9] = [0.0, -0.25, 0.0] # Middle MCP
        lm[13] = [-0.1, -0.2, 0.0] # Ring MCP
        lm[17] = [-0.2, -0.15, 0.0] # Pinky MCP
        lm[2] = [0.15, -0.1, 0.0] # Thumb MCP

        # Configure finger extension based on gesture type
        if gesture_type in ["OPEN_PALM", "PALM_FLAT", "SIGN_B"]:
            lm[8] = [0.15, -0.5, 0.0]
            lm[12] = [0.0, -0.55, 0.0]
            lm[16] = [-0.15, -0.5, 0.0]
            lm[20] = [-0.25, -0.45, 0.0]
            lm[4] = [0.3, -0.3, 0.0] if gesture_type == "OPEN_PALM" else [0.1, -0.3, 0.0]

        elif gesture_type in ["FIST", "SIGN_S", "PINCH_CLOSED"]:
            lm[8] = [0.1, -0.25, 0.05]
            lm[12] = [0.0, -0.28, 0.05]
            lm[16] = [-0.1, -0.25, 0.05]
            lm[20] = [-0.2, -0.2, 0.05]
            lm[4] = [0.05, -0.25, 0.05]

        elif gesture_type == "THUMBS_UP":
            lm[8] = [0.1, -0.25, 0.05]
            lm[12] = [0.0, -0.28, 0.05]
            lm[16] = [-0.1, -0.25, 0.05]
            lm[20] = [-0.2, -0.2, 0.05]
            lm[4] = [0.2, -0.5, 0.0]

        elif gesture_type == "THUMBS_DOWN":
            lm[8] = [0.1, -0.25, 0.05]
            lm[12] = [0.0, -0.28, 0.05]
            lm[16] = [-0.1, -0.25, 0.05]
            lm[20] = [-0.2, -0.2, 0.05]
            lm[4] = [0.2, 0.3, 0.0]

        elif gesture_type in ["PEACE", "SIGN_V"]:
            lm[8] = [0.2, -0.5, 0.0]
            lm[12] = [-0.05, -0.5, 0.0]
            lm[16] = [-0.1, -0.25, 0.05]
            lm[20] = [-0.2, -0.2, 0.05]
            lm[4] = [0.05, -0.25, 0.05]

        elif gesture_type == "I_LOVE_YOU":
            lm[8] = [0.2, -0.5, 0.0]
            lm[12] = [0.0, -0.28, 0.05]
            lm[16] = [-0.1, -0.25, 0.05]
            lm[20] = [-0.3, -0.45, 0.0]
            lm[4] = [0.35, -0.3, 0.0]

        elif gesture_type == "ROCK_ON":
            lm[8] = [0.2, -0.5, 0.0]
            lm[12] = [0.0, -0.28, 0.05]
            lm[16] = [-0.1, -0.25, 0.05]
            lm[20] = [-0.3, -0.45, 0.0]
            lm[4] = [0.05, -0.25, 0.05]

        elif gesture_type == "OK_SIGN":
            lm[8] = [0.15, -0.3, 0.0]
            lm[4] = [0.15, -0.3, 0.0]  # Tips touching
            lm[12] = [0.0, -0.55, 0.0]
            lm[16] = [-0.15, -0.5, 0.0]
            lm[20] = [-0.25, -0.45, 0.0]

        elif gesture_type == "POINTING":
            lm[8] = [0.1, -0.5, 0.0]
            lm[12] = [0.0, -0.28, 0.05]
            lm[16] = [-0.1, -0.25, 0.05]
            lm[20] = [-0.2, -0.2, 0.05]
            lm[4] = [0.05, -0.25, 0.05]

        elif gesture_type == "SIGN_L":
            lm[8] = [0.1, -0.5, 0.0]
            lm[4] = [0.35, -0.2, 0.0]
            lm[12] = [0.0, -0.28, 0.05]
            lm[16] = [-0.1, -0.25, 0.05]
            lm[20] = [-0.2, -0.2, 0.05]

        elif gesture_type in ["CALL_ME", "SIGN_Y"]:
            lm[8] = [0.1, -0.25, 0.05]
            lm[12] = [0.0, -0.28, 0.05]
            lm[16] = [-0.1, -0.25, 0.05]
            lm[20] = [-0.3, -0.45, 0.0]
            lm[4] = [0.35, -0.3, 0.0]

        elif gesture_type in ["THREE_FINGERS", "SIGN_W"]:
            lm[8] = [0.2, -0.5, 0.0]
            lm[12] = [0.0, -0.55, 0.0]
            lm[16] = [-0.2, -0.5, 0.0]
            lm[20] = [-0.2, -0.2, 0.05]
            lm[4] = [0.05, -0.25, 0.05]

        elif gesture_type == "FOUR_FINGERS":
            lm[8] = [0.2, -0.5, 0.0]
            lm[12] = [0.0, -0.55, 0.0]
            lm[16] = [-0.15, -0.5, 0.0]
            lm[20] = [-0.25, -0.45, 0.0]
            lm[4] = [0.05, -0.25, 0.05]

        else:
            # Default generic extended/curled gesture
            lm[8] = [0.1, -0.4, 0.0]
            lm[12] = [0.0, -0.4, 0.0]
            lm[16] = [-0.1, -0.4, 0.0]
            lm[20] = [-0.2, -0.4, 0.0]
            lm[4] = [0.2, -0.3, 0.0]

        # Add Gaussian noise for variation
        noise = np.random.normal(0, 0.015, lm.shape).astype(np.float32)
        lm += noise

        feat = extract_feature_vector(lm)
        data.append(feat)

    return data


def train_and_save_model():
    """
    Trains Scikit-Learn Random Forest Classifier for gesture recognition and saves model artifact.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    gesture_classes = [
        "OPEN_PALM", "FIST", "THUMBS_UP", "THUMBS_DOWN", "PEACE",
        "I_LOVE_YOU", "ROCK_ON", "OK_SIGN", "POINTING", "SIGN_L",
        "CALL_ME", "THREE_FINGERS", "FOUR_FINGERS", "PALM_FLAT"
    ]

    X_train = []
    y_train = []

    print("Generating synthetic gesture dataset...")
    for cls in gesture_classes:
        samples = generate_synthetic_landmarks(cls, samples=250)
        X_train.extend(samples)
        y_train.extend([cls] * len(samples))

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train)

    print(f"Training Scikit-Learn Random Forest Model on {len(X_train)} samples across {len(gesture_classes)} classes...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
    clf.fit(X_train, y_train)

    train_acc = clf.score(X_train, y_train)
    print(f"Training Accuracy: {train_acc * 100:.2f}%")

    joblib.dump(clf, MODEL_PATH)
    print(f"Saved Machine Learning Model Artifact to '{MODEL_PATH}'!")
    return clf



if __name__ == '__main__':
    train_and_save_model()
