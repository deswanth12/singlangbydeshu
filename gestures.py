import numpy as np
from utils import get_finger_extension_ratio, angle_3d, euclidean_dist, hand_distance_and_center

# Landmark Indices
WRIST = 0
THUMB_TIP, THUMB_IP, THUMB_MCP, THUMB_CMC = 4, 3, 2, 1
INDEX_TIP, INDEX_DIP, INDEX_PIP, INDEX_MCP = 8, 7, 6, 5
MIDDLE_TIP, MIDDLE_DIP, MIDDLE_PIP, MIDDLE_MCP = 12, 11, 10, 9
RING_TIP, RING_DIP, RING_PIP, RING_MCP = 16, 15, 14, 13
PINKY_TIP, PINKY_DIP, PINKY_PIP, PINKY_MCP = 20, 19, 18, 17


def extract_hand_state(lm3d):
    """
    Extracts scale, rotation, and camera-distance invariant feature states for a (21, 3) hand.
    """
    wrist = lm3d[WRIST]
    scale_ref = np.linalg.norm(lm3d[MIDDLE_MCP] - wrist) + 1e-6
    lm = (lm3d - wrist) / scale_ref

    # 1. Finger Extension Ratios
    r_idx = get_finger_extension_ratio(lm, INDEX_TIP, INDEX_MCP)
    r_mid = get_finger_extension_ratio(lm, MIDDLE_TIP, MIDDLE_MCP)
    r_rng = get_finger_extension_ratio(lm, RING_TIP, RING_MCP)
    r_pky = get_finger_extension_ratio(lm, PINKY_TIP, PINKY_MCP)
    r_thm = euclidean_dist(lm[THUMB_TIP], lm[INDEX_MCP])

    # Boolean extension flags (with soft tolerance)
    idx_ext = r_idx > 1.30
    mid_ext = r_mid > 1.30
    rng_ext = r_rng > 1.30
    pky_ext = r_pky > 1.30
    thm_ext = r_thm > 0.45 or angle_3d(lm[THUMB_TIP], lm[THUMB_MCP], lm[INDEX_MCP]) > 1.0

    ext_count = sum([idx_ext, mid_ext, rng_ext, pky_ext])

    # 2. Inter-fingertip Distances
    d_thm_idx = euclidean_dist(lm[THUMB_TIP], lm[INDEX_TIP])
    d_thm_mid = euclidean_dist(lm[THUMB_TIP], lm[MIDDLE_TIP])
    d_idx_mid = euclidean_dist(lm[INDEX_TIP], lm[MIDDLE_TIP])
    d_mid_rng = euclidean_dist(lm[MIDDLE_TIP], lm[RING_TIP])

    # 3. Y-Heights relative to Wrist
    thumb_y = lm[THUMB_TIP][1]
    mcp_y = lm[THUMB_MCP][1]
    wrist_y = lm[WRIST][1]

    return {
        'lm': lm,
        'r_idx': r_idx, 'r_mid': r_mid, 'r_rng': r_rng, 'r_pky': r_pky, 'r_thm': r_thm,
        'idx_ext': idx_ext, 'mid_ext': mid_ext, 'rng_ext': rng_ext, 'pky_ext': pky_ext, 'thm_ext': thm_ext,
        'ext_count': ext_count,
        'd_thm_idx': d_thm_idx, 'd_thm_mid': d_thm_mid, 'd_idx_mid': d_idx_mid, 'd_mid_rng': d_mid_rng,
        'thumb_y': thumb_y, 'mcp_y': mcp_y, 'wrist_y': wrist_y
    }


from ml_classifier import classify_with_ml


def classify_single_hand(lm3d):
    """
    Classifies single-hand gestures using ML Model predictions first, falling back to candidate templates.
    """
    # 1. Machine Learning Model Inference
    try:
        ml_label, ml_text, ml_conf = classify_with_ml(lm3d)
        if ml_label and ml_conf >= 0.45:
            return ml_label, ml_text, ml_conf
    except Exception as e:
        pass

    s = extract_hand_state(lm3d)
    lm = s['lm']

    candidates = []


    # --- OPEN PALM ("Hello") ---
    if s['ext_count'] >= 4:
        score = 0.85 + (0.1 if s['thm_ext'] else 0.05)
        candidates.append(("OPEN_PALM", "Hello", score))

    # --- PALM FLAT ("Stop") ---
    if s['ext_count'] >= 4 and not s['thm_ext']:
        candidates.append(("PALM_FLAT", "Hold / Stop", 0.92))

    # --- THUMBS UP ("Good") ---
    if s['ext_count'] == 0 and s['thm_ext'] and s['thumb_y'] < s['mcp_y']:
        candidates.append(("THUMBS_UP", "Good / Like", 0.96))

    # --- THUMBS DOWN ("Bad") ---
    if s['ext_count'] == 0 and s['thm_ext'] and s['thumb_y'] > s['wrist_y']:
        candidates.append(("THUMBS_DOWN", "Bad / Dislike", 0.94))

    # --- FIST ("Yes") ---
    if s['ext_count'] == 0 and not s['thm_ext']:
        candidates.append(("FIST", "Yes", 0.92))

    # --- PEACE / VICTORY ("Victory / Peace") ---
    if s['idx_ext'] and s['mid_ext'] and not s['rng_ext'] and not s['pky_ext']:
        score = 0.95 if s['d_idx_mid'] > 0.15 else 0.90
        candidates.append(("PEACE", "Victory / Peace", score))

    # --- I LOVE YOU ---
    if s['idx_ext'] and s['pky_ext'] and s['thm_ext'] and not s['mid_ext'] and not s['rng_ext']:
        candidates.append(("I_LOVE_YOU", "I Love You", 0.97))

    # --- ROCK ON ---
    if s['idx_ext'] and s['pky_ext'] and not s['mid_ext'] and not s['rng_ext'] and not s['thm_ext']:
        candidates.append(("ROCK_ON", "Rock / Cool", 0.94))

    # --- OK SIGN ---
    if s['d_thm_idx'] < 0.22 and s['mid_ext'] and s['rng_ext'] and s['pky_ext']:
        candidates.append(("OK_SIGN", "OK", 0.95))

    # --- POINTING ---
    if s['idx_ext'] and not s['mid_ext'] and not s['rng_ext'] and not s['pky_ext'] and not s['thm_ext']:
        candidates.append(("POINTING", "You / Pointing", 0.93))

    # --- ASL LETTER L ---
    if s['idx_ext'] and s['thm_ext'] and not s['mid_ext'] and not s['rng_ext'] and not s['pky_ext']:
        candidates.append(("SIGN_L", "Letter L", 0.95))

    # --- CALL ME / ASL Y ---
    if s['pky_ext'] and s['thm_ext'] and not s['idx_ext'] and not s['mid_ext'] and not s['rng_ext']:
        candidates.append(("CALL_ME", "Call Me / Y", 0.94))

    # --- THREE FINGERS / ASL W ---
    if s['idx_ext'] and s['mid_ext'] and s['rng_ext'] and not s['pky_ext']:
        candidates.append(("THREE_FINGERS", "Three / W", 0.92))

    # --- FOUR FINGERS / ASL B ---
    if s['idx_ext'] and s['mid_ext'] and s['rng_ext'] and s['pky_ext'] and not s['thm_ext']:
        candidates.append(("FOUR_FINGERS", "Four / B", 0.91))

    # --- PINCH ("Small") ---
    if s['d_thm_idx'] < 0.16 and s['ext_count'] == 0:
        candidates.append(("PINCH", "Small / A bit", 0.90))

    # --- ASL LETTER C ---
    if 0.9 < s['r_idx'] < 1.3 and 0.9 < s['r_mid'] < 1.3 and s['d_thm_idx'] > 0.25:
        candidates.append(("SIGN_C", "Letter C", 0.88))

    # --- ASL LETTER D ---
    if s['idx_ext'] and not s['mid_ext'] and not s['rng_ext'] and not s['pky_ext'] and s['d_thm_mid'] < 0.22:
        candidates.append(("SIGN_D", "Letter D", 0.91))

    # --- ASL LETTER I ---
    if s['pky_ext'] and not s['idx_ext'] and not s['mid_ext'] and not s['rng_ext'] and not s['thm_ext']:
        candidates.append(("SIGN_I", "Letter I", 0.90))

    if not candidates:
        return "UNKNOWN", "Detecting...", 0.30

    # Pick candidate with highest score
    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates[0]


def classify_dual_hand(lm3d_left, lm3d_right, handedness=None):
    """
    Classifies two-handed gestures using spatial relationships, hand shapes, and relative distance.
    `lm3d_left` and `lm3d_right` are (21, 3) landmark arrays.
    """
    dist, center = hand_distance_and_center(lm3d_left, lm3d_right)

    left_s = extract_hand_state(lm3d_left)
    right_s = extract_hand_state(lm3d_right)

    # 1. Pray / Thank You / Please (Palms touching or close in center)
    if dist < 0.45 and left_s['ext_count'] >= 3 and right_s['ext_count'] >= 3:
        return "PRAY_HANDS", "Thank You / Please", 0.98

    # 2. Cross Hands (No / Stop / Cancel)
    if dist < 0.55 and left_s['ext_count'] >= 3 and right_s['ext_count'] >= 3:
        if lm3d_left[WRIST][0] > lm3d_right[WRIST][0]:
            return "CROSS_HANDS", "No / Cancel", 0.96

    # 3. Heart Hands (Thumbs and index tips touching)
    if dist < 0.40 and left_s['thm_ext'] and right_s['thm_ext']:
        d_thumbs = euclidean_dist(lm3d_left[THUMB_TIP], lm3d_right[THUMB_TIP])
        d_indexes = euclidean_dist(lm3d_left[INDEX_TIP], lm3d_right[INDEX_TIP])
        if d_thumbs < 0.30 and d_indexes < 0.30:
            return "HEART_HANDS", "Love / Heart", 0.98

    # 4. Double Thumbs Up (Both thumbs pointing up)
    if left_s['ext_count'] == 0 and left_s['thm_ext'] and right_s['ext_count'] == 0 and right_s['thm_ext']:
        if left_s['thumb_y'] < left_s['mcp_y'] and right_s['thumb_y'] < right_s['mcp_y']:
            return "DOUBLE_THUMBS_UP", "Awesome / Great", 0.98

    # 5. Open Book (Palms side by side facing up)
    if dist < 0.40 and left_s['ext_count'] >= 4 and right_s['ext_count'] >= 4:
        d_wrists = euclidean_dist(lm3d_left[WRIST], lm3d_right[WRIST])
        if d_wrists < 0.30:
            return "OPEN_BOOK", "Book / Read", 0.95

    # 6. Roof / House (Index & Middle tips touching forming a triangle roof)
    if dist < 0.45 and left_s['idx_ext'] and right_s['idx_ext']:
        d_indexes = euclidean_dist(lm3d_left[INDEX_TIP], lm3d_right[INDEX_TIP])
        if d_indexes < 0.20:
            return "ROOF_HOUSE", "Home / House", 0.94

    # 7. Clapping / Applause
    if dist < 0.35 and left_s['ext_count'] >= 4 and right_s['ext_count'] >= 4:
        return "CLAPPING", "Applause / Bravo", 0.93

    # 8. Dual Peace / Victory
    if left_s['idx_ext'] and left_s['mid_ext'] and right_s['idx_ext'] and right_s['mid_ext']:
        if left_s['ext_count'] == 2 and right_s['ext_count'] == 2:
            return "DUAL_PEACE", "Double Victory", 0.97

    # 9. Dual Open Palms (Welcome)
    if left_s['ext_count'] >= 4 and right_s['ext_count'] >= 4:
        return "DUAL_OPEN_PALMS", "Welcome / Open", 0.92

    # Fallback: classify both hands individually and combine their results
    lbl_l, txt_l, conf_l = classify_single_hand(lm3d_left)
    lbl_r, txt_r, conf_r = classify_single_hand(lm3d_right)

    if txt_l and txt_r and txt_l != "Detecting..." and txt_r != "Detecting...":
        if txt_l == txt_r:
            return f"DUAL_{lbl_l}", f"Dual {txt_l}", (conf_l + conf_r) / 2.0
        return f"{lbl_l}+{lbl_r}", f"{txt_l} & {txt_r}", (conf_l + conf_r) / 2.0

    return lbl_r if conf_r >= conf_l else lbl_l, txt_r if conf_r >= conf_l else txt_l, max(conf_l, conf_r)


def classify_static(landmarks_list, handedness=None):
    """
    Main entry point for single or dual hand classification.
    `landmarks_list` is a list containing 1 or 2 hand landmark arrays.
    """
    if not landmarks_list or len(landmarks_list) == 0:
        return None, None

    if len(landmarks_list) == 1:
        lm3d = np.array(landmarks_list[0], dtype=np.float32)
        return classify_single_hand(lm3d)[:2]

    # Two hands present
    lm3d_1 = np.array(landmarks_list[0], dtype=np.float32)
    lm3d_2 = np.array(landmarks_list[1], dtype=np.float32)

    # Assign left and right hands based on handedness or spatial X coordinate
    if handedness and len(handedness) >= 2:
        if handedness[0].lower() == 'left':
            lm3d_left, lm3d_right = lm3d_1, lm3d_2
        else:
            lm3d_left, lm3d_right = lm3d_2, lm3d_1
    else:
        # Spatial ordering: hand with smaller X is screen left
        if lm3d_1[WRIST][0] < lm3d_2[WRIST][0]:
            lm3d_left, lm3d_right = lm3d_1, lm3d_2
        else:
            lm3d_left, lm3d_right = lm3d_2, lm3d_1

    return classify_dual_hand(lm3d_left, lm3d_right, handedness)[:2]



def get_gesture_dictionary():
    return [
        {
            "id": "OPEN_PALM",
            "name": "Hello",
            "category": "Conversational",
            "description": "Open your hand with all 5 fingers fully extended toward the camera.",
            "tip": "Keep palm flat and fingers spread."
        },
        {
            "id": "FIST",
            "name": "Yes / Affirmative",
            "category": "Conversational",
            "description": "Make a clean fist with all fingers curled into your palm.",
            "tip": "Fold thumb over your knuckles."
        },
        {
            "id": "THUMBS_UP",
            "name": "Good / Like",
            "category": "Conversational",
            "description": "Make a fist and point your thumb straight upward.",
            "tip": "Point thumb upwards cleanly."
        },
        {
            "id": "THUMBS_DOWN",
            "name": "Bad / Dislike",
            "category": "Conversational",
            "description": "Make a fist and point your thumb downward.",
            "tip": "Point thumb downwards clearly."
        },
        {
            "id": "PEACE",
            "name": "Victory / Peace (V)",
            "category": "Conversational / ASL",
            "description": "Extend your index and middle fingers upward in a V shape.",
            "tip": "Keep ring and pinky fingers folded down."
        },
        {
            "id": "I_LOVE_YOU",
            "name": "I Love You",
            "category": "Conversational",
            "description": "Extend thumb, index finger, and pinky finger simultaneously.",
            "tip": "Keep middle and ring fingers curled down."
        },
        {
            "id": "OK_SIGN",
            "name": "OK",
            "category": "Conversational",
            "description": "Touch the tip of your thumb to your index finger, forming a circle.",
            "tip": "Keep the other three fingers extended."
        },
        {
            "id": "POINTING",
            "name": "You / Pointing",
            "category": "Conversational",
            "description": "Extend your index finger while keeping other fingers curled.",
            "tip": "Point index finger clearly forward or up."
        },
        {
            "id": "ROCK_ON",
            "name": "Rock / Cool",
            "category": "Conversational",
            "description": "Extend index and pinky fingers while holding thumb over middle & ring fingers.",
            "tip": "Keep middle and ring fingers tucked."
        },
        {
            "id": "CALL_ME",
            "name": "Call Me / Letter Y",
            "category": "Conversational / ASL",
            "description": "Extend thumb and pinky out while curling index, middle, and ring fingers.",
            "tip": "Resembles a phone receiver."
        },
        {
            "id": "PRAY_HANDS",
            "name": "Thank You / Please",
            "category": "Dual-Hand",
            "description": "Bring both open palms together in front of the camera.",
            "tip": "Press both hands together vertically."
        },
        {
            "id": "CROSS_HANDS",
            "name": "No / Stop / Cancel",
            "category": "Dual-Hand",
            "description": "Cross both wrists or open palms across each other.",
            "tip": "Cross wrists in an X shape."
        },
        {
            "id": "HEART_HANDS",
            "name": "Love / Heart",
            "category": "Dual-Hand",
            "description": "Form a heart shape with index fingers and thumbs touching.",
            "tip": "Bring thumbs and index fingertips together."
        },
        {
            "id": "SIGN_L",
            "name": "Letter L",
            "category": "ASL Alphabet",
            "description": "Extend index finger up and thumb sideways to form an 'L'.",
            "tip": "90 degree angle between thumb and index."
        },
        {
            "id": "THREE_FINGERS",
            "name": "Three / Letter W",
            "category": "ASL Numbers",
            "description": "Extend index, middle, and ring fingers upward.",
            "tip": "Keep thumb and pinky tucked."
        }
    ]


