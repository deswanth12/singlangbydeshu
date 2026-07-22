import time
import json
import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from gestures import classify_static, get_gesture_dictionary
import database as db

app = Flask(__name__, static_folder='static', template_folder='templates')

session_stats = {
    'start_time': time.time(),
    'classifications_count': 0,
    'unique_gestures_seen': set()
}

CUSTOM_SIGNS_FILE = 'custom_signs.json'


def load_custom_signs():
    if os.path.exists(CUSTOM_SIGNS_FILE):
        try:
            with open(CUSTOM_SIGNS_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_custom_signs_to_file(signs):
    with open(CUSTOM_SIGNS_FILE, 'w') as f:
        json.dump(signs, f, indent=2)


@app.route('/')
def index():
    """Serves the main sign language translator dashboard."""
    return render_template('index.html')


@app.route('/classify', methods=['POST'])
def classify():
    """
    Receives landmark data for 1 or 2 hands, classifies it using ML Model,
    and logs recognized signs into SQLite database.
    """
    data = request.get_json()
    if not data or 'landmarks' not in data:
        return jsonify({'error': 'Missing landmarks data'}), 400

    raw_landmarks = data['landmarks']
    handedness = data.get('handedness', None)

    if not raw_landmarks:
        return jsonify({'label': None, 'text': None})

    session_stats['classifications_count'] += 1
    hand_mode = 'Dual-Hand' if (isinstance(raw_landmarks[0][0], list) and len(raw_landmarks) > 1) else 'Single-Hand'

    if isinstance(raw_landmarks[0][0], list):
        label, text = classify_static(raw_landmarks, handedness)
    else:
        label, text = classify_static([raw_landmarks], handedness)

    if label:
        session_stats['unique_gestures_seen'].add(label)
        # Log recognized sign to SQLite DB
        try:
            db.record_translation(text or label, confidence=0.90, hand_mode=hand_mode)
        except Exception as err:
            print(f"DB Logging Error: {err}")

    return jsonify({
        'label': label,
        'text': text,
        'total_classified': session_stats['classifications_count']
    })


@app.route('/api/gestures', methods=['GET'])
def get_gestures():
    """Returns built-in plus user-created custom gesture dictionary."""
    built_in = get_gesture_dictionary()
    custom = load_custom_signs()
    return jsonify({'gestures': built_in + custom})


@app.route('/api/save_custom_sign', methods=['POST'])
def save_custom_sign():
    """Saves a new user-recorded custom gesture."""
    data = request.get_json()
    if not data or 'name' not in data or 'landmarks' not in data:
        return jsonify({'error': 'Invalid payload'}), 400

    name = data['name'].strip()
    category = data.get('category', 'Custom')
    description = data.get('description', 'User-defined custom sign gesture.')
    tip = data.get('tip', 'Perform custom landmark gesture.')

    custom_signs = load_custom_signs()
    new_sign = {
        'id': f"CUSTOM_{name.upper().replace(' ', '_')}",
        'name': name,
        'category': category,
        'description': description,
        'tip': tip,
        'landmarks': data['landmarks']
    }
    custom_signs.append(new_sign)
    save_custom_signs_to_file(custom_signs)

    return jsonify({'status': 'success', 'sign': new_sign})


@app.route('/api/history', methods=['GET', 'DELETE'])
def history_endpoint():
    """GET returns persistent translation history from SQLite DB; DELETE clears history."""
    if request.method == 'DELETE':
        db.clear_history()
        return jsonify({'status': 'cleared'})
    
    logs = db.get_translation_history(limit=50)
    return jsonify({'history': logs})


@app.route('/api/practice', methods=['GET', 'POST'])
def practice_endpoint():
    """POST records quiz practice match; GET returns practice history."""
    if request.method == 'POST':
        data = request.get_json() or {}
        target = data.get('target_sign', 'Unknown')
        matched = data.get('matched_sign', 'Unknown')
        score = data.get('score', 0)
        streak = data.get('streak', 0)
        db.record_practice_session(target, matched, score, streak)
        return jsonify({'status': 'recorded'})

    logs = db.get_practice_history(limit=50)
    return jsonify({'practice_history': logs})


@app.route('/api/stats', methods=['GET'])
def get_stats():
    uptime = int(time.time() - session_stats['start_time'])
    return jsonify({
        'uptime_seconds': uptime,
        'total_classifications': session_stats['classifications_count'],
        'unique_gestures_detected': len(session_stats['unique_gestures_seen'])
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

