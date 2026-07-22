import cv2
import numpy as np
import pyttsx3
import mediapipe as mp
import time

from gestures import classify_static, get_gesture_dictionary
from utils import TemporalSmoother


def draw_hud_panel(img, current_text, sentence_text, tts_enabled, fps):
    """Draws a modern HUD banner over the OpenCV webcam video frame."""
    h, w, _ = img.shape

    # Top overlay bar
    cv2.rectangle(img, (0, 0), (w, 60), (13, 19, 33), -1)
    cv2.line(img, (0, 60), (w, 60), (0, 242, 254), 2)

    cv2.putText(img, "AI SIGN LANGUAGE TRANSLATOR — DESKTOP HUD", (15, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    status_str = f"TTS: {'ON' if tts_enabled else 'OFF'} | FPS: {fps}"
    cv2.putText(img, status_str, (w - 230, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 242, 254), 2, cv2.LINE_AA)

    # Bottom sentence bar
    cv2.rectangle(img, (0, h - 70), (w, h), (13, 19, 33), -1)
    cv2.line(img, (0, h - 70), (w, h - 70), (0, 242, 254), 2)

    sign_display = f"Current Sign: {current_text or 'Waiting...'}"
    cv2.putText(img, sign_display, (15, h - 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 135), 2, cv2.LINE_AA)

    sent_display = f"Sentence: {sentence_text or '...'}"
    cv2.putText(img, sent_display, (15, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("❌ Could not open webcam.")

    # Initialize PyTTSx3 engine
    try:
        tts_engine = pyttsx3.init()
        tts_engine.setProperty('rate', 160)
    except Exception as e:
        print(f"⚠️ Warning: Could not initialize pyttsx3 TTS engine: {e}")
        tts_engine = None

    tts_enabled = True

    # Initialize MediaPipe Hands (Max 2 hands)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.65,
        min_tracking_confidence=0.65
    )
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    smoother = TemporalSmoother(maxlen=9, min_confidence=0.5)
    last_spoken = ""
    current_sign_text = ""

    prev_time = time.time()
    fps = 0

    print("=========================================================")
    print("✅ AI Sign Language Translator (Desktop Mode Started)")
    print("Controls: 't' toggle TTS | 'c' clear sentence | 'q' quit")
    print("=========================================================")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Calculate FPS
            curr_time = time.time()
            fps = int(1.0 / (curr_time - prev_time + 1e-6))
            prev_time = curr_time

            # Mirror frame
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            res = hands.process(rgb)
            landmarks_list = []

            if res.multi_hand_landmarks:
                for hand_landmarks in res.multi_hand_landmarks:
                    # Draw landmarks with custom styling
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style()
                    )

                    lm = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
                    landmarks_list.append(lm)

                # Classify landmarks
                label, text = classify_static(landmarks_list)
                if text:
                    smoother.add(text)
                    dom_label, conf = smoother.get()
                    current_sign_text = dom_label or text

                    # Update sentence buffer if stable
                    if smoother.update_sentence(current_sign_text):
                        if tts_enabled and tts_engine and current_sign_text != last_spoken:
                            try:
                                tts_engine.say(current_sign_text)
                                tts_engine.runAndWait()
                                last_spoken = current_sign_text
                            except Exception as err:
                                print(f"TTS error: {err}")
            else:
                current_sign_text = ""

            # Draw Overlay HUD
            sentence_str = smoother.get_sentence_text()
            draw_hud_panel(frame, current_sign_text, sentence_str, tts_enabled, fps)

            cv2.imshow("AI Sign Language Translator — OpenCV Desktop", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('t'):
                tts_enabled = not tts_enabled
                last_spoken = ""
                print(f"TTS toggled: {'ON' if tts_enabled else 'OFF'}")

            elif key == ord('c'):
                smoother.clear()
                last_spoken = ""
                print("Sentence cleared.")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()