import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import time
from pyorbbecsdk import Pipeline, Config, OBSensorType

STEPS = {
    '1': 'Step 1 — Palm to palm',
    '2': 'Step 2 — Right over left',
    '3': 'Step 3 — Fingers interlaced',
    '4': 'Step 4 — Backs of fingers',
    '5': 'Step 5 — Rotational thumbs',
    '6': 'Step 6 — Fingertip rubbing',
    '7': 'Step 7 — Wrists'
}

os.makedirs('my_data_orbbec', exist_ok=True)

collected   = {str(i): 0 for i in range(1, 8)}
recording   = False
current     = None
writers     = {}
filehandles = {}

# Start Orbbec
print("Starting Orbbec camera...")
pipeline = Pipeline()
config   = Config()
profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
profile  = profiles.get_video_stream_profile(0)
config.enable_stream(profile)
pipeline.start(config)
print("✅ Orbbec ready!")

def get_frame():
    frames = pipeline.wait_for_frames(200)
    if not frames:
        return None
    color_frame = frames.get_color_frame()
    if not color_frame:
        return None
    data  = np.asarray(color_frame.get_data(), dtype=np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is not None:
        frame = cv2.flip(frame, 1)
    return frame

mp_hands  = mp.solutions.hands
mp_draw   = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

print("="*50)
print("  WHO Hand Hygiene — Orbbec Data Collection")
print("="*50)
print("  Press 1-7 to record that step")
print("  Press S   to stop current step")
print("  Press Q   to quit and save")
print("="*50)

with mp_hands.Hands(
    model_complexity=0,
    max_num_hands=2,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.3
) as hands:

    while True:
        frame = get_frame()
        if frame is None:
            continue

        h, w = frame.shape[:2]
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        hand_count = 0
        if results.multi_hand_landmarks:
            hand_count = len(results.multi_hand_landmarks)
            for hand_lm in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, hand_lm,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style())

        # Save landmarks if recording
        if recording and current and \
           results.multi_hand_landmarks and results.multi_handedness:
            left_lm  = [0.0] * 63
            right_lm = [0.0] * 63

            for hand_lm, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness
            ):
                label  = handedness.classification[0].label
                coords = [c for lm in hand_lm.landmark
                          for c in [lm.x, lm.y, lm.z]]
                if label == 'Left':
                    left_lm  = coords
                else:
                    right_lm = coords

            writers[current].writerow(left_lm + right_lm)
            collected[current] += 1

        # UI overlay
        cv2.rectangle(frame, (0, 0), (w, 100), (20, 20, 20), -1)
        cv2.putText(frame, "Orbbec Gemini 336L — Data Collection",
                    (12, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (100, 200, 255), 1, cv2.LINE_AA)

        if recording and current:
            count = collected[current]
            color = (29, 158, 117) if count >= 200 else (186, 117, 23)
            cv2.putText(frame,
                        f"RECORDING: {STEPS[current]}",
                        (12, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        color, 2, cv2.LINE_AA)
            cv2.putText(frame,
                        f"Frames: {count}/300   Press S to stop",
                        (12, 72),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        color, 1, cv2.LINE_AA)
            pct = min(count / 300, 1.0)
            bw  = int((w - 24) * pct)
            cv2.rectangle(frame, (12, 80), (12 + bw, 92), color, -1)
            cv2.rectangle(frame, (12, 80), (w - 12, 92), (60,60,60), 1)
        else:
            cv2.putText(frame,
                        "Press 1-7 to record a step",
                        (12, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (120, 120, 120), 1, cv2.LINE_AA)

        hc_col = (29, 158, 117) if hand_count == 2 else (60, 60, 200)
        cv2.putText(frame, f"Hands: {hand_count}/2",
                    (w - 130, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    hc_col, 1, cv2.LINE_AA)

        summary = "  ".join(
            [f"S{i}:{collected[str(i)]}" for i in range(1, 8)])
        cv2.rectangle(frame, (0, h - 30), (w, h), (20, 20, 20), -1)
        cv2.putText(frame, summary,
                    (12, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (150, 150, 150), 1, cv2.LINE_AA)

        cv2.imshow("Orbbec Data Collection — WHO Hand Hygiene", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key in range(ord('1'), ord('8')):
            step = chr(key)
            if step not in filehandles:
                f = open(f'my_data_orbbec/step{step}_orbbec.csv',
                         'w', newline='')
                filehandles[step] = f
                writers[step]     = csv.writer(f)
            current   = step
            recording = True
            print(f"Recording Step {step}: {STEPS[step]}")
        elif key == ord('s'):
            if current:
                print(f"Stopped Step {current} — "
                      f"{collected[current]} frames saved")
            recording = False
            current   = None

pipeline.stop()
cv2.destroyAllWindows()

for f in filehandles.values():
    f.close()

print("\n" + "="*45)
print("  Collection complete — Orbbec data")
print("="*45)
for i in range(1, 8):
    count  = collected[str(i)]
    status = "✓ Good" if count >= 200 else \
             ("✗ Need more" if count < 100 else "~ OK")
    print(f"  Step {i}: {count:4d} frames  {status}")
print("="*45)
print("  Files saved in: C:\\dissertation\\my_data_orbbec\\")