import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import time

mp_hands  = mp.solutions.hands
mp_draw   = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

STEPS = {
    '1': 'Step 1 — Palm to palm',
    '2': 'Step 2 — Right over left',
    '3': 'Step 3 — Fingers interlaced',
    '4': 'Step 4 — Backs of fingers',
    '5': 'Step 5 — Rotational thumbs',
    '6': 'Step 6 — Fingertip rubbing',
    '7': 'Step 7 — Wrists'
}

os.makedirs('my_data', exist_ok=True)

collected  = {str(i): 0 for i in range(1, 8)}
recording  = False
current    = None
writers    = {}
filehandles = {}

cap = cv2.VideoCapture(0)
print("=" * 50)
print("  WHO Hand Hygiene — Data Collection Tool")
print("=" * 50)
print("  Press 1-7 to record that step")
print("  Press S   to stop current step")
print("  Press Q   to quit and save")
print("=" * 50)

with mp_hands.Hands(
    model_complexity=0,
    max_num_hands=2,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.3
) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame   = cv2.flip(frame, 1)
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        h, w = frame.shape[:2]

        # Draw landmarks
        hand_count = 0
        if results.multi_hand_landmarks:
            hand_count = len(results.multi_hand_landmarks)
            for hand_lm in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style())

        # Save landmarks if recording
        if recording and current and results.multi_hand_landmarks and results.multi_handedness:
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

        if recording and current:
            count = collected[current]
            color = (29, 158, 117) if count >= 200 else (186, 117, 23)
            cv2.putText(frame,
                        f"RECORDING: {STEPS[current]}",
                        (12, 32), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color, 2, cv2.LINE_AA)
            cv2.putText(frame,
                        f"Frames: {count}/300   Press S to stop",
                        (12, 62), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1, cv2.LINE_AA)
            # Progress bar
            pct = min(count / 300, 1.0)
            bw  = int((w - 24) * pct)
            cv2.rectangle(frame, (12, 75), (12 + bw, 88), color, -1)
            cv2.rectangle(frame, (12, 75), (w - 12, 88), (60, 60, 60), 1)
        else:
            cv2.putText(frame,
                        "Press 1-7 to record a step",
                        (12, 42), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (120, 120, 120), 1, cv2.LINE_AA)

        # Hand count indicator
        hc_col = (29, 158, 117) if hand_count == 2 else (60, 60, 200)
        cv2.putText(frame, f"Hands: {hand_count}/2",
                    (w - 130, 32), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, hc_col, 1, cv2.LINE_AA)

        # Bottom summary
        summary = "  ".join([f"S{i}:{collected[str(i)]}" for i in range(1, 8)])
        cv2.rectangle(frame, (0, h - 30), (w, h), (20, 20, 20), -1)
        cv2.putText(frame, summary,
                    (12, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (150, 150, 150), 1, cv2.LINE_AA)

        cv2.imshow("Data Collection — WHO Hand Hygiene", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif chr(key) in '1234567' if key < 128 else False:
            step = chr(key)
            if step not in filehandles:
                f = open(f'my_data/step{step}_me.csv', 'w', newline='')
                filehandles[step] = f
                writers[step]     = csv.writer(f)
            current   = step
            recording = True
            print(f"Recording Step {step}: {STEPS[step]}")

        elif key == ord('s'):
            if current:
                print(f"Stopped Step {current} — {collected[current]} frames saved")
            recording = False
            current   = None

cap.release()
cv2.destroyAllWindows()

for f in filehandles.values():
    f.close()

print("\n" + "="*45)
print("  Collection complete!")
print("="*45)
for i in range(1, 8):
    count = collected[str(i)]
    status = "✓ Good" if count >= 200 else ("✗ Need more" if count < 100 else "~ OK")
    print(f"  Step {i}: {count:4d} frames  {status}")
print("="*45)
print("  Files saved in: C:\\dissertation\\my_data\\")