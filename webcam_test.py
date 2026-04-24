import cv2
import numpy as np
import pickle
import time
from collections import deque
import mediapipe as mp

with open('xgb_model_smote.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

STEPS = {
    0: "Step 1: Palm to palm",
    1: "Step 2: Right over left",
    2: "Step 3: Fingers interlaced",
    3: "Step 4: Backs of fingers",
    4: "Step 5: Rotational thumbs",
    5: "Step 6: Fingertip rubbing",
    6: "Step 7: Wrists"
}

# How many consecutive seconds a step must be detected to count as DONE
HOLD_SECONDS = 3

# --- Colours (BGR) ---
GREEN       = (29, 158, 117)
GREEN_DIM   = (15, 80, 50)
AMBER       = (30, 140, 210)
GRAY        = (100, 100, 100)
WHITE       = (220, 220, 220)
DARK        = (20, 20, 20)
DARK2       = (30, 30, 30)
PANEL_BG    = (25, 25, 25)

mp_drawing        = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands_mod      = mp.solutions.hands

pred_buffer   = deque(maxlen=15)
completed     = set()          # steps that are done
step_start    = {}             # when we started holding current step
start_time    = time.time()

cap = cv2.VideoCapture(0)
cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

PANEL_W = 340
WIN_W   = cam_w + PANEL_W
WIN_H   = cam_h

print("✅ Camera started. Press Q to quit.")

def draw_panel(canvas, current_pred, confidence, hand_count):
    px = cam_w
    pw = PANEL_W
    ph = WIN_H

    # Panel background
    cv2.rectangle(canvas, (px, 0), (px + pw, ph), PANEL_BG, -1)
    cv2.line(canvas, (px, 0), (px, ph), (50, 50, 50), 1)

    # Title bar
    cv2.rectangle(canvas, (px, 0), (px + pw, 48), DARK, -1)
    cv2.putText(canvas, "WHO Hand Hygiene",
                (px + 12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)
    cv2.putText(canvas, "7-Step Protocol",
                (px + 12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.42, GRAY, 1, cv2.LINE_AA)

    # Steps list
    step_h  = 52
    start_y = 58

    for i in range(7):
        y = start_y + i * step_h
        is_done    = i in completed
        is_active  = (i == current_pred) and hand_count == 2

        # Row background
        bg = GREEN_DIM if is_done else ((15, 30, 20) if is_active else (28, 28, 28))
        cv2.rectangle(canvas, (px + 8, y), (px + pw - 8, y + step_h - 4), bg, -1)

        # Active border
        if is_active and not is_done:
            cv2.rectangle(canvas, (px + 8, y), (px + pw - 8, y + step_h - 4), GREEN, 1)

        # Tick circle
        cx = px + 28
        cy = y + (step_h - 4) // 2

        if is_done:
            cv2.circle(canvas, (cx, cy), 11, GREEN, -1)
            cv2.putText(canvas, "OK", (cx - 8, cy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        elif is_active:
            cv2.circle(canvas, (cx, cy), 11, GREEN, 2)
            cv2.circle(canvas, (cx, cy), 5, GREEN, -1)
        else:
            cv2.circle(canvas, (cx, cy), 11, (60, 60, 60), 1)

        # Step name
        name_color = GREEN if is_done else (WHITE if is_active else GRAY)
        cv2.putText(canvas, STEPS[i],
                    (px + 46, y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, name_color, 1, cv2.LINE_AA)

        # Hold progress bar (only for active step)
        if is_active and not is_done:
            held = time.time() - step_start.get(i, time.time())
            pct  = min(held / HOLD_SECONDS, 1.0)
            bar_x1 = px + 46
            bar_x2 = px + pw - 14
            bar_y  = y + 34
            cv2.rectangle(canvas, (bar_x1, bar_y), (bar_x2, bar_y + 5), (50, 50, 50), -1)
            cv2.rectangle(canvas, (bar_x1, bar_y),
                          (bar_x1 + int((bar_x2 - bar_x1) * pct), bar_y + 5), GREEN, -1)
            cv2.putText(canvas, f"Hold {HOLD_SECONDS - int(held)}s",
                        (bar_x2 - 38, bar_y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, GREEN, 1, cv2.LINE_AA)

        elif is_done:
            # Show time taken
            taken = step_start.get(f"done_{i}", 0)
            if taken:
                cv2.putText(canvas, f"{taken:.0f}s",
                            (px + pw - 36, y + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, GREEN, 1, cv2.LINE_AA)

    # Stats bar at bottom
    stats_y = start_y + 7 * step_h + 10
    elapsed = int(time.time() - start_time)
    done_count = len(completed)
    pct_done = int(done_count / 7 * 100)

    cv2.rectangle(canvas, (px + 8, stats_y), (px + pw - 8, stats_y + 56), DARK, -1)

    stats = [
        (f"{done_count}/7", "Steps"),
        (f"{pct_done}%",    "Done"),
        (f"{int(confidence*100)}%", "Conf"),
        (f"{elapsed}s",     "Time"),
    ]
    sw = (pw - 24) // 4
    for j, (val, lbl) in enumerate(stats):
        sx = px + 12 + j * sw
        cv2.putText(canvas, val, (sx, stats_y + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, GREEN, 1, cv2.LINE_AA)
        cv2.putText(canvas, lbl, (sx, stats_y + 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, GRAY, 1, cv2.LINE_AA)

    # All done banner
    if len(completed) == 7:
        banner_y = stats_y + 65
        cv2.rectangle(canvas, (px + 8, banner_y), (px + pw - 8, banner_y + 60), GREEN_DIM, -1)
        cv2.rectangle(canvas, (px + 8, banner_y), (px + pw - 8, banner_y + 60), GREEN, 1)
        cv2.putText(canvas, "All steps complete!",
                    (px + 20, banner_y + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, GREEN, 1, cv2.LINE_AA)
        cv2.putText(canvas, "Hand hygiene done correctly",
                    (px + 20, banner_y + 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, GRAY, 1, cv2.LINE_AA)

    # Hand count indicator
    hc = hand_count if hand_count else 0
    hc_color = GREEN if hc == 2 else (0, 80, 200)
    cv2.putText(canvas, f"Hands: {hc}/2",
                (px + pw - 90, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, hc_color, 1, cv2.LINE_AA)


with mp_hands_mod.Hands(
    model_complexity=1,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = hands.process(rgb)
        rgb.flags.writeable = True

        current_pred = -1
        confidence   = 0.0
        hand_count   = 0

        if results.multi_hand_landmarks and results.multi_handedness:
            left_lm  = np.zeros(63)
            right_lm = np.zeros(63)
            hand_count = len(results.multi_hand_landmarks)

            for hand_lm, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness
            ):
                label  = handedness.classification[0].label
                coords = [c for lm in hand_lm.landmark
                          for c in [lm.x, lm.y, lm.z]]
                if label == 'Left':
                    left_lm  = np.array(coords)
                else:
                    right_lm = np.array(coords)

                mp_drawing.draw_landmarks(
                    frame, hand_lm, mp_hands_mod.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

            if hand_count == 2:
                features        = np.concatenate([left_lm, right_lm]).reshape(1, -1)
                features_scaled = scaler.transform(features)
                proba           = model.predict_proba(features_scaled)[0]
                pred            = int(np.argmax(proba))
                confidence      = float(proba[pred])

                if confidence > 0.55:
                    pred_buffer.append(pred)

                if pred_buffer:
                    current_pred = max(set(pred_buffer), key=pred_buffer.count)

                    # Track hold time per step
                    if current_pred not in completed:
                        if current_pred not in step_start:
                            step_start[current_pred] = time.time()
                        elif time.time() - step_start[current_pred] >= HOLD_SECONDS:
                            completed.add(current_pred)
                            step_start[f"done_{current_pred}"] = (
                                time.time() - start_time
                            )
                    else:
                        # Reset timer if switched to different step
                        for k in list(step_start.keys()):
                            if isinstance(k, int) and k != current_pred:
                                del step_start[k]

        # Camera overlay — current step name at top
        cv2.rectangle(frame, (0, 0), (cam_w, 45), DARK, -1)
        step_label = STEPS.get(current_pred, "Show both hands") if current_pred >= 0 else "Show both hands"
        label_col  = GREEN if current_pred >= 0 and hand_count == 2 else GRAY
        cv2.putText(frame, step_label,
                    (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, label_col, 2, cv2.LINE_AA)

        # Confidence bar on camera
        if confidence > 0:
            bw = int((cam_w - 24) * confidence)
            cv2.rectangle(frame, (12, cam_h - 14), (12 + bw, cam_h - 6), GREEN, -1)
            cv2.rectangle(frame, (12, cam_h - 14), (cam_w - 12, cam_h - 6), (60, 60, 60), 1)

        # Quit hint
        cv2.putText(frame, "Q to quit",
                    (12, cam_h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, GRAY, 1, cv2.LINE_AA)

        # Stitch camera + panel side by side
        canvas = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
        canvas[:, :cam_w] = frame
        draw_panel(canvas, current_pred, confidence, hand_count)

        cv2.imshow("WHO Hand Hygiene Detection", canvas)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("Session complete.")
print(f"Steps completed: {sorted([s+1 for s in completed])}")