import cv2
import numpy as np
import pickle
import time
from collections import deque
import mediapipe as mp
from pyorbbecsdk import Pipeline, Config, OBSensorType

with open('xgb_model_orbbec.pkl', 'rb') as f:
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

HOLD_SECONDS   = 2.5
VOTE_THRESHOLD = 0.55

GREEN     = (29, 158, 117)
GREEN_DIM = (10, 60, 35)
AMBER     = (30, 150, 220)
GRAY      = (110, 110, 110)
WHITE     = (220, 220, 220)
DARK      = (18, 18, 18)
PANEL_BG  = (22, 22, 22)
RED       = (60, 60, 220)
PANEL_W   = 340

mp_drawing        = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands_mod      = mp.solutions.hands

pred_buffer    = deque(maxlen=5)
session_start  = time.time()
step_start     = {}
step_done_at   = {}
step_scan_time = {}
step_votes     = {}
step_total     = {}
completed      = set()
current_step   = 0

def predict(left_lm, right_lm):
    features = np.concatenate([left_lm, right_lm]).reshape(1, -1)
    scaled   = scaler.transform(features)
    proba    = model.predict_proba(scaled)[0]
    pred     = int(np.argmax(proba))
    return pred, float(proba[pred])

def fmt_time(seconds):
    return f"{seconds:.1f}s" if seconds else "--"

def get_frame():
    frames = pipeline.wait_for_frames(200)
    if not frames:
        return None, None
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()
    frame = None
    depth_map = None
    if color_frame:
        data  = np.asarray(color_frame.get_data(), dtype=np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if frame is not None:
            frame = cv2.flip(frame, 1)
    if depth_frame:
        try:
            depth_data = np.asarray(
                depth_frame.get_data(), dtype=np.uint16)
            depth_map = depth_data.reshape(
                depth_frame.get_height(),
                depth_frame.get_width())
        except:
            depth_map = None
    return frame, depth_map

def draw_panel(canvas, smoothed_pred, confidence, hand_count, cam_w):
    px    = cam_w
    WIN_H = canvas.shape[0]

    cv2.rectangle(canvas, (px, 0),
                  (px + PANEL_W, WIN_H), PANEL_BG, -1)
    cv2.line(canvas, (px, 0), (px, WIN_H), (45, 45, 45), 1)

    # Title bar
    cv2.rectangle(canvas, (px, 0),
                  (px + PANEL_W, 55), DARK, -1)
    cv2.putText(canvas, "WHO Hand Hygiene",
                (px + 10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, GREEN, 1, cv2.LINE_AA)
    cv2.putText(canvas, "Orbbec Gemini 336L — Depth Camera",
                (px + 10, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                (100, 200, 255), 1, cv2.LINE_AA)

    # Hand count
    hc_col = GREEN if hand_count >= 1 else RED
    cv2.putText(canvas, f"Hands:{hand_count}/2",
                (px + PANEL_W - 90, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, hc_col, 1, cv2.LINE_AA)

    # Column headers
    cv2.rectangle(canvas, (px + 7, 58),
                  (px + PANEL_W - 7, 74), (35, 35, 35), -1)
    cv2.putText(canvas, "Step",
                (px + 14, 69),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, GRAY, 1, cv2.LINE_AA)
    cv2.putText(canvas, "Status",
                (px + 155, 69),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, GRAY, 1, cv2.LINE_AA)
    cv2.putText(canvas, "Time",
                (px + 268, 69),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, GRAY, 1, cv2.LINE_AA)

    step_h  = 50
    start_y = 78

    for i in range(7):
        y         = start_y + i * step_h
        is_done   = i in completed
        is_active = (smoothed_pred == i) and (i == current_step)

        if is_done:
            bg = GREEN_DIM
        elif is_active:
            bg = (10, 35, 20)
        else:
            bg = (26, 26, 26)

        cv2.rectangle(canvas, (px + 7, y),
                      (px + PANEL_W - 7, y + step_h - 3), bg, -1)

        if is_active and not is_done:
            cv2.rectangle(canvas, (px + 7, y),
                          (px + PANEL_W - 7, y + step_h - 3), GREEN, 1)

        # Circle indicator
        cx, cy = px + 22, y + (step_h - 3) // 2
        if is_done:
            cv2.circle(canvas, (cx, cy), 10, GREEN, -1)
            cv2.putText(canvas, "OK", (cx - 8, cy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                        (255, 255, 255), 1, cv2.LINE_AA)
        elif is_active:
            cv2.circle(canvas, (cx, cy), 10, GREEN, 2)
            cv2.circle(canvas, (cx, cy), 4,  GREEN, -1)
        elif i == current_step:
            cv2.circle(canvas, (cx, cy), 10, (80, 120, 80), 1)
        else:
            cv2.circle(canvas, (cx, cy), 10, (50, 50, 50), 1)

        # Step name
        nc = GREEN if is_done else \
             (WHITE if is_active or i == current_step else (65, 65, 65))
        cv2.putText(canvas, STEPS[i],
                    (px + 36, y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, nc, 1, cv2.LINE_AA)

        if is_active and not is_done:
            held     = time.time() - step_start.get(i, time.time())
            votes    = step_votes.get(i, 0)
            total    = max(step_total.get(i, 1), 1)
            pct      = min(held / HOLD_SECONDS, 1.0)
            vote_pct = votes / total

            # Progress bar
            bx1, bx2, by = px + 36, px + 225, y + 32
            cv2.rectangle(canvas, (bx1, by),
                          (bx2, by + 5), (40, 40, 40), -1)
            bar_col = GREEN if vote_pct >= VOTE_THRESHOLD else AMBER
            cv2.rectangle(canvas, (bx1, by),
                          (bx1 + int((bx2 - bx1) * pct), by + 5),
                          bar_col, -1)

            # Big countdown timer
            remain = max(0, HOLD_SECONDS - held)
            cv2.putText(canvas, f"{remain:.1f}s",
                        (px + PANEL_W - 58, y + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        GREEN, 2, cv2.LINE_AA)

            status     = "Holding..."
            status_col = bar_col

        elif is_done:
            status     = "Done"
            status_col = GREEN
            t = step_scan_time.get(i)
            if t:
                cv2.putText(canvas, f"{t:.1f}s",
                            (px + PANEL_W - 50, y + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                            GREEN, 1, cv2.LINE_AA)
        elif i == current_step:
            status     = "< Do this"
            status_col = (80, 120, 80)
        else:
            status     = "Waiting"
            status_col = (55, 55, 55)

        cv2.putText(canvas, status,
                    (px + 152, y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                    status_col, 1, cv2.LINE_AA)

    # Summary stats
    sy = start_y + 7 * step_h + 8
    cv2.rectangle(canvas, (px + 7, sy),
                  (px + PANEL_W - 7, sy + 75), DARK, -1)

    elapsed    = time.time() - session_start
    done_count = len(completed)
    times      = list(step_scan_time.values())
    avg_t      = sum(times) / len(times) if times else 0

    cv2.putText(canvas, "Session summary",
                (px + 14, sy + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, GRAY, 1, cv2.LINE_AA)

    summary = [
        (f"{done_count}/7",           "Steps"),
        (f"{int(done_count/7*100)}%", "Done"),
        (fmt_time(avg_t),             "Avg"),
        (fmt_time(elapsed),           "Total"),
    ]
    sw = (PANEL_W - 20) // 4
    for j, (val, lbl) in enumerate(summary):
        sx = px + 10 + j * sw
        cv2.putText(canvas, val, (sx, sy + 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                    GREEN, 1, cv2.LINE_AA)
        cv2.putText(canvas, lbl, (sx, sy + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                    GRAY, 1, cv2.LINE_AA)

    # Completion banner
    if done_count == 7:
        by = sy + 82
        cv2.rectangle(canvas, (px + 7, by),
                      (px + PANEL_W - 7, by + 52), GREEN_DIM, -1)
        cv2.rectangle(canvas, (px + 7, by),
                      (px + PANEL_W - 7, by + 52), GREEN, 1)
        cv2.putText(canvas, "All 7 steps complete!",
                    (px + 14, by + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, GREEN, 1, cv2.LINE_AA)
        cv2.putText(canvas,
                    f"Total:{fmt_time(elapsed)} Avg:{fmt_time(avg_t)}/step",
                    (px + 14, by + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, GRAY, 1, cv2.LINE_AA)


# ── Start Orbbec ──────────────────────────────────────────────
print("Starting Orbbec Gemini 336L...")
pipeline = Pipeline()
config   = Config()

try:
    profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    profile  = profiles.get_video_stream_profile(0)
    config.enable_stream(profile)
    print(f"✅ Colour: {profile.get_width()}x{profile.get_height()} "
          f"@ {profile.get_fps()}fps")
except Exception as e:
    config.enable_stream(OBSensorType.COLOR_SENSOR)
    print(f"✅ Colour stream enabled")

has_depth = False
try:
    d_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    d_profile  = d_profiles.get_video_stream_profile(0)
    config.enable_stream(d_profile)
    has_depth = True
    print(f"✅ Depth: {d_profile.get_width()}x{d_profile.get_height()} "
          f"@ {d_profile.get_fps()}fps")
except Exception as e:
    print(f"Depth not enabled: {e}")

pipeline.start(config)
print("✅ Camera ready! Press Q to quit, R to reset.")

# ── Main loop ─────────────────────────────────────────────────
with mp_hands_mod.Hands(
    model_complexity=0,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.3
) as hands:

    while True:
        frame, depth_map = get_frame()
        if frame is None:
            continue

        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = hands.process(rgb)
        rgb.flags.writeable = True

        smoothed_pred = -1
        confidence    = 0.0
        hand_count    = 0

        if results.multi_hand_landmarks and results.multi_handedness:
            left_lm    = np.zeros(63)
            right_lm   = np.zeros(63)
            hand_count = len(results.multi_hand_landmarks)

            for hand_lm, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness
            ):
                label  = handedness.classification[0].label
                coords = []

                for lm in hand_lm.landmark:
                    lm_x = lm.x
                    lm_y = lm.y
                    if depth_map is not None:
                        px_x   = min(int(lm.x * w), w - 1)
                        px_y   = min(int(lm.y * h), h - 1)
                        real_z = float(depth_map[px_y, px_x]) / 1000.0
                        lm_z   = real_z if real_z > 0 else lm.z
                    else:
                        lm_z = lm.z
                    coords.extend([lm_x, lm_y, lm_z])

                if label == 'Left':
                    left_lm  = np.array(coords)
                else:
                    right_lm = np.array(coords)

                mp_drawing.draw_landmarks(
                    frame, hand_lm,
                    mp_hands_mod.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            raw_pred, confidence = predict(left_lm, right_lm)
            pred_buffer.append(raw_pred)
            smoothed_pred = max(set(pred_buffer), key=pred_buffer.count)

            # Voting window detection
            if current_step not in completed:
                if smoothed_pred == current_step:
                    if current_step not in step_start:
                        step_start[current_step] = time.time()
                        step_votes[current_step] = 0
                        step_total[current_step] = 0
                    step_votes[current_step] += 1
                    step_total[current_step] += 1
                    elapsed_hold = time.time() - step_start[current_step]
                    if elapsed_hold >= HOLD_SECONDS:
                        vote_ratio = (step_votes[current_step] /
                                      max(step_total[current_step], 1))
                        if vote_ratio >= VOTE_THRESHOLD:
                            step_scan_time[current_step] = elapsed_hold
                            step_done_at[current_step]   = (
                                time.time() - session_start)
                            completed.add(current_step)
                            pred_buffer.clear()
                            step_start.pop(current_step, None)
                            step_votes.pop(current_step, None)
                            step_total.pop(current_step, None)
                            if current_step < 6:
                                current_step += 1
                        else:
                            step_start.pop(current_step, None)
                            step_votes.pop(current_step, None)
                            step_total.pop(current_step, None)
                else:
                    if current_step in step_total:
                        step_total[current_step] += 1

        # Camera overlay
        cv2.rectangle(frame, (0, 0), (w, 52), DARK, -1)
        cv2.putText(frame, "Orbbec Gemini 336L  |  Depth Camera",
                    (12, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (100, 200, 255), 1, cv2.LINE_AA)

        if len(completed) == 7:
            label_text = "All 7 steps complete!"
            label_col  = GREEN
        elif smoothed_pred == current_step and hand_count >= 1:
            label_text = STEPS.get(current_step, "")
            label_col  = GREEN
        else:
            label_text = f"Do: {STEPS.get(current_step, '')}"
            label_col  = AMBER

        cv2.putText(frame, label_text,
                    (12, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                    label_col, 2, cv2.LINE_AA)

        # Session timer top right
        elapsed = int(time.time() - session_start)
        cv2.putText(frame, f"{elapsed}s",
                    (w - 55, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    GRAY, 1, cv2.LINE_AA)

        # Confidence bar
        if confidence > 0:
            bw = int((w - 24) * confidence)
            cv2.rectangle(frame, (12, h-10),
                          (12 + bw, h - 4), GREEN, -1)
            cv2.rectangle(frame, (12, h-10),
                          (w - 12, h - 4), (55, 55, 55), 1)

        cv2.putText(frame, "Q=quit  R=reset",
                    (12, h - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                    GRAY, 1, cv2.LINE_AA)

        # Depth minimap bottom-right corner
        if depth_map is not None:
            try:
                small    = cv2.resize(depth_map, (200, 150))
                norm     = cv2.normalize(
                    small, None, 0, 255,
                    cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                coloured = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
                frame[h-155:h-5, w-205:w-5] = coloured
                cv2.putText(frame, "Depth view",
                            (w - 200, h - 160),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.35, GRAY, 1, cv2.LINE_AA)
            except:
                pass

        # Build canvas
        canvas = np.zeros((h, w + PANEL_W, 3), dtype=np.uint8)
        canvas[:, :w] = frame
        draw_panel(canvas, smoothed_pred, confidence, hand_count, w)

        cv2.imshow("WHO Hand Hygiene — Orbbec Gemini 336L", canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            completed.clear()
            step_start.clear()
            step_done_at.clear()
            step_scan_time.clear()
            step_votes.clear()
            step_total.clear()
            pred_buffer.clear()
            current_step  = 0
            session_start = time.time()
            print("Reset!")

pipeline.stop()
cv2.destroyAllWindows()

print("\n" + "="*47)
print("  WHO Hand Hygiene — Orbbec Session Results")
print("="*47)
print(f"  {'Step':<30} {'Scan Time':>10}")
print("-"*47)
for i in range(7):
    t    = step_scan_time.get(i)
    done = "✓" if i in completed else " "
    print(f"  {done} {STEPS[i]:<29} "
          f"{f'{t:.1f}s' if t else 'not done':>10}")
print("-"*47)
times = list(step_scan_time.values())
if times:
    print(f"  {'Average scan time':<30} {sum(times)/len(times):.1f}s")
    print(f"  {'Steps completed':<30} {len(completed)}/7")
print("="*47)