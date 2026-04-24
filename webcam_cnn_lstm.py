import cv2
import numpy as np
import pickle
import subprocess
import json
import base64
from collections import deque
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

print("Loading CNN+LSTM model...")
model = tf.keras.models.load_model('cnn_lstm_model.h5')
print("✅ CNN+LSTM ready!")

SEQ_LEN    = 30
MODEL_NAME = "CNN + LSTM"

STEPS = {
    0: "Step 1: Palm to palm",
    1: "Step 2: Right over left",
    2: "Step 3: Fingers interlaced",
    3: "Step 4: Backs of fingers",
    4: "Step 5: Rotational thumbs",
    5: "Step 6: Fingertip rubbing",
    6: "Step 7: Wrists"
}

COLOR = (180, 120, 220)
GRAY  = (120, 120, 120)
DARK  = (20, 20, 20)

frame_buffer = deque(maxlen=SEQ_LEN)
pred_buffer  = deque(maxlen=5)
step_text    = "Buffering frames..."
confidence   = 0.0

def predict_sequence():
    if len(frame_buffer) < SEQ_LEN:
        return -1, 0.0
    seq   = np.array(frame_buffer, dtype=np.float32).reshape(1, SEQ_LEN, 126)
    proba = model.predict(seq, verbose=0)[0]
    pred  = int(np.argmax(proba))
    return pred, float(proba[pred])

print("Starting landmark helper...")
helper = subprocess.Popen(
    [r'env_xgboost\Scripts\python.exe', 'landmark_helper_with_frame.py'],
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
    text=True,
    bufsize=1
)
print("✅ Helper running!")
print("✅ Camera window opening... Press Q on the window to quit.")

# Create window first
cv2.namedWindow(f"WHO Detection — {MODEL_NAME}", cv2.WINDOW_NORMAL)

try:
    while True:
        # Read one line from helper
        line = helper.stdout.readline()
        if not line:
            print("Helper stopped sending data")
            break

        line = line.strip()
        if not line:
            continue

        try:
            data = json.loads(line)

            # Decode frame
            frame_bytes = base64.b64decode(data['frame'])
            frame_arr   = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame       = cv2.imdecode(frame_arr, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            # Process landmarks
            left_lm  = np.array(data['left'],  dtype=np.float32)
            right_lm = np.array(data['right'], dtype=np.float32)

            features = scaler.transform(
                np.concatenate([left_lm, right_lm]).reshape(1, -1))
            frame_buffer.append(features.flatten())

            # Predict every frame
            pred, confidence = predict_sequence()
            if pred >= 0:
                pred_buffer.append(pred)
            if pred_buffer:
                smoothed  = max(set(pred_buffer), key=pred_buffer.count)
                step_text = STEPS.get(smoothed, "Unknown")

        except (json.JSONDecodeError, KeyError, ValueError):
            continue

        # Draw UI
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 95), DARK, -1)

        cv2.putText(frame, MODEL_NAME,
                    (15, 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, COLOR, 1, cv2.LINE_AA)

        cv2.putText(frame, step_text,
                    (15, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85, COLOR, 2, cv2.LINE_AA)

        if confidence > 0:
            bw = int((w - 30) * confidence)
            cv2.rectangle(frame, (15, 62), (15 + bw, 75), COLOR, -1)
            cv2.rectangle(frame, (15, 62), (w - 15, 75), (80, 80, 80), 1)
            cv2.putText(frame, f"{confidence*100:.0f}%",
                        (w - 55, 73),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (200, 200, 200), 1, cv2.LINE_AA)

        buf_pct = len(frame_buffer) / SEQ_LEN
        bw2     = int((w - 30) * buf_pct)
        cv2.rectangle(frame, (15, 80), (15 + bw2, 88), (100, 60, 160), -1)
        cv2.rectangle(frame, (15, 80), (w - 15,   88), (50,  50,  50), 1)

        if len(frame_buffer) < SEQ_LEN:
            label = f"Buffering {len(frame_buffer)}/{SEQ_LEN}..."
        else:
            label = "Press Q on window to quit"

        cv2.putText(frame, label,
                    (15, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, GRAY, 1, cv2.LINE_AA)

        # Show frame
        cv2.imshow(f"WHO Detection — {MODEL_NAME}", frame)

        # Check for Q key — must be pressed on the CV2 window
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # Q or ESC
            break

finally:
    helper.terminate()
    cv2.destroyAllWindows()
    print("Camera closed.")