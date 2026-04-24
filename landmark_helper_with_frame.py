import cv2
import mediapipe as mp
import numpy as np
import json
import base64

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    model_complexity=0,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.3
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame   = cv2.flip(frame, 1)
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        left_lm  = [0.0] * 63
        right_lm = [0.0] * 63

        if results.multi_hand_landmarks and results.multi_handedness:
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

                mp_draw.draw_landmarks(
                    frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style())

        # Encode frame as base64 to send via stdout
        _, buf     = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        frame_b64  = base64.b64encode(buf).decode('utf-8')

        data = {
            'left':  left_lm,
            'right': right_lm,
            'frame': frame_b64
        }
        print(json.dumps(data), flush=True)

cap.release()