"""
=============================================================
Orbbec Gemini 336L — Camera Test & Recording Script
=============================================================
Tests the Orbbec camera and records RGB + depth video.

Controls:
  R = Start/Stop recording
  S = Take a screenshot
  D = Toggle depth overlay on/off
  Q or ESC = Quit

Output saved to: C:\\dissertation\\orbbec_recordings\\

Run in env_xgboost:
  cd C:\\dissertation
  env_xgboost\\Scripts\\activate
  python test_orbbec_record.py
=============================================================
"""

import cv2
import numpy as np
import os
import time
from datetime import datetime

# ── Try importing Orbbec SDK ──────────────────────────────
try:
    from pyorbbecsdk import (
        Pipeline, Config,
        OBSensorType, OBFormat,
        VideoStreamProfile
    )
    ORBBEC_AVAILABLE = True
    print("Orbbec SDK found")
except ImportError:
    ORBBEC_AVAILABLE = False
    print("Orbbec SDK not found — falling back to laptop camera")

# ── Output folder ─────────────────────────────────────────
OUTPUT_DIR = r"C:\dissertation\orbbec_recordings"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def colorise_depth(depth_frame):
    """Convert raw depth to a colourised heatmap for display."""
    if depth_frame is None:
        return None
    depth_array = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
    depth_array = depth_array.reshape(
        depth_frame.get_height(), depth_frame.get_width()
    )
    depth_clipped = np.clip(depth_array, 0, 3000).astype(np.float32)
    depth_norm = (depth_clipped / 3000.0 * 255).astype(np.uint8)
    depth_colour = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
    return depth_colour, depth_array

def run_orbbec():
    """Main loop using Orbbec camera."""
    pipeline = Pipeline()
    config = Config()

    try:
        colour_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        colour_profile = colour_profiles.get_video_stream_profile(1280, 720, OBFormat.MJPG, 10)
        config.enable_stream(colour_profile)
        print("Colour stream: 1280x720 MJPEG @ 10 FPS")
    except Exception as e:
        print(f"Could not set colour profile: {e}")
        config.enable_all_stream()

    try:
        depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_profile = depth_profiles.get_default_video_stream_profile()
        config.enable_stream(depth_profile)
        print("Depth stream enabled")
    except Exception as e:
        print(f"Could not enable depth: {e}")

    pipeline.start(config)
    print("\nOrbbec camera started!")
    print("Controls: R=Record  S=Screenshot  D=Depth overlay  Q=Quit\n")

    recording    = False
    show_depth   = False
    video_writer = None
    frame_count  = 0
    start_time   = time.time()
    rec_start    = None

    while True:
        frames = pipeline.wait_for_frames(100)
        if frames is None:
            continue

        colour_frame = frames.get_color_frame()
        if colour_frame is None:
            continue

        colour_data = np.frombuffer(colour_frame.get_data(), dtype=np.uint8)
        if colour_frame.get_format() == OBFormat.MJPG:
            bgr = cv2.imdecode(colour_data, cv2.IMREAD_COLOR)
        else:
            bgr = colour_data.reshape(colour_frame.get_height(), colour_frame.get_width(), 3)
        if bgr is None:
            continue

        display = bgr.copy()

        depth_frame  = frames.get_depth_frame()
        depth_colour = None
        depth_array  = None
        if depth_frame:
            result = colorise_depth(depth_frame)
            if result:
                depth_colour, depth_array = result

        if show_depth and depth_colour is not None:
            depth_resized = cv2.resize(depth_colour, (display.shape[1], display.shape[0]))
            display = cv2.addWeighted(display, 0.6, depth_resized, 0.4, 0)

        frame_count += 1
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0

        cv2.putText(display, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if depth_array is not None:
            cx, cy = display.shape[1]//2, display.shape[0]//2
            depth_scaled = depth_array.astype(np.float32)
            depth_resized_arr = cv2.resize(depth_scaled, (display.shape[1], display.shape[0]))
            centre_depth = depth_resized_arr[cy, cx]
            cv2.putText(display, f"Centre depth: {centre_depth:.0f}mm", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.drawMarker(display, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)

        status = "RECORDING" if recording else "PREVIEW"
        colour = (0, 0, 255) if recording else (0, 255, 0)
        cv2.putText(display, status, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)

        if recording and rec_start:
            rec_elapsed = time.time() - rec_start
            cv2.putText(display, f"REC {rec_elapsed:.1f}s", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        depth_label = "DEPTH ON" if show_depth else "DEPTH OFF"
        cv2.putText(display, depth_label, (10, display.shape[0]-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(display, "R=Rec  S=Screenshot  D=Depth  Q=Quit",
                    (10, display.shape[0]-45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        if recording and video_writer:
            video_writer.write(bgr)

        cv2.imshow("Orbbec Gemini 336L - Test & Record", display)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):
            break
        elif key == ord('r'):
            if not recording:
                fname = os.path.join(OUTPUT_DIR, f"orbbec_{get_timestamp()}.avi")
                h, w = bgr.shape[:2]
                video_writer = cv2.VideoWriter(fname, cv2.VideoWriter_fourcc(*'XVID'), 10, (w, h))
                recording = True
                rec_start = time.time()
                print(f"Recording started -> {fname}")
            else:
                recording = False
                if video_writer:
                    video_writer.release()
                    video_writer = None
                print(f"Recording stopped. Saved to {OUTPUT_DIR}")
        elif key == ord('s'):
            fname = os.path.join(OUTPUT_DIR, f"screenshot_{get_timestamp()}.png")
            cv2.imwrite(fname, bgr)
            print(f"Screenshot saved -> {fname}")
            if depth_colour is not None:
                dfname = os.path.join(OUTPUT_DIR, f"depth_{get_timestamp()}.png")
                cv2.imwrite(dfname, depth_colour)
                print(f"Depth map saved  -> {dfname}")
        elif key == ord('d'):
            show_depth = not show_depth
            print(f"Depth overlay: {'ON' if show_depth else 'OFF'}")

    if recording and video_writer:
        video_writer.release()
    pipeline.stop()
    cv2.destroyAllWindows()
    print(f"\nDone. Files saved in: {OUTPUT_DIR}")


def run_laptop():
    """Fallback: use laptop camera if Orbbec not available."""
    print("\nUsing laptop camera (Orbbec not detected)")
    print("Controls: R=Record  S=Screenshot  Q=Quit\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No camera found!")
        return

    recording    = False
    video_writer = None
    frame_count  = 0
    start_time   = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0

        display = frame.copy()
        cv2.putText(display, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        status = "RECORDING" if recording else "PREVIEW (Laptop)"
        colour = (0, 0, 255) if recording else (0, 255, 0)
        cv2.putText(display, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
        cv2.putText(display, "R=Rec  S=Screenshot  Q=Quit",
                    (10, display.shape[0]-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        if recording and video_writer:
            video_writer.write(frame)

        cv2.imshow("Camera Test & Record", display)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):
            break
        elif key == ord('r'):
            if not recording:
                fname = os.path.join(OUTPUT_DIR, f"laptop_{get_timestamp()}.avi")
                h, w = frame.shape[:2]
                video_writer = cv2.VideoWriter(fname, cv2.VideoWriter_fourcc(*'XVID'), 30, (w, h))
                recording = True
                print(f"Recording -> {fname}")
            else:
                recording = False
                if video_writer:
                    video_writer.release()
                    video_writer = None
                print(f"Stopped. Saved to {OUTPUT_DIR}")
        elif key == ord('s'):
            fname = os.path.join(OUTPUT_DIR, f"screenshot_{get_timestamp()}.png")
            cv2.imwrite(fname, frame)
            print(f"Screenshot -> {fname}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone. Files in: {OUTPUT_DIR}")


if __name__ == "__main__":
    print("=" * 50)
    print("  Orbbec Gemini 336L - Camera Test & Recorder")
    print("  Cardiff Met Dissertation 2026")
    print("=" * 50)

    if ORBBEC_AVAILABLE:
        try:
            run_orbbec()
        except Exception as e:
            print(f"\nOrbbec error: {e}")
            print("Falling back to laptop camera...")
            run_laptop()
    else:
        run_laptop()
