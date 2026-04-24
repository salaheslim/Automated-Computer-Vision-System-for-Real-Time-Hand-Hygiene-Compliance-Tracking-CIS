from pyorbbecsdk import Pipeline, Config, OBSensorType
import cv2
import numpy as np

pipeline = Pipeline()
config   = Config()

# Enable colour and depth streams
config.enable_stream(OBSensorType.COLOR_SENSOR)
config.enable_stream(OBSensorType.DEPTH_SENSOR)
pipeline.start(config)

print("✅ Orbbec camera started. Press Q to quit.")

while True:
    frames = pipeline.wait_for_frames(100)
    if not frames:
        continue

    # Get colour frame
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    if color_frame:
        color_data = np.asarray(color_frame.get_data(), dtype=np.uint8)
        color_img  = color_data.reshape(
            color_frame.get_height(),
            color_frame.get_width(), 3)
        cv2.imshow("Colour", color_img)

    if depth_frame:
        depth_data = np.asarray(depth_frame.get_data(), dtype=np.uint16)
        depth_img  = depth_data.reshape(
            depth_frame.get_height(),
            depth_frame.get_width())
        # Normalise for display
        depth_display = cv2.normalize(
            depth_img, None, 0, 255,
            cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_colour = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
        cv2.imshow("Depth", depth_colour)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pipeline.stop()
cv2.destroyAllWindows()