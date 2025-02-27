import cv2
import time
import ml_model


def capture_and_detect() :
    """Capture image and run pest detection model."""
    cam = cv2.VideoCapture(0)
    time.sleep(2)
    ret, frame = cam.read()

    if ret :
        img_path = f"data/images/live.jpg"
        cv2.imwrite(img_path, frame)
        print("✅ Image Captured!")

        # Run pest detection
        result = ml_model.predict_pest(img_path)
        print(f"🚨 Detection Result: {result}")

    cam.release()


capture_and_detect()

# python camera.py
# ✔️ Captures an image using Raspberry Pi camera.
# ✔️ Runs ML model to detect pests or diseases.
# ✔️ Prints results in real-time.
#
