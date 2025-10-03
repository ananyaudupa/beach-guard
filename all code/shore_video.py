import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image
import os

# ========================
# CONFIG
# ========================
IMG_SIZE = 640
VIDEO_PATH = "inputs/data.mp4"   # <-- input video
OUTPUT_FOLDER = "video_outputs"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
FINAL_OUTPUT_PATH = os.path.join(OUTPUT_FOLDER, "final_shoreline_overlay.mp4")

# ========================
# Load ONNX model
# ========================
session = ort.InferenceSession("unet_beach.onnx")
input_name = session.get_inputs()[0].name

# ========================
# Preprocess function
# ========================
def preprocess(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    img_array = image.astype(np.float32) / 255.0
    img_array = np.transpose(img_array, (2,0,1))
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, image  # return both tensor & resized original (RGB)

# ========================
# Video setup
# ========================
cap = cv2.VideoCapture(VIDEO_PATH)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(FINAL_OUTPUT_PATH, fourcc, fps, (width, height))

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    print(f"Processing frame {frame_count}...")

    # preprocess
    input_tensor, resized_original = preprocess(frame)

    # inference
    output = session.run(None, {input_name: input_tensor})
    output_tensor = output[0]  # [1, NUM_CLASSES, H, W]
    mask = np.argmax(output_tensor, axis=1).squeeze()

    # find water mask
    h, w = mask.shape
    water = (mask == 2).astype(np.uint8) * 255
    contours, _ = cv2.findContours(water, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # overlay shoreline
    overlay = cv2.cvtColor(resized_original, cv2.COLOR_RGB2BGR)
    for cnt in contours:
        for pt in cnt:
            x, y = pt[0]
            if y > h - 5 or x < 5 or x > w - 5:  # ignore bottom & edges
                continue
            cv2.circle(overlay, (x,y), 2, (255,255,255), -1)  # shoreline (thick white)

    # resize overlay back to original frame size
    overlay = cv2.resize(overlay, (width, height), interpolation=cv2.INTER_LINEAR)

    # save frame
    out.write(overlay)

cap.release()
out.release()
print(f"✅ Final shoreline video saved at {FINAL_OUTPUT_PATH}")
