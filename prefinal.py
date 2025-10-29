"""
Real-time Beach Safety (video) — YOLOv8 + ONNX UNet segmentation + pygame audio
Features:
 - Live video processing and display
 - Counts SAFE / MODERATE / DANGER per frame
 - Plays only one alert at a time (DANGER priority)
 - Stops alert for a zone after 100 frames of no detections (~3s, adjustable)
 - Saves processed video to disk
"""

import os
import cv2
import math
import time
import warnings
import numpy as np
import torch
import onnxruntime as ort
import pygame
from ultralytics import YOLO

warnings.filterwarnings("ignore", category=FutureWarning)

# ========================
# CONFIG (update these)
# ========================
IMG_SIZE = 640  # for ONNX preprocess
SAFE_DISTANCE_PX = 100  # pixel threshold for danger
HALT_FRAMES = 100  # frames of no detection to stop alarm (~3s at 30fps)
VIDEO_SOURCE = r"C:\Users\THANUSH SHETTY\Desktop\beach\vedio_beach.mp4"  # or 0 for webcam
OUTPUT_VIDEO_PATH = r"C:\Users\THANUSH SHETTY\Desktop\beach\vedio_beach_out.mp4"

YOLO_WEIGHTS = r"C:\Users\THANUSH SHETTY\Desktop\beach\yolov8m.pt"
ONNX_MODEL_PATH = r"C:\Users\THANUSH SHETTY\Desktop\beach\unet_beach.onnx"

MODERATE_SOUND = r"C:\Users\THANUSH SHETTY\Desktop\beach\moderate_alert.mp3"
DANGER_SOUND = r"C:\Users\THANUSH SHETTY\Desktop\beach\danger_alert.mp3"

# Zones distance thresholds (you can tune)
SAFE_THRESHOLD = SAFE_DISTANCE_PX / 2
MODERATE_THRESHOLD = SAFE_DISTANCE_PX

# ========================
# Basic checks
# ========================
if not os.path.exists(YOLO_WEIGHTS):
    raise FileNotFoundError(f"YOLO weights not found: {YOLO_WEIGHTS}")
if not os.path.exists(ONNX_MODEL_PATH):
    raise FileNotFoundError(f"ONNX segmentation model not found: {ONNX_MODEL_PATH}")
if not os.path.exists(MODERATE_SOUND):
    print("⚠ moderate sound not found — moderate alert will be silent.")
if not os.path.exists(DANGER_SOUND):
    print("⚠ danger sound not found — danger alert will be silent.")

# ========================
# Initialize audio (pygame)
# ========================
pygame.init()
try:
    pygame.mixer.init()
except Exception as e:
    print(f"⚠ pygame.mixer.init() failed: {e}")

def play_alert(sound_path):
    """Play a sound using pygame (non-blocking). Only one sound at a time."""
    try:
        if not os.path.exists(sound_path):
            return
        # stop any current music and play new
        pygame.mixer.music.stop()
        pygame.mixer.music.unload() if hasattr(pygame.mixer.music, "unload") else None
        pygame.mixer.music.load(sound_path)
        pygame.mixer.music.play(-1)  # loop until explicitly stopped
        # we loop so we can stop after HALT_FRAMES
    except Exception as e:
        print(f"⚠ Could not play sound '{sound_path}': {e}")

def stop_alert():
    try:
        pygame.mixer.music.stop()
        # some pygame versions need unloading
        if hasattr(pygame.mixer.music, "unload"):
            pygame.mixer.music.unload()
    except Exception:
        pass

# ========================
# Load Models
# ========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

yolo_model = YOLO(YOLO_WEIGHTS).to(device)
session = ort.InferenceSession(ONNX_MODEL_PATH, providers=["CPUExecutionProvider"])
onnx_input_name = session.get_inputs()[0].name

# ========================
# Video source / writer
# ========================
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    raise FileNotFoundError(f"Video source not found or cannot be opened: {VIDEO_SOURCE}")

fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_w, frame_h))

# ========================
# Helpers
# ========================
def preprocess_for_onnx(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)
    return img

def find_shoreline_points(mask):
    """Return mapped shoreline points (in original frame coords) from water mask (mask==2)."""
    water_mask = (mask == 2).astype(np.uint8) * 255
    contours, _ = cv2.findContours(water_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts = []
    for cnt in contours:
        for pt in cnt:
            x, y = pt[0]
            # filter border noise
            if 5 < x < IMG_SIZE - 5 and y < IMG_SIZE - 5:
                x_m = int(x * frame_w / IMG_SIZE)
                y_m = int(y * frame_h / IMG_SIZE)
                pts.append((x_m, y_m))
    return pts

# ========================
# State for alerts
# ========================
frame_count = 0
last_danger_frame = -9999
last_moderate_frame = -9999
current_alert = None  # "danger", "moderate", or None

print("✅ Starting real-time processing. Press 'q' in the display window to quit.")

# ========================
# Main loop
# ========================
while True:
    ret, frame = cap.read()
    if not ret:
        print("🎬 End of video / stream.")
        break
    frame_count += 1
    overlay = frame.copy()

    # 1) YOLO detection (people only)
    results = yolo_model.predict(frame, device=device, imgsz=IMG_SIZE, classes=[0], conf=0.25, iou=0.3, verbose=False)
    detections = []
    if len(results) > 0 and len(results[0].boxes) > 0:
        for box in results[0].boxes:
            # use box.xyxy (tensor) -> numpy
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)
            conf = float(box.conf[0].cpu().numpy())
            detections.append((x1, y1, x2, y2, conf))

    # 2) ONNX segmentation
    in_tensor = preprocess_for_onnx(frame)
    out = session.run(None, {onnx_input_name: in_tensor})
    mask = np.argmax(out[0], axis=1).squeeze()  # expected shape (H,W)

    # 3) Shoreline extraction
    shoreline_points = find_shoreline_points(mask)
    # draw some shoreline dots (optional)
    for p in shoreline_points[::max(1, len(shoreline_points)//300)]:  # sample for speed
        cv2.circle(overlay, p, 2, (255, 0, 0), -1)

    # 4) For each detection compute distance to shoreline & categorize
    safe_count = 0
    moderate_count = 0
    danger_count = 0

    for (x1, y1, x2, y2, conf) in detections:
        person_feet = ((x1 + x2) // 2, y2)
        # map feet to mask coords
        feet_x = int(person_feet[0] * IMG_SIZE / frame_w)
        feet_y = int(person_feet[1] * IMG_SIZE / frame_h)
        if not (0 <= feet_x < IMG_SIZE and 0 <= feet_y < IMG_SIZE):
            continue

        region_class = int(mask[feet_y, feet_x])  # 0=bg,1=sky,2=water,3=sand etc.
        # Only care people near/inside water or sand (beach area)
        if region_class not in [2, 3]:
            # draw faint box for non-beach people
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (160, 160, 160), 1)
            continue

        # if shoreline empty, classify as moderate (conservative)
        if not shoreline_points:
            color = (0, 255, 255)
            text = "MODERATE (?)"
            moderate_count += 1
        else:
            # nearest shoreline
            nearest = min(shoreline_points, key=lambda p: math.hypot(p[0]-person_feet[0], p[1]-person_feet[1]))
            min_dist = math.hypot(nearest[0]-person_feet[0], nearest[1]-person_feet[1])

            # draw line to shoreline
            cv2.line(overlay, person_feet, nearest, (255, 255, 0), 1)

            if min_dist < SAFE_THRESHOLD:
                color = (0, 255, 0)
                text = f"SAFE ({int(min_dist)}px)"
                safe_count += 1
            elif SAFE_THRESHOLD <= min_dist < MODERATE_THRESHOLD:
                color = (0, 255, 255)
                text = f"MODERATE ({int(min_dist)}px)"
                moderate_count += 1
            else:
                color = (0, 0, 255)
                text = f"DANGER ({int(min_dist)}px)"
                danger_count += 1
                # update last seen danger frame
                last_danger_frame = frame_count

        # draw box and label
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        cv2.putText(overlay, text, (x1, max(10, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 5) Decide which alert to play (DANGER priority)
    # If danger_count > 0 => play danger siren and set last_danger_frame
    # Else if moderate_count > 0 => play moderate alarm and set last_moderate_frame
    # Stop corresponding alarm if no detection for HALT_FRAMES
    if danger_count > 0:
        # start danger alarm if not already playing danger
        if current_alert != "danger":
            stop_alert()
            play_alert(DANGER_SOUND)
            current_alert = "danger"
        else:
            # ensure it keeps playing; update last_danger_frame already set above
            pass
        last_danger_frame = frame_count
    else:
        # no danger currently — check if we should stop danger alarm
        if current_alert == "danger" and (frame_count - last_danger_frame) > HALT_FRAMES:
            stop_alert()
            current_alert = None

        # if no danger playing, consider moderate
        if (current_alert is None or current_alert == "moderate") and moderate_count > 0:
            # start moderate alarm if not already playing moderate
            if current_alert != "moderate":
                stop_alert()
                play_alert(MODERATE_SOUND)
                current_alert = "moderate"
            last_moderate_frame = frame_count
        else:
            # stop moderate if it timed out
            if current_alert == "moderate" and (frame_count - last_moderate_frame) > HALT_FRAMES:
                stop_alert()
                current_alert = None

    # 6) Overlay counts and status
    cv2.putText(overlay, f"SAFE: {safe_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(overlay, f"MODERATE: {moderate_count}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 200), 2)
    cv2.putText(overlay, f"DANGER: {danger_count}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Flashing banner if danger active
    if current_alert == "danger":
        # flashing effect
        if (frame_count // 10) % 2 == 0:
            cv2.rectangle(overlay, (0, frame_h - 60), (frame_w, frame_h), (0, 0, 255), -1)
            cv2.putText(overlay, "!!! DANGER ZONE ACTIVE !!!", (30, frame_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 3)
    elif current_alert == "moderate":
        if (frame_count // 15) % 2 == 0:
            cv2.rectangle(overlay, (0, frame_h - 60), (frame_w, frame_h), (0, 200, 200), -1)
            cv2.putText(overlay, "MODERATE ALERT", (30, frame_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 3)

    # 7) Show window & write output frame
    cv2.imshow("Beach Safety - Live", overlay)
    writer.write(overlay)

    # 8) User quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# cleanup
stop_alert()
cap.release()
writer.release()
cv2.destroyAllWindows()
print(f"✅ Output saved to: {OUTPUT_VIDEO_PATH}")
print("👋 Done — stay safe, Beach Hero!")
