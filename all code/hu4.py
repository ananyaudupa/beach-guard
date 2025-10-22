import cv2
import torch
import numpy as np
import math
from ultralytics import YOLO
import onnxruntime as ort

# ========================
# CONFIG
# ========================
IMG_SIZE = 1920 # 1280, 1536
SAFE_DISTANCE = 100  # pixel threshold
input_image = "yes.jpeg"
output_image = "photo_with_distance1.jpg"
onnx_model_path = "unet_beach.onnx"

# ========================
# Load models
# ========================
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load YOLOv12 model — change path to your YOLOv12 weights
yolo_model = YOLO("yolo11m.pt")  # example
yolo_model.to(device)

session = ort.InferenceSession(onnx_model_path)
input_name = session.get_inputs()[0].name

def preprocess_frame(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    img_array = image.astype(np.float32) / 255.0
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ========================
# Read input image
# ========================
frame = cv2.imread(input_image)
if frame is None:
    raise FileNotFoundError("❌ Image not found!")

overlay = frame.copy()

# ========================
# YOLOv12 Person Detection
# ========================
# Note: classes=[0] for “person” in COCO labels (assuming your model uses COCO)
results = yolo_model.predict(frame, device=device, imgsz=1920, classes=[0], conf=0.1, iou=0.3, verbose=False)
detections = []
if len(results[0].boxes) > 0:
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        conf = float(box.conf[0].cpu().numpy())
        detections.append((x1, y1, x2, y2, conf))

# ========================
# UNet Segmentation
# ========================
input_tensor = preprocess_frame(frame)
output = session.run(None, {input_name: input_tensor})
mask = np.argmax(output[0], axis=1).squeeze()  # e.g., 0=background,1=sky,2=water,3=sand

# ========================
# Extract Shoreline from Water Region
# ========================
water_mask = (mask == 2).astype(np.uint8) * 255
contours, _ = cv2.findContours(water_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
shoreline_points = []

for cnt in contours:
    for pt in cnt:
        x, y = pt[0]
        if y > IMG_SIZE - 5 or x < 5 or x > IMG_SIZE - 5:
            continue
        x_mapped = int(x * frame.shape[1] / IMG_SIZE)
        y_mapped = int(y * frame.shape[0] / IMG_SIZE)
        shoreline_points.append((x_mapped, y_mapped))
        cv2.circle(overlay, (x_mapped, y_mapped), 2, (255, 0, 0), -1)  # blue shoreline dots

# ========================
# Distance Calculation (Only for People in WATER)
# ========================
for i, (x1, y1, x2, y2, conf) in enumerate(detections):
    person_feet = ((x1 + x2) // 2, y2)

    # Map feet to mask coordinates
    feet_x = int(person_feet[0] * IMG_SIZE / frame.shape[1])
    feet_y = int(person_feet[1] * IMG_SIZE / frame.shape[0])

    # Skip invalid points
    if not (0 <= feet_x < IMG_SIZE and 0 <= feet_y < IMG_SIZE):
        continue

    # Get UNet class at feet
    region_class = mask[feet_y, feet_x]

    # ✅ Only process people in water (class 2)
    if region_class != 3:
        continue

    # Distance Calculation
    color = (0, 255, 0)
    text = ""

    if len(shoreline_points) > 0:
        nearest_point = min(
            shoreline_points,
            key=lambda p: math.hypot(p[0] - person_feet[0], p[1] - person_feet[1])
        )
        min_dist = math.hypot(nearest_point[0] - person_feet[0], nearest_point[1] - person_feet[1])
        cv2.line(overlay, person_feet, nearest_point, (255, 255, 0), 2)

        # ✅ Danger if far from shore (≥ SAFE_DISTANCE)
        if min_dist >= SAFE_DISTANCE:
            color = (0, 0, 255)  # red
            #text = f"DANGER ({int(min_dist)}px)"
        else:
            color = (0, 255, 0)  # green
            #text = f"SAFE ({int(min_dist)}px)"

    # Draw box & label
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
    cv2.putText(overlay, text, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

# =======================
# Save Output
# =======================
cv2.imwrite(output_image, overlay)
print(f"✅ Output saved at {output_image}")