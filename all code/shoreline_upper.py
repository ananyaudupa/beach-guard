import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image
import os

# ========================
# CONFIG
# ========================
IMG_SIZE = 640
IMG_PATH = "/content/drive/MyDrive/test/img5007_jpg.rf.f35e05ac446725cec8dad735343f2bf0.jpg"
OUTPUT_FOLDER = "/content/drive/MyDrive/out"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
FINAL_OUTPUT_PATH = os.path.join(OUTPUT_FOLDER, "final_overlay_with_boundary.png")

# ========================
# Load ONNX model
# ========================
session = ort.InferenceSession("/content/drive/MyDrive/model done/unet_beach.onnx")
input_name = session.get_inputs()[0].name

# ========================
# Preprocess function
# ========================
def preprocess(img_path):
    image = Image.open(img_path).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image).astype(np.float32) / 255.0
    img_array = np.transpose(img_array, (2,0,1))
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, np.array(image)

input_tensor, resized_original = preprocess(IMG_PATH)

# ========================
# Run ONNX inference
# ========================
output = session.run(None, {input_name: input_tensor})
output_tensor = output[0]  # [1, NUM_CLASSES, H, W]
mask = np.argmax(output_tensor, axis=1).squeeze()

# ========================
# Find water region boundary
# ========================
h, w = mask.shape
water = (mask == 2).astype(np.uint8) * 255
contours, _ = cv2.findContours(water, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# ========================
# Overlay boundary on original
# ========================
overlay = cv2.cvtColor(resized_original, cv2.COLOR_RGB2BGR)  # original resized
for cnt in contours:
    for pt in cnt:
        x, y = pt[0]
        # ignore boundaries touching bottom or sides
        if y > h - 5 or x < 5 or x > w - 5:
            continue
        # draw upper shoreline in WHITE (thicker line instead of dots)
        cv2.circle(overlay, (x,y), 2, (255,255,255), -1)

# ========================
# Save final result
# ========================
cv2.imwrite(FINAL_OUTPUT_PATH, overlay)
print(f"✅ Final OVERLAY with UPPER shoreline saved at {FINAL_OUTPUT_PATH}")
