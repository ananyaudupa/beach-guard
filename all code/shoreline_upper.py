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
FINAL_OUTPUT_PATH = os.path.join(OUTPUT_FOLDER, "mask_with_upper_water_boundary.png")

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
    return img_array

input_tensor = preprocess(IMG_PATH)

# ========================
# Run ONNX inference
# ========================
output = session.run(None, {input_name: input_tensor})
output_tensor = output[0]  # [1, NUM_CLASSES, H, W]
mask = np.argmax(output_tensor, axis=1).squeeze()

# ========================
# Color-coded mask
# ========================
colors = {
    0: (0,0,0),       # background
    1: (0,255,0),     # sand
    2: (0,0,255),     # water
    3: (0,165,255)    # sky
}
h, w = mask.shape
color_mask = np.zeros((h,w,3), dtype=np.uint8)
for cls, color in colors.items():
    color_mask[mask==cls] = color

# ========================
# Find water region boundary
# ========================
water = (mask == 2).astype(np.uint8) * 255
contours, _ = cv2.findContours(water, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# ========================
# Filter & draw only *upper* boundary
# ========================
for cnt in contours:
    for pt in cnt:
        x, y = pt[0]
        # ignore boundaries that are touching bottom or left/right edges
        if y > h - 5 or x < 5 or x > w - 5:
            continue
        # draw upper shoreline only
        cv2.circle(color_mask, (x,y), 1, (255,255,255), -1)

# ========================
# Save final mask
# ========================
cv2.imwrite(FINAL_OUTPUT_PATH, color_mask)
print(f"✅ Final UPPER shoreline mask saved at {FINAL_OUTPUT_PATH}")
