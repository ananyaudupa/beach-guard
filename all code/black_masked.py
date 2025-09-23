import os
import numpy as np
from PIL import Image

# ==============================
# 1. Define your input & output paths
# ==============================
input_folder = "/content/drive/MyDrive/train/masked"      # Folder with colored masks
output_folder = "/content/drive/MyDrive/train/blackout"     # Folder to save class-number masks
os.makedirs(output_folder, exist_ok=True)

# ==============================
# 2. Define color map (RGB -> class number)
# ==============================
COLOR_MAP = {
    (0,0,0):0,
    (102,20,188): 1,   # Purple → Sky
    (71,106,14): 2,   # Yellow → Water
    (214,210,121): 3      # Green → Sand
}

# ==============================
# 3. Function to convert RGB mask → class mask
# ==============================
def rgb_to_class(mask_path):
    mask = np.array(Image.open(mask_path).convert("RGB"))  # Load mask
    class_mask = np.zeros(mask.shape[:2], dtype=np.uint8)  # Empty (H,W)

    for rgb, class_idx in COLOR_MAP.items():
        matches = np.all(mask == rgb, axis=-1)
        class_mask[matches] = class_idx

    return class_mask

# ==============================
# 4. Process all mask images
# ==============================
for filename in os.listdir(input_folder):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        mask_path = os.path.join(input_folder, filename)

        # Convert
        class_mask = rgb_to_class(mask_path)

        # Save in new folder
        save_path = os.path.join(output_folder, filename)
        Image.fromarray(class_mask).save(save_path)

        print(f"Saved class mask: {save_path}")

print("✅ All masks converted and saved!")
