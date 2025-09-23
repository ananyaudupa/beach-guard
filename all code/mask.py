import os
import random
import shutil
from tqdm import tqdm
import numpy as np

def split_dataset(images_dir, masks_dir, output_dir, train_ratio=0.8, val_ratio=0.2):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create subdirectories for train and val (each with images & masks)
    for split in ['train', 'val']:
        os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, "masks"), exist_ok=True)

    # List all image files (only .jpg in images_dir)
    all_images = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]
    all_images.sort()
    random.shuffle(all_images)

    total_files = len(all_images)
    train_end = int(total_files * train_ratio)

    train_files = all_images[:train_end]
    val_files = all_images[train_end:]

    def copy_pairs(file_list, split):
        for file_name in tqdm(file_list, desc=f"Copying {split}"):
            # Copy image
            src_img = os.path.join(images_dir, file_name)
            dst_img = os.path.join(output_dir, split, "images", file_name)

            # Build mask filename (replace .jpg with .png if masks are .png)
            mask_name = os.path.splitext(file_name)[0] + ".png"
            src_mask = os.path.join(masks_dir, mask_name)
            dst_mask = os.path.join(output_dir, split, "masks", mask_name)

            if os.path.exists(src_img) and os.path.exists(src_mask):
                shutil.copy(src_img, dst_img)
                shutil.copy(src_mask, dst_mask)
            else:
                print(f"⚠️ Skipping {file_name}, mask or image missing.")

    copy_pairs(train_files, "train")
    copy_pairs(val_files, "val")

    print(f"✅ Dataset split completed. Train: {len(train_files)}, Val: {len(val_files)}")

if __name__ == "__main__":
    # 🔧 Hardcoded paths
    images_dir = "train"   # folder containing .jpg images
    masks_dir = "mask"     # folder containing .png masks
    output_dir = "output"  # folder where split dataset will be saved

    split_dataset(images_dir, masks_dir, output_dir, train_ratio=0.8, val_ratio=0.2)
