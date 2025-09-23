import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

# =========================
# CONFIG
# =========================
TRAIN_IMG_DIR = "/kaggle/input/majorproject/finaldata/train/original"
TRAIN_MASK_DIR = "/kaggle/input/majorproject/finaldata/train/blackouts"
VALID_IMG_DIR = "/kaggle/input/majorproject/finaldata/val/original"
VALID_MASK_DIR = "/kaggle/input/majorproject/finaldata/val/blackouts"

NUM_CLASSES = 4   # background, sand, water, sky
IMG_SIZE = 640
BATCH_SIZE = 4
EPOCHS = 25
LR = 1e-2

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print(f"Using device: {DEVICE}")

# =========================
# DATASET CLASS
# =========================
class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=640):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]
        self.img_size = img_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_file = self.images[idx]

        # 🔑 Extract ID before .rf. (e.g., "img52_jpg")
        img_id = img_file.split(".rf.")[0].replace(".jpg", "")
        mask_file = f"{img_id}_mask.png"

        img_path = os.path.join(self.img_dir, img_file)
        mask_path = os.path.join(self.mask_dir, mask_file)

        # Load
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Resize
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        # To tensor
        image = transforms.ToTensor()(image)
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))

        return image, mask

# =========================
# DATALOADERS
# =========================
train_dataset = SegmentationDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, IMG_SIZE)
valid_dataset = SegmentationDataset(VALID_IMG_DIR, VALID_MASK_DIR, IMG_SIZE)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

# =========================
# MODEL (U-Net)
# =========================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_classes):
        super(UNet, self).__init__()
        self.enc1 = DoubleConv(3, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(512, 1024)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        self.out = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)

# =========================
# METRICS
# =========================
def dice_score(pred, target, num_classes):
    pred = torch.argmax(pred, dim=1)
    dice = 0
    for cls in range(num_classes):
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()
        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()
        if union.item() > 0:
            dice += (2. * intersection) / union
    return dice / num_classes

def iou_score(pred, target, num_classes):
    pred = torch.argmax(pred, dim=1)
    iou = 0
    for cls in range(num_classes):
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()
        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum() - intersection
        if union.item() > 0:
            iou += intersection / union
    return iou / num_classes

# =========================
# TRAINING
# =========================
model = UNet(n_classes=NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

best_dice = 0.0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for imgs, masks in train_loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss, val_dice, val_iou = 0, 0, 0
    with torch.no_grad():
        for imgs, masks in valid_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
            val_dice += dice_score(outputs, masks, NUM_CLASSES)
            val_iou += iou_score(outputs, masks, NUM_CLASSES)

    val_dice /= len(valid_loader)
    val_iou /= len(valid_loader)

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss: {train_loss/len(train_loader):.4f} | "
          f"Val Loss: {val_loss/len(valid_loader):.4f} | "
          f"Dice: {val_dice:.4f} | IoU: {val_iou:.4f}")

    if val_dice > best_dice:
        best_dice = val_dice
        torch.save(model.state_dict(), "unet_beach_best.pth")
        print(" ✅ Best model updated!")

torch.save(model.state_dict(), "unet_beach_final.pth")
print("🎉 Training finished. Final model saved.")

# =========================
# EXPORT TO ONNX
# =========================
model = UNet(n_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load("unet_beach_final.pth", map_location=DEVICE))
model.eval()

dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
onnx_path = "unet_beach.onnx"

torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size", 2: "height", 3: "width"},
        "output": {0: "batch_size", 2: "height", 3: "width"},
    }
)

print(f"📦 ONNX model saved as {onnx_path}")
