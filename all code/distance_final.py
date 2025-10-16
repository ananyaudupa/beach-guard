import cv2
import numpy as np
from ultralytics import YOLO
import math

# ======== CONFIG ==========
IMAGE_PATH = r"C:\Users\ANANYA UDUPA\Downloads\major\WhatsApp Image 2025-10-16 at 2.57.37 PM.jpeg"
MODEL_PATH = "yolov8n.pt"
OUTPUT_PATH = "out_image_with_distance.jpg"

MARKER_SIZE_M = 1.0
S = MARKER_SIZE_M
square_world_corners = np.array([
    [0.0, 0.0],
    [2.0, 0.0],
    [2.0, 2.0],
    [0.0, 2.0]
], dtype=np.float32)

# ======== LOAD YOLO ==========
model = YOLO(MODEL_PATH)

# ======== HELPER FUNCTIONS ==========
def compute_homography_from_square_corners(corner_pixels, corner_world):
    H, mask = cv2.findHomography(corner_pixels.astype(np.float32), corner_world.astype(np.float32), method=cv2.RANSAC)
    return H

def pixel_to_world(pt_px, H):
    p = np.array([[[pt_px[0], pt_px[1]]]], dtype=np.float32)
    p_w = cv2.perspectiveTransform(p, H)
    return p_w[0,0]

# ======== MOUSE CLICK HANDLER ==========
clicked_points = []

def click_event(event, x, y, flags, param):
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        cv2.circle(vis, (x, y), 5, (0,0,255), -1)
        cv2.imshow("Click Two Points", vis)
        if len(clicked_points) == 2:
            # Compute distance
            pt1_world = pixel_to_world(clicked_points[0], Hmat)
            pt2_world = pixel_to_world(clicked_points[1], Hmat)
            dist = math.hypot(pt2_world[0]-pt1_world[0], pt2_world[1]-pt1_world[1])
            print(f"Distance between points: {dist:.2f} meters")
            cv2.putText(vis, f"{dist:.2f} m", (min(clicked_points[0][0], clicked_points[1][0]), 
                                               min(clicked_points[0][1], clicked_points[1][1])-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
            cv2.imshow("Click Two Points", vis)

# ======== MAIN ==========
image = cv2.imread(IMAGE_PATH)
vis = image.copy()

# ---- Pixel coordinates of the reference square ----
square_pixel_corners = np.array([
    [544,506],  # bottom-left
    [700,512],  # bottom-right
    [702,552],  # top-right
    [533,540]   # top-left
], dtype=np.float32)

# Compute homography
Hmat = compute_homography_from_square_corners(square_pixel_corners, square_world_corners)

# Show image and wait for two clicks
cv2.imshow("Click Two Points", vis)
cv2.setMouseCallback("Click Two Points", click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()
