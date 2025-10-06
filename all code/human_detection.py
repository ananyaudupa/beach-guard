from ultralytics import YOLO
import cv2

# Load YOLOv8 pretrained model
model = YOLO("yolov8m.pt")  # or yolov8s.pt for better accuracy
model.to("cuda:0")  # Send model to GPU 0

# Input and output video paths
input_video = "./d2.mp4"
output_video = "./output.mp4"

# Open video file
cap = cv2.VideoCapture(input_video)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create VideoWriter to save output
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# Process video frame-by-frame
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run detection (only persons)
    results = model.predict(frame, classes=[0,3,8], conf=0.2, iou=0.4, verbose=False)
    
    # Draw detections
    annotated_frame = results[0].plot()
    
    # Write frame to output video
    out.write(annotated_frame)

# Release resources
cap.release()
out.release()
print(f"Processed video saved as {output_video}")