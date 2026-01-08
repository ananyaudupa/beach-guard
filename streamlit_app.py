"""
Streamlit Frontend for Beach Safety Monitoring System
Based on chat.py with video display and admin panel
"""

import os
import cv2
import math
import time
import warnings
import numpy as np
import torch
import onnxruntime as ort
import requests
import streamlit as st
import tempfile
import threading
import pygame
import base64
from ultralytics import YOLO
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=FutureWarning)

# ========================
# ENVIRONMENT SETUP
# ========================
load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN", "8554429290:AAE91vK9OMbS4ORzi4kP7BOVpDXHhc-dpxA")
CHAT_ID = os.getenv("CHAT_ID","-1003263432307")

IMG_SIZE = 640
HALT_FRAMES = 100

YOLO_WEIGHTS = "yolov8m.pt"
ONNX_MODEL_PATH = "unet_beach.onnx"
MODERATE_SOUND = "moderate_alert.mp3"
DANGER_SOUND = "danger_alert.mp3"

# Default values (can be changed by admin)
DEFAULT_SAFE_DISTANCE_PX = 100
DEFAULT_SAFE_THRESHOLD = 50
DEFAULT_PIXELS_PER_METER = 10  # Default conversion: 20 pixels = 1 meter

# ========================
# Session State Initialization
# ========================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "admin_username" not in st.session_state:
    st.session_state.admin_username = "admin"
if "admin_password" not in st.session_state:
    st.session_state.admin_password = "admin123"  # Change this in production
if "safe_distance_px" not in st.session_state:
    st.session_state.safe_distance_px = DEFAULT_SAFE_DISTANCE_PX
if "safe_threshold" not in st.session_state:
    st.session_state.safe_threshold = DEFAULT_SAFE_THRESHOLD
if "pixels_per_meter" not in st.session_state:
    st.session_state.pixels_per_meter = DEFAULT_PIXELS_PER_METER
if "processing" not in st.session_state:
    st.session_state.processing = False
if "video_path" not in st.session_state:
    st.session_state.video_path = None
if "processed_frames" not in st.session_state:
    st.session_state.processed_frames = []
if "stats" not in st.session_state:
    st.session_state.stats = {"safe": 0, "moderate": 0, "danger": 0, "frame_count": 0}
if "input_mode" not in st.session_state:
    st.session_state.input_mode = "video"  # "video" or "camera"
if "camera_active" not in st.session_state:
    st.session_state.camera_active = False
if "camera_cap" not in st.session_state:
    st.session_state.camera_cap = None

# ========================
# Load Models
# ========================
@st.cache_resource
def load_models():
    """Load YOLO and ONNX models"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Check if model files exist
    if not os.path.exists(YOLO_WEIGHTS):
        raise FileNotFoundError(f"YOLO weights not found: {YOLO_WEIGHTS}")
    if not os.path.exists(ONNX_MODEL_PATH):
        raise FileNotFoundError(f"ONNX model not found: {ONNX_MODEL_PATH}")
    
    yolo_model = YOLO(YOLO_WEIGHTS).to(device)
    session = ort.InferenceSession(ONNX_MODEL_PATH, providers=["CPUExecutionProvider"])
    onnx_input_name = session.get_inputs()[0].name
    
    return yolo_model, session, onnx_input_name, device

# Models will be loaded using @st.cache_resource decorator

# ========================
# Audio Setup
# ========================
pygame.init()
try:
    pygame.mixer.init()
except Exception as e:
    st.warning(f"⚠ pygame.mixer.init() failed: {e}")

def play_alert(sound_path):
    """Play alert sound (non-blocking)"""
    try:
        if os.path.exists(sound_path):
            pygame.mixer.music.stop()
            if hasattr(pygame.mixer.music, "unload"):
                pygame.mixer.music.unload()
            pygame.mixer.music.load(sound_path)
            pygame.mixer.music.play(-1)  # Loop until stopped
    except Exception as e:
        # Silently fail - audio errors shouldn't interrupt video processing
        pass

def stop_alert():
    """Stop any currently playing alert sound"""
    try:
        pygame.mixer.music.stop()
        if hasattr(pygame.mixer.music, "unload"):
            pygame.mixer.music.unload()
    except Exception as e:
        # Silently fail
        pass

# ========================
# Helper Functions
# ========================
def pixels_to_meters(pixels, pixels_per_meter):
    """Convert pixel distance to meters"""
    return pixels / pixels_per_meter

def preprocess_for_onnx(frame):
    """Preprocess frame for ONNX model"""
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, 0)

def find_shoreline_points(mask, frame_w, frame_h):
    """Find shoreline points from segmentation mask"""
    water_mask = (mask == 2).astype(np.uint8) * 255
    contours, _ = cv2.findContours(water_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts = []
    for cnt in contours:
        for pt in cnt:
            x, y = pt[0]
            if 5 < x < IMG_SIZE - 5 and y < IMG_SIZE - 5:
                x_m = int(x * frame_w / IMG_SIZE)
                y_m = int(y * frame_h / IMG_SIZE)
                pts.append((x_m, y_m))
    return pts

def send_telegram_message(text):
    """Send text message to Telegram bot chat (non-blocking)"""
    def _send():
        try:
            if not BOT_TOKEN or not CHAT_ID:
                return
            url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
            data = {"chat_id": CHAT_ID, "text": text}
            requests.post(url, data=data, timeout=10)
        except Exception:
            # Silently fail - Telegram errors shouldn't interrupt video processing
            pass
    
    # Send in background thread to avoid blocking
    thread = threading.Thread(target=_send, daemon=True)
    thread.start()

def send_telegram_photo(frame, caption=""):
    """Send OpenCV frame image to Telegram (non-blocking, compressed)"""
    def _send():
        try:
            if not BOT_TOKEN or not CHAT_ID:
                return
            
            # Resize and compress image to reduce size and prevent timeouts
            height, width = frame.shape[:2]
            max_dimension = 800  # Max width or height
            if width > max_dimension or height > max_dimension:
                scale = max_dimension / max(width, height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame_resized = cv2.resize(frame, (new_width, new_height))
            else:
                frame_resized = frame
            
            # Encode with compression
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 75]  # Reduce quality to 75% for smaller file
            _, img_encoded = cv2.imencode('.jpg', frame_resized, encode_params)
            
            files = {'photo': ('frame.jpg', img_encoded.tobytes())}
            data = {"chat_id": CHAT_ID, "caption": caption}
            requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto", 
                         data=data, files=files, timeout=15)  # Increased timeout
        except Exception:
            # Silently fail - Telegram errors shouldn't interrupt video processing
            pass
    
    # Send in background thread to avoid blocking
    thread = threading.Thread(target=_send, daemon=True)
    thread.start()

# ========================
# Video Processing Function
# ========================
def process_video_frame(frame, frame_count, safe_distance_px, safe_threshold, 
                        yolo_model, session, onnx_input_name, device, pixels_per_meter):
    """Process a single video frame"""
    overlay = frame.copy()
    frame_w = frame.shape[1]
    frame_h = frame.shape[0]
    
    # Calculate thresholds
    moderate_threshold = safe_distance_px
    
    # YOLO detection
    results = yolo_model.predict(frame, device=device, imgsz=IMG_SIZE,
                                 classes=[0], conf=0.25, iou=0.3, verbose=False)
    detections = []
    if len(results) > 0 and len(results[0].boxes) > 0:
        for box in results[0].boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)
            detections.append((x1, y1, x2, y2))
    
    # UNet segmentation
    in_tensor = preprocess_for_onnx(frame)
    out = session.run(None, {onnx_input_name: in_tensor})
    mask = np.argmax(out[0], axis=1).squeeze()
    
    # Shoreline
    shoreline_points = find_shoreline_points(mask, frame_w, frame_h)
    for p in shoreline_points[::max(1, len(shoreline_points)//300)]:
        cv2.circle(overlay, p, 2, (255, 0, 0), -1)
    
    safe_count = moderate_count = danger_count = 0
    current_alert = None
    
    # Person classification
    for (x1, y1, x2, y2) in detections:
        person_feet = ((x1 + x2)//2, y2)
        feet_x = int(person_feet[0] * IMG_SIZE / frame_w)
        feet_y = int(person_feet[1] * IMG_SIZE / frame_h)
        
        if not (0 <= feet_x < IMG_SIZE and 0 <= feet_y < IMG_SIZE):
            continue
        
        region_class = int(mask[feet_y, feet_x])
        
        if region_class not in [2, 3]:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (120, 120, 120), 1)
            continue
        
        color = (0, 255, 255)
        text = "MODERATE (?)"
        
        if shoreline_points:
            nearest = min(shoreline_points,
                          key=lambda p: math.hypot(p[0]-person_feet[0], p[1]-person_feet[1]))
            min_dist = math.hypot(nearest[0]-person_feet[0], nearest[1]-person_feet[1])
            min_dist_meters = pixels_to_meters(min_dist, pixels_per_meter)
            cv2.line(overlay, person_feet, nearest, (255, 255, 0), 1)
            
            if min_dist < safe_threshold:
                color = (0, 255, 0)
                text = f"SAFE ({min_dist_meters:.1f}m)"
                safe_count += 1
            elif safe_threshold <= min_dist < moderate_threshold:
                color = (0, 255, 255)
                text = f"MODERATE ({min_dist_meters:.1f}m)"
                moderate_count += 1
            else:
                color = (0, 0, 255)
                text = f"DANGER ({min_dist_meters:.1f}m)"
                danger_count += 1
        else:
            moderate_count += 1
        
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        cv2.putText(overlay, text, (x1, max(10, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Display stats on overlay with background for better visibility
    # Safe count - Green
    cv2.rectangle(overlay, (15, 15), (200, 55), (0, 0, 0), -1)  # Black background
    cv2.rectangle(overlay, (15, 15), (200, 55), (0, 255, 0), 2)  # Green border
    cv2.putText(overlay, f"SAFE: {safe_count}", (20, 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Moderate count - Yellow
    cv2.rectangle(overlay, (15, 60), (250, 100), (0, 0, 0), -1)  # Black background
    cv2.rectangle(overlay, (15, 60), (250, 100), (0, 255, 255), 2)  # Yellow border
    cv2.putText(overlay, f"MODERATE: {moderate_count}", (20, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    
    # Danger count - Red
    cv2.rectangle(overlay, (15, 105), (220, 145), (0, 0, 0), -1)  # Black background
    cv2.rectangle(overlay, (15, 105), (220, 145), (0, 0, 255), 2)  # Red border
    cv2.putText(overlay, f"DANGER: {danger_count}", (20, 135), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    # Alert banners
    if danger_count > 0 and (frame_count // 10) % 2 == 0:
        cv2.rectangle(overlay, (0, frame_h - 60), (frame_w, frame_h), (0, 0, 255), -1)
        cv2.putText(overlay, "!!! DANGER ZONE ACTIVE !!!", (30, frame_h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        current_alert = "danger"
    elif moderate_count > 0 and (frame_count // 15) % 2 == 0:
        cv2.rectangle(overlay, (0, frame_h - 60), (frame_w, frame_h), (0, 200, 200), -1)
        cv2.putText(overlay, "MODERATE ALERT", (30, frame_h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)
        current_alert = "moderate"
    
    return overlay, {
        "safe": safe_count,
        "moderate": moderate_count,
        "danger": danger_count,
        "alert": current_alert
    }

# ========================
# Streamlit UI
# ========================
st.set_page_config(
    page_title="Beach Guard",
    page_icon="🏖️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1E88E5;
        padding: 20px;
        background: linear-gradient(90deg, #E3F2FD 0%, #BBDEFB 100%);
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .stats-box {
        background: #F5F5F5;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        border-left: 5px solid;
    }
    .safe-box { border-left-color: #4CAF50; }
    .safe-box h3 { color: #4CAF50 !important; }
    .safe-box .count-number { color: #4CAF50 !important; font-size: 2.5em !important; font-weight: bold !important; }
    .moderate-box { border-left-color: #FFC107; }
    .moderate-box h3 { color: #FFC107 !important; }
    .moderate-box .count-number { color: #FFC107 !important; font-size: 2.5em !important; font-weight: bold !important; }
    .danger-box { border-left-color: #F44336; }
    .danger-box h3 { color: #F44336 !important; }
    .danger-box .count-number { color: #F44336 !important; font-size: 2.5em !important; font-weight: bold !important; }
    .stats-box span.count-number {
        display: block !important;
        width: 100% !important;
    }
    .admin-panel {
        background: #FFF3E0;
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #FF9800;
    }
    
    /* Landing Page Styles */
    .landing-container {
        position: relative;
        min-height: 70vh;
        overflow: hidden;
        padding: 0;
        margin: -1rem;
    }
    
    .video-overlay {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.4);
        z-index: 2;
    }
    
    .landing-content {
        position: relative;
        z-index: 10;
        padding: 3rem 2rem;
        text-align: center;
        color: white;
    }
    
    .landing-title {
        font-size: 4rem;
        font-weight: bold;
        color: white;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.7);
    }
    
    .landing-subtitle {
        font-size: 1.5rem;
        color: white;
        margin-bottom: 3rem;
        text-shadow: 1px 1px 4px rgba(0,0,0,0.7);
    }
    
    .get-started-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 3rem;
        font-size: 1.3rem;
        font-weight: bold;
        border: none;
        border-radius: 50px;
        cursor: pointer;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        text-decoration: none;
        display: inline-block;
    }
    
    .get-started-btn:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.6);
    }
    
    .guidelines-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2.5rem;
        margin: 3rem auto;
        max-width: 800px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        position: relative;
        z-index: 10;
    }
    
    .guideline-title {
        font-size: 2rem;
        color: #34495E;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    .guideline-item {
        margin-bottom: 1rem;
        padding: 0.8rem 0;
        font-size: 1.1rem;
        color: #34495E;
        line-height: 1.8;
        list-style: none;
    }
    
    .guideline-item strong {
        color: #1E3A5F;
        font-weight: 600;
    }
    
    /* Hide default Streamlit elements on landing page */
    .landing-page [data-testid="stSidebar"] {
        display: none !important;
    }
    
    /* Full width for landing page */
    .landing-page .main {
        margin-left: 0 !important;
        max-width: 100% !important;
        padding: 0 !important;
    }
    
    /* Style the Streamlit button to match our design */
    .landing-page button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        padding: 1rem 3rem !important;
        font-size: 1.3rem !important;
        font-weight: bold !important;
        border: none !important;
        border-radius: 50px !important;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4) !important;
        transition: all 0.3s ease !important;
        width: auto !important;
        margin: 4rem auto 2rem auto !important;
        display: block !important;
        position: relative;
        z-index: 10;
    }
    
    .landing-page button[kind="primary"]:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.6) !important;
    }
    
    /* Ensure landing content is properly positioned */
    .landing-page .stMarkdown {
        margin: 0 !important;
    }
    
    .landing-page [data-testid="column"] {
        background: transparent !important;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("🏖️ Beach Guard")

# Initialize page in session state if not exists
if "current_page" not in st.session_state:
    st.session_state.current_page = "🏖️ Landing"

# Handle navigation from button click
if "navigate_to" in st.session_state:
    st.session_state.current_page = st.session_state.navigate_to
    del st.session_state.navigate_to

# Get current page from session state or sidebar
page_options = ["🏖️ Landing", "🏠 Dashboard", "⚙️ Admin Panel"]
default_index = page_options.index(st.session_state.current_page) if st.session_state.current_page in page_options else 0
page = st.sidebar.radio("Navigation", page_options, index=default_index)

# Update session state when sidebar changes
if page != st.session_state.current_page:
    st.session_state.current_page = page

# ========================
# Landing Page
# ========================
if page == "🏖️ Landing":
    # Hide sidebar using JavaScript
    st.markdown("""
    <script>
        window.addEventListener('load', function() {
            const sidebar = document.querySelector('[data-testid="stSidebar"]');
            if (sidebar) {
                sidebar.style.display = 'none';
            }
            const main = document.querySelector('.main');
            if (main) {
                main.style.marginLeft = '0';
            }
        });
    </script>
    <style>
        [data-testid="stSidebar"] { display: none !important; }
        .main { margin-left: 0 !important; }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="landing-page">', unsafe_allow_html=True)
    
    # Load background image (1.jpg)
    image_path = "1.jpg"
    background_image = None
    
    # Try to find and load the image
    if not os.path.exists(image_path):
        # Try in assets folder
        assets_image = os.path.join("assets", "1.jpeg")
        if os.path.exists(assets_image):
            image_path = assets_image
        # Try parent directory
        elif os.path.exists(os.path.join("..", image_path)):
            image_path = os.path.join("..", image_path)
    
    # Load and convert image to base64
    if os.path.exists(image_path):
        try:
            with open(image_path, "rb") as img_file:
                img_data = img_file.read()
                img_base64 = base64.b64encode(img_data).decode('utf-8')
                # Determine image type from extension
                img_ext = os.path.splitext(image_path)[1].lower()
                img_type = "image/jpeg" if img_ext in [".jpg", ".jpeg"] else "image/png"
                background_image = f"data:{img_type};base64,{img_base64}"
        except Exception as e:
            # If image loading fails, continue with gradient fallback
            pass
    
    # Create the landing page structure with background
    if background_image:
        # Use image as background
        st.markdown(f"""
        <div class="landing-container" style="background-image: url('{background_image}'); background-size: cover; background-position: center; background-repeat: no-repeat;">
            <div class="video-overlay"></div>
            <div class="landing-content">
                <h1 class="landing-title">🏖️ Beach Guard</h1>
                <p class="landing-subtitle">SMART SURVEILLANCE FOR BEACH SAFETY AND
 EMERGENCY RESPONSE</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Fallback to gradient background if image not found
        st.markdown("""
        <div class="landing-container" style="background: linear-gradient(180deg, #87CEEB 0%, #E0F6FF 30%, #FFF8DC 60%, #FFE4B5 100%);">
            <div class="landing-content">
                <h1 class="landing-title">🏖️ Beach Guard</h1>
                <p class="landing-subtitle">SMART SURVEILLANCE FOR BEACH SAFETY AND
 EMERGENCY RESPONSE</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Get Started button - centered with spacing
    st.markdown("<div style='margin-top: 3rem;'></div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Get Started →", type="primary", use_container_width=True, key="get_started_btn"):
            st.session_state.current_page = "🏠 Dashboard"
            st.rerun()
    
    # Guidelines section - using container with proper HTML
    guidelines_html = """
    <div class="guidelines-container">
        <h2 class="guideline-title" style="color: black;">Usage Guidelines</h2>
        <div class="guideline-item"><strong>Upload only beach surveillance videos</strong> for accurate analysis</div>
        <div class="guideline-item"><strong>Ensure stable internet connection</strong> for real-time alerts</div>
        <div class="guideline-item"><strong>Use admin credentials</strong> for accessing configuration settings</div>
        <div class="guideline-item"><strong>Keep audio enabled</strong> to receive danger alerts</div>
        <div class="guideline-item"><strong>Maintain proper visibility</strong> in uploaded footage for correct detection</div>
    </div>
    """
    st.markdown(guidelines_html, unsafe_allow_html=True)
    
    # Close landing-page div
    st.markdown('</div>', unsafe_allow_html=True)

# ========================
# Dashboard Page (formerly Home)
# ========================
elif page == "🏠 Dashboard":
    st.markdown('<div class="main-header"><h1>🏖️ BEACH GUARD  Dashboard</h1></div>', 
                unsafe_allow_html=True)
    
    # Display current settings (read-only for non-admin)
    safe_dist_m = pixels_to_meters(st.session_state.safe_distance_px, st.session_state.pixels_per_meter)
    safe_thresh_m = pixels_to_meters(st.session_state.safe_threshold, st.session_state.pixels_per_meter)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"**Safe Distance:** {safe_dist_m:.1f}m ({st.session_state.safe_distance_px} px)")
    with col2:
        st.info(f"**Safe Threshold:** {safe_thresh_m:.1f}m ({st.session_state.safe_threshold} px)")
    with col3:
        st.info(f"**Pixels/Meter:** {st.session_state.pixels_per_meter}")
    
    # Input mode selection
    st.subheader("📹 Input Source")
    input_mode = st.radio(
        "Select input source:",
        ["📁 Upload Video File", "📷 Real-time Camera"],
        horizontal=True,
        key="input_mode_radio"
    )
    
    # Update session state
    if "Upload Video File" in input_mode:
        st.session_state.input_mode = "video"
    else:
        st.session_state.input_mode = "camera"
    
    uploaded_file = None
    
    if st.session_state.input_mode == "video":
        # Video upload
        uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])
    
    # Load models (cached by @st.cache_resource)
    try:
        with st.spinner("Loading AI models... This may take a moment."):
            yolo_model, session, onnx_input_name, device = load_models()
        st.success(f"✅ Models loaded successfully on {device}!")
    except FileNotFoundError as e:
        st.error(f"❌ {str(e)}")
        st.info("Please ensure the model files are in the correct directory.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.stop()
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(uploaded_file.read())
                st.session_state.video_path = tmp_file.name
            
            st.success(f"✅ Video uploaded: {uploaded_file.name}")
            
            # Show video info
            cap_test = cv2.VideoCapture(st.session_state.video_path)
            if cap_test.isOpened():
                fps = cap_test.get(cv2.CAP_PROP_FPS) or 20.0
                total_frames = int(cap_test.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap_test.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap_test.get(cv2.CAP_PROP_FRAME_HEIGHT))
                duration = total_frames / fps if fps > 0 else 0
                cap_test.release()
                
                st.info(f"📊 Video Info: {width}x{height} @ {fps:.2f} FPS, {total_frames} frames (~{duration:.1f}s)")
            
            # Process button
            if st.button("▶️ Process Video", type="primary", use_container_width=True):
                st.session_state.processing = True
                st.session_state.processed_frames = []
                st.session_state.stats = {"safe": 0, "moderate": 0, "danger": 0, "frame_count": 0}
                
                # Open video
                cap = cv2.VideoCapture(st.session_state.video_path)
                if not cap.isOpened():
                    st.error("❌ Cannot open video file! Please try uploading again.")
                    st.session_state.processing = False
                else:
                    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    if total_frames == 0:
                        st.error("❌ Invalid video file or unable to read frame count!")
                        cap.release()
                        st.session_state.processing = False
                    else:
                        # Progress bar and placeholders
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        video_placeholder = st.empty()
                        
                        # Stats columns (created once outside loop)
                        stats_col1, stats_col2, stats_col3 = st.columns(3)
                        safe_placeholder = stats_col1.empty()
                        moderate_placeholder = stats_col2.empty()
                        danger_placeholder = stats_col3.empty()
                        
                        frame_count = 0
                        last_danger_frame = -9999
                        last_moderate_frame = -9999
                        current_alert = None
                        
                        # Process video
                        try:
                            while True:
                                ret, frame = cap.read()
                                if not ret:
                                    break
                                
                                frame_count += 1
                                
                                # Process frame
                                processed_frame, frame_stats = process_video_frame(
                                    frame, frame_count,
                                    st.session_state.safe_distance_px,
                                    st.session_state.safe_threshold,
                                    yolo_model, session, onnx_input_name, device,
                                    st.session_state.pixels_per_meter
                                )
                                
                                # Update stats
                                st.session_state.stats["safe"] = max(st.session_state.stats["safe"], frame_stats["safe"])
                                st.session_state.stats["moderate"] = max(st.session_state.stats["moderate"], frame_stats["moderate"])
                                st.session_state.stats["danger"] = max(st.session_state.stats["danger"], frame_stats["danger"])
                                st.session_state.stats["frame_count"] = frame_count
                                
                                # Telegram alerts and audio alerts (only send once per alert type)
                                if frame_stats["danger"] > 0:
                                    if current_alert != "danger":
                                        send_telegram_message("🚨 Danger Alert: People detected in the danger zone!")
                                        send_telegram_photo(processed_frame, caption="🚨 DANGER DETECTED")
                                        play_alert(DANGER_SOUND)
                                        current_alert = "danger"
                                    last_danger_frame = frame_count
                                elif frame_stats["moderate"] > 0:
                                    if current_alert != "moderate":
                                        send_telegram_message("⚠ Moderate risk detected — watch carefully.")
                                        send_telegram_photo(processed_frame, caption="⚠ Moderate risk area.")
                                        play_alert(MODERATE_SOUND)
                                        current_alert = "moderate"
                                    last_moderate_frame = frame_count
                                else:
                                    if current_alert == "danger" and (frame_count - last_danger_frame) > HALT_FRAMES:
                                        send_telegram_message("✅ Danger cleared. Area safe again.")
                                        stop_alert()
                                        current_alert = None
                                    elif current_alert == "moderate" and (frame_count - last_moderate_frame) > HALT_FRAMES:
                                        send_telegram_message("✅ Moderate zone cleared.")
                                        stop_alert()
                                        current_alert = None
                                
                                # Convert BGR to RGB for display
                                display_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                                
                                # Display video
                                video_placeholder.image(display_frame, use_container_width=True, channels="RGB")
                                
                                # Update stats display
                                safe_count_html = f'<div class="stats-box safe-box"><h3 style="color: #4CAF50; margin: 5px 0;">🟢 Safe</h3><span class="count-number" style="color: #4CAF50 !important; font-size: 2.5em !important; font-weight: bold !important; display: block !important; margin: 10px 0 !important; text-align: center !important;">{frame_stats["safe"]}</span></div>'
                                safe_placeholder.markdown(safe_count_html, unsafe_allow_html=True)
                                
                                moderate_count_html = f'<div class="stats-box moderate-box"><h3 style="color: #FFC107; margin: 5px 0;">🟡 Moderate</h3><span class="count-number" style="color: #FFC107 !important; font-size: 2.5em !important; font-weight: bold !important; display: block !important; margin: 10px 0 !important; text-align: center !important;">{frame_stats["moderate"]}</span></div>'
                                moderate_placeholder.markdown(moderate_count_html, unsafe_allow_html=True)
                                
                                danger_count_html = f'<div class="stats-box danger-box"><h3 style="color: #F44336; margin: 5px 0;">🔴 Danger</h3><span class="count-number" style="color: #F44336 !important; font-size: 2.5em !important; font-weight: bold !important; display: block !important; margin: 10px 0 !important; text-align: center !important;">{frame_stats["danger"]}</span></div>'
                                danger_placeholder.markdown(danger_count_html, unsafe_allow_html=True)
                                
                                # Update progress
                                progress = frame_count / total_frames
                                progress_bar.progress(progress)
                                status_text.text(f"Processing frame {frame_count}/{total_frames} ({progress*100:.1f}%)")
                                
                                # Small delay for display
                                time.sleep(0.03)
                            
                            cap.release()
                            progress_bar.progress(1.0)
                            status_text.text(f"✅ Processing complete! Processed {frame_count} frames.")
                            st.success("🎉 Video processing completed!")
                            st.session_state.processing = False
                            
                        except Exception as e:
                            st.error(f"❌ Error during video processing: {str(e)}")
                            cap.release()
                            st.session_state.processing = False
                        
                        finally:
                            # Cleanup
                            if os.path.exists(st.session_state.video_path):
                                try:
                                    os.unlink(st.session_state.video_path)
                                except:
                                    pass
        except Exception as e:
            st.error(f"❌ Error uploading video: {str(e)}")
            st.session_state.video_path = None
    
    # Real-time camera processing
    elif st.session_state.input_mode == "camera":
        st.info("📷 **Real-time Camera Mode**: Use your webcam for live beach monitoring. Click 'Start Live Processing' to begin.")
        
        # Initialize camera capture
        if st.button("📷 Initialize Camera", type="primary", use_container_width=True):
            if st.session_state.camera_cap is None:
                st.session_state.camera_cap = cv2.VideoCapture(0)
                if not st.session_state.camera_cap.isOpened():
                    st.error("❌ Cannot access camera! Please check if your camera is connected and permissions are granted.")
                    st.session_state.camera_cap = None
                else:
                    # Set camera properties for better performance
                    st.session_state.camera_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    st.session_state.camera_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    st.session_state.camera_cap.set(cv2.CAP_PROP_FPS, 30)
                    st.success("✅ Camera initialized successfully!")
                    st.rerun()
        
        # Start/Stop button
        if st.session_state.camera_cap is not None:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("▶️ Start Live Processing", type="primary", use_container_width=True):
                    st.session_state.processing = True
                    st.session_state.stats = {"safe": 0, "moderate": 0, "danger": 0, "frame_count": 0}
                    if "last_danger_frame" not in st.session_state:
                        st.session_state.last_danger_frame = -9999
                    if "last_moderate_frame" not in st.session_state:
                        st.session_state.last_moderate_frame = -9999
                    if "current_alert" not in st.session_state:
                        st.session_state.current_alert = None
                    st.rerun()
            with col2:
                if st.button("⏹️ Stop Processing", use_container_width=True):
                    st.session_state.processing = False
                    if st.session_state.camera_cap is not None:
                        st.session_state.camera_cap.release()
                        st.session_state.camera_cap = None
                    stop_alert()
                    st.rerun()
        
        # Process camera feed (one frame per rerun for Streamlit compatibility)
        if st.session_state.processing and st.session_state.camera_cap is not None:
            # Create placeholders for display
            status_text = st.empty()
            video_placeholder = st.empty()
            
            # Stats columns
            stats_col1, stats_col2, stats_col3 = st.columns(3)
            safe_placeholder = stats_col1.empty()
            moderate_placeholder = stats_col2.empty()
            danger_placeholder = stats_col3.empty()
            
            # Initialize frame count if not exists
            if "camera_frame_count" not in st.session_state:
                st.session_state.camera_frame_count = 0
            
            try:
                ret, frame = st.session_state.camera_cap.read()
                if not ret:
                    st.error("❌ Failed to read from camera!")
                    st.session_state.processing = False
                else:
                    st.session_state.camera_frame_count += 1
                    frame_count = st.session_state.camera_frame_count
                    
                    # Process frame
                    processed_frame, frame_stats = process_video_frame(
                        frame, frame_count,
                        st.session_state.safe_distance_px,
                        st.session_state.safe_threshold,
                        yolo_model, session, onnx_input_name, device,
                        st.session_state.pixels_per_meter
                    )
                    
                    # Update stats
                    st.session_state.stats["safe"] = max(st.session_state.stats["safe"], frame_stats["safe"])
                    st.session_state.stats["moderate"] = max(st.session_state.stats["moderate"], frame_stats["moderate"])
                    st.session_state.stats["danger"] = max(st.session_state.stats["danger"], frame_stats["danger"])
                    st.session_state.stats["frame_count"] = frame_count
                    
                    # Telegram alerts and audio alerts
                    if frame_stats["danger"] > 0:
                        if st.session_state.current_alert != "danger":
                            send_telegram_message("🚨 Danger Alert: People detected in the danger zone!")
                            send_telegram_photo(processed_frame, caption="🚨 DANGER DETECTED")
                            play_alert(DANGER_SOUND)
                            st.session_state.current_alert = "danger"
                        st.session_state.last_danger_frame = frame_count
                    elif frame_stats["moderate"] > 0:
                        if st.session_state.current_alert != "moderate":
                            send_telegram_message("⚠ Moderate risk detected — watch carefully.")
                            send_telegram_photo(processed_frame, caption="⚠ Moderate risk area.")
                            play_alert(MODERATE_SOUND)
                            st.session_state.current_alert = "moderate"
                        st.session_state.last_moderate_frame = frame_count
                    else:
                        if st.session_state.current_alert == "danger" and (frame_count - st.session_state.last_danger_frame) > HALT_FRAMES:
                            send_telegram_message("✅ Danger cleared. Area safe again.")
                            stop_alert()
                            st.session_state.current_alert = None
                        elif st.session_state.current_alert == "moderate" and (frame_count - st.session_state.last_moderate_frame) > HALT_FRAMES:
                            send_telegram_message("✅ Moderate zone cleared.")
                            stop_alert()
                            st.session_state.current_alert = None
                    
                    # Convert BGR to RGB for display
                    display_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    
                    # Display video
                    video_placeholder.image(display_frame, use_container_width=True, channels="RGB")
                    
                    # Update stats display
                    safe_count_html = f'<div class="stats-box safe-box"><h3 style="color: #4CAF50; margin: 5px 0;">🟢 Safe</h3><span class="count-number" style="color: #4CAF50 !important; font-size: 2.5em !important; font-weight: bold !important; display: block !important; margin: 10px 0 !important; text-align: center !important;">{frame_stats["safe"]}</span></div>'
                    safe_placeholder.markdown(safe_count_html, unsafe_allow_html=True)
                    
                    moderate_count_html = f'<div class="stats-box moderate-box"><h3 style="color: #FFC107; margin: 5px 0;">🟡 Moderate</h3><span class="count-number" style="color: #FFC107 !important; font-size: 2.5em !important; font-weight: bold !important; display: block !important; margin: 10px 0 !important; text-align: center !important;">{frame_stats["moderate"]}</span></div>'
                    moderate_placeholder.markdown(moderate_count_html, unsafe_allow_html=True)
                    
                    danger_count_html = f'<div class="stats-box danger-box"><h3 style="color: #F44336; margin: 5px 0;">🔴 Danger</h3><span class="count-number" style="color: #F44336 !important; font-size: 2.5em !important; font-weight: bold !important; display: block !important; margin: 10px 0 !important; text-align: center !important;">{frame_stats["danger"]}</span></div>'
                    danger_placeholder.markdown(danger_count_html, unsafe_allow_html=True)
                    
                    # Update status
                    status_text.text(f"🔄 Live Processing... Frame: {frame_count} | Safe: {frame_stats['safe']} | Moderate: {frame_stats['moderate']} | Danger: {frame_stats['danger']}")
                    
                    # Auto-rerun for continuous processing (with small delay)
                    time.sleep(0.033)  # ~30 FPS
                    st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error during camera processing: {str(e)}")
                if st.session_state.camera_cap is not None:
                    st.session_state.camera_cap.release()
                    st.session_state.camera_cap = None
                st.session_state.processing = False
                stop_alert()

# ========================
# Admin Panel
# ========================
elif page == "⚙️ Admin Panel":
    st.markdown('<div class="main-header"><h1>⚙️ Admin Panel</h1></div>', 
                unsafe_allow_html=True)
    
    # Authentication
    if not st.session_state.authenticated:
        st.markdown('<div class="admin-panel">', unsafe_allow_html=True)
        st.subheader("🔐 Admin Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("Login", type="primary"):
                if username == st.session_state.admin_username and password == st.session_state.admin_password:
                    st.session_state.authenticated = True
                    st.success("✅ Login successful!")
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials!")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Admin Settings
        st.success(f"✅ Logged in as: {st.session_state.admin_username}")
        
        st.markdown('<div class="admin-panel">', unsafe_allow_html=True)
        st.subheader("⚙️ Configuration Settings")
        
        # Safe Distance Configuration
        st.markdown("### 📏 Safe Distance Configuration")
        col1, col2, col3 = st.columns(3)
        
        with col3:
            new_pixels_per_meter = st.number_input(
                "Pixels per Meter",
                min_value=1,
                max_value=100,
                value=st.session_state.pixels_per_meter,
                step=1,
                help="Conversion factor: how many pixels equal 1 meter in the video"
            )
        
        with col1:
            new_safe_distance = st.number_input(
                "Safe Distance (pixels)",
                min_value=10,
                max_value=500,
                value=st.session_state.safe_distance_px,
                step=10,
                help="Maximum distance in pixels considered safe from shoreline"
            )
            safe_dist_m = pixels_to_meters(new_safe_distance, new_pixels_per_meter)
            st.caption(f"≈ {safe_dist_m:.1f} meters")
        
        with col2:
            new_safe_threshold = st.number_input(
                "Safe Threshold (pixels)",
                min_value=5,
                max_value=250,
                value=st.session_state.safe_threshold,
                step=5,
                help="Distance threshold below which people are considered safe"
            )
            safe_thresh_m = pixels_to_meters(new_safe_threshold, new_pixels_per_meter)
            st.caption(f"≈ {safe_thresh_m:.1f} meters")
        
        # Save button
        if st.button("💾 Save Settings", type="primary", use_container_width=True):
            st.session_state.safe_distance_px = int(new_safe_distance)
            st.session_state.safe_threshold = int(new_safe_threshold)
            st.session_state.pixels_per_meter = float(new_pixels_per_meter)
            st.success("✅ Settings saved successfully!")
            st.rerun()
        
        # Display current settings
        st.markdown("### 📊 Current Settings")
        safe_dist_m = pixels_to_meters(st.session_state.safe_distance_px, st.session_state.pixels_per_meter)
        safe_thresh_m = pixels_to_meters(st.session_state.safe_threshold, st.session_state.pixels_per_meter)
        moderate_dist_m = pixels_to_meters(st.session_state.safe_distance_px, st.session_state.pixels_per_meter)
        st.info(f"""
        - **Safe Distance:** {safe_dist_m:.1f}m ({st.session_state.safe_distance_px} px)
        - **Safe Threshold:** {safe_thresh_m:.1f}m ({st.session_state.safe_threshold} px)
        - **Moderate Threshold:** {moderate_dist_m:.1f}m ({st.session_state.safe_distance_px} px) (auto-calculated)
        - **Pixels per Meter:** {st.session_state.pixels_per_meter}
        """)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Logout button
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.success("Logged out successfully!")
            st.rerun()

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Beach Guard")
st.sidebar.markdown("SMART SURVEILLANCE FOR BEACH SAFETY AND EMERGENCY RESPONSE")
