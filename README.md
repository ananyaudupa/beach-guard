
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:1e90ff,100:00bfff&height=220&section=header&text=BEACHGUARD&fontSize=48&fontColor=ffffff&animation=fadeIn" width="100%" />

# 🏖️ BEACHGUARD  
## Smart Surveillance for Beach Safety and Emergency Response

> **An AI-powered real-time beach monitoring system designed to prevent drowning incidents and enhance emergency response using Computer Vision and Deep Learning.**

---

## 📌 Project Overview

**BeachGuard** is an intelligent beach safety surveillance system developed to overcome the limitations of traditional lifeguard-based monitoring. Drowning incidents often occur due to delayed detection, human fatigue, poor visibility, and the difficulty of monitoring large coastal areas.

This project introduces an **automated, real-time AI-driven solution** that continuously analyzes beach video feeds, detects people near or inside water, evaluates drowning risk based on shoreline proximity, and triggers **instant alerts** to lifeguards and authorities.

---

## 🧠 Core Idea

BeachGuard integrates multiple AI components into a single safety pipeline:

- **YOLOv8** → Real-time human detection  
- **U-Net (ONNX)** → Accurate shoreline segmentation  
- **Distance-based risk analysis** → Zone classification  
- **Audio + Telegram alerts** → Immediate emergency response  
- **Streamlit dashboard** → Live monitoring and control  

---

## 📄 Abstract

Drowning is a major worldwide tragedy, with over 300,000 deaths annually. Lifeguards must manually monitor large coastal areas, which is inefficient and error-prone. BeachGuard solves this problem using computer vision and deep learning.

The system uses a **U-Net segmentation model** to identify sand and water regions and accurately extract the shoreline in real time. A **YOLOv8m object detection model** detects people in the scene. The shortest distance between each detected person and the shoreline is computed and used to classify risk into **Safe**, **Moderate**, and **Danger** zones.

When a person enters the Moderate zone, a warning alert is generated. When a person enters the Danger zone, an emergency alert is immediately triggered, significantly reducing rescue response time and increasing survival chances.

---

## 🎯 Objectives

- Improve beach safety through AI-based real-time surveillance
- Assist lifeguards with automated alerts and decision support
- Detect drowning-risk situations without human intervention
- Reduce emergency response time
- Promote safer tourism and public awareness

---

## 🚨 Risk Zones Classification

| Zone | Description |
|----|------------|
| 🟢 **Safe Zone** | Person is far from shoreline |
| 🟡 **Moderate Zone** | Person is near shoreline |
| 🔴 **Danger Zone** | Person is inside or very close to water |

---

## 🏗️ System Architecture & Data Flow

The following diagram illustrates how video input flows through detection, segmentation, distance computation, alert generation, and display modules.

<p align="center">
 <img width="900" height="800" alt="Simple Use case diagram" src="https://github.com/user-attachments/assets/1de348b9-0943-4345-bed1-4d43327d31ee" />
</p>

**Explanation (High Level):**
- User uploads video → frames extracted
- YOLOv8 detects people
- U-Net segments shoreline
- Distance to water is computed
- Risk is classified
- Alerts are triggered (sound + Telegram)
- Results displayed on Streamlit dashboard

---

## 🔄 Detailed Processing Flowchart

The complete operational flow of the system from start to end is shown below.

<p align="center">
 <img width="900" height="750" alt="Blank diagram" src="https://github.com/user-attachments/assets/c35eb88d-1954-48e9-95bb-b7b102ba036c" />
</p>

**Key Steps:**
1. System initialization and settings load  
2. Video stream processing frame-by-frame  
3. Person detection and shoreline segmentation  
4. Distance computation to water  
5. Risk classification (Safe / Moderate / Danger)  
6. Alert triggering and overlay rendering  
7. Output video saving and live display  

---

## 🧪 Methodology

### 🔹 Human Detection
- YOLOv8 detects humans in real time
- Foot position derived from bounding box midpoint

### 🔹 Shoreline Detection
- ONNX-based U-Net segments water and sand regions
- Shoreline extracted from class boundaries

### 🔹 Distance Measurement
- Euclidean distance between foot point and nearest shoreline pixel

### 🔹 Alert Mechanism
- **Moderate Alert** → Warning sound + Telegram message  
- **Danger Alert** → Emergency sound + Telegram alert  
- Alerts are non-blocking and de-duplicated

---

## 🖥️ Technologies Used

### Programming & Frameworks
- Python 3.8+
- Streamlit

### AI & Computer Vision
- YOLOv8 (Ultralytics)
- U-Net (ONNX)
- OpenCV
- PyTorch
- ONNX Runtime

### Alerts & UI
- Pygame (Audio alerts)
- Telegram Bot API

---

## 💻 Hardware Requirements

| Component | Minimum | Recommended |
|---------|--------|-------------|
| CPU | Intel i5 / Ryzen 5 | Intel i7 / Ryzen 7 |
| GPU | GTX 1050 (4GB) | RTX 2060+ |
| RAM | 8 GB | 16 GB+ |
| Storage | 10 GB | SSD |
| Camera | HD Webcam | 1080p CCTV |

---

## ⚙️ Software Requirements

- Windows 10 / Ubuntu 20.04+
- Python 3.8+
- OpenCV, NumPy, PyTorch
- Ultralytics YOLOv8
- CUDA Toolkit (optional)

---

## 📊 Results

- Accurate real-time detection of swimmers
- Clear visualization of shoreline and safety zones
- Instant alerts for dangerous conditions
- Stable performance at 15–20 FPS
- Successfully validated using Black Box and White Box testing

---

## 🔮 Future Scope

- Rip current detection
- Drone-based beach surveillance
- Mobile app for lifeguards
- Multi-beach centralized monitoring
- Weather and tide data integration
- Real-world distance calibration using GPS

---

## 👨‍💻 Project Team

| Name | USN |
|----|----|
| Ananya Udupa | 4SN22AI009 |
| Likhitha | 4SN22AI034 |
| Shreyas Nayak | 4SN22AI054 |
| **Thanush** | 4SN22AI059 |

---

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:00bfff,100:1e90ff&height=180&section=footer" width="100%" />
