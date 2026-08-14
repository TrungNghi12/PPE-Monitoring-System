# 🦺 PPE Monitoring System (Personal Protective Equipment Analytics)

[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![YOLO26s](https://img.shields.io/badge/YOLO26s-Ultralytics-FF9900.svg?style=flat)](https://github.com/ultralytics/ultralytics)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB.svg?style=flat&logo=python)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8.svg?style=flat&logo=opencv)](https://opencv.org/)

An end-to-end Computer Vision & AI Engineering solution for automated Personal Protective Equipment (PPE) compliance monitoring on images, powered by YOLO26s, a custom Spatial Logic Engine, FastAPI backend, and an interactive Glassmorphism Web Dashboard.

![PPE Monitoring System Demo](docs/demo.gif)

---

## 🌟 Key Features

- 🤖 **AI Object Detection**: Detects 5 target object classes using a fine-tuned YOLO26s model (`Person`, `Helmet`, `No-Helmet`, `Vest`, `No-Vest`).
- 🧠 **Spatial Logic Engine**: Uses spatial overlap (IoU) and fallback spatial bounding box inference to associate PPE gear items with detected workers.
- ⚖️ **Safety Compliance & Risk Evaluation**:
  - 🟢 **COMPLIANT**: Worker equipped with both Helmet AND Vest.
  - 🔴 **VIOLATION**: Worker missing protective gear or directly flagged with `No-Helmet` / `No-Vest`.
  - 📊 **Risk Level Rating**: Automatically assesses overall scene risk (`LOW`, `MEDIUM`, `HIGH`).
- 🎨 **Visual Annotations**: Overlays green 🟢 bounding boxes for compliant workers and red 🔴 boxes detailing missing gear for violators.
- ⚡ **FastAPI RESTful Service**: Exposes API endpoint for image processing (`/predict`).
- 🖥️ **Glassmorphism Web Dashboard**: Modern responsive Dark Mode UI featuring drag-and-drop file upload, live confidence threshold slider, real-time KPI summary cards, and a detailed worker compliance breakdown table.

---

## 🏗️ System Architecture

```text
[ Input: Image ]
          │
          ▼
[ FastAPI Server (app/main.py) ]
          │
          ▼
[ YOLO26s Inference Engine (src/inference.py) ]
  ├── Detections: Person, Helmet, Vest, No-Helmet, No-Vest
          │
          ▼
[ Spatial Logic Engine (src/violation.py) ]
  ├── Spatial IoU Overlap Matching & Gear Association per Person
  └── Fallback Synthetic Person Box Generation
          │
          ▼
[ Visual Annotator (src/visualization.py) ]
  └── Renders Bounding Boxes & Encodes Output to Base64 JPEG
          │
          ▼
[ Web Dashboard (frontend/) ]
  └── Displays KPI Summary Cards, Image Preview & Breakdown Table
```

---

## 📁 Repository Structure

```text
ppe-monitoring-system/
├── app/
│   ├── config.py           # Model paths, CORS origins, default thresholds
│   └── main.py             # FastAPI app endpoints (/predict, /app)
├── src/
│   ├── __init__.py
│   ├── inference.py        # PpeDetector class for YOLO model loading & inference
│   ├── violation.py        # Safety logic engine & Spatial IoU matching
│   └── visualization.py    # Custom bounding box visualizer & Base64 encoder
├── frontend/
│   ├── index.html          # Web Dashboard UI HTML layout
│   ├── styles.css          # Glassmorphism Dark Mode styling
│   └── app.js              # Frontend API client & dynamic DOM renderer
├── models/
│   └── best.pt             # Trained YOLO26s PyTorch model weights
├── notebooks/
│   ├── 01_EDA.ipynb        # Exploratory data analysis notebook
│   ├── 02_Preprocessing.ipynb # Data preprocessing & annotation formatting
│   ├── 03_Training.ipynb   # Model training pipeline notebook
│   └── 04_Evaluation.ipynb # Model evaluation metrics notebook
├── scratch/                # Analysis & verification helper scripts
├── requirements.txt        # Python dependency specifications
├── run.py                  # Single-command server startup script
└── README.md               # Project documentation
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python `3.9` or higher
- PyTorch & OpenCV compatible environment (CPU or CUDA GPU)

### 2. Installation
Install all required dependencies:
```bash
pip install -r requirements.txt
```

### 3. Running the Server & Web Dashboard
Launch the application with a single command:
```bash
python run.py
```

### 4. Access in Web Browser
Once running, open your browser to:
- 🖥️ **Web Dashboard**: [http://localhost:8000/app/](http://localhost:8000/app/)
- 📖 **Swagger API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- 🟢 **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

---

## 📡 RESTful API Endpoints

### 1. `GET /health`
Returns the status of the server and current UTC timestamp.

### 2. `POST /predict`
Performs safety compliance analysis on an uploaded image file (`.jpg`, `.png`).

- **Form Data**: `file` (Image upload)
- **Query Params**: `conf` (Confidence threshold, default: `0.35`)
- **Response Sample**:
```json
{
  "compliance_status": "VIOLATION",
  "risk_level": "HIGH",
  "people_count": 2,
  "compliant_count": 1,
  "violation_count": 1,
  "annotated_image": "data:image/jpeg;base64,...",
  "persons": [
    {
      "person_id": 1,
      "box": [120.5, 80.0, 350.2, 700.0],
      "has_helmet": true,
      "has_vest": false,
      "is_compliant": false,
      "detected_items": ["Helmet"],
      "missing_items": ["Vest"]
    }
  ],
  "timestamp": "2026-08-14T13:18:00+00:00"
}
```

