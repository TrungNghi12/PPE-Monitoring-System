from datetime import datetime, timezone
from typing import List, Dict, Any
import tempfile
import os
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import CORS_ORIGINS, MODEL_PATH, CONF_THRESHOLD
from src.inference import PpeDetector, detection_to_dict
from src.violation import evaluate_compliance
from src.visualization import draw_ppe_annotations

app = FastAPI(
    title="PPE Monitoring System API",
    description="API for inspecting personal protective equipment (Helmet, Vest) using YOLO26s & Spatial Logic Engine",
    version="1.0.0",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize PpeDetector
detector = PpeDetector(MODEL_PATH)

# Mount static files for Frontend Web Dashboard
if os.path.exists("frontend"):
    from fastapi.staticfiles import StaticFiles
    app.mount("/app", StaticFiles(directory="frontend", html=True), name="frontend")


@app.get("/")
def root():
    """Check status of Server & YOLO Model."""
    return {
        "status": "online",
        "service": "PPE Monitoring System API",
        "model_loaded": True,
        "classes": detector.model.names,
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    conf: float = Query(CONF_THRESHOLD, ge=0.1, le=1.0, description="Confidence threshold"),
):
    """Predict and evaluate safety compliance for uploaded image file.

    - Returns Base64 annotated image (🟢 Compliant / 🔴 Violation)
    - Returns Detections list and detailed person breakdown report.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image file format. Content-Type: {file.content_type}",
        )

    try:
        contents = await file.read()
        inference = detector.predict_from_bytes(contents, conf=conf)
        compliance = evaluate_compliance(inference.detections)

        # Draw custom visual bounding boxes (Green for passed, Red for violation)
        if inference.raw_image is not None:
            custom_annotated_b64 = draw_ppe_annotations(
                inference.raw_image, inference.detections, compliance
            )
        else:
            custom_annotated_b64 = inference.annotated_image_b64

        # Convert persons detail to dict list
        persons_detail = []
        for p in compliance.persons:
            persons_detail.append(
                {
                    "person_id": p.person_id,
                    "box": [round(float(v), 2) for v in p.box],
                    "has_helmet": p.has_helmet,
                    "has_vest": p.has_vest,
                    "is_compliant": p.is_compliant,
                    "detected_items": p.detected_items,
                    "missing_items": p.missing_items,
                }
            )

        return JSONResponse(
            {
                "annotated_image": custom_annotated_b64,
                "detections": [detection_to_dict(d) for d in inference.detections],
                "compliance_status": compliance.compliance_status,
                "risk_level": compliance.risk_level,
                "people_count": compliance.people_count,
                "compliant_count": compliance.compliant_count,
                "violation_count": compliance.violation_count,
                "persons": persons_detail,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing error: {str(e)}")


