import base64
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import cv2
import numpy as np
from ultralytics import YOLO

from app.config import MODEL_PATH, CONF_THRESHOLD
from src.visualization import draw_ppe_annotations


@dataclass
class Detection:
    box: List[float]  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str


@dataclass
class InferenceResult:
    detections: List[Detection]
    annotated_image_b64: str
    image_height: int
    image_width: int
    raw_image: Optional[np.ndarray] = None


def detection_to_dict(detection: Detection) -> Dict[str, Any]:
    """Convert Detection object to dictionary for JSON serialization."""
    return {
        "box": [round(float(v), 2) for v in detection.box],
        "confidence": round(float(detection.confidence), 4),
        "class_id": int(detection.class_id),
        "class_name": str(detection.class_name),
    }


class PpeDetector:
    def __init__(self, model_path: str = MODEL_PATH):
        """Initialize YOLO model for PPE Monitoring."""
        self.model_path = model_path
        self.model = YOLO(model_path)

    def predict(
        self, image: np.ndarray, conf: float = CONF_THRESHOLD
    ) -> InferenceResult:
        """Execute inference on input image numpy array (BGR)."""
        if image is None or image.size == 0:
            raise ValueError("Input image is invalid or empty.")

        height, width = image.shape[:2]

        # Running YOLO inference
        results = self.model(image, conf=conf)[0]

        detections: List[Detection] = []

        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)

            for box, conf_val, cls_id in zip(boxes, confidences, class_ids):
                cls_name = self.model.names.get(cls_id, f"class_{cls_id}")
                detections.append(
                    Detection(
                        box=[float(x) for x in box],
                        confidence=float(conf_val),
                        class_id=int(cls_id),
                        class_name=str(cls_name),
                    )
                )

        # Plot default YOLO annotated image
        annotated_frame = results.plot()

        # Convert annotated frame to base64 JPEG
        _, buffer = cv2.imencode(".jpg", annotated_frame)
        b64_str = base64.b64encode(buffer).decode("utf-8")
        annotated_image_b64 = f"data:image/jpeg;base64,{b64_str}"

        return InferenceResult(
            detections=detections,
            annotated_image_b64=annotated_image_b64,
            image_height=height,
            image_width=width,
            raw_image=image,
        )

    def predict_from_bytes(
        self, image_bytes: bytes, conf: float = CONF_THRESHOLD
    ) -> InferenceResult:
        """Decode raw byte array from uploaded file and perform prediction."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image data from bytes.")
        return self.predict(image, conf=conf)
