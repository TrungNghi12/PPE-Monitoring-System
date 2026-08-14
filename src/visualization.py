import base64
from typing import List, Any
import cv2
import numpy as np


def draw_ppe_annotations(
    image: np.ndarray, detections: List[Any], compliance_result: Any
) -> str:
    """Draw high-quality bounding boxes and visual labels based on compliance evaluation (🟢 Compliant / 🔴 Violation).

    Returns a Base64 string formatted as `data:image/jpeg;base64,...`
    """
    annotated = image.copy()
    height, width = annotated.shape[:2]

    # BGR Color definitions
    COLOR_COMPLIANT = (76, 217, 100)  # Smooth Green
    COLOR_VIOLATION = (59, 59, 238)  # Bright Red
    COLOR_GEAR = (240, 180, 41)  # Orange/Yellow
    COLOR_TEXT = (255, 255, 255)  # White

    # 1. Draw protective gear bounding boxes first (Helmet, Vest, No-Helmet, No-Vest)
    for d in detections:
        cls_name = getattr(d, "class_name", d.get("class_name") if isinstance(d, dict) else "")
        box = getattr(d, "box", d.get("box") if isinstance(d, dict) else [])
        conf = getattr(d, "confidence", d.get("confidence", 0.0) if isinstance(d, dict) else 0.0)

        if not box or len(box) < 4:
            continue

        if cls_name.lower() == "person":
            continue

        x1, y1, x2, y2 = [int(v) for v in box[:4]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)

        # Color selection for gear
        if cls_name.lower() in ["no-helmet", "no-vest"]:
            gear_color = COLOR_VIOLATION
        else:
            gear_color = COLOR_GEAR

        # Draw gear bounding box border
        cv2.rectangle(annotated, (x1, y1), (x2, y2), gear_color, 2)

        # Label tag for gear
        label = f"{cls_name} {int(conf * 100)}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(annotated, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), gear_color, -1)
        cv2.putText(
            annotated,
            label,
            (x1 + 3, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            COLOR_TEXT,
            1,
            cv2.LINE_AA,
        )

    # 2. Draw Person bounding boxes based on compliance_result
    if hasattr(compliance_result, "persons") and compliance_result.persons:
        for p in compliance_result.persons:
            box = p.box
            if not box or len(box) < 4:
                continue

            x1, y1, x2, y2 = [int(v) for v in box[:4]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)

            if p.is_compliant:
                box_color = COLOR_COMPLIANT
                status_text = f"PASSED: FULL PPE (#{p.person_id})"
            else:
                box_color = COLOR_VIOLATION
                missing_str = ", ".join(p.missing_items) if p.missing_items else "PPE"
                status_text = f"VIOLATION: Missing {missing_str} (#{p.person_id})"

            # Draw prominent Person bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 3)

            # Label banner above box
            (tw, th), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            banner_y1 = max(0, y1 - th - 10)
            banner_y2 = y1
            cv2.rectangle(annotated, (x1, banner_y1), (min(width, x1 + tw + 12), banner_y2), box_color, -1)
            cv2.putText(
                annotated,
                status_text,
                (x1 + 6, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                COLOR_TEXT,
                2,
                cv2.LINE_AA,
            )

    # Encode output image to Base64 JPEG
    _, buffer = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    b64_str = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"
