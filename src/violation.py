from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class PersonCompliance:
    person_id: int
    box: List[float]
    has_helmet: bool
    has_vest: bool
    is_compliant: bool
    detected_items: List[str]
    missing_items: List[str]


@dataclass
class ComplianceResult:
    compliance_status: str  # "COMPLIANT", "VIOLATION", "NO_PERSON_DETECTED"
    risk_level: str  # "LOW", "MEDIUM", "HIGH"
    people_count: int
    compliant_count: int
    violation_count: int
    persons: List[PersonCompliance] = field(default_factory=list)


def is_gear_inside_person(
    gear_box: List[float], person_box: List[float], overlap_thresh: float = 0.35
) -> bool:
    """Check if protective gear bounding box (gear_box) is inside worker region (person_box)."""
    gx1, gy1, gx2, gy2 = gear_box
    px1, py1, px2, py2 = person_box

    # Intersection coordinates
    ix1 = max(gx1, px1)
    iy1 = max(gy1, py1)
    ix2 = min(gx2, px2)
    iy2 = min(gy2, py2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    intersection_area = iw * ih

    gear_area = max(1.0, (gx2 - gx1) * (gy2 - gy1))

    # Intersection ratio over gear area
    overlap_ratio = intersection_area / gear_area

    # Gear center point
    cx = (gx1 + gx2) / 2.0
    cy = (gy1 + gy2) / 2.0

    margin_y = max(15.0, (py2 - py1) * 0.20)  # Top margin extension for helmet
    center_inside = (px1 <= cx <= px2) and ((py1 - margin_y) <= cy <= (py2 + margin_y))

    return overlap_ratio >= overlap_thresh or center_inside


def infer_person_box_from_gear(gear_box: List[float], class_name: str) -> List[float]:
    """Infer worker bounding box based on detected protective gear item.

    Example: If Vest is detected, extend top margin for head/helmet and bottom margin for torso/legs.
    """
    gx1, gy1, gx2, gy2 = gear_box
    gw = gx2 - gx1
    gh = gy2 - gy1

    cname = class_name.lower()

    if "vest" in cname:
        # Vest is upper torso: extend top margin 75% for head, bottom margin 160% for legs
        px1 = max(0.0, gx1 - 0.25 * gw)
        py1 = max(0.0, gy1 - 0.75 * gh)
        px2 = gx2 + 0.25 * gw
        py2 = gy2 + 1.60 * gh
    elif "helmet" in cname:
        # Helmet is head: extend bottom margin 500% for full body
        px1 = max(0.0, gx1 - 0.75 * gw)
        py1 = max(0.0, gy1 - 0.10 * gh)
        px2 = gx2 + 0.75 * gw
        py2 = gy2 + 5.00 * gh
    else:
        # Default fallback
        px1 = max(0.0, gx1 - 0.3 * gw)
        py1 = max(0.0, gy1 - 0.5 * gh)
        px2 = gx2 + 0.3 * gw
        py2 = gy2 + 2.0 * gh

    return [px1, py1, px2, py2]


def evaluate_compliance(detections: List[Any]) -> ComplianceResult:
    """Evaluate PPE compliance status based on YOLO detections list.

    Automatically infers worker position even when YOLO misses the 'Person' class.
    """
    person_boxes: List[List[float]] = []
    gear_detections: List[Any] = []

    # Categorize detections
    for d in detections:
        cls_name = getattr(d, "class_name", d.get("class_name") if isinstance(d, dict) else "")
        box = getattr(d, "box", d.get("box") if isinstance(d, dict) else [])

        if not box or len(box) < 4:
            continue

        if cls_name.lower() == "person":
            person_boxes.append(box)
        else:
            gear_detections.append(d)

    # If YOLO missed Person box but detected Vest/Helmet/No-Helmet/No-Vest
    for g in gear_detections:
        g_box = getattr(g, "box", g.get("box") if isinstance(g, dict) else [])
        g_cls = getattr(g, "class_name", g.get("class_name") if isinstance(g, dict) else "")

        # Check if gear_box is already inside an existing Person box
        matched = any(is_gear_inside_person(g_box, p_box) for p_box in person_boxes)

        if not matched:
            # Generate synthetic inferred Person box based on gear location
            inferred_pbox = infer_person_box_from_gear(g_box, g_cls)
            person_boxes.append(inferred_pbox)

    if not person_boxes:
        return ComplianceResult(
            compliance_status="NO_PERSON_DETECTED",
            risk_level="LOW",
            people_count=0,
            compliant_count=0,
            violation_count=0,
            persons=[],
        )

    persons_compliance: List[PersonCompliance] = []
    compliant_count = 0
    violation_count = 0

    for idx, p_box in enumerate(person_boxes, 1):
        detected_items = []
        missing_items = []

        # Filter protective gear belonging to this worker
        matched_gears = [
            g
            for g in gear_detections
            if is_gear_inside_person(
                getattr(g, "box", g.get("box") if isinstance(g, dict) else []), p_box
            )
        ]

        matched_names = [
            getattr(g, "class_name", g.get("class_name") if isinstance(g, dict) else "").lower()
            for g in matched_gears
        ]

        # 1. Check Helmet
        has_no_helmet = "no-helmet" in matched_names
        has_helmet = "helmet" in matched_names and not has_no_helmet

        if has_helmet:
            detected_items.append("Helmet")
        else:
            missing_items.append("Helmet")

        # 2. Check Safety Vest
        has_no_vest = "no-vest" in matched_names
        has_vest = "vest" in matched_names and not has_no_vest

        if has_vest:
            detected_items.append("Vest")
        else:
            missing_items.append("Vest")

        # Compliance condition: Must have Helmet AND Vest, and no negative flags (No-Helmet / No-Vest)
        is_compliant = has_helmet and has_vest

        if is_compliant:
            compliant_count += 1
        else:
            violation_count += 1

        persons_compliance.append(
            PersonCompliance(
                person_id=idx,
                box=p_box,
                has_helmet=has_helmet,
                has_vest=has_vest,
                is_compliant=is_compliant,
                detected_items=detected_items,
                missing_items=missing_items,
            )
        )

    # Overall summary stats
    total_people = len(person_boxes)

    if violation_count == 0:
        overall_status = "COMPLIANT"
        risk_level = "LOW"
    else:
        overall_status = "VIOLATION"
        violation_ratio = violation_count / total_people
        risk_level = "HIGH" if violation_ratio >= 0.5 else "MEDIUM"

    return ComplianceResult(
        compliance_status=overall_status,
        risk_level=risk_level,
        people_count=total_people,
        compliant_count=compliant_count,
        violation_count=violation_count,
        persons=persons_compliance,
    )
