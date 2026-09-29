"""
PPE compliance matching logic.

This module determines whether Roboflow-detected PPE objects
correspond to the anatomical PPE regions generated from a
pose estimator.

Important design rules:

1. Roboflow is authoritative for Person, Hard Hat, and
   Safety Vest detections.

2. MediaPipe Pose and YOLO Pose provide anatomical
   information only.

3. "Missing" PPE may only be inferred when a valid
   pose-derived expected region exists.

4. If no valid pose-derived region exists, PPE status is
   "unknown", not "missing".

5. The same matching logic is used for MediaPipe Pose and
   YOLO Pose so the pose estimator remains the experimental
   variable.
"""

from .iou import (
    intersection_over_union,
    intersection_over_detection,
    center_in_box,
)


# ============================================================
# FIND BEST PPE MATCH
# ============================================================

def best_ppe_match(
    detections,
    expected_region,
    person_box,
    threshold,
):
    """
    Find the Roboflow PPE detection that best corresponds to
    a pose-derived expected PPE region.

    A PPE detection must first belong spatially to the
    authoritative Roboflow Person box.

    Matching considers:

        1. IoU with expected region
        2. Detection containment in expected region
        3. Detection center inside expected region

    Returns:
        best matching detection dictionary

    or:
        None
    """

    if expected_region is None:
        return None

    if person_box is None:
        return None

    best_match = None
    best_score = 0.0

    for detection in detections:

        detection_box = (
            detection.get("box")
        )

        if detection_box is None:
            continue

        # ----------------------------------------------------
        # PPE MUST BELONG TO THIS PERSON
        # ----------------------------------------------------
        #
        # The center of the PPE detection must fall inside
        # the authoritative Roboflow Person bounding box.
        # This reduces accidental assignment of PPE belonging
        # to neighboring workers.
        # ----------------------------------------------------

        if not center_in_box(
            detection_box,
            person_box,
        ):
            continue

        # ----------------------------------------------------
        # SPATIAL MATCH METRICS
        # ----------------------------------------------------

        iou = (
            intersection_over_union(
                detection_box,
                expected_region,
            )
        )

        containment = (
            intersection_over_detection(
                detection_box,
                expected_region,
            )
        )

        center_match = (
            center_in_box(
                detection_box,
                expected_region,
            )
        )

        # ----------------------------------------------------
        # COMBINED MATCH SCORE
        # ----------------------------------------------------

        score = max(
            iou,
            containment,
        )

        # If the center of the PPE detection lies within the
        # expected anatomical region, allow it to satisfy the
        # threshold even when differently sized boxes result
        # in a relatively low IoU.
        if center_match:

            score = max(
                score,
                threshold,
            )

        # ----------------------------------------------------
        # KEEP BEST VALID MATCH
        # ----------------------------------------------------

        if (
            score >= threshold
            and score > best_score
        ):

            best_score = score

            best_match = {
                **detection,

                "match_score":
                    round(
                        float(score),
                        4,
                    ),

                "iou":
                    round(
                        float(iou),
                        4,
                    ),

                "containment":
                    round(
                        float(containment),
                        4,
                    ),

                "center_in_expected_region":
                    bool(
                        center_match
                    ),
            }

    return best_match


# ============================================================
# EVALUATE ONE PPE TYPE
# ============================================================

def evaluate_ppe_type(
    ppe_name,
    detections,
    expected_region,
    person_box,
    threshold,
):
    """
    Determine detected / missing / unknown status for one
    PPE type.

    Safety rule:

    No expected pose-derived region
        -> UNKNOWN

    Expected region + matching Roboflow PPE
        -> DETECTED

    Expected region + no matching Roboflow PPE
        -> MISSING
    """

    # --------------------------------------------------------
    # NO VALID ANATOMICAL REGION
    # --------------------------------------------------------

    if expected_region is None:

        return {
            "status":
                "unknown",

            "reason":
                (
                    "person_detected_but_"
                    "pose_region_unavailable"
                ),

            "expected_region":
                None,

            "matched_detection":
                None,
        }

    # --------------------------------------------------------
    # LOOK FOR ROBOFLOW PPE
    # --------------------------------------------------------

    match = best_ppe_match(
        detections=detections,

        expected_region=(
            expected_region
        ),

        person_box=(
            person_box
        ),

        threshold=(
            threshold
        ),
    )

    # --------------------------------------------------------
    # PPE DETECTED
    # --------------------------------------------------------

    if match is not None:

        return {
            "status":
                "detected",

            "reason":
                (
                    "roboflow_detection_"
                    "matched_expected_region"
                ),

            "expected_region":
                expected_region,

            "matched_detection":
                match,
        }

    # --------------------------------------------------------
    # PPE MISSING
    # --------------------------------------------------------
    #
    # We only reach this state because a valid anatomical
    # expected region exists.
    # --------------------------------------------------------

    return {
        "status":
            "missing",

        "reason":
            (
                "valid_pose_region_but_no_"
                "matching_roboflow_detection"
            ),

        "expected_region":
            expected_region,

        "matched_detection":
            None,
    }


# ============================================================
# EVALUATE ONE PERSON
# ============================================================

def evaluate_person(
    person_id,
    person_detection,
    pose,
    regions,
    detections_by_class,
    cfg,
):
    """
    Evaluate PPE compliance for ONE authoritative Roboflow
    Person detection.

    This interface is intentionally shared by both pose
    backends.

    Args:

        person_id:
            Stable sequential ID assigned by infer.py.

        person_detection:
            Authoritative Roboflow Person detection.

        pose:
            Pose result mapped back to original image
            coordinates, or None.

        regions:
            Pose-derived expected PPE regions.

        detections_by_class:
            Roboflow detections grouped into:
                person
                hard_hat
                safety_vest

        cfg:
            Project configuration.

    Returns:

        Dictionary containing the Person detection,
        pose availability, expected regions, and PPE status.
    """

    person_box = (
        person_detection.get(
            "box"
        )
    )

    pose_available = (
        pose is not None
    )

    # --------------------------------------------------------
    # EXPECTED REGIONS
    # --------------------------------------------------------

    hard_hat_region = (
        regions.get(
            "hard_hat"
        )
        if regions
        else None
    )

    safety_vest_region = (
        regions.get(
            "safety_vest"
        )
        if regions
        else None
    )

    # --------------------------------------------------------
    # MATCHING THRESHOLDS
    # --------------------------------------------------------

    matching_cfg = (
        cfg.get(
            "matching",
            {},
        )
    )

    hard_hat_threshold = (
        matching_cfg.get(
            "helmet_iou_threshold",
            0.10,
        )
    )

    safety_vest_threshold = (
        matching_cfg.get(
            "vest_iou_threshold",
            0.10,
        )
    )

    # --------------------------------------------------------
    # HARD HAT
    # --------------------------------------------------------

    hard_hat_result = (
        evaluate_ppe_type(
            ppe_name="hard_hat",

            detections=(
                detections_by_class.get(
                    "hard_hat",
                    [],
                )
            ),

            expected_region=(
                hard_hat_region
            ),

            person_box=(
                person_box
            ),

            threshold=(
                hard_hat_threshold
            ),
        )
    )

    # --------------------------------------------------------
    # SAFETY VEST
    # --------------------------------------------------------

    safety_vest_result = (
        evaluate_ppe_type(
            ppe_name="safety_vest",

            detections=(
                detections_by_class.get(
                    "safety_vest",
                    [],
                )
            ),

            expected_region=(
                safety_vest_region
            ),

            person_box=(
                person_box
            ),

            threshold=(
                safety_vest_threshold
            ),
        )
    )

    # --------------------------------------------------------
    # PERSON RESULT
    # --------------------------------------------------------

    return {
        "person_id":
            person_id,

        "person_detection":
            person_detection,

        "person_box":
            person_box,

        "person_source":
            "Roboflow Workflow",

        "pose_available":
            pose_available,

        "regions": {
            "hard_hat":
                hard_hat_region,

            "safety_vest":
                safety_vest_region,
        },

        "ppe": {
            "hard_hat":
                hard_hat_result,

            "safety_vest":
                safety_vest_result,
        },
    }