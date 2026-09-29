"""
Visualization utilities for Pose-Aware PPE Compliance Detection.

Output behavior
---------------

DETECTED PPE
    Green bounding box around the actual Roboflow detection.
    Includes the Roboflow confidence score.

MISSING HARD HAT
    Blue box around the pose-derived expected head region.

MISSING SAFETY VEST
    Red box around the pose-derived expected torso region.

MULTI-PERSON SCENES
    Labels are prefixed P1, P2, P3, etc.

Examples:

    Hard Hat 0.94
    Safety Vest 0.91

    Missing Hard Hat
    Missing Safety Vest

Multi-person:

    P1 Hard Hat 0.94
    P1 Safety Vest 0.91

    P2 Missing Hard Hat
    P2 Safety Vest 0.88
"""

import cv2


# ============================================================
# COLORS
# ============================================================
#
# OpenCV uses BGR, not RGB.
# ============================================================

COLORS = {

    # Actual PPE detection
    "detected": (
        0,
        180,
        0,
    ),

    # Pose-derived missing hard-hat region
    "missing_hard_hat": (
        255,
        0,
        0,
    ),

    # Pose-derived missing vest region
    "missing_safety_vest": (
        0,
        0,
        255,
    ),

    # Reserved
    "unknown": (
        0,
        165,
        255,
    ),
}


# ============================================================
# LABEL COLLISION
# ============================================================

def rectangles_overlap(
    a,
    b,
    padding=2,
):

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    return not (
        ax2 + padding < bx1
        or bx2 + padding < ax1
        or ay2 + padding < by1
        or by2 + padding < ay1
    )


# ============================================================
# BOX CLAMPING
# ============================================================

def clamp_box(
    box,
    width,
    height,
):

    x1, y1, x2, y2 = map(
        int,
        box,
    )

    x1 = max(
        0,
        min(
            x1,
            width - 1,
        ),
    )

    y1 = max(
        0,
        min(
            y1,
            height - 1,
        ),
    )

    x2 = max(
        0,
        min(
            x2,
            width - 1,
        ),
    )

    y2 = max(
        0,
        min(
            y2,
            height - 1,
        ),
    )

    return (
        x1,
        y1,
        x2,
        y2,
    )


# ============================================================
# LABEL POSITIONING
# ============================================================

def choose_label_rectangle(
    box,
    label_width,
    label_height,
    image_width,
    image_height,
    occupied,
):
    """
    Find a label position that avoids overlapping existing
    labels whenever possible.
    """

    x1, y1, x2, y2 = box

    candidates = [

        # Above - right aligned
        (
            x2 - label_width,
            y1 - label_height,
            x2,
            y1,
        ),

        # Above - left aligned
        (
            x1,
            y1 - label_height,
            x1 + label_width,
            y1,
        ),

        # Inside - top right
        (
            x2 - label_width,
            y1,
            x2,
            y1 + label_height,
        ),

        # Inside - top left
        (
            x1,
            y1,
            x1 + label_width,
            y1 + label_height,
        ),

        # Below - right aligned
        (
            x2 - label_width,
            y2,
            x2,
            y2 + label_height,
        ),

        # Below - left aligned
        (
            x1,
            y2,
            x1 + label_width,
            y2 + label_height,
        ),
    ]

    # --------------------------------------------------------
    # TRY PREFERRED LOCATIONS
    # --------------------------------------------------------

    for candidate in candidates:

        lx1, ly1, lx2, ly2 = (
            candidate
        )

        if (
            lx1 < 0
            or ly1 < 0
            or lx2 >= image_width
            or ly2 >= image_height
        ):
            continue

        collision = any(
            rectangles_overlap(
                candidate,
                existing,
                padding=3,
            )
            for existing in occupied
        )

        if not collision:
            return candidate

    # --------------------------------------------------------
    # SEARCH FOR AN OPEN LOCATION
    # --------------------------------------------------------

    start_x = max(
        0,
        min(
            x2 - label_width,
            image_width
            - label_width
            - 1,
        ),
    )

    for y in range(
        0,
        max(
            1,
            image_height
            - label_height,
        ),
        label_height + 2,
    ):

        candidate = (
            start_x,
            y,
            start_x + label_width,
            y + label_height,
        )

        collision = any(
            rectangles_overlap(
                candidate,
                existing,
                padding=3,
            )
            for existing in occupied
        )

        if not collision:
            return candidate

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    return (
        max(
            0,
            x1,
        ),
        max(
            0,
            y1,
        ),
        min(
            image_width - 1,
            x1 + label_width,
        ),
        min(
            image_height - 1,
            y1 + label_height,
        ),
    )


# ============================================================
# DRAW ANNOTATION
# ============================================================

def draw_reference_annotation(
    image,
    box,
    text,
    color,
    occupied_labels,
):
    """
    Draw:

        thin bounding box
        filled label tab
        white text
        collision-aware label
    """

    if box is None:
        return

    image_height, image_width = (
        image.shape[:2]
    )

    x1, y1, x2, y2 = clamp_box(
        box,
        image_width,
        image_height,
    )

    # --------------------------------------------------------
    # BOX
    # --------------------------------------------------------

    cv2.rectangle(
        image,
        (
            x1,
            y1,
        ),
        (
            x2,
            y2,
        ),
        color,
        2,
        cv2.LINE_AA,
    )

    # --------------------------------------------------------
    # TEXT
    # --------------------------------------------------------

    font = (
        cv2.FONT_HERSHEY_SIMPLEX
    )

    font_scale = 0.42
    font_thickness = 1

    (
        text_width,
        text_height,
    ), baseline = (
        cv2.getTextSize(
            text,
            font,
            font_scale,
            font_thickness,
        )
    )

    pad_x = 5
    pad_y = 3

    label_width = (
        text_width
        + pad_x * 2
    )

    label_height = (
        text_height
        + baseline
        + pad_y * 2
    )

    label_width = min(
        label_width,
        image_width - 1,
    )

    # --------------------------------------------------------
    # CHOOSE LABEL POSITION
    # --------------------------------------------------------

    label_rect = (
        choose_label_rectangle(
            (
                x1,
                y1,
                x2,
                y2,
            ),
            label_width,
            label_height,
            image_width,
            image_height,
            occupied_labels,
        )
    )

    lx1, ly1, lx2, ly2 = (
        label_rect
    )

    # --------------------------------------------------------
    # FILLED LABEL TAB
    # --------------------------------------------------------

    cv2.rectangle(
        image,
        (
            lx1,
            ly1,
        ),
        (
            lx2,
            ly2,
        ),
        color,
        -1,
        cv2.LINE_AA,
    )

    # --------------------------------------------------------
    # WHITE LABEL TEXT
    # --------------------------------------------------------

    text_x = (
        lx1
        + pad_x
    )

    text_y = (
        ly1
        + pad_y
        + text_height
    )

    cv2.putText(
        image,
        text,
        (
            text_x,
            text_y,
        ),
        font,
        font_scale,
        (
            255,
            255,
            255,
        ),
        font_thickness,
        cv2.LINE_AA,
    )

    occupied_labels.append(
        label_rect
    )


# ============================================================
# LABEL PREFIX
# ============================================================

def person_prefix(
    person_id,
    multiple_people,
):

    if not multiple_people:
        return ""

    return (
        f"P{person_id} "
    )


# ============================================================
# FORMAT CONFIDENCE
# ============================================================

def confidence_text(
    detection,
):

    if not detection:
        return ""

    confidence = detection.get(
        "confidence"
    )

    if confidence is None:
        return ""

    try:

        return (
            f" {float(confidence):.2f}"
        )

    except (
        TypeError,
        ValueError,
    ):

        return ""


# ============================================================
# DRAW COMPLETE RESULT
# ============================================================

def draw_results(
    image,
    report,
):
    """
    Render one pose backend's compliance results.

    Expected report:

        {
            "people": [...]
        }

    Each person contains:

        person_id

        ppe:
            hard_hat:
                status
                expected_region
                matched_detection

            safety_vest:
                status
                expected_region
                matched_detection
    """

    canvas = (
        image.copy()
    )

    occupied_labels = []

    people = report.get(
        "people",
        [],
    )

    multiple_people = (
        len(people) > 1
    )

    # ========================================================
    # PROCESS EACH PERSON
    # ========================================================

    for person in people:

        person_id = person.get(
            "person_id",
            "?",
        )

        prefix = (
            person_prefix(
                person_id,
                multiple_people,
            )
        )

        ppe = person.get(
            "ppe",
            {},
        )

        # ====================================================
        # HARD HAT
        # ====================================================

        hard_hat = ppe.get(
            "hard_hat",
            {},
        )

        hard_hat_status = (
            hard_hat.get(
                "status",
                "unknown",
            )
        )

        # ----------------------------------------------------
        # HARD HAT DETECTED
        # ----------------------------------------------------

        if (
            hard_hat_status
            == "detected"
        ):

            detection = (
                hard_hat.get(
                    "matched_detection"
                )
            )

            if detection:

                text = (
                    prefix
                    + "Hard Hat"
                    + confidence_text(
                        detection
                    )
                )

                draw_reference_annotation(
                    canvas,
                    detection.get(
                        "box"
                    ),
                    text,
                    COLORS[
                        "detected"
                    ],
                    occupied_labels,
                )

        # ----------------------------------------------------
        # HARD HAT MISSING
        # ----------------------------------------------------

        elif (
            hard_hat_status
            == "missing"
        ):

            expected_region = (
                hard_hat.get(
                    "expected_region"
                )
            )

            # Missing is only visualized when the compliance
            # system produced a valid expected region.
            if (
                expected_region
                is not None
            ):

                text = (
                    prefix
                    + "Missing Hard Hat"
                )

                draw_reference_annotation(
                    canvas,
                    expected_region,
                    text,
                    COLORS[
                        "missing_hard_hat"
                    ],
                    occupied_labels,
                )

        # ====================================================
        # SAFETY VEST
        # ====================================================

        vest = ppe.get(
            "safety_vest",
            {},
        )

        vest_status = (
            vest.get(
                "status",
                "unknown",
            )
        )

        # ----------------------------------------------------
        # VEST DETECTED
        # ----------------------------------------------------

        if (
            vest_status
            == "detected"
        ):

            detection = (
                vest.get(
                    "matched_detection"
                )
            )

            if detection:

                text = (
                    prefix
                    + "Safety Vest"
                    + confidence_text(
                        detection
                    )
                )

                draw_reference_annotation(
                    canvas,
                    detection.get(
                        "box"
                    ),
                    text,
                    COLORS[
                        "detected"
                    ],
                    occupied_labels,
                )

        # ----------------------------------------------------
        # VEST MISSING
        # ----------------------------------------------------

        elif (
            vest_status
            == "missing"
        ):

            expected_region = (
                vest.get(
                    "expected_region"
                )
            )

            if (
                expected_region
                is not None
            ):

                text = (
                    prefix
                    + "Missing Safety Vest"
                )

                draw_reference_annotation(
                    canvas,
                    expected_region,
                    text,
                    COLORS[
                        "missing_safety_vest"
                    ],
                    occupied_labels,
                )

    return canvas