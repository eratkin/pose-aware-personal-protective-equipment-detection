"""
infer.py
========

Batch inference and comparison pipeline for the
Pose-Aware PPE Compliance Detection project.

Pipeline:

    Image
      |
      v
    Roboflow Workflow
      |
      +-- Person
      +-- Hard Hat
      +-- Safety Vest
      |
      v
    Roboflow Person Crops
      |
      +----------------------+
      |                      |
      v                      v
    MediaPipe Pose        YOLO Pose
      |                      |
      v                      v
    Expected Regions      Expected Regions
      |                      |
      +----------+-----------+
                 |
                 v
          PPE Spatial Matching
                 |
                 v
        Compare Pose Backends
                 |
                 v
       Images + JSON Reports

Important design decision:

Roboflow establishes which people and PPE objects exist.
YOLO Pose and MediaPipe are used only for anatomical
keypoints / expected PPE regions.

Both pose backends receive the same Roboflow person crops
and use the same downstream compliance logic.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2

from config import load_config, ROOT

from ppe.ppe_detector import PPEDetector

from pose.yolo_pose import YOLOPoseBackend
from pose.mediapipe_pose import MediaPipePoseBackend

from matching.regions import expected_regions
from matching.compliance import evaluate_person

from visualization import draw_results


# ============================================================
# SUPPORTED INPUT TYPES
# ============================================================

SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# PERSON-CROP SETTINGS
# ============================================================

# Very small Roboflow Person detections may not contain enough
# information for reliable pose estimation.
MIN_PERSON_WIDTH = 25
MIN_PERSON_HEIGHT = 50

# Add context around each Roboflow Person bounding box.
CROP_PADDING_RATIO = 0.15

# Small person crops are enlarged before being passed to
# MediaPipe and YOLO Pose.
#
# BOTH pose backends receive the same processed crop.
MIN_POSE_CROP_HEIGHT = 320


# ============================================================
# DETECTION HELPERS
# ============================================================

def group_detections(detections):
    """
    Group normalized Roboflow detections by canonical class.

    Expected canonical classes:

        person
        hard_hat
        safety_vest
    """

    grouped = {
        "person": [],
        "hard_hat": [],
        "safety_vest": [],
    }

    for detection in detections:

        class_name = detection.get(
            "class"
        )

        if class_name in grouped:

            grouped[
                class_name
            ].append(
                detection
            )

    return grouped


# ============================================================
# INPUT DISCOVERY
# ============================================================

def find_images(input_path):
    """
    Accept either a single image:

        input/image1.png

    or an entire directory:

        input/

    Directory processing is intentionally non-recursive so
    output images accidentally placed in subdirectories are
    not picked up as new inputs.
    """

    path = Path(
        input_path
    )

    if not path.exists():

        raise FileNotFoundError(
            "Input path does not exist: "
            f"{path}"
        )

    # --------------------------------------------------------
    # SINGLE IMAGE
    # --------------------------------------------------------

    if path.is_file():

        if (
            path.suffix.lower()
            not in SUPPORTED_IMAGE_EXTENSIONS
        ):

            raise ValueError(
                "Unsupported image format: "
                f"{path.suffix}"
            )

        return [
            path
        ]

    # --------------------------------------------------------
    # DIRECTORY
    # --------------------------------------------------------

    images = sorted(
        [
            file
            for file in path.iterdir()
            if (
                file.is_file()
                and file.suffix.lower()
                in SUPPORTED_IMAGE_EXTENSIONS
            )
        ],
        key=lambda file: file.name.lower(),
    )

    if not images:

        raise FileNotFoundError(
            "No supported images were found in: "
            f"{path}"
        )

    return images


# ============================================================
# PERSON CROPPING
# ============================================================

def padded_crop(
    image,
    box,
    padding_ratio=CROP_PADDING_RATIO,
):
    """
    Create a padded crop around one authoritative
    Roboflow Person detection.

    Returns:

        crop
        metadata

    metadata is later used to translate pose coordinates
    back into the original image coordinate system.
    """

    image_height, image_width = (
        image.shape[:2]
    )

    x1, y1, x2, y2 = box

    person_width = (
        x2 - x1
    )

    person_height = (
        y2 - y1
    )

    padding_x = (
        person_width
        * padding_ratio
    )

    padding_y = (
        person_height
        * padding_ratio
    )

    crop_x1 = max(
        0,
        int(
            x1 - padding_x
        ),
    )

    crop_y1 = max(
        0,
        int(
            y1 - padding_y
        ),
    )

    crop_x2 = min(
        image_width,
        int(
            x2 + padding_x
        ),
    )

    crop_y2 = min(
        image_height,
        int(
            y2 + padding_y
        ),
    )

    crop = image[
        crop_y1:crop_y2,
        crop_x1:crop_x2,
    ].copy()

    metadata = {

        "x_offset":
            crop_x1,

        "y_offset":
            crop_y1,

        "original_width":
            crop_x2 - crop_x1,

        "original_height":
            crop_y2 - crop_y1,

        "scale":
            1.0,
    }

    return (
        crop,
        metadata,
    )


# ============================================================
# UPSCALE SMALL PERSON CROPS
# ============================================================

def upscale_pose_crop(
    crop,
    metadata,
):
    """
    Enlarge small person crops before pose estimation.

    This is especially useful for workers who occupy a small
    portion of a larger construction-site image.

    The scale factor is retained so keypoints can later be
    mapped back into the original image.
    """

    height, width = (
        crop.shape[:2]
    )

    if (
        height <= 0
        or width <= 0
    ):

        return (
            crop,
            metadata,
        )

    # No enlargement required.
    if (
        height
        >= MIN_POSE_CROP_HEIGHT
    ):

        return (
            crop,
            metadata,
        )

    scale = (
        MIN_POSE_CROP_HEIGHT
        / float(height)
    )

    new_width = max(
        1,
        int(
            width * scale
        ),
    )

    new_height = max(
        1,
        int(
            height * scale
        ),
    )

    resized = cv2.resize(
        crop,
        (
            new_width,
            new_height,
        ),
        interpolation=cv2.INTER_CUBIC,
    )

    metadata["scale"] = (
        scale
    )

    return (
        resized,
        metadata,
    )


# ============================================================
# POSE COORDINATE TRANSLATION
# ============================================================

def translate_pose_to_image(
    pose,
    metadata,
):
    """
    Translate pose coordinates from the processed person crop
    back into the coordinate system of the original image.
    """

    if pose is None:
        return None

    x_offset = metadata[
        "x_offset"
    ]

    y_offset = metadata[
        "y_offset"
    ]

    scale = metadata.get(
        "scale",
        1.0,
    )

    if scale <= 0:
        scale = 1.0

    translated_points = {}

    # --------------------------------------------------------
    # KEYPOINTS
    # --------------------------------------------------------

    for (
        name,
        point,
    ) in pose.get(
        "points",
        {},
    ).items():

        if (
            point is None
            or len(point) < 3
        ):
            continue

        x, y, confidence = (
            point
        )

        original_x = (
            x / scale
            + x_offset
        )

        original_y = (
            y / scale
            + y_offset
        )

        translated_points[
            name
        ] = (
            original_x,
            original_y,
            confidence,
        )

    # --------------------------------------------------------
    # POSE BOUNDING BOX
    # --------------------------------------------------------

    translated_box = None

    if (
        pose.get("box")
        is not None
    ):

        x1, y1, x2, y2 = (
            pose["box"]
        )

        translated_box = [

            x1 / scale
            + x_offset,

            y1 / scale
            + y_offset,

            x2 / scale
            + x_offset,

            y2 / scale
            + y_offset,
        ]

    return {

        "id":
            pose.get(
                "id",
                1,
            ),

        "box":
            translated_box,

        "points":
            translated_points,
    }


# ============================================================
# ESTIMATE POSE FOR ONE ROBOFLOW PERSON
# ============================================================

def estimate_pose_for_person(
    image,
    person_detection,
    pose_backend,
):
    """
    Run one pose estimator on one Roboflow-detected person.

    Roboflow remains authoritative for person existence.

    The pose estimator only provides anatomical landmarks.
    """

    box = person_detection[
        "box"
    ]

    x1, y1, x2, y2 = (
        box
    )

    person_width = (
        x2 - x1
    )

    person_height = (
        y2 - y1
    )

    # --------------------------------------------------------
    # MINIMUM SIZE CHECK
    # --------------------------------------------------------

    if (
        person_width
        < MIN_PERSON_WIDTH
        or person_height
        < MIN_PERSON_HEIGHT
    ):

        return (
            None,
            "person_box_too_small",
        )

    # --------------------------------------------------------
    # CREATE PERSON CROP
    # --------------------------------------------------------

    crop, metadata = (
        padded_crop(
            image,
            box,
        )
    )

    if (
        crop is None
        or crop.size == 0
    ):

        return (
            None,
            "empty_person_crop",
        )

    # --------------------------------------------------------
    # UPSCALE IF NEEDED
    # --------------------------------------------------------

    pose_input, metadata = (
        upscale_pose_crop(
            crop,
            metadata,
        )
    )

    # --------------------------------------------------------
    # RUN POSE ESTIMATOR
    # --------------------------------------------------------

    poses = (
        pose_backend.predict(
            pose_input
        )
    )

    if not poses:

        return (
            None,
            "pose_not_detected",
        )

    # Because the crop originated from one Roboflow Person
    # detection, we use the first pose returned by the backend.
    pose = poses[0]

    # --------------------------------------------------------
    # MAP BACK TO ORIGINAL IMAGE
    # --------------------------------------------------------

    translated_pose = (
        translate_pose_to_image(
            pose,
            metadata,
        )
    )

    if translated_pose is None:

        return (
            None,
            "pose_translation_failed",
        )

    return (
        translated_pose,
        None,
    )


# ============================================================
# RUN ONE POSE BACKEND
# ============================================================

def run_pose_backend(
    backend_name,
    pose_backend,
    image,
    person_detections,
    grouped_detections,
    cfg,
):
    """
    Evaluate every Roboflow Person detection using one pose
    backend.

    Both MediaPipe and YOLO Pose call this SAME function.
    """

    people_results = []

    successful_poses = 0

    # --------------------------------------------------------
    # PPE COUNTERS
    # --------------------------------------------------------

    detected_hard_hats = 0
    missing_hard_hats = 0
    unknown_hard_hats = 0

    detected_vests = 0
    missing_vests = 0
    unknown_vests = 0

    # --------------------------------------------------------
    # PROCESS PEOPLE
    # --------------------------------------------------------

    for (
        person_index,
        person_detection,
    ) in enumerate(
        person_detections,
        start=1,
    ):

        pose, failure_reason = (
            estimate_pose_for_person(
                image=image,

                person_detection=(
                    person_detection
                ),

                pose_backend=(
                    pose_backend
                ),
            )
        )

        # ----------------------------------------------------
        # CREATE EXPECTED PPE REGIONS
        # ----------------------------------------------------

        if pose is not None:

            successful_poses += 1

            regions = (
                expected_regions(
                    pose,
                    image.shape,
                    cfg,
                )
            )

        else:

            # No valid anatomical region means we cannot
            # responsibly infer that PPE is missing.
            regions = {
                "hard_hat": None,
                "safety_vest": None,
            }

        # ----------------------------------------------------
        # PPE MATCHING
        # ----------------------------------------------------

        result = evaluate_person(
            person_id=(
                person_index
            ),

            person_detection=(
                person_detection
            ),

            pose=pose,

            regions=regions,

            detections_by_class=(
                grouped_detections
            ),

            cfg=cfg,
        )

        result[
            "pose_backend"
        ] = backend_name

        result[
            "pose_failure_reason"
        ] = failure_reason

        if pose is not None:

            result[
                "pose"
            ] = pose

        people_results.append(
            result
        )

        # ----------------------------------------------------
        # READ DOWNSTREAM PPE STATUS
        # ----------------------------------------------------

        hard_hat_status = (
            result
            .get("ppe", {})
            .get("hard_hat", {})
            .get(
                "status",
                "unknown",
            )
        )

        vest_status = (
            result
            .get("ppe", {})
            .get("safety_vest", {})
            .get(
                "status",
                "unknown",
            )
        )

        # ----------------------------------------------------
        # HARD HAT COUNTERS
        # ----------------------------------------------------

        if (
            hard_hat_status
            == "detected"
        ):

            detected_hard_hats += 1

        elif (
            hard_hat_status
            == "missing"
        ):

            missing_hard_hats += 1

        else:

            unknown_hard_hats += 1

        # ----------------------------------------------------
        # SAFETY VEST COUNTERS
        # ----------------------------------------------------

        if (
            vest_status
            == "detected"
        ):

            detected_vests += 1

        elif (
            vest_status
            == "missing"
        ):

            missing_vests += 1

        else:

            unknown_vests += 1

    # --------------------------------------------------------
    # BACKEND SUMMARY
    # --------------------------------------------------------

    total_people = len(
        person_detections
    )

    pose_success_rate = (
        successful_poses
        / total_people
        if total_people
        else 0.0
    )

    return {

        "summary": {

            "backend":
                backend_name,

            "total_roboflow_people":
                total_people,

            "successful_poses":
                successful_poses,

            "failed_poses":
                total_people
                - successful_poses,

            "pose_success_rate":
                round(
                    pose_success_rate,
                    4,
                ),

            "hard_hat": {

                "detected":
                    detected_hard_hats,

                "missing":
                    missing_hard_hats,

                "unknown":
                    unknown_hard_hats,
            },

            "safety_vest": {

                "detected":
                    detected_vests,

                "missing":
                    missing_vests,

                "unknown":
                    unknown_vests,
            },
        },

        "people":
            people_results,
    }


# ============================================================
# COMPARE MEDIAPIPE AND YOLO POSE
# ============================================================

def compare_backends(
    mediapipe_results,
    yolo_results,
):
    """
    Compare the downstream PPE statuses generated using
    MediaPipe Pose versus YOLO Pose.

    IMPORTANT:

    Agreement is NOT the same thing as accuracy.

    Accuracy requires labeled ground truth.
    """

    mp_people = (
        mediapipe_results[
            "people"
        ]
    )

    yolo_people = (
        yolo_results[
            "people"
        ]
    )

    comparisons = []

    full_agreement_count = 0
    hard_hat_agreement_count = 0
    vest_agreement_count = 0

    total = min(
        len(mp_people),
        len(yolo_people),
    )

    # --------------------------------------------------------
    # PERSON-BY-PERSON COMPARISON
    # --------------------------------------------------------

    for index in range(
        total
    ):

        mp_person = (
            mp_people[index]
        )

        yolo_person = (
            yolo_people[index]
        )

        person_id = (
            mp_person[
                "person_id"
            ]
        )

        # ----------------------------------------------------
        # MEDIAPIPE STATUS
        # ----------------------------------------------------

        mp_hat = (
            mp_person
            .get("ppe", {})
            .get("hard_hat", {})
            .get(
                "status",
                "unknown",
            )
        )

        mp_vest = (
            mp_person
            .get("ppe", {})
            .get("safety_vest", {})
            .get(
                "status",
                "unknown",
            )
        )

        # ----------------------------------------------------
        # YOLO POSE STATUS
        # ----------------------------------------------------

        yolo_hat = (
            yolo_person
            .get("ppe", {})
            .get("hard_hat", {})
            .get(
                "status",
                "unknown",
            )
        )

        yolo_vest = (
            yolo_person
            .get("ppe", {})
            .get("safety_vest", {})
            .get(
                "status",
                "unknown",
            )
        )

        # ----------------------------------------------------
        # AGREEMENT
        # ----------------------------------------------------

        hat_agreement = (
            mp_hat
            == yolo_hat
        )

        vest_agreement = (
            mp_vest
            == yolo_vest
        )

        full_agreement = (
            hat_agreement
            and vest_agreement
        )

        if hat_agreement:

            hard_hat_agreement_count += 1

        if vest_agreement:

            vest_agreement_count += 1

        if full_agreement:

            full_agreement_count += 1

        comparisons.append(
            {
                "person_id":
                    person_id,

                "mediapipe_pose_available":
                    mp_person.get(
                        "pose_available",
                        False,
                    ),

                "yolo_pose_available":
                    yolo_person.get(
                        "pose_available",
                        False,
                    ),

                "hard_hat": {

                    "mediapipe":
                        mp_hat,

                    "yolo_pose":
                        yolo_hat,

                    "agreement":
                        hat_agreement,
                },

                "safety_vest": {

                    "mediapipe":
                        mp_vest,

                    "yolo_pose":
                        yolo_vest,

                    "agreement":
                        vest_agreement,
                },

                "full_ppe_status_agreement":
                    full_agreement,
            }
        )

    # --------------------------------------------------------
    # SAFE RATE CALCULATION
    # --------------------------------------------------------

    def rate(
        value
    ):

        if total == 0:
            return 0.0

        return round(
            value / total,
            4,
        )

    return {

        "people_compared":
            total,

        "hard_hat_agreement_count":
            hard_hat_agreement_count,

        "hard_hat_agreement_rate":
            rate(
                hard_hat_agreement_count
            ),

        "safety_vest_agreement_count":
            vest_agreement_count,

        "safety_vest_agreement_rate":
            rate(
                vest_agreement_count
            ),

        "full_agreement_count":
            full_agreement_count,

        "full_agreement_rate":
            rate(
                full_agreement_count
            ),

        "per_person":
            comparisons,

        "interpretation": (
            "Agreement means MediaPipe Pose "
            "and YOLO Pose produced the same "
            "downstream PPE status for the "
            "same Roboflow-detected person. "
            "Agreement does not establish "
            "ground-truth accuracy."
        ),
    }


# ============================================================
# SAVE ANNOTATED IMAGE
# ============================================================

def save_annotated_image(
    image,
    detections,
    backend_results,
    backend_name,
    image_stem,
    image_output_dir,
):
    """
    Render positive Roboflow PPE detections plus inferred
    missing-PPE regions for one pose backend.
    """

    renderer_report = {

        "roboflow_detections":
            detections,

        "people":
            backend_results[
                "people"
            ],
    }

    annotated = (
        draw_results(
            image,
            renderer_report,
        )
    )

    output_path = (
        image_output_dir
        / (
            f"{image_stem}_"
            f"{backend_name}.jpg"
        )
    )

    success = cv2.imwrite(
        str(output_path),
        annotated,
    )

    if not success:

        raise RuntimeError(
            "Could not save annotated image: "
            f"{output_path}"
        )

    return output_path


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_image(
    image_path,
    detector,
    mediapipe_backend,
    yolo_backend,
    cfg,
    output_root,
):
    """
    Run the complete comparison experiment for ONE image.
    """

    start_time = (
        time.perf_counter()
    )

    print()
    print("=" * 70)

    print(
        f"PROCESSING: "
        f"{image_path.name}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    image = cv2.imread(
        str(
            image_path
        )
    )

    if image is None:

        raise RuntimeError(
            "OpenCV could not read image: "
            f"{image_path}"
        )

    # ========================================================
    # ROBOFLOW WORKFLOW
    # ========================================================

    print()
    print(
        "Running Roboflow Workflow..."
    )

    roboflow_start = (
        time.perf_counter()
    )

    detections = (
        detector.predict(
            str(
                image_path
            )
        )
    )

    roboflow_seconds = (
        time.perf_counter()
        - roboflow_start
    )

    grouped = (
        group_detections(
            detections
        )
    )

    # Roboflow Person detections are authoritative.
    person_detections = sorted(
        grouped[
            "person"
        ],

        key=lambda item: item.get(
            "confidence",
            0.0,
        ),

        reverse=True,
    )

    print(
        "  Persons:      "
        f"{len(person_detections)}"
    )

    print(
        "  Hard Hats:    "
        f"{len(grouped['hard_hat'])}"
    )

    print(
        "  Safety Vests: "
        f"{len(grouped['safety_vest'])}"
    )

    # ========================================================
    # MEDIAPIPE POSE
    # ========================================================

    print()
    print(
        "Running MediaPipe Pose..."
    )

    mediapipe_start = (
        time.perf_counter()
    )

    mediapipe_results = (
        run_pose_backend(
            backend_name=(
                "mediapipe"
            ),

            pose_backend=(
                mediapipe_backend
            ),

            image=image,

            person_detections=(
                person_detections
            ),

            grouped_detections=(
                grouped
            ),

            cfg=cfg,
        )
    )

    mediapipe_seconds = (
        time.perf_counter()
        - mediapipe_start
    )

    # ========================================================
    # YOLO POSE
    # ========================================================

    print(
        "Running YOLO Pose..."
    )

    yolo_start = (
        time.perf_counter()
    )

    yolo_results = (
        run_pose_backend(
            backend_name=(
                "yolo"
            ),

            pose_backend=(
                yolo_backend
            ),

            image=image,

            person_detections=(
                person_detections
            ),

            grouped_detections=(
                grouped
            ),

            cfg=cfg,
        )
    )

    yolo_seconds = (
        time.perf_counter()
        - yolo_start
    )

    # ========================================================
    # COMPARE BACKENDS
    # ========================================================

    comparison = (
        compare_backends(
            mediapipe_results,
            yolo_results,
        )
    )

    # ========================================================
    # CREATE IMAGE OUTPUT DIRECTORY
    #
    # Example:
    #
    # output/
    #   image1/
    # ========================================================

    image_output_dir = (
        output_root
        / image_path.stem
    )

    image_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # SAVE MEDIAPIPE ANNOTATED IMAGE
    # ========================================================

    mediapipe_image_path = (
        save_annotated_image(
            image=image,

            detections=detections,

            backend_results=(
                mediapipe_results
            ),

            backend_name=(
                "mediapipe"
            ),

            image_stem=(
                image_path.stem
            ),

            image_output_dir=(
                image_output_dir
            ),
        )
    )

    # ========================================================
    # SAVE YOLO POSE ANNOTATED IMAGE
    # ========================================================

    yolo_image_path = (
        save_annotated_image(
            image=image,

            detections=detections,

            backend_results=(
                yolo_results
            ),

            backend_name=(
                "yolo"
            ),

            image_stem=(
                image_path.stem
            ),

            image_output_dir=(
                image_output_dir
            ),
        )
    )

    total_seconds = (
        time.perf_counter()
        - start_time
    )

    # ========================================================
    # COMPLETE PER-IMAGE REPORT
    # ========================================================

    report = {

        "image":
            str(
                image_path
            ),

        "image_name":
            image_path.name,

        "experiment":
            (
                "MediaPipe Pose vs "
                "YOLO Pose"
            ),

        "object_detection": {

            "source":
                "Roboflow Workflow",

            "workflow":
                (
                    "ppeDetection "
                    "vppedetection-zk8qc-1-"
                    "rfdetr-small-t1 Logic"
                ),

            "person_count":
                len(
                    person_detections
                ),

            "hard_hat_count":
                len(
                    grouped[
                        "hard_hat"
                    ]
                ),

            "safety_vest_count":
                len(
                    grouped[
                        "safety_vest"
                    ]
                ),

            "detections":
                detections,
        },

        "controlled_pipeline": {

            "person_detection":
                "Roboflow Workflow",

            "ppe_detection":
                "Roboflow Workflow",

            "person_crop_strategy":
                (
                    "Same padded Roboflow "
                    "Person crop supplied to "
                    "both pose backends"
                ),

            "region_generation":
                (
                    "Same expected-region "
                    "logic"
                ),

            "ppe_matching":
                (
                    "Same spatial/IoU "
                    "matching logic"
                ),

            "variable_compared":
                "pose estimator",
        },

        "mediapipe":
            mediapipe_results,

        "yolo_pose":
            yolo_results,

        "comparison":
            comparison,

        "timing_seconds": {

            "roboflow_workflow":
                round(
                    roboflow_seconds,
                    4,
                ),

            "mediapipe_pose_pipeline":
                round(
                    mediapipe_seconds,
                    4,
                ),

            "yolo_pose_pipeline":
                round(
                    yolo_seconds,
                    4,
                ),

            "total_image_pipeline":
                round(
                    total_seconds,
                    4,
                ),
        },

        "output_files": {

            "mediapipe_image":
                str(
                    mediapipe_image_path
                ),

            "yolo_pose_image":
                str(
                    yolo_image_path
                ),
        },

        "interpretation_note": (
            "MediaPipe Pose and YOLO Pose "
            "are evaluated using the same "
            "Roboflow detections and the same "
            "downstream PPE compliance logic. "
            "Pose coverage measures whether a "
            "backend produced a usable pose. "
            "Agreement measures whether both "
            "backends produced the same PPE "
            "status. Neither metric alone is "
            "ground-truth accuracy."
        ),
    }

    # ========================================================
    # SAVE PER-IMAGE JSON
    # ========================================================

    report_path = (
        image_output_dir
        / (
            f"{image_path.stem}_"
            "comparison.json"
        )
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # TERMINAL RESULT
    # ========================================================

    mp_summary = (
        mediapipe_results[
            "summary"
        ]
    )

    yolo_summary = (
        yolo_results[
            "summary"
        ]
    )

    print()
    print(
        "Image results:"
    )

    print(
        "  MediaPipe poses: "
        f"{mp_summary['successful_poses']}/"
        f"{mp_summary['total_roboflow_people']}"
    )

    print(
        "  YOLO poses:      "
        f"{yolo_summary['successful_poses']}/"
        f"{yolo_summary['total_roboflow_people']}"
    )

    print(
        "  Hard-hat agreement: "
        f"{comparison['hard_hat_agreement_count']}/"
        f"{comparison['people_compared']}"
    )

    print(
        "  Vest agreement:     "
        f"{comparison['safety_vest_agreement_count']}/"
        f"{comparison['people_compared']}"
    )

    print(
        "  Full agreement:     "
        f"{comparison['full_agreement_count']}/"
        f"{comparison['people_compared']}"
    )

    print(
        "  Processing time:    "
        f"{total_seconds:.2f}s"
    )

    print(
        "  Output: "
        f"{image_output_dir}"
    )

    # ========================================================
    # RETURN SMALL BATCH RECORD
    # ========================================================

    return {

        "image":
            image_path.name,

        "status":
            "success",

        "roboflow_people":
            len(
                person_detections
            ),

        "hard_hat_detections":
            len(
                grouped[
                    "hard_hat"
                ]
            ),

        "safety_vest_detections":
            len(
                grouped[
                    "safety_vest"
                ]
            ),

        "mediapipe_successful_poses":
            mp_summary[
                "successful_poses"
            ],

        "yolo_successful_poses":
            yolo_summary[
                "successful_poses"
            ],

        "mediapipe_pose_success_rate":
            mp_summary[
                "pose_success_rate"
            ],

        "yolo_pose_success_rate":
            yolo_summary[
                "pose_success_rate"
            ],

        "hard_hat_agreement_rate":
            comparison[
                "hard_hat_agreement_rate"
            ],

        "safety_vest_agreement_rate":
            comparison[
                "safety_vest_agreement_rate"
            ],

        "full_agreement_rate":
            comparison[
                "full_agreement_rate"
            ],

        "processing_seconds":
            round(
                total_seconds,
                4,
            ),

        "report":
            str(
                report_path
            ),
    }


# ============================================================
# BUILD BATCH SUMMARY
# ============================================================

def build_batch_summary(
    results,
):
    """
    Aggregate all successfully processed images into one
    experiment-level summary.
    """

    successful = [
        result
        for result in results
        if (
            result.get("status")
            == "success"
        )
    ]

    failed = [
        result
        for result in results
        if (
            result.get("status")
            == "failed"
        )
    ]

    # --------------------------------------------------------
    # TOTAL PEOPLE
    # --------------------------------------------------------

    total_people = sum(
        result.get(
            "roboflow_people",
            0,
        )
        for result in successful
    )

    # --------------------------------------------------------
    # POSE TOTALS
    # --------------------------------------------------------

    total_mp_poses = sum(
        result.get(
            "mediapipe_successful_poses",
            0,
        )
        for result in successful
    )

    total_yolo_poses = sum(
        result.get(
            "yolo_successful_poses",
            0,
        )
        for result in successful
    )

    # --------------------------------------------------------
    # POSE SUCCESS RATES
    # --------------------------------------------------------

    overall_mp_rate = (
        total_mp_poses
        / total_people
        if total_people
        else 0.0
    )

    overall_yolo_rate = (
        total_yolo_poses
        / total_people
        if total_people
        else 0.0
    )

    # --------------------------------------------------------
    # DETECTION TOTALS
    # --------------------------------------------------------

    total_hard_hats = sum(
        result.get(
            "hard_hat_detections",
            0,
        )
        for result in successful
    )

    total_vests = sum(
        result.get(
            "safety_vest_detections",
            0,
        )
        for result in successful
    )

    # --------------------------------------------------------
    # IMAGE-LEVEL AGREEMENT AVERAGES
    #
    # These are descriptive averages across successfully
    # processed images. They are NOT accuracy metrics.
    # --------------------------------------------------------

    if successful:

        average_hat_agreement = (
            sum(
                result.get(
                    "hard_hat_agreement_rate",
                    0.0,
                )
                for result in successful
            )
            / len(successful)
        )

        average_vest_agreement = (
            sum(
                result.get(
                    "safety_vest_agreement_rate",
                    0.0,
                )
                for result in successful
            )
            / len(successful)
        )

        average_full_agreement = (
            sum(
                result.get(
                    "full_agreement_rate",
                    0.0,
                )
                for result in successful
            )
            / len(successful)
        )

    else:

        average_hat_agreement = 0.0
        average_vest_agreement = 0.0
        average_full_agreement = 0.0

    return {

        "images_attempted":
            len(
                results
            ),

        "images_successful":
            len(
                successful
            ),

        "images_failed":
            len(
                failed
            ),

        "total_roboflow_people":
            total_people,

        "total_roboflow_hard_hat_detections":
            total_hard_hats,

        "total_roboflow_safety_vest_detections":
            total_vests,

        "mediapipe": {

            "successful_poses":
                total_mp_poses,

            "overall_pose_success_rate":
                round(
                    overall_mp_rate,
                    4,
                ),
        },

        "yolo_pose": {

            "successful_poses":
                total_yolo_poses,

            "overall_pose_success_rate":
                round(
                    overall_yolo_rate,
                    4,
                ),
        },

        "backend_agreement": {

            "average_hard_hat_agreement_rate":
                round(
                    average_hat_agreement,
                    4,
                ),

            "average_safety_vest_agreement_rate":
                round(
                    average_vest_agreement,
                    4,
                ),

            "average_full_agreement_rate":
                round(
                    average_full_agreement,
                    4,
                ),
        },

        "images":
            results,

        "interpretation": (
            "Pose success rate measures "
            "whether a pose backend produced "
            "a usable pose for a Roboflow-"
            "detected person. Backend agreement "
            "measures whether MediaPipe Pose "
            "and YOLO Pose produced the same "
            "downstream PPE status. These "
            "measurements are descriptive and "
            "are not substitutes for validation "
            "against labeled ground truth."
        ),
    }


# ============================================================
# PUBLIC PIPELINE FUNCTION
# ============================================================

def run_pipeline(
    input_path=None,
    config_path=None,
):
    """
    Run the ENTIRE PPE experiment.

    This is the function called by the root-level main.py.

    Steps:

        1. Discover input images
        2. Load configuration
        3. Initialize Roboflow
        4. Initialize MediaPipe Pose
        5. Initialize YOLO Pose
        6. Process every image
        7. Save per-image annotations
        8. Save per-image comparison JSON
        9. Create batch summary
        10. Return batch summary
    """

    # ========================================================
    # DEFAULT PATHS
    # ========================================================

    if input_path is None:

        input_path = (
            ROOT / "input"
        )

    if config_path is None:

        config_path = (
            ROOT
            / "configs"
            / "config.yaml"
        )

    input_path = Path(
        input_path
    )

    config_path = Path(
        config_path
    )

    # ========================================================
    # LOAD CONFIG
    # ========================================================

    cfg = load_config(
        str(
            config_path
        )
    )

    # ========================================================
    # FIND INPUT IMAGES
    # ========================================================

    images = find_images(
        input_path
    )

    print()
    print("=" * 70)

    print(
        "POSE-AWARE PPE BATCH COMPARISON"
    )

    print("=" * 70)

    print()

    print(
        f"Input: {input_path}"
    )

    print(
        f"Images found: {len(images)}"
    )

    print()

    for (
        index,
        image_path,
    ) in enumerate(
        images,
        start=1,
    ):

        print(
            f"  {index}. "
            f"{image_path.name}"
        )

    # ========================================================
    # OUTPUT ROOT
    # ========================================================

    output_root = (
        ROOT
        / cfg[
            "output"
        ][
            "directory"
        ]
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # INITIALIZE MODELS ONCE
    #
    # We intentionally do NOT reload these for every image.
    # ========================================================

    print()
    print(
        "Initializing Roboflow Workflow..."
    )

    detector = (
        PPEDetector(
            confidence=cfg[
                "inference"
            ][
                "ppe_confidence"
            ]
        )
    )

    print(
        "Loading MediaPipe Pose..."
    )

    mediapipe_backend = (
        MediaPipePoseBackend(
            cfg[
                "mediapipe"
            ][
                "min_detection_confidence"
            ],

            cfg[
                "mediapipe"
            ][
                "min_tracking_confidence"
            ],
        )
    )

    print(
        "Loading YOLO Pose..."
    )

    yolo_backend = (
        YOLOPoseBackend(
            cfg[
                "models"
            ][
                "yolo_pose_weights"
            ],

            cfg[
                "inference"
            ][
                "pose_confidence"
            ],
        )
    )

    print(
        "Models initialized."
    )

    # ========================================================
    # RUN BATCH
    # ========================================================

    results = []

    batch_start = (
        time.perf_counter()
    )

    try:

        for (
            index,
            image_path,
        ) in enumerate(
            images,
            start=1,
        ):

            print()
            print(
                "#" * 70
            )

            print(
                f"IMAGE "
                f"{index}/"
                f"{len(images)}"
            )

            print(
                "#" * 70
            )

            try:

                result = (
                    process_image(
                        image_path=(
                            image_path
                        ),

                        detector=(
                            detector
                        ),

                        mediapipe_backend=(
                            mediapipe_backend
                        ),

                        yolo_backend=(
                            yolo_backend
                        ),

                        cfg=cfg,

                        output_root=(
                            output_root
                        ),
                    )
                )

                results.append(
                    result
                )

            except Exception as exc:

                # One failed image should not terminate the
                # rest of the batch.
                print()
                print(
                    "ERROR processing "
                    f"{image_path.name}"
                )

                print(
                    "  "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

                results.append(
                    {

                        "image":
                            image_path.name,

                        "status":
                            "failed",

                        "error_type":
                            type(
                                exc
                            ).__name__,

                        "error":
                            str(
                                exc
                            ),
                    }
                )

    finally:

        # ====================================================
        # CLEAN UP MODELS
        # ====================================================

        if hasattr(
            mediapipe_backend,
            "close",
        ):

            mediapipe_backend.close()

        if hasattr(
            yolo_backend,
            "close",
        ):

            yolo_backend.close()

    # ========================================================
    # BATCH SUMMARY
    # ========================================================

    batch_seconds = (
        time.perf_counter()
        - batch_start
    )

    batch_summary = (
        build_batch_summary(
            results
        )
    )

    batch_summary[
        "total_processing_seconds"
    ] = round(
        batch_seconds,
        4,
    )

    # ========================================================
    # SAVE BATCH SUMMARY
    # ========================================================

    batch_summary_path = (
        output_root
        / "batch_summary.json"
    )

    batch_summary_path.write_text(
        json.dumps(
            batch_summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # FINAL TERMINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)

    print(
        "PIPELINE COMPLETE"
    )

    print("=" * 70)

    print()

    print(
        "Images attempted:  "
        f"{batch_summary['images_attempted']}"
    )

    print(
        "Images successful: "
        f"{batch_summary['images_successful']}"
    )

    print(
        "Images failed:     "
        f"{batch_summary['images_failed']}"
    )

    print()

    print(
        "Total Roboflow people: "
        f"{batch_summary['total_roboflow_people']}"
    )

    print()

    # --------------------------------------------------------
    # MEDIAPIPE SUMMARY
    # --------------------------------------------------------

    print(
        "MediaPipe pose coverage: "
        f"{batch_summary['mediapipe']['successful_poses']}/"
        f"{batch_summary['total_roboflow_people']} "
        "("
        f"{batch_summary['mediapipe']['overall_pose_success_rate']:.1%}"
        ")"
    )

    # --------------------------------------------------------
    # YOLO SUMMARY
    # --------------------------------------------------------

    print(
        "YOLO Pose coverage:      "
        f"{batch_summary['yolo_pose']['successful_poses']}/"
        f"{batch_summary['total_roboflow_people']} "
        "("
        f"{batch_summary['yolo_pose']['overall_pose_success_rate']:.1%}"
        ")"
    )

    # --------------------------------------------------------
    # AGREEMENT SUMMARY
    # --------------------------------------------------------

    print()

    print(
        "Average hard-hat agreement: "
        f"{batch_summary['backend_agreement']['average_hard_hat_agreement_rate']:.1%}"
    )

    print(
        "Average vest agreement:     "
        f"{batch_summary['backend_agreement']['average_safety_vest_agreement_rate']:.1%}"
    )

    print(
        "Average full agreement:     "
        f"{batch_summary['backend_agreement']['average_full_agreement_rate']:.1%}"
    )

    print()

    print(
        "NOTE: Agreement is not accuracy."
    )

    print(
        "Ground-truth labels are required "
        "to determine which pose backend "
        "is actually more accurate."
    )

    print()

    print(
        "Total processing time: "
        f"{batch_seconds:.2f} seconds"
    )

    print()

    print(
        "Batch summary:"
    )

    print(
        f"  {batch_summary_path}"
    )

    print()

    print(
        "Per-image results:"
    )

    print(
        f"  {output_root}"
    )

    # Allows main.py or future code to consume the results.
    return batch_summary


# ============================================================
# DIRECT COMMAND-LINE ENTRY POINT
# ============================================================

def main():
    """
    Allows infer.py to still be run directly.

    Examples:

        python src/infer.py

        python src/infer.py --input input

        python src/infer.py --input input/image1.png

    Normal project use can simply be:

        python main.py
    """

    parser = argparse.ArgumentParser(
        description=(
            "Pose-aware PPE detection with "
            "MediaPipe vs YOLO Pose comparison."
        )
    )

    parser.add_argument(
        "--input",

        default=str(
            ROOT / "input"
        ),

        help=(
            "Single image or directory "
            "containing input images."
        ),
    )

    parser.add_argument(
        "--config",

        default=str(
            ROOT
            / "configs"
            / "config.yaml"
        ),

        help=(
            "Path to configuration YAML."
        ),
    )

    args = (
        parser.parse_args()
    )

    run_pipeline(
        input_path=(
            args.input
        ),

        config_path=(
            args.config
        ),
    )


if __name__ == "__main__":
    main()