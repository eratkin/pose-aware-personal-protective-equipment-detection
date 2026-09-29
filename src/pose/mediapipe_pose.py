from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# MediaPipe Pose Landmarker uses 33 body landmarks.
# These indices map the MediaPipe landmarks we need into the same
# semantic names used by the YOLO Pose backend.
MEDIAPIPE_LANDMARKS = {
    "nose": 0,
    "left_eye": 2,
    "right_eye": 5,
    "left_ear": 7,
    "right_ear": 8,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}


class MediaPipePoseBackend:
    """
    MediaPipe Tasks Pose Landmarker backend.

    Converts MediaPipe's normalized landmarks into the same internal
    pixel-coordinate structure used by the YOLO Pose backend.
    """

    def __init__(
        self,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_path="models/pose_landmarker_lite.task",
    ):
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe pose model not found at: {self.model_path}\n"
                "Expected: models\\pose_landmarker_lite.task"
            )

        base_options = python.BaseOptions(
            model_asset_path=str(self.model_path)
        )

        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=10,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self.landmarker = vision.PoseLandmarker.create_from_options(options)

    def predict(self, image):
        """
        Detect poses and return them in the project's standard format:

        [
            {
                "id": 1,
                "box": [...],
                "points": {
                    "nose": (x, y, confidence),
                    ...
                }
            }
        ]
        """

        if image is None:
            raise ValueError("Input image is None.")

        height, width = image.shape[:2]

        # OpenCV uses BGR. MediaPipe expects RGB.
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )

        result = self.landmarker.detect(mp_image)

        people = []

        for person_index, landmarks in enumerate(result.pose_landmarks):
            points = {}

            all_x = []
            all_y = []

            for name, landmark_index in MEDIAPIPE_LANDMARKS.items():
                landmark = landmarks[landmark_index]

                # MediaPipe coordinates are normalized 0-1.
                # Convert them to actual image pixels.
                x = float(landmark.x * width)
                y = float(landmark.y * height)

                visibility = float(
                    getattr(landmark, "visibility", 1.0)
                )

                presence = float(
                    getattr(landmark, "presence", 1.0)
                )

                # Use the lower value so a landmark must be both
                # visible and present to receive high confidence.
                confidence = min(visibility, presence)

                points[name] = (x, y, confidence)

                if confidence > 0:
                    all_x.append(x)
                    all_y.append(y)

            # MediaPipe does not directly provide the same person
            # bounding box as YOLO Pose, so construct one from landmarks.
            person_box = None

            if all_x and all_y:
                padding_x = width * 0.03
                padding_y = height * 0.03

                x1 = max(0.0, min(all_x) - padding_x)
                y1 = max(0.0, min(all_y) - padding_y)
                x2 = min(float(width - 1), max(all_x) + padding_x)
                y2 = min(float(height - 1), max(all_y) + padding_y)

                person_box = [x1, y1, x2, y2]

            people.append(
                {
                    "id": person_index + 1,
                    "box": person_box,
                    "points": points,
                }
            )

        return people

    def close(self):
        if hasattr(self, "landmarker"):
            self.landmarker.close()