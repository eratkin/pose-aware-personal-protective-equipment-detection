import os
import time
from pathlib import Path

from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient, InferenceConfiguration


class RoboflowWorkflowError(RuntimeError):
    """Raised when the Roboflow Workflow cannot return usable detections."""


class PPEDetector:
    """
    PPE/person detector backed by a hosted Roboflow Workflow.

    Roboflow is authoritative for object detections:
        - Person
        - Hard Hat
        - Safety Vest

    Pose models are NOT used to establish whether a person exists.
    """

    SUPPORTED_CLASSES = {
        "person": "person",
        "hard hat": "hard_hat",
        "hard_hat": "hard_hat",
        "hardhat": "hard_hat",
        "safety vest": "safety_vest",
        "safety_vest": "safety_vest",
        "safetyvest": "safety_vest",
    }

    def __init__(
        self,
        weights=None,
        confidence=0.25,
        class_aliases=None,
        retries=2,
    ):
        # weights is intentionally accepted for backward compatibility
        # with the old local-detector constructor, but it is not used.
        self.confidence = confidence
        self.retries = retries
        self.class_aliases = class_aliases or {}

        # Load private environment variables from the project root.
        project_root = Path(__file__).resolve().parents[2]
        load_dotenv(project_root / ".env")

        # Roboflow deployment configuration is intentionally stored
        # outside the source code so account-specific values are not
        # exposed in the public repository.
        api_key = os.getenv("ROBOFLOW_API_KEY")
        api_url = os.getenv("ROBOFLOW_API_URL")
        workspace_name = os.getenv("ROBOFLOW_WORKSPACE")
        workflow_id = os.getenv("ROBOFLOW_WORKFLOW_ID")

        required_settings = {
            "ROBOFLOW_API_KEY": api_key,
            "ROBOFLOW_API_URL": api_url,
            "ROBOFLOW_WORKSPACE": workspace_name,
            "ROBOFLOW_WORKFLOW_ID": workflow_id,
        }

        missing_settings = [
            name
            for name, value in required_settings.items()
            if not value
        ]

        if missing_settings:
            raise RoboflowWorkflowError(
                "Missing required Roboflow environment setting(s): "
                + ", ".join(missing_settings)
                + ". Add them to the project's .env file."
            )

        # Store only the values needed later when running the workflow.
        self.workspace_name = workspace_name
        self.workflow_id = workflow_id

        self.client = InferenceHTTPClient(
            api_url=api_url,
            api_key=api_key,
        )

        # Use header-based authentication.
        self.client.configure(
            InferenceConfiguration(
                api_key_transport="header"
            )
        )

    @staticmethod
    def normalize_class_name(name):
        return (
            str(name)
            .strip()
            .lower()
            .replace("-", " ")
            .replace("_", " ")
        )

    def canonical_class(self, name):
        normalized = self.normalize_class_name(name)

        canonical = self.SUPPORTED_CLASSES.get(normalized)

        if canonical:
            return canonical

        return normalized.replace(" ", "_")

    @staticmethod
    def center_box_to_xyxy(prediction):
        """
        Roboflow returns bounding boxes using:

            x      = box center x
            y      = box center y
            width
            height

        Convert to the internal format:

            [x1, y1, x2, y2]
        """

        x = float(prediction["x"])
        y = float(prediction["y"])
        width = float(prediction["width"])
        height = float(prediction["height"])

        x1 = x - width / 2.0
        y1 = y - height / 2.0
        x2 = x + width / 2.0
        y2 = y + height / 2.0

        return [x1, y1, x2, y2]

    def _find_predictions(self, value):
        """
        Search defensively for the prediction list.

        The Workflow response can be wrapped under a Workflow-specific
        output name, so we do not hard-code that output key.
        """

        if isinstance(value, dict):

            predictions = value.get("predictions")

            if isinstance(predictions, list):
                # Confirm it resembles the expected object-detection schema.
                if not predictions:
                    return predictions

                first = predictions[0]

                if isinstance(first, dict):
                    required = {
                        "x",
                        "y",
                        "width",
                        "height",
                        "class",
                    }

                    if required.issubset(first.keys()):
                        return predictions

            for child in value.values():
                found = self._find_predictions(child)

                if found is not None:
                    return found

        elif isinstance(value, list):

            for child in value:
                found = self._find_predictions(child)

                if found is not None:
                    return found

        return None

    def _run_workflow(self, image_path):
        last_error = None

        for attempt in range(self.retries + 1):
            try:
                return self.client.run_workflow(
                    workspace_name=self.workspace_name,
                    workflow_id=self.workflow_id,
                    images={
                        "image": str(image_path)
                    },
                    parameters={},
                    use_cache=False,
                )

            except Exception as exc:
                last_error = exc

                if attempt >= self.retries:
                    break

                # Small exponential backoff:
                # 1 sec, then 2 sec.
                time.sleep(2 ** attempt)

        raise RoboflowWorkflowError(
            "Roboflow Workflow inference failed after "
            f"{self.retries + 1} attempts: {last_error}"
        )

    def predict(self, image_or_path):
        """
        Run the hosted Workflow and return normalized detections.

        Internal format:

        {
            "class": "person",
            "raw_class": "Person",
            "confidence": 0.977,
            "box": [x1, y1, x2, y2],
            ...
        }
        """

        # The Workflow SDK can send the local image itself.
        if isinstance(image_or_path, (str, Path)):
            image_path = Path(image_or_path)

        else:
            raise TypeError(
                "Hosted Roboflow inference requires the image path. "
                "Pass the image filename to PPEDetector.predict()."
            )

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        response = self._run_workflow(image_path)

        predictions = self._find_predictions(response)

        if predictions is None:
            raise RoboflowWorkflowError(
                "The Workflow completed, but no object-detection "
                "'predictions' list matching the expected Roboflow "
                "schema was found in its response."
            )

        detections = []

        for prediction in predictions:

            try:
                confidence = float(
                    prediction.get("confidence", 0.0)
                )

                if confidence < self.confidence:
                    continue

                raw_class = str(prediction["class"])

                canonical = self.canonical_class(
                    raw_class
                )

                # Ignore unrelated classes if the Workflow ever grows.
                if canonical not in {
                    "person",
                    "hard_hat",
                    "safety_vest",
                }:
                    continue

                detections.append(
                    {
                        "class": canonical,
                        "raw_class": raw_class,
                        "class_id": prediction.get(
                            "class_id"
                        ),
                        "confidence": confidence,
                        "detection_id": prediction.get(
                            "detection_id"
                        ),
                        "box": self.center_box_to_xyxy(
                            prediction
                        ),
                    }
                )

            except (
                KeyError,
                TypeError,
                ValueError,
            ) as exc:

                raise RoboflowWorkflowError(
                    "A Roboflow detection did not match "
                    "the expected detection schema."
                ) from exc

        return detections