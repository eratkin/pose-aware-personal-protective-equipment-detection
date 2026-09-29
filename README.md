# Pose-Aware PPE Compliance Detection

A computer-vision pipeline that checks whether workers in an image are wearing
required PPE (hard hats and safety vests) by combining a Roboflow-trained
object detector with human pose estimation. Pose keypoints are used to
estimate *where* PPE should appear on a detected person, so the system can
tell the difference between "no PPE detected" and "PPE could not be
evaluated" rather than guessing.

## How it works

```
Image
  |
  v
Roboflow Workflow  --> Person, Hard Hat, Safety Vest detections
  |
  v
Roboflow Person Crops
  |
  +----------------------+
  |                       |
  v                       v
MediaPipe Pose         YOLO Pose
  |                       |
  v                       v
Expected Regions       Expected Regions
  |                       |
  +-----------+-----------+
              |
              v
      PPE Spatial Matching
              |
              v
     Compare Pose Backends
              |
              v
    Images + JSON Reports
```

**Design principles**

- **Roboflow is the source of truth for existence.** Whether a person, hard
  hat, or safety vest is present in the image is decided entirely by the
  Roboflow-trained detector. Pose estimation never adds or removes a
  detection.
- **Pose only supplies anatomical geometry.** YOLO Pose and MediaPipe Pose
  are run on the same Roboflow person crop and used solely to estimate head
  and torso regions.
- **Three-way PPE status, not two.** Each PPE item is `detected`, `missing`,
  or `unknown`. `missing` is only reported when a valid pose-derived region
  exists — if pose estimation fails for a person, their PPE status is
  `unknown` rather than assumed missing.
- **Same downstream logic for both pose backends.** MediaPipe and YOLO Pose
  are evaluated through identical region-generation and matching code, so
  the pose estimator is the only variable being compared.

## Project structure

```
.
├── main.py                    # Entry point: python main.py
├── configs/
│   └── config.yaml            # Roboflow, model, and matching settings
├── models/                    # Local weights (YOLO Pose, MediaPipe task file)
├── input/                     # Images to process
├── output/                    # Annotated images + JSON reports (generated)
└── src/
    ├── config.py              # Loads config.yaml and .env
    ├── dataset.py             # Downloads the training dataset from Roboflow
    ├── train.py               # Trains the custom YOLO PPE detector
    ├── infer.py               # Batch inference + pose-backend comparison
    ├── compare_pose.py        # Convenience script: runs infer.py twice
    ├── visualization.py       # Draws annotated output images
    ├── ppe/
    │   └── ppe_detector.py    # Wraps the Roboflow PPE detection workflow
    ├── pose/
    │   ├── yolo_pose.py       # YOLO Pose backend
    │   └── mediapipe_pose.py  # MediaPipe Pose backend
    └── matching/
        ├── iou.py             # Box geometry helpers (IoU, containment, etc.)
        ├── regions.py         # Pose keypoints -> expected PPE regions
        ├── association.py     # Matches poses to Roboflow person detections
        └── compliance.py      # Detected / missing / unknown decision logic
```

## Setup

1. **Install dependencies.**
   ```
   pip install ultralytics roboflow opencv-python mediapipe pyyaml python-dotenv numpy
   ```
   (Pin versions in a `requirements.txt` for reproducible installs.)

2. **Add your Roboflow key.** Copy `.env.example` to `.env` and set:
   ```
   ROBOFLOW_API_KEY=your_private_key
   ```

3. **Add model weights to `models/`.**
   - YOLO Pose weights (e.g. `yolo11n-pose.pt`) — downloaded automatically by
     `ultralytics` if not present.
   - MediaPipe Pose Landmarker task file, e.g. `pose_landmarker_lite.task`
     (download from Google's MediaPipe model index).

4. **Configure `configs/config.yaml`.** Example shape, based on the keys the
   code reads:
   ```yaml
   roboflow:
     workspace: your-workspace
     project: your-project
     version: 1
     dataset_format: yolov8
     download_dir: data/raw

   models:
     yolo_pose_weights: models/yolo11n-pose.pt

   mediapipe:
     min_detection_confidence: 0.5
     min_tracking_confidence: 0.5

   inference:
     ppe_confidence: 0.5
     pose_confidence: 0.5

   matching:
     helmet_iou_threshold: 0.10
     vest_iou_threshold: 0.10

   regions:
     head_width_scale: 1.0
     head_height_scale: 1.0
     torso_width_scale: 1.0
     torso_height_scale: 1.0

   output:
     directory: output
   ```

## Usage

**Download the training dataset from Roboflow:**
```
python src/dataset.py
```

**Train the PPE detector:**
```
python src/train.py --data path/to/data.yaml --epochs 100 --device 0
```
Copy the resulting `runs/ppe_detection/train/weights/best.pt` to
`models/ppe_best.pt` for use during inference.

**Run the full pipeline on the `input/` folder (or a single image):**
```
python main.py
python main.py --input input/image1.png
```

**Compare pose backends directly on one image:**
```
python src/compare_pose.py --image input/image1.png
```

## Output

For each processed image, the pipeline writes:
- An annotated image per pose backend, with detected PPE in green, missing
  hard hats/vests flagged in their expected region, and per-person labels
  when multiple people are in frame.
- A `*_comparison.json` report with per-person PPE status, pose-match
  scores, and a comparison between the MediaPipe and YOLO Pose results.
- A `batch_summary.json` with aggregate pose-coverage and backend-agreement
  statistics across the whole run.

Backend agreement measures whether MediaPipe and YOLO Pose reached the same
PPE conclusion — it is **not** a measure of ground-truth accuracy, since no
labeled evaluation set is used at inference time.

## Notes

- Person crops below a minimum size are skipped for pose estimation
  (`MIN_PERSON_WIDTH`, `MIN_PERSON_HEIGHT` in `infer.py`) and are reported as
  `unknown` rather than `missing`.
- Small crops are upscaled before pose estimation to improve keypoint
  quality on distant workers, and keypoints are mapped back to original
  image coordinates afterward.
- Secrets (the Roboflow API key) are kept in `.env`, which should be
  git-ignored; `configs/config.yaml` holds only non-secret settings.