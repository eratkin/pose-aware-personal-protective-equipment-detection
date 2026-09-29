from ultralytics import YOLO

COCO = ['nose','left_eye','right_eye','left_ear','right_ear','left_shoulder','right_shoulder',
        'left_elbow','right_elbow','left_wrist','right_wrist','left_hip','right_hip',
        'left_knee','right_knee','left_ankle','right_ankle']


class YOLOPoseBackend:
    def __init__(self, weights='yolo11n-pose.pt', confidence=0.5):
        self.model = YOLO(weights)
        self.confidence = confidence

    def predict(self, image):
        r = self.model.predict(image, conf=self.confidence, verbose=False)[0]
        people = []
        if r.keypoints is None:
            return people
        xy = r.keypoints.xy.cpu().numpy()
        conf = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else None
        boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else [None] * len(xy)
        for i, pts in enumerate(xy):
            points = {}
            for j, name in enumerate(COCO):
                c = float(conf[i][j]) if conf is not None else 1.0
                points[name] = (float(pts[j][0]), float(pts[j][1]), c)
            people.append({'id': i + 1, 'box': boxes[i].tolist() if boxes[i] is not None else None, 'points': points})
        return people
