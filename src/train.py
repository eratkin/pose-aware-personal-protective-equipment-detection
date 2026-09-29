import argparse
from pathlib import Path
from ultralytics import YOLO
from config import ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='Path to Roboflow-generated data.yaml')
    ap.add_argument('--model', default='yolov8n.pt', help='Starting pretrained detection model')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--device', default='cpu', help='cpu or GPU index such as 0')
    args = ap.parse_args()

    model = YOLO(args.model)
    result = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=args.device,
        project=str(ROOT / 'runs' / 'ppe_detection'),
        name='train',
    )
    print('Training complete. Copy runs/ppe_detection/train/weights/best.pt to models/ppe_best.pt for inference.')
    return result


if __name__ == '__main__':
    main()
