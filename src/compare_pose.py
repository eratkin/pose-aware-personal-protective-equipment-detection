import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description='Run the same image through both pose backends.')
    ap.add_argument('--image', required=True)
    ap.add_argument('--weights', default=None)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    for backend in ('yolo', 'mediapipe'):
        cmd = [sys.executable, str(root/'src'/'infer.py'), '--image', args.image, '--pose-backend', backend]
        if args.weights:
            cmd += ['--weights', args.weights]
        print(f'\n=== {backend.upper()} ===')
        subprocess.run(cmd, check=True)
    print('\nCompare the two JSON files in output/. Focus on keypoint availability, missing-PPE decisions, IoU/containment, and failure cases.')


if __name__ == '__main__':
    main()
