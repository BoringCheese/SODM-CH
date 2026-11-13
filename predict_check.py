# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Quick sanity check: build yolo26-lks-smoe-123 from YAML and run one forward pass."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch

from ultralytics import YOLO

MODEL_YAML = Path(__file__).parent / "ultralytics" / "cfg" / "models" / "custom" / "yolo26-lks-smoe-123.yaml"


def main():
    model = YOLO(str(MODEL_YAML))
    model.info()
    model.eval()
    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        y = model.model(x)
    print("forward OK, output type:", type(y))


if __name__ == "__main__":
    main()
