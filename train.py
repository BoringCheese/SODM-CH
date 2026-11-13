# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Train yolo26-lks-smoe-123.

Usage:
    python train.py                     # 默认参数训练
    python train.py data=xxx.yaml epochs=100 imgsz=640
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))  # 保证使用本仓库内的 ultralytics

from ultralytics import YOLO

MODEL_YAML = Path(__file__).parent / "ultralytics" / "cfg" / "models" / "custom" / "yolo26-lks-smoe-123.yaml"


def main():
    # 解析 key=value 形式的命令行参数
    overrides = {}
    for arg in sys.argv[1:]:
        if "=" in arg:
            k, v = arg.split("=", 1)
            try:
                v = eval(v)  # noqa: S307 - 允许 100 / 0.01 / True 之类的字面量
            except Exception:
                pass
            overrides[k] = v

    model = YOLO(str(MODEL_YAML))  # 从 YAML 构建模型（默认 s 档，可加 scale=n 等）
    model.train(
        data=overrides.pop("data", "coco128.yaml"),  # 换成你的数据集 yaml
        epochs=overrides.pop("epochs", 100),
        imgsz=overrides.pop("imgsz", 640),
        batch=overrides.pop("batch", 16),
        **overrides,
    )


if __name__ == "__main__":
    main()
