# yolo26-lks-smoe-123

基于 Ultralytics（8.4.x）的目标检测模型仓库，仅保留运行 **yolo26-lks-smoe-123** 所需的代码。

## 模型结构

配置文件：`ultralytics/cfg/models/custom/yolo26-lks-smoe-123.yaml`

在 YOLO26 检测框架上引入三个自定义模块：

| 模块 | 位置 | 作用 |
| ---- | ---- | ---- |
| `DWTLSK` | `ultralytics/nn/modules/conv.py` | Stem：Haar 小波分解 + 大核选择性门控（LSK），有序/无序双分支加权融合 |
| `SFDDown` | `ultralytics/nn/modules/conv.py` | 下采样：抗混叠低频 + 无损 space-to-depth + Haar 高频细节，轻量内容门控融合 |
| `SAHSEF` | `ultralytics/nn/modules/moe.py` | Neck 三尺度融合：尺度对齐 → 尺度加权 → 异构专家（detail/context/frequency）top-k 稀疏路由 → 轻量重组 |

s 档约 203 万参数（632 层）。

## 安装

```bash
pip install -r requirements.txt
# 或者以可编辑模式安装为包
pip install -e .
```

需要 PyTorch（CPU 或 CUDA 均可），以及 `pytorch-wavelets`（`DWTLSK` 的小波分解依赖）。

## 使用

### 构建并前向验证

```bash
python predict_check.py
```

### 训练

```bash
python train.py data=你的数据集.yaml epochs=100 imgsz=640 batch=16
```

也可以在 Python 中直接使用：

```python
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/custom/yolo26-lks-smoe-123.yaml")
model.train(data="你的数据集.yaml", epochs=100, imgsz=640)
```

### 推理 / 验证

```python
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")  # 训练得到的权重
model.val(data="你的数据集.yaml")
model.predict(source="图片或目录路径")
```

## 许可

AGPL-3.0（沿用 Ultralytics 许可证），详见 `LICENSE`。
