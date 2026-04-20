# 鑫光板材瑕疵检测项目

> 基于 YOLOv8 + OpenVINO 的工业板材瑕疵检测系统

---

## 项目简介

本项目旨在利用鑫光板材数据集训练一个高精度的 YOLOv8 目标检测模型，实现三类板材瑕疵（defect_2, defect_3, defect_6）的实时检测。项目已完成 Baseline 模型训练与 OpenVINO 推理链路验证，为 DK2500 开发板部署奠定基础。

**项目背景**：英特尔杯嵌入式AI专题赛参赛项目

---

## 环境要求

- **操作系统**：Windows / Linux
- **Python**：3.8+
- **CUDA**（可选）：支持 GPU 加速训练

### 安装步骤

```bash
# 克隆项目
git clone <项目仓库地址>
cd XinGuang

# 创建虚拟环境（推荐）
conda create -n xinguang python=3.10
conda activate xinguang

# 安装依赖
pip install -r requirements.txt
```

---

## 项目结构

```
XinGuang/
├── scripts/                 # 可执行 Python 脚本
│   ├── prepare_dataset.py   # 数据集预处理（LabelMe → YOLO）
│   ├── start_training.py    # 模型训练启动脚本
│   ├── val_baseline.py      # 精度验证脚本
│   ├── infer_openvino.py    # OpenVINO 推理与速度测试
│   └── full_compare_openvino.py  # 模型对比测试
├── configs/                 # 配置文件
│   ├── xinguang.yaml        # 数据集配置
│   └── yolov8n_cbam_wiou.yaml  # CBAM+WIoU 魔改模型配置
├── docs/                    # 文档
│   ├── PROJECT_SUMMARY.md   # 项目总结
│   └── 关于鑫光板材检测数据集的情况说明.pdf
├── datasets/                # YOLO 格式数据集（不移动）
│   ├── images/train/
│   ├── images/val/
│   ├── labels/train/
│   └── labels/val/
├── runs/                    # 训练结果与推理输出（不移动）
│   └── detect/
├── outputs/                 # 预留输出目录
├── requirements.txt         # 依赖清单
└── README.md                # 项目说明
```

---

## 快速开始

### 1. 数据预处理

将 LabelMe 格式的原始标注转换为 YOLO 格式，并通过滑窗裁剪生成 640×640 子图：

```bash
python scripts/prepare_dataset.py
```

### 2. 模型训练

启动 Baseline 或魔改模型训练：

```bash
# 使用 CBAM + WIoU 魔改配置训练
python scripts/start_training.py
```

训练参数：
- `epochs=100`：训练轮数
- `batch=8`：批次大小
- `imgsz=640`：输入尺寸
- `device='cpu'`：CPU 训练

### 3. 模型验证

验证训练模型的精度指标：

```bash
python scripts/val_baseline.py
```

### 4. OpenVINO 推理

测试模型在 Intel CPU 上的推理性能：

```bash
python scripts/infer_openvino.py
```

### 5. 模型对比

对比官方 YOLOv8n 与自训练模型的精度和速度：

```bash
python scripts/full_compare_openvino.py
```

---

## 模型性能

### Baseline 模型指标

| 指标 | 值 |
|------|-----|
| mAP@0.5 | **0.5865** |
| mAP@0.5:0.95 | 0.3017 |
| defect_2 mAP@0.5 | 0.417 |

### OpenVINO 推理性能（Intel CPU）

| 指标 | 值 |
|------|-----|
| 纯推理 FPS | **58.08** |
| 平均单张耗时 | **17.22 ms** |

### 模型对比报告

| 模型 | mAP50 | OpenVINO FPS |
|------|-------|-------------|
| 官方 YOLOv8n | 0.0050 | ~70 |
| 自训练 Baseline | **0.5865** | ~58 |

> **结论**：定制化训练的模型在本数据集上精度远超官方预训练模型，证明了领域定制训练的必要性。

---

## 后续优化方向

1. **CBAM + WIoU 魔改训练**：启动第二轮训练，预期 mAP 提升 5-10%
2. **DK2500 部署**：将最优模型部署至 Intel Core Ultra 5 225U 开发板
3. **NPU 加速**：利用 OpenVINO NPU 后端进一步提升推理速度

---

## 技术栈

- **AI 框架**：PyTorch, Ultralytics YOLOv8
- **推理加速**：OpenVINO 2026.1.0
- **开发环境**：Anaconda, VS Code
- **目标硬件**：DK2500 (Intel Core Ultra 5 225U)

---

## 作者信息

- **责任编辑**：袁康睿
- **团队成员**：彭日骏、刘宇杭（后续DK2500 部署）

---

*项目持续更新中，欢迎贡献代码和建议。*
