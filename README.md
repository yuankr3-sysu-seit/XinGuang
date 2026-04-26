# 鑫光板材瑕疵检测系统

> 基于 YOLOv8 + OpenVINO 的工业板材表面瑕疵实时检测系统  
> 英特尔杯嵌入式 AI 专题赛参赛项目 | 目标硬件：DK2500 (Intel Core Ultra 5 225U)

---

## 项目简介

本项目针对鑫光板材工业线扫图像，训练 YOLOv8n 目标检测模型，识别三类板材表面瑕疵（点状瑕疵、中型区域瑕疵、大型/边缘瑕疵），并通过 OpenVINO 在 CPU 上实现高速推理，为 DK2500 开发套件端侧部署奠定基础。

**技术栈**：PyTorch + Ultralytics YOLOv8 + OpenVINO 2026.1.0  
**目标硬件**：DK2500 (Intel Core Ultra 5 225U，NPU 加速)

**瑕疵类别**：

| 类别 | 描述 | 示例 |
|------|------|------|
| defect_2 | 点状小瑕疵（针孔、气泡） | 10×10 px 级微小目标 |
| defect_3 | 中型区域瑕疵（划痕、凹陷） | 中等面积缺陷 |
| defect_6 | 大型或边缘瑕疵 | 大面积或板材边缘缺陷 |

---

## 环境要求

- **Python**：3.8+（推荐 3.10）
- **操作系统**：Windows / Linux
- **CUDA**（可选）：GPU 加速训练

### 安装步骤

```bash
# 创建虚拟环境（推荐）
conda create -n xinguang python=3.10
conda activate xinguang

# 安装依赖
pip install -r requirements.txt
```

### 关键依赖

| 包 | 用途 |
|------|------|
| ultralytics>=8.0.0 | YOLOv8 目标检测框架 |
| openvino>=2026.1.0 | Intel OpenVINO 推理引擎 |
| opencv-python>=4.8.0 | 图像处理 |
| numpy>=1.24.0 | 数值计算 |
| tqdm>=4.66.0 | 进度条 |
| PyYAML>=6.0 | YAML 配置文件解析 |

---

## 项目结构

```
XinGuang/
├── scripts/                              # 可执行 Python 脚本
│   ├── prepare_dataset.py                # LabelMe JSON → YOLO 格式 + 滑窗裁剪
│   ├── start_training.py                 # 模型训练启动（Baseline / 魔改）
│   ├── train_v2.py                       # 魔改模型第二轮训练（CBAM+WIoU, 100 epochs）
│   ├── resume_training.py                # 恢复中断的训练
│   ├── val_baseline.py                   # 精度验证（mAP, P, R）
│   ├── infer_openvino.py                 # OpenVINO 推理与性能基准测试
│   ├── full_compare_openvino.py          # 官方 vs 自训练全面对比（精度+速度）
│   └── export_int8.py                    # INT8 量化导出
├── configs/                              # 配置文件
│   ├── xinguang.yaml                     # 数据集配置（3类, 路径）
│   ├── yolov8n_cbam_wiou.yaml            # CBAM+WIoU 魔改模型配置（已废弃）
│   └── yolov8n-p2.yaml                   # P2 高分辨率检测头配置（待验证）
├── datasets/                             # YOLO 格式数据集
│   ├── images/train/                     # 训练图片
│   ├── images/val/                       # 验证图片
│   ├── labels/train/                     # 训练标注
│   └── labels/val/                       # 验证标注
├── runs/detect/                          # 训练结果
│   ├── baseline_train/                   # Exp-01: Baseline 训练结果
│   │   └── weights/
│   │       └── best_openvino_model/      # OpenVINO IR 模型 (best.xml + best.bin)
│   ├── cbam_wiou_v1/                     # Exp-02: CBAM+WIoU 魔改结果
│   └── baseline_aug_v12/                 # Exp-03: 数据增强调整结果
├── reports/                              # 验证指标报告
│   └── ov_ir_evaluation_report.txt       # IR 模型验证指标
├── baseline_best_openvino_model/         # 4月22日Baseline OpenVINO 导出模型
│   ├── best.xml
│   └── best.bin
├── backup/baseline_best.pt/              # Baseline 训练备份（含图表）
├── convert_infer_evaluate.py             # 一键 IR 模型导出与验证
├── docs/                                 # 项目文档
│   ├── SUM_0426.md                       # 完整项目总结报告
│   └── PROJECT_SUMMARY.md                # 早期项目总结
├── requirements.txt                      # 依赖清单
└── README.md                             # 项目说明
```

### 脚本功能说明

| 脚本 | 功能 |
|------|------|
| `prepare_dataset.py` | 将 LabelMe JSON 标注转为 YOLO 格式，滑窗裁剪 640×640 子图（步长 500），双重保留机制确保微小瑕疵不丢失，负样本 10% 概率保留 |
| `start_training.py` | Baseline/魔改训练启动，支持 YOLOv8 官方配置和自定义 yaml 模型文件 |
| `val_baseline.py` | PyTorch 精度验证，输出 mAP@0.5、mAP@0.5:0.95、Precision、Recall |
| `infer_openvino.py` | OpenVINO 推理 + 全量验证集性能基准测试（输出纯推理 FPS、端到端 FPS） |
| `full_compare_openvino.py` | 官方 YOLOv8n vs Baseline 全面对比（精度 + OpenVINO 速度） |
| `export_int8.py` | OpenVINO INT8 量化导出（需验证集图片进行 NNCF 校准） |

---

## 已完成工作总结

### 数据工程

- **原始数据**：工业线扫长图（分辨率 1326×5006），LabelMe 矩形框标注
- **预处理**：`scripts/prepare_dataset.py` 实现滑窗切割（640×640，步长 500），边缘窗口自动贴边 + 黑色像素补齐
- **标注保留**：双重保留机制——交集面积占原框面积 > 20% 或交集绝对面积 > 200 px，确保微小瑕疵不丢失
- **负样本**：无瑕疵窗口按 10% 概率保留，用于评估误报率
- **数据集划分**：8:2 随机划分（随机种子 42），正负样本同时分布于训练集与验证集
- **最终规模**：训练集 ~2630 张，验证集 ~658 张，3 个类别

### 实验历程

本项目共完成三轮完整训练实验：

| 实验 | 名称 | 模型配置 | 轮次 | mAP50 | 结论 |
|:---|:---|:---|:---:|:---:|:---|
| Exp-01 | `baseline_train` | **YOLOv8n**，默认参数 | 50 | **0.5855** | **当前最优基线模型** |
| Exp-02 | `cbam_wiou_v1` | YOLOv8n + CBAM + WIoU | 100 | 0.5711 | CBAM 对小目标反效果，WIoU 小数据集收敛困难 |
| Exp-03 | `baseline_aug_v12` | YOLOv8n，降 mosaic + 加 copy_paste | 50 | 0.4466 | mosaic 骤降导致样本多样性不足 |

**失败归因**：
- **Exp-02**：CBAM 全局池化淹没微小瑕疵特征，随机初始化权重破坏预训练模型；WIoU 默认超参数（γ=1.9）针对 COCO 调优，小数据集上收敛困难
- **Exp-03**：mosaic 从 1.0 骤降至 0.3，训练样本多样性断崖式下跌，学习率未同步调整

### 部署验证

成功将 Baseline 模型导出为 OpenVINO 格式，在 Intel Core Ultra 7 155H（CPU）上实现：
- **纯推理 FPS**：**107.32**
- **平均单张推理耗时**：**9.32 ms**
- **部署链路已完全打通**，推理速度远超实时检测需求（目标 FPS ≥ 50）

### 官方模型对比

| 模型 | mAP50 | OpenVINO FPS |
|:---|:---:|:---:|
| 官方 YOLOv8n | **0.005** | ~13 |
| Baseline (Exp-01) | **0.5855** | ~107 |

官方预训练模型在本任务上完全失效（精度趋近于零），证明了特定工业场景定制训练的不可替代性。

---

## 当前模型性能

### Baseline 总体指标

| 指标 | 值 |
|------|:---:|
| mAP@0.5 | **0.5855** |
| mAP@0.5:0.95 | 0.3017 |
| Precision | 0.6752 |
| Recall | 0.5149 |

### 各类别 AP50

| 类别 | 描述 | AP50 | 状态 |
|:---|:---|---:|:---|
| defect_2 | 点状小瑕疵 | **0.417** | ⚠️ 核心短板 |
| defect_3 | 中型瑕疵 | **0.689** | ✅ 已接近可用 |
| defect_6 | 大型/边缘瑕疵 | **0.653** | ✅ 已接近可用 |

### IR 模型验证指标

**数据来源**：`reports/ov_ir_evaluation_report.txt`

| 指标 | 值 |
|------|:---:|
| mAP@0.5 | **0.5835** |
| mAP@0.5:0.95 | 0.3044 |

| 类别 | AP50 |
|:---|:---:|
| defect_2 | 0.4183 |
| defect_3 | 0.6783 |
| defect_6 | 0.6539 |

### OpenVINO 推理性能

| 测试平台 | 推理框架 | 平均耗时 | 纯推理 FPS |
|:---|:---|---:|:---:|
| Core Ultra 7 155H (CPU) | OpenVINO FP32 | **9.32 ms** | **107.32** |
| Core Ultra 5 225U / DK2500 (目标硬件) | OpenVINO | 待测 | 待测 |

### 精度瓶颈分析

**defect_2 小目标漏检是限制整体精度的核心矛盾。** 原因：
1. **物理像素瓶颈**：YOLOv8n 默认最小检测头 P3 下采样 8 倍，10×10 px 的 defect_2 在 P3 层仅剩约 1.25×1.25 px 特征
2. **背景干扰**：复杂木纹纹理与小瑕疵在特征上高度相似，模型难以区分

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 数据预处理

将 LabelMe JSON 标注 + 原始长图转换为 YOLO 格式数据集：

```bash
python scripts/prepare_dataset.py
```

预处理参数（在脚本顶部配置）：
- 滑窗尺寸：640×640
- 滑窗步长：500
- 正样本保留阈值：IoU > 20% 或交集面积 > 200 px
- 负样本保留概率：10%
- 训练/验证集划分：8:2

### 3. 模型训练

启动 Baseline 训练（推荐先从此开始）：

```bash
python scripts/start_training.py
```

训练参数（硬编码在 `start_training.py` 中）：

| 参数 | 值 |
|------|:---:|
| epochs | 50 |
| batch | 8 |
| imgsz | 640 |
| device | cpu |
| workers | 0 |

### 4. 精度验证

```bash
python scripts/val_baseline.py
```

输出：mAP@0.5、mAP@0.5:0.95、Precision、Recall。

### 5. OpenVINO 推理与性能测试

```bash
python scripts/infer_openvino.py
```

自动完成：PT 模型 → OpenVINO 导出 → 全量验证集推理 → 性能报告（FPS、平均耗时）。

### 6. 模型对比

对比官方 YOLOv8n 与自训练 Baseline 的精度和速度：

```bash
python scripts/full_compare_openvino.py
```

### 7. INT8 量化导出（实验性）

```bash
python scripts/export_int8.py
```

> **注意**：当前 `export_int8.py` 指向 `cbam_wiou_v1` 模型，如需量化 Baseline 模型请修改脚本中的 `model_path`。

### 8. IR 模型导出与验证

一键完成 PT 模型 → IR 模型导出，并对验证集进行推理验证：

```bash
python convert_infer_evaluate.py
```

执行后将自动完成：
1. 将 `best.pt` 导出为 OpenVINO IR 格式（生成 `.xml` 与 `.bin` 文件）
2. 使用 IR 模型在验证集上推理
3. 输出并保存精度指标报告

生成的文件：
- IR 模型：`runs/detect/baseline_train/weights/best_openvino_model/best.xml` 与 `best.bin`
- 指标报告：`reports/ov_ir_evaluation_report.txt`

---

## 下一步工作

### 最高优先级：P2 检测头验证

**问题**：defect_2 小目标漏检是当前主要瓶颈。P2 高分辨率检测头（下采样 4 倍而非 8 倍）可显著提升小目标特征分辨率。

**配置已就绪**：`configs/yolov8n-p2.yaml`（四检测头 P2/P3/P4/P5，无注意力模块）

**操作步骤**：
1. 确保 `start_training.py` 中加载 `configs/yolov8n-p2.yaml`
2. 设置输出名称为 `baseline_p2_v1`
3. 运行 `python scripts/start_training.py`，训练 50 轮
4. 验收标准：defect_2 AP50 > **0.50**，整体 mAP50 > **0.60**

**预期收益**：defect_2 AP50 提升 10~15 个百分点，整体 mAP50 冲击 0.62~0.65  
**预期耗时**：CPU 约 8 小时 / 50 轮

### DK2500 部署验证

- IR 模型已就绪，路径为 `runs/detect/baseline_train/weights/best_openvino_model/`
- 队友可直接使用 `best.xml` 和 `best.bin` 进行部署
- 在 DK2500 上运行推理，确认 FPS ≥ 50
- 验证 NPU 后端推理速度

### 云端 GPU 充分训练

若 P2 路线验证成功，建议在 AutoDL 等平台租用 RTX 4090 将模型跑满 300 轮：
- 预计耗时：3-5 小时
- 预计费用：10-15 元
- 预期 mAP50：冲击 **0.75~0.78**

### 其他优化方向

| 优先级 | 优化项 | 预期收益 | 风险 |
|:---|:---|---:|:---|
| 中 | 余弦退火学习率（`cos_lr=True`） | +1~2 点 | 低 |
| 中 | K-means 定制锚框 | +2~3 点 | 低 |
| 低 | CoordAtt 注意力模块 | +3~5 点 | 需验证 OpenVINO NPU 兼容性 |
| ❌ | CBAM 注意力 | — | **已证伪，对小目标反效果** |
| ❌ | WIoU 损失函数 | — | **小数据集上水土不服** |

### 核心经验教训

1. **先消融，后全量**：任何结构或超参数改动，先通过 20-30 轮消融实验验证趋势
2. **验证硬件兼容性**：非标准算子可能导致 OpenVINO NPU 推理速度断崖式下跌
3. **负样本不可缺**：预处理阶段保留无瑕疵背景图，是防止产线误报的基础保障

---

## 作者信息

- **彭日骏** — 项目统筹，软硬协同
- **刘宇杭** — DK2500硬件配套部署等
- **袁康睿** — 前期模型训练，项目文档

**团队**：英特尔杯嵌入式 AI 专题赛参赛团队  
**院校**：中山大学

---

*项目持续更新中。*
