# 鑫光板材瑕疵检测项目

## 项目概述

基于 YOLOv8 + OpenVINO 的工业板材瑕疵检测系统，英特尔杯嵌入式AI专题赛参赛项目。
检测三类板材瑕疵：defect_2, defect_3, defect_6。
目标部署硬件：DK2500 (Intel Core Ultra 5 225U)。

## 技术栈

- **AI 框架**: PyTorch, Ultralytics YOLOv8
- **推理加速**: OpenVINO 2026.1.0
- **图像处理**: OpenCV, NumPy
- **工具**: tqdm, PyYAML
- **开发环境**: Anaconda, VS Code, Windows + Python 3.10

## 项目结构

```
XinGuang/
├── scripts/                          # 可执行 Python 脚本
│   ├── prepare_dataset.py            # LabelMe JSON → YOLO 格式转换 + 滑窗裁剪
│   ├── start_training.py             # 模型训练启动（Baseline / 魔改）
│   ├── train_v2.py                   # 魔改模型第二轮训练（CBAM+WIoU, 100 epochs）
│   ├── resume_training.py            # 恢复中断的训练
│   ├── val_baseline.py               # 精度验证（mAP, P, R）
│   ├── infer_openvino.py             # OpenVINO 推理与性能基准测试
│   ├── infer_openvino_compare.py     # 双模型 OpenVINO 推理对比
│   ├── full_compare_openvino.py      # 官方 vs 自训练全面对比（精度+速度）
│   └── export_int8.py                # INT8 量化导出
├── configs/
│   ├── xinguang.yaml                 # 数据集配置（3类, 路径）
│   └── yolov8n_cbam_wiou.yaml        # CBAM + WIoU 魔改模型配置
├── datasets/
│   ├── images/train, images/val      # YOLO 格式图片
│   └── labels/train, labels/val      # YOLO 格式标注
├── runs/detect/                      # 训练结果 + 推理输出（不移动）
├── backup/baseline_best.pt/          # Baseline 训练备份（含图表）
├── baseline_best_openvino_model/     # Baseline OpenVINO 导出模型
├── yolov8n_openvino_model_official/  # 官方 YOLOv8n OpenVINO 模型
├── docs/                             # 项目文档
├── requirements.txt                  # 依赖清单
├── export_and_infer.py               # 快速导出+推理脚本（实验用）
├── full_compare.py                   # 初始对比脚本（实验用）
├── infer_openvino_compare.py         # 初始对比脚本（实验用）
└── yolov8n.pt                        # 官方预训练权重
```

## 关键脚本说明

### 数据预处理
- `prepare_dataset.py`: 将 LabelMe JSON 标注转为 YOLO 格式，滑窗裁剪生成 640×640 子图（步长 500）
  - 正样本保留条件：交集/原框面积 > 20% 或 交集绝对面积 > 200px
  - 负样本保留概率 10%
  - 训练/验证集 8:2 随机划分

### 模型训练
- `start_training.py`: Baseline/魔改训练启动（CPU, batch=8, workers=0, epochs=50）
  - 小目标增强策略：mosaic=0.3, copy_paste=0.3, scale=0.3, 关闭旋转/MixUp
- `train_v2.py`: 第二轮训练（加载 v1 last.pt，100 epochs, workers=6）
- `resume_training.py`: 用 `model.train(resume=True)` 恢复中断训练

### 验证与推理
- `val_baseline.py`: PyTorch 精度验证（mAP@0.5, mAP@0.5:0.95, P, R）
- `infer_openvino.py`: OpenVINO 推理 + 性能基准测试（纯推理 FPS, 端到端 FPS）
- `full_compare_openvino.py`: 官方 YOLOv8n vs Baseline 全面对比（精度+OpenVINO速度）

### 模型导出
- `export_int8.py`: OpenVINO INT8 量化导出（需验证集图片进行校准）

## 编码约定

- **路径**: 使用绝对路径 `D:/files_1/PythonProject/XinGuang/...`
- **中文路径安全**: 统一用 `cv2.imdecode(np.fromfile(...))` 和 `cv2.imencode(...).tofile(...)` 读写图片
- **Windows 安全**: 训练脚本开头添加 `multiprocessing.freeze_support()`
- **注释**: Python 注释用中文
- **日志**: 统一使用 `print(f"[INFO] ...")` / `print(f"[WARN] ...")` / `print(f"[ERROR] ...")` 格式
- **tqdm**: 推理循环使用 tqdm 进度条，显示当前推理耗时
- **模型加载**: 通过 `YOLO(path)` 加载，支持 `.pt` 和 OpenVINO 目录

## 训练参数惯例

| 参数 | Baseline | 魔改 v2 |
|------|----------|---------|
| epochs | 50 | 100 |
| batch | 8 | 8 |
| imgsz | 640 | 640 |
| device | cpu | cpu |
| workers | 4 (初版) / 0 (start_training) | 6 |
| 数据增强 | mosaic=1.0 | mosaic=0.3, copy_paste=0.3 |

## 模型性能基线

| 指标 | Baseline |
|------|----------|
| mAP@0.5 | 0.5865 |
| mAP@0.5:0.95 | 0.3017 |
| OpenVINO CPU FPS | ~58 |
| 平均推理耗时 | ~17.22 ms |

## 操作日志

**每次执行关键操作后，必须更新 `docs/OPERATION_LOG.md`。** 记录内容包括：
- 操作时间
- 操作人（用户或 Claude Code）
- 具体操作内容
- 操作结果
- 当前项目状态快照

## 行动准则

### 小事自主，大事确认

**改代码、运行验证、改文档等日常操作，Claude Code 直接执行，不用问用户。** 但需遵守：
- **动手前**：简要说明要做什么
- **完成后**：汇报做了什么、结果如何

**长时间运行的任务（如模型训练）必须由用户手动启动，Claude Code 不得自动执行。** Claude Code 只负责准备配置、确认参数，然后告知用户手动运行命令。

## 实验方法论

**核心原则：前测 → 单变量 → 后测**（详见 `docs/EXPERIMENT_GUIDE.md`）

1. **前测**：改动前先获取完整 Baseline 指标（mAP、每类 AP、每类 Recall、负样本误报率）
2. **单变量**：每次只改一个因素（模型结构 OR 损失函数 OR 数据增强，不混改）
3. **快速消融**：先 20-30 轮验证趋势，确认有效再跑全量
4. **后测**：用相同指标体系对比，量化收益

**教训**：Exp-02 (CBAM+WIoU) 两个改动一起上 + 没有消融验证 → 100 轮精度反降

## 注意事项

- 不要在未确认的情况下修改 datasets/ 和 runs/ 下的内容（它们体积大且是训练产物）
- 实验性脚本（根目录下的 .py 文件）与正式脚本（scripts/ 下的）功能可能重叠，改动前先确认
- 魔改模型依赖 ultralytics 源码中手动添加的 CBAM 模块和 WIoU 损失修改
- OpenVINO INT8 量化导出依赖验证集图片进行校准
