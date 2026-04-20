# 鑫光板材瑕疵检测项目总结

> 整理时间：2026-04-20  
> 负责人：袁康睿  
> 项目背景：英特尔杯嵌入式AI专题赛，基于DK2500开发套件的工业板材瑕疵检测系统。

## 一、项目目标

- 利用鑫光板材数据集，训练一个 YOLOv8 目标检测模型，识别三类板材瑕疵（defect_2, defect_3, defect_6）。
- 在 Intel CPU 上验证 OpenVINO 推理链路，确保模型可顺利迁移至 DK2500 开发板。
- 建立可量化的 Baseline，并在此基础上实施 CBAM + WIoU 魔改，提升检测精度。

## 二、已完成工作清单

### 1. 数据获取与工程化准备
- 从百度网盘免费获取 10.5GB 原始数据集。
- 使用 Trae Builder 扫描数据集目录，自动生成《关于鑫光板材检测数据集的情况说明.pdf》。
- 识别中文路径潜在风险，将全部文件夹及文件名改为英文命名规范。
- 编写/使用 `prepare_dataset.py` 将 LabelMe JSON 标注转为 YOLO 格式，并通过滑窗裁剪生成 640×640 子图，最终形成 3287 张训练图片与对应标注。

### 2. 训练环境搭建与 Baseline 训练
- 在 Windows + Anaconda 环境下配置 PyTorch + Ultralytics。
- 编写 `start_training.py`，针对 CPU 训练优化参数（`batch=8`, `workers=4`, `epochs=50`）。
- 成功完成 Baseline 模型训练，产出 `runs/detect/baseline_train/weights/best.pt`。

### 3. 模型验证与精度评估
- 编写 `val_baseline.py`，对 Baseline 模型进行精度验证。
- 获得关键指标：
  - mAP@0.5: **0.5865**
  - mAP@0.5:0.95: 0.3017
  - 其中小瑕疵 defect_2 的 mAP@0.5 为 0.417，验证了引入注意力机制的必要性。

### 4. OpenVINO 推理链路打通
- 编写 `infer_openvino.py`，实现从 `.pt` 到 OpenVINO 格式的自动导出与推理。
- Baseline 模型在 OpenVINO CPU 后端上推理速度：
  - 纯推理 FPS: **58.08**
  - 平均单张耗时: **17.22 ms**
- 证明模型在 Intel CPU 上具备实时检测能力，为 DK2500 部署奠定基础。

### 5. 官方模型 vs Baseline 全面对比
- 编写 `full_compare_openvino.py`，完成官方 `yolov8n.pt` 与自训练 `best.pt` 的精度、速度双维度对比。
- 对比报告核心结论：
  | 指标 | 官方 YOLOv8n | Baseline |
  |---|---|---|
  | mAP50 | 0.0050 | **0.5865** |
  | OpenVINO FPS | ~70 | ~58 |
- 证明定制化训练的必要性，产出报告 `openvino_full_comparison_report.txt`。

### 6. 魔改模型源码准备
- 在 `ultralytics` 源码中完成以下修改：
  - 添加 CBAM 注意力模块（`nn/modules/cbam.py`，并注册到 `tasks.py`）。
  - 修改 `metrics.py` 与 `loss.py`，支持 WIoU 损失函数。
- 创建魔改配置文件 `yolov8n_cbam_wiou.yaml`。
- 魔改模型已通过加载测试，待启动第二轮训练。

## 三、当前项目关键文件说明

| 文件/目录 | 用途 |
|---|---|
| `datasets/` | 处理后的 YOLO 格式数据集（images + labels） |
| `xinguang.yaml` | 数据集配置文件（类别数、路径） |
| `prepare_dataset.py` | 原始数据集预处理脚本 |
| `start_training.py` | Baseline / 魔改训练启动脚本 |
| `infer_openvino.py` | OpenVINO 推理与速度测试脚本 |
| `val_baseline.py` | PyTorch 精度验证脚本 |
| `full_compare_openvino.py` | 官方 vs Baseline 双模型对比脚本 |
| `yolov8n_cbam_wiou.yaml` | CBAM + WIoU 魔改模型配置文件 |
| `runs/detect/baseline_train/` | Baseline 训练结果（含 best.pt） |
| `runs/detect/openvino_full_compare/` | 对比测试结果图片与报告 |
| `PROJECT_SUMMARY.md` | 本文件 |

## 四、下一步计划

1. 启动 CBAM + WIoU 魔改模型训练（`epochs=100`），产出 `cbam_wiou_v1/best.pt`。
2. 对魔改模型进行精度验证，与 Baseline 对比 mAP 提升幅度。
3. 将最优模型交付队友刘宇杭，在 DK2500 开发板上完成 OpenVINO NPU 部署。
4. 完善项目文档，上传至 GitHub。

## 五、技术栈与工具链

- **AI 框架**：PyTorch, Ultralytics YOLOv8
- **推理加速**：OpenVINO 2026.1.0
- **开发环境**：Anaconda, VS Code, Trae IDE
- **辅助工具**：Gemini 3.1 Pro, Claude 4.7, DeepSeek
- **目标硬件**：DK2500 (Intel Core Ultra 5 225U)

---

*此文档记录了从数据获取到模型部署预演的全过程，可作为项目交接与技术复盘的唯一事实来源。*