# 操作日志

> 记录项目中的关键操作、决策与变更，便于回溯与交接。

---

## 2026-05-28

### 项目全貌梳理

**操作人**：Claude Code（AI 助手）

**操作内容**：
- 全面走访项目所有文件与目录
- 确认项目状态：Baseline 模型训练完成（mAP50=0.5855），OpenVINO 部署链路打通
- 确认下一步计划：P2 检测头验证为最高优先级
- 创建本操作日志文件，从今日开始记录所有关键操作

**当前项目状态**：
- 最优模型：Baseline (Exp-01)，mAP50 = 0.5855
- P2 检测头配置已就绪：`configs/yolov8n-p2.yaml`
- `start_training.py` 已修改为加载 P2 模型，输出名称为 `baseline_p2_v1`
- IR 模型已导出：`runs/detect/baseline_train/weights/best_openvino_model/`

**待执行**：
- [ ] 运行 P2 检测头训练（50 轮，CPU 约 8 尀时）
- [ ] 验证 defect_2 AP50 > 0.50，整体 mAP50 > 0.60
- [ ] DK2500 硬件实测

### 行动准则确认

**操作人**：用户

**准则一**：长时间运行的任务（如模型训练）必须由用户手动启动，Claude Code 不得自动执行。

**准则二**：改代码、运行验证、改文档等日常操作，Claude Code 直接执行不用问。动手前说一声要做什么，完成后汇报做了什么。

**已更新至**：`CLAUDE.md` → 行动准则章节

### 整理 DeepSeek 实验方法论建议

**操作人**：Claude Code

**操作内容**：
- 阅读并领会 DeepSeek 提供的科学实验方法论
- 创建 `docs/EXPERIMENT_GUIDE.md`，记录核心原则、优化路线图、推荐实验计划
- 关键发现：defect_2 平均框 191×194px，可能不是纯小目标问题，需进一步分析尺寸分布

**核心要点**：
- 前测→单变量→后测，每次只改一个因素
- 先 20-30 轮快速消融，确认有效再跑全量
- 当前缺失指标：每类 Recall、defect_2 尺寸分布中位数、负样本误报率

### 补充前测指标

**操作人**：Claude Code

**操作内容**：
- 扩展 `scripts/analyze_dataset.py`，新增 defect_2 尺寸中位数和分段统计
- 创建 `scripts/val_detailed.py`，输出每类 AP50/Precision/Recall + 负样本误报率
- 修复负样本识别 bug（无标注文件的图片才是负样本，非空 .txt）

**关键发现**：

| 指标 | 值 | 意义 |
|:---|:---|:---|
| defect_2 Recall | 0.3785 | 62% 的 defect_2 被漏检，核心问题 |
| defect_2 中位数尺寸 | 149.6×139.5 px | 不是小目标！P2 收益可能有限 |
| <32px 的 defect_2 | 仅 4% | 真正小目标极少 |
| 负样本 FPPI | 0.0729 | PASS，误报率达标 |

**结论修正**：defect_2 的主要问题不是"小目标漏检"，而是"中大目标漏检"（Recall 0.3785），原因可能是对比度低、背景干扰。

---

## 2026-05-31

### 项目目录整理

**操作人**：Claude Code

**操作内容**：
- 将根目录散落的 3 个实验脚本移入 `scripts/`：`convert_infer_evaluate.py`、`full_compare.py`、`infer_openvino_compare.py`
- 新建 `models/` 目录，统一归集模型相关文件：
  - `models/weights/yolov8n.pt`（官方预训练权重）
  - `models/baseline_best_openvino_model/`（Baseline OpenVINO 模型）
  - `models/yolov8n_openvino_model_official/`（官方 OpenVINO 模型）
  - `models/backup/baseline_best.pt/`（Baseline 训练备份含图表）
- 将 `reports/ov_ir_evaluation_report.txt` 和 `kernel.errors.txt` 移入 `docs/`
- 删除空目录：`outputs/`、`infer/`
- 更新所有脚本中的路径引用（6 个文件共 9 处路径修改）
- 更新 `CLAUDE.md` 项目结构描述

**整理前后对比**：

| 整理前（根目录） | 整理后 |
|:---|:---|
| 3 个散落 .py 脚本 | 全部移入 `scripts/` |
| `yolov8n.pt` 在根目录 | → `models/weights/` |
| `baseline_best_openvino_model/` | → `models/` |
| `yolov8n_openvino_model_official/` | → `models/` |
| `backup/` | → `models/backup/` |
| `reports/`、`kernel.errors.txt` | → `docs/` |
| `outputs/`、`infer/`（空目录） | 已删除 |

**整理后一级目录**：仅 6 个（`scripts/`、`configs/`、`models/`、`datasets/`、`runs/`、`docs/`）+ 根目录配置文件

### 三模型对照实验脚本

**操作人**：Claude Code

**操作内容**：
- 创建 `scripts/three_model_compare.py`，覆盖三模型（Official YOLOv8n / Baseline / CBAM+WIoU）对照评估
- 评估指标共 7 项 + 速度：
  1. mAP@0.5
  2. mAP@0.5:0.95
  3. Precision（整体）
  4. Recall（整体）
  5. F1 Score
  6. 每类 AP50（defect_2 / defect_3 / defect_6）
  7. FPPI（负样本误报率）
  + OpenVINO 平均推理耗时 / FPS
- 输出：控制台表格 + `runs/detect/three_model_compare/comparison_report.txt`

**指令来源**：彭日骏 → 袁康睿（2026-05-31）
