# -*- coding: utf-8 -*-
"""
一键转换、推理与指标输出脚本 (方案一)
功能：
  1. 将训练好的 best.pt 模型导出为 OpenVINO IR 格式 (生成 .xml 与 .bin)
  2. 使用导出的 IR 模型对验证集进行推理
  3. 输出精度指标 (mAP50, mAP50-95 以及各类别 AP50)
  4. 将指标保存到文本文件

作者：技术助理
运行方式：python convert_infer_evaluate.py
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO

# ============================================================
# 一、配置路径
# ============================================================
# 项目根目录 (请根据实际情况修改)
PROJECT_ROOT = r"D:/files_1/PythonProject/XinGuang"

# Baseline 最佳权重路径
BEST_PT = os.path.join(PROJECT_ROOT, "runs", "detect", "baseline_train", "weights", "best.pt")

# OpenVINO 模型输出目录 (将会在此生成 .xml 和 .bin)
OV_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "runs", "detect", "baseline_train", "weights", "best_openvino_model")

# 数据集配置文件
DATA_YAML = os.path.join(PROJECT_ROOT, "configs", "xinguang.yaml")

# 推理结果保存目录
INFER_SAVE_DIR = os.path.join(PROJECT_ROOT, "infer", "detect", "ov_ir_infer")

# 指标报告保存路径
REPORT_PATH = os.path.join(PROJECT_ROOT, "reports", "ov_ir_evaluation_report.txt")

# ============================================================
# 二、导出 OpenVINO IR 模型
# ============================================================
def export_to_ir():
    """将 PyTorch 模型导出为 OpenVINO IR (.xml + .bin)"""
    print(f"[1/3] 正在加载 PyTorch 模型: {BEST_PT}")
    model = YOLO(BEST_PT)
    
    print("[2/3] 正在导出为 OpenVINO IR 格式 ...")
    # 导出 OpenVINO 模型，half=True 使用 FP16 以加速推理
    model.export(format="openvino", imgsz=640, half=True)
    
    # 确认 IR 文件生成
    xml_file = os.path.join(OV_OUTPUT_DIR, "best.xml")
    bin_file = os.path.join(OV_OUTPUT_DIR, "best.bin")
    if os.path.exists(xml_file) and os.path.exists(bin_file):
        print(f"[3/3] ✅ IR 模型已成功生成: {xml_file} 和 {bin_file}")
    else:
        print("[WARNING] 导出完成，但未在预期位置找到 .xml/.bin 文件，请检查 Ultralytics 实际导出目录。")
    return OV_OUTPUT_DIR

# ============================================================
# 三、使用 IR 模型进行推理并计算指标
# ============================================================
def evaluate_ir_model(ir_dir):
    """使用导出的 IR 模型运行验证并输出指标"""
    print(f"\n[评估] 正在加载 IR 模型: {ir_dir}")
    # 加载 OpenVINO 模型
    model = YOLO(ir_dir, task="detect")
    
    # 预热 (可选，保证首次推理计时准确)
    print("[评估] 正在运行验证 (需要几分钟，请耐心等待) ...")
    
    # 调用 val 获取完整指标
    metrics = model.val(
        data=DATA_YAML,
        imgsz=640,
        device="cpu",          # 使用 OpenVINO 的 CPU 插件
        plots=False,           # 不生成图片以加速
        save_json=False,
        verbose=False
    )
    
    # 提取结果
    map50 = metrics.box.map50
    map50_95 = metrics.box.map
    # 各类别 AP50
    ap50_per_class = metrics.box.ap50.tolist() if hasattr(metrics.box.ap50, 'tolist') else metrics.box.ap50
    
    # 类别名称
    class_names = {0: "defect_2", 1: "defect_3", 2: "defect_6"}
    
    # 组装报告字符串
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("   OpenVINO IR 模型验证报告")
    report_lines.append("=" * 60)
    report_lines.append(f"   IR 模型路径       : {ir_dir}")
    report_lines.append(f"   数据集             : {DATA_YAML}")
    report_lines.append("-" * 60)
    report_lines.append(f"   {'类别':<15} {'AP50':<10}")
    report_lines.append("-" * 60)
    for i, ap in enumerate(ap50_per_class):
        name = class_names.get(i, f"class_{i}")
        report_lines.append(f"   {name:<15} {ap:<10.4f}")
    report_lines.append("-" * 60)
    report_lines.append(f"   整体 mAP50        : {map50:<10.4f}")
    report_lines.append(f"   整体 mAP50-95     : {map50_95:<10.4f}")
    report_lines.append("=" * 60)
    
    # 打印到终端
    for line in report_lines:
        print(line)
    
    # 保存报告
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\n[报告] 指标已保存至: {REPORT_PATH}")
    return report_lines

# ============================================================
# 四、主流程
# ============================================================
if __name__ == "__main__":
    # 1. 导出 IR 模型
    ir_dir = export_to_ir()
    
    # 2. 使用 IR 模型评估
    evaluate_ir_model(ir_dir)
    
    print("\n[DONE] 一键转换、推理、指标输出全部完成！")