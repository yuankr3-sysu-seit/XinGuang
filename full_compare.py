# -*- coding: utf-8 -*-
"""
full_compare.py
全面的模型对比脚本：官方 YOLOv8n vs. 自训练 Baseline best.pt
对比维度：精度（mAP, P, R）与 速度（推理耗时, FPS）
"""

import os
import time
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
import numpy as np
import cv2

# ============================================================
# 一、全局配置
# ============================================================

PROJECT_ROOT = r"D:\files_1\PythonProject\XinGuang"
VAL_IMG_DIR = os.path.join(PROJECT_ROOT, "datasets", "images", "val")
DATA_YAML = os.path.join(PROJECT_ROOT, "xinguang.yaml")

# 模型配置列表
MODEL_CONFIGS = [
    {
        "name": "Official_YOLOv8n",
        "pt_path": os.path.join(PROJECT_ROOT, "yolov8n.pt"),
    },
    {
        "name": "Baseline_best",
        "pt_path": os.path.join(PROJECT_ROOT, "runs", "detect", "baseline_train", "weights", "best.pt"),
    }
]

# 验证参数
DEVICE = "cpu"
IMG_SIZE = 640
CONF_THRES = 0.25
IOU_THRES = 0.45
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# 结果保存目录
SAVE_DIR_BASE = os.path.join(PROJECT_ROOT, "runs", "detect", "full_compare")
REPORT_FILE = os.path.join(SAVE_DIR_BASE, "full_comparison_report.txt")


# ============================================================
# 二、工具函数 (图片读取与推理计时)
# ============================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def imread_unicode(path: str):
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None

def collect_images(img_dir: str):
    if not os.path.isdir(img_dir):
        return []
    files = []
    for name in os.listdir(img_dir):
        if os.path.splitext(name)[1].lower() in IMG_EXTS:
            files.append(os.path.join(img_dir, name))
    files.sort()
    return files

def format_speed_report(model_name, n_imgs, total_sec, infer_ms_list):
    infer_total_ms = sum(infer_ms_list) if infer_ms_list else 0.0
    avg_infer_ms = (infer_total_ms / len(infer_ms_list)) if infer_ms_list else 0.0
    fps_pure = (1000.0 / avg_infer_ms) if avg_infer_ms > 0 else 0.0
    fps_e2e = (n_imgs / total_sec) if total_sec > 0 else 0.0

    lines = []
    lines.append(f"模型名称          : {model_name}")
    lines.append(f"测试图片数        : {n_imgs}")
    lines.append(f"端到端总耗时      : {total_sec:.2f} s")
    lines.append(f"纯推理总耗时      : {infer_total_ms / 1000.0:.2f} s")
    lines.append(f"平均单张推理耗时  : {avg_infer_ms:.2f} ms")
    lines.append(f"纯推理 FPS        : {fps_pure:.2f}")
    lines.append(f"端到端 FPS        : {fps_e2e:.2f}")
    return "\n".join(lines)

def test_model_speed(model, img_paths):
    """
    测试模型在验证集上的推理速度。
    """
    infer_ms_list = []
    t_start = time.perf_counter()

    pbar = tqdm(img_paths, desc="   测速中", ncols=100, unit="img")
    for img_path in pbar:
        img = imread_unicode(img_path)
        if img is None:
            continue

        try:
            results = model.predict(
                source=img,
                device=DEVICE,
                imgsz=IMG_SIZE,
                conf=CONF_THRES,
                iou=IOU_THRES,
                verbose=False,
            )
        except Exception:
            continue

        if not results:
            continue

        result = results[0]
        infer_ms = float(result.speed.get("inference", 0.0))
        infer_ms_list.append(infer_ms)
        pbar.set_postfix_str(f"inf={infer_ms:.1f}ms")

    total_sec = time.perf_counter() - t_start
    ok_cnt = len(infer_ms_list)
    return ok_cnt, total_sec, infer_ms_list


# ============================================================
# 三、主流程
# ============================================================

def main():
    # 准备图片列表
    img_paths = collect_images(VAL_IMG_DIR)
    if not img_paths:
        print(f"[ERROR] 验证集目录 {VAL_IMG_DIR} 中没有图片。")
        return
    print(f"[INFO] 共发现 {len(img_paths)} 张验证图片，用于速度测试。")

    ensure_dir(SAVE_DIR_BASE)

    all_results = []
    for cfg in MODEL_CONFIGS:
        name = cfg["name"]
        pt_path = cfg["pt_path"]

        print(f"\n{'='*50}")
        print(f"开始全面评估模型：{name}")
        print(f"{'='*50}")

        # 1. 加载模型
        print(f"[INFO] 加载模型: {pt_path}")
        model = YOLO(pt_path)

        # 2. 精度验证 (PyTorch)
        print(f"[INFO] 正在进行精度验证...")
        metrics = model.val(data=DATA_YAML, imgsz=IMG_SIZE, device=DEVICE, plots=True, verbose=False)
        
        # 提取指标
        map50 = metrics.box.map50
        map50_95 = metrics.box.map
        precision = metrics.box.p[0]
        recall = metrics.box.r[0]
        print(f"   [DONE] mAP50: {map50:.4f}, mAP50-95: {map50_95:.4f}, P: {precision:.4f}, R: {recall:.4f}")

        # 3. 速度测试
        print(f"[INFO] 正在进行速度测试...")
        n_imgs, total_sec, infer_ms_list = test_model_speed(model, img_paths)
        speed_report = format_speed_report(name, n_imgs, total_sec, infer_ms_list)
        
        avg_infer_ms = (sum(infer_ms_list) / len(infer_ms_list)) if infer_ms_list else 0.0
        fps = 1000.0 / avg_infer_ms if avg_infer_ms > 0 else 0.0
        print(f"   [DONE] 纯推理 FPS: {fps:.2f}")

        # 存储结果
        all_results.append({
            "name": name,
            "map50": map50,
            "map50_95": map50_95,
            "precision": precision,
            "recall": recall,
            "fps": fps,
            "speed_report": speed_report
        })

    # 4. 汇总写入报告文件
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("     官方 YOLOv8n vs. 自训练 Baseline 模型 全面对比报告\n")
        f.write("=" * 70 + "\n")
        f.write(f"验证数据集       : {DATA_YAML}\n")
        f.write(f"测试设备         : {DEVICE}\n")
        f.write(f"图片尺寸         : {IMG_SIZE}\n")
        f.write("=" * 70 + "\n\n")

        # 精度对比表格
        f.write("【一、精度指标对比】\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'模型':<20} {'mAP50':<10} {'mAP50-95':<10} {'Precision':<10} {'Recall':<10}\n")
        f.write("-" * 70 + "\n")
        for res in all_results:
            f.write(f"{res['name']:<20} {res['map50']:<10.4f} {res['map50_95']:<10.4f} {res['precision']:<10.4f} {res['recall']:<10.4f}\n")
        f.write("-" * 70 + "\n\n")

        # 速度对比表格
        f.write("【二、推理速度对比 (纯推理 FPS)】\n")
        f.write("-" * 70 + "\n")
        for res in all_results:
            f.write(f"  {res['name']:<20s} : {res['fps']:.2f} FPS\n")
        f.write("-" * 70 + "\n\n")

        # 详细速度报告
        f.write("【三、详细速度报告】\n")
        f.write("=" * 70 + "\n")
        for res in all_results:
            f.write(res['speed_report'])
            f.write("\n" + "-" * 40 + "\n\n")

    print(f"\n[INFO] 全面对比报告已保存至：{REPORT_FILE}")
    print("[DONE] 模型全面对比测试完成。")

if __name__ == "__main__":
    main()