# -*- coding: utf-8 -*-
"""
full_compare_openvino.py
终极对比脚本：官方 YOLOv8n vs. 自训练 Baseline best.pt

功能：
1. 分别将两个模型导出为 OpenVINO 格式（Intel 加速引擎）
2. 在验证集上测试 OpenVINO 推理速度（找回 70+ FPS）
3. 在验证集上测试 PyTorch 精度（mAP, P, R）
4. 生成一份包含精度和速度的全面对比报告
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
DATA_YAML = os.path.join(PROJECT_ROOT, "configs", "xinguang.yaml")

# 模型配置列表
MODEL_CONFIGS = [
    {
        "name": "Official_YOLOv8n",
        "pt_path": os.path.join(PROJECT_ROOT, "yolov8n.pt"),
        "ov_dir": os.path.join(PROJECT_ROOT, "yolov8n_openvino_model_official")
    },
    {
        "name": "Baseline_best",
        "pt_path": os.path.join(PROJECT_ROOT, "runs", "detect", "baseline_train", "weights", "best.pt"),
        "ov_dir": os.path.join(PROJECT_ROOT, "baseline_best_openvino_model")
    }
]

# 推理参数
DEVICE = "cpu"          # OpenVINO 会自动调用 CPU 插件
IMG_SIZE = 640
CONF_THRES = 0.25
IOU_THRES = 0.45
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# 结果保存目录
SAVE_DIR_BASE = os.path.join(PROJECT_ROOT, "runs", "detect", "openvino_full_compare")
REPORT_FILE = os.path.join(SAVE_DIR_BASE, "openvino_full_comparison_report.txt")


# ============================================================
# 二、工具函数
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

def export_to_openvino(pt_path: str, ov_dir: str):
    """导出 PyTorch 模型为 OpenVINO 格式"""
    if os.path.isdir(ov_dir) and len(os.listdir(ov_dir)) > 0:
        print(f"   [INFO] OpenVINO 模型已存在，跳过导出：{ov_dir}")
        return ov_dir
    print(f"   [INFO] 正在导出 OpenVINO 模型...")
    model = YOLO(pt_path)
    model.export(format="openvino", imgsz=IMG_SIZE, half=False)
    # Ultralytics 默认导出到 pt 同目录下的 <stem>_openvino_model
    expected_dir = str(Path(pt_path).parent / (Path(pt_path).stem + "_openvino_model"))
    
    if os.path.isdir(expected_dir):
        # 如果导出到了默认位置，移动到我们指定的 ov_dir
        if expected_dir != ov_dir:
            import shutil
            if os.path.exists(ov_dir):
                shutil.rmtree(ov_dir)
            shutil.move(expected_dir, ov_dir)
    return ov_dir

def test_openvino_speed(ov_dir: str, img_paths: list, save_dir: str):
    """使用 OpenVINO 模型测试推理速度并保存结果图"""
    print(f"   [INFO] 加载 OpenVINO 模型：{ov_dir}")
    model = YOLO(ov_dir, task="detect")

    # 预热
    warmup_img = imread_unicode(img_paths[0])
    if warmup_img is not None:
        _ = model.predict(warmup_img, device=DEVICE, imgsz=IMG_SIZE,
                          conf=CONF_THRES, iou=IOU_THRES, verbose=False)

    infer_ms_list = []
    t_start = time.perf_counter()
    pbar = tqdm(img_paths, desc="   OpenVINO 测速", ncols=100, unit="img")
    for img_path in pbar:
        img = imread_unicode(img_path)
        if img is None:
            continue
        try:
            results = model.predict(img, device=DEVICE, imgsz=IMG_SIZE,
                                    conf=CONF_THRES, iou=IOU_THRES, verbose=False)
        except Exception:
            continue
        if not results:
            continue
        result = results[0]
        infer_ms = float(result.speed.get("inference", 0.0))
        infer_ms_list.append(infer_ms)

        # 保存结果图
        plotted = result.plot()
        save_name = Path(img_path).stem + ".jpg"
        save_path = os.path.join(save_dir, save_name)
        cv2.imencode(".jpg", plotted)[1].tofile(save_path)  # 兼容中文路径

        pbar.set_postfix_str(f"{infer_ms:.1f}ms")

    total_sec = time.perf_counter() - t_start
    return infer_ms_list, total_sec

def format_speed_report(model_name, n_imgs, total_sec, infer_ms_list):
    infer_total_ms = sum(infer_ms_list) if infer_ms_list else 0.0
    avg_infer_ms = infer_total_ms / len(infer_ms_list) if infer_ms_list else 0.0
    fps_pure = 1000.0 / avg_infer_ms if avg_infer_ms > 0 else 0.0
    fps_e2e = n_imgs / total_sec if total_sec > 0 else 0.0

    lines = [
        f"模型名称          : {model_name}",
        f"测试图片数        : {n_imgs}",
        f"端到端总耗时      : {total_sec:.2f} s",
        f"纯推理总耗时      : {infer_total_ms / 1000.0:.2f} s",
        f"平均单张推理耗时  : {avg_infer_ms:.2f} ms",
        f"纯推理 FPS        : {fps_pure:.2f}",
        f"端到端 FPS        : {fps_e2e:.2f}"
    ]
    return "\n".join(lines)


# ============================================================
# 三、主流程
# ============================================================

def main():
    print("=" * 70)
    print("         OpenVINO 终极对比测试：官方 vs Baseline")
    print("=" * 70)

    # 准备图片列表
    img_paths = collect_images(VAL_IMG_DIR)
    if not img_paths:
        print(f"[ERROR] 验证集目录 {VAL_IMG_DIR} 中没有图片。")
        return
    print(f"[INFO] 共发现 {len(img_paths)} 张验证图片。")

    ensure_dir(SAVE_DIR_BASE)
    all_results = []

    for cfg in MODEL_CONFIGS:
        name = cfg["name"]
        pt_path = cfg["pt_path"]
        ov_dir = cfg["ov_dir"]

        print(f"\n{'='*50}")
        print(f"处理模型：{name}")
        print(f"{'='*50}")

        # ---------- 1. 导出 OpenVINO ----------
        ov_path = export_to_openvino(pt_path, ov_dir)

        # ---------- 2. OpenVINO 速度测试 ----------
        print(f"   [INFO] 开始 OpenVINO 速度测试...")
        save_subdir = os.path.join(SAVE_DIR_BASE, f"{name}_images")
        ensure_dir(save_subdir)
        infer_ms_list, total_sec = test_openvino_speed(ov_path, img_paths, save_subdir)
        n_ok = len(infer_ms_list)
        avg_infer_ms = sum(infer_ms_list) / n_ok if n_ok else 0.0
        fps = 1000.0 / avg_infer_ms if avg_infer_ms > 0 else 0.0
        speed_report = format_speed_report(name, n_ok, total_sec, infer_ms_list)
        print(f"   [DONE] 纯推理 FPS = {fps:.2f}")

        # ---------- 3. PyTorch 精度验证 ----------
        print(f"   [INFO] 开始精度验证 (PyTorch)...")
        model_pt = YOLO(pt_path)
        metrics = model_pt.val(data=DATA_YAML, imgsz=IMG_SIZE, device="cpu", verbose=False, plots=False)
        map50 = metrics.box.map50
        map50_95 = metrics.box.map
        precision = metrics.box.p[0] if len(metrics.box.p) > 0 else 0.0
        recall = metrics.box.r[0] if len(metrics.box.r) > 0 else 0.0
        print(f"   [DONE] mAP50={map50:.4f}, mAP50-95={map50_95:.4f}, P={precision:.4f}, R={recall:.4f}")

        all_results.append({
            "name": name,
            "map50": map50,
            "map50_95": map50_95,
            "precision": precision,
            "recall": recall,
            "fps": fps,
            "speed_report": speed_report
        })

    # ---------- 4. 生成对比报告 ----------
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("        OpenVINO 终极对比报告：官方模型 vs 自训练 Baseline\n")
        f.write("=" * 80 + "\n")
        f.write(f"验证数据集       : {DATA_YAML}\n")
        f.write(f"推理设备         : OpenVINO CPU\n")
        f.write(f"图片尺寸         : {IMG_SIZE}\n")
        f.write("=" * 80 + "\n\n")

        f.write("【一、精度指标对比 (PyTorch)】\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'模型':<20} {'mAP50':<10} {'mAP50-95':<10} {'Precision':<10} {'Recall':<10}\n")
        f.write("-" * 80 + "\n")
        for res in all_results:
            f.write(f"{res['name']:<20} {res['map50']:<10.4f} {res['map50_95']:<10.4f} {res['precision']:<10.4f} {res['recall']:<10.4f}\n")
        f.write("-" * 80 + "\n\n")

        f.write("【二、推理速度对比 (OpenVINO)】\n")
        f.write("-" * 80 + "\n")
        for res in all_results:
            f.write(f"  {res['name']:<20s} : {res['fps']:.2f} FPS\n")
        f.write("-" * 80 + "\n\n")

        f.write("【三、详细速度报告 (OpenVINO)】\n")
        f.write("=" * 80 + "\n")
        for res in all_results:
            f.write(res['speed_report'])
            f.write("\n" + "-" * 40 + "\n\n")

    print("\n" + "=" * 70)
    print(f"[DONE] 终极对比报告已保存至：\n{REPORT_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()