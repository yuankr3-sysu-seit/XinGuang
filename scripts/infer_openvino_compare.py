# -*- coding: utf-8 -*-
"""
infer_openvino_compare.py
双模型 OpenVINO 推理对比脚本（官方 yolov8n vs 自训练 best.pt）

功能：
1. 分别导出官方 yolov8n.pt 和自训练 best.pt 为 OpenVINO 格式
2. 对同一验证集进行推理，记录性能指标
3. 将两份报告汇总输出到文本文件
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

# 模型配置列表：每个元素为 (模型名称, 权重文件路径, OpenVINO导出目录)
MODEL_CONFIGS = [
    {
        "name": "Official_YOLOv8n",
        "pt_path": os.path.join(PROJECT_ROOT, "models", "weights", "yolov8n.pt"),
        "ov_dir": os.path.join(PROJECT_ROOT, "models", "yolov8n_openvino_model_official")
    },
    {
        "name": "Baseline_best",
        "pt_path": os.path.join(PROJECT_ROOT, "runs", "detect", "baseline_train", "weights", "best.pt"),
        "ov_dir": os.path.join(PROJECT_ROOT, "models", "baseline_best_openvino_model")
    }
]

# 推理参数
DEVICE = "cpu"
IMG_SIZE = 640
CONF_THRES = 0.25
IOU_THRES = 0.45
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# 结果保存目录
SAVE_DIR_BASE = os.path.join(PROJECT_ROOT, "runs", "detect", "openvino_compare")
REPORT_FILE = os.path.join(SAVE_DIR_BASE, "comparison_report.txt")


# ============================================================
# 二、工具函数（保持原样，略作调整）
# ============================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def imread_unicode(path: str):
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None

def imwrite_unicode(path: str, img) -> bool:
    ext = os.path.splitext(path)[1]
    if not ext:
        ext = ".jpg"
        path = path + ext
    ok, buf = cv2.imencode(ext, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        return False
    try:
        buf.tofile(path)
        return True
    except Exception:
        return False

def collect_images(img_dir: str):
    if not os.path.isdir(img_dir):
        return []
    files = []
    for name in os.listdir(img_dir):
        if os.path.splitext(name)[1].lower() in IMG_EXTS:
            files.append(os.path.join(img_dir, name))
    files.sort()
    return files

def export_openvino_if_needed(pt_path: str, ov_dir: str) -> str:
    if os.path.isdir(ov_dir) and len(os.listdir(ov_dir)) > 0:
        print(f"   [INFO] 复用已存在的 OpenVINO 模型：{ov_dir}")
        return ov_dir
    if not os.path.isfile(pt_path):
        raise FileNotFoundError(f"未找到权重文件：{pt_path}")
    print(f"   [INFO] 导出 {os.path.basename(pt_path)} 为 OpenVINO 格式 ...")
    tmp_model = YOLO(pt_path)
    exported = tmp_model.export(format="openvino", imgsz=IMG_SIZE, half=False)
    if not os.path.isdir(ov_dir):
        if isinstance(exported, (str, os.PathLike)) and os.path.isdir(str(exported)):
            return str(exported)
        raise RuntimeError(f"导出后未找到 OpenVINO 模型目录：{ov_dir}")
    return ov_dir

def format_single_report(model_name, n_imgs, total_sec, infer_ms_list):
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


# ============================================================
# 三、单模型测试函数
# ============================================================

def test_model(model_cfg, img_paths, save_subdir):
    """
    对单个模型进行全量推理，返回性能指标字典和报告字符串。
    """
    name = model_cfg["name"]
    pt_path = model_cfg["pt_path"]
    ov_dir = model_cfg["ov_dir"]

    print(f"\n{'='*50}")
    print(f"开始测试模型：{name}")
    print(f"{'='*50}")

    # 1. 导出/加载 OpenVINO 模型
    ov_path = export_openvino_if_needed(pt_path, ov_dir)
    print(f"   [INFO] 加载 OpenVINO 模型：{ov_path}")
    model = YOLO(ov_path, task="detect")

    # 2. 预热
    warmup_img = imread_unicode(img_paths[0])
    if warmup_img is not None:
        _ = model.predict(source=warmup_img, device=DEVICE, imgsz=IMG_SIZE,
                          conf=CONF_THRES, iou=IOU_THRES, verbose=False)
    print("   [INFO] 预热完成，开始推理...")

    # 3. 推理循环
    infer_ms_list = []
    failed_cnt = 0
    t_start = time.perf_counter()

    pbar = tqdm(img_paths, desc=f"   {name}", ncols=100, unit="img")
    for img_path in pbar:
        img = imread_unicode(img_path)
        if img is None:
            failed_cnt += 1
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
            failed_cnt += 1
            continue

        if not results:
            continue

        result = results[0]
        infer_ms = float(result.speed.get("inference", 0.0))
        infer_ms_list.append(infer_ms)

        # 保存结果图（可选）
        plotted = result.plot()
        save_name = f"{name}_{Path(img_path).stem}.jpg"
        save_path = os.path.join(save_subdir, save_name)
        imwrite_unicode(save_path, plotted)

        pbar.set_postfix_str(f"inf={infer_ms:.1f}ms")

    total_sec = time.perf_counter() - t_start
    ok_cnt = len(infer_ms_list)

    # 4. 生成报告字符串
    report_str = format_single_report(name, ok_cnt, total_sec, infer_ms_list)
    print(f"\n   [DONE] {name} 测试完成，有效图片 {ok_cnt} 张。")

    return {
        "name": name,
        "n_imgs": ok_cnt,
        "total_sec": total_sec,
        "infer_ms_list": infer_ms_list,
        "failed": failed_cnt,
        "report": report_str
    }


# ============================================================
# 四、主流程
# ============================================================

def main():
    # 准备图片列表
    img_paths = collect_images(VAL_IMG_DIR)
    if not img_paths:
        print(f"[ERROR] 验证集目录 {VAL_IMG_DIR} 中没有图片。")
        return
    print(f"[INFO] 共发现 {len(img_paths)} 张验证图片。")

    # 准备输出目录
    ensure_dir(SAVE_DIR_BASE)

    all_results = []
    for cfg in MODEL_CONFIGS:
        # 为每个模型创建独立的保存子目录
        sub_dir = os.path.join(SAVE_DIR_BASE, cfg["name"])
        ensure_dir(sub_dir)
        result = test_model(cfg, img_paths, sub_dir)
        all_results.append(result)

    # 汇总写入报告文件
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("     OpenVINO 双模型推理性能对比报告\n")
        f.write("=" * 60 + "\n")
        f.write(f"测试图片总数 : {len(img_paths)}\n")
        f.write(f"推理设备     : {DEVICE}\n")
        f.write(f"图片尺寸     : {IMG_SIZE}\n")
        f.write("=" * 60 + "\n\n")

        for res in all_results:
            f.write(res["report"])
            f.write("\n" + "-" * 40 + "\n\n")

        # 附加简要对比
        f.write("简要对比 (纯推理 FPS):\n")
        for res in all_results:
            avg_ms = sum(res["infer_ms_list"]) / len(res["infer_ms_list"]) if res["infer_ms_list"] else 0
            fps = 1000.0 / avg_ms if avg_ms > 0 else 0
            f.write(f"  {res['name']:20s} : {fps:.2f} FPS\n")

    print(f"\n[INFO] 对比报告已保存至：{REPORT_FILE}")
    print("[DONE] 双模型对比测试完成。")


if __name__ == "__main__":
    main()