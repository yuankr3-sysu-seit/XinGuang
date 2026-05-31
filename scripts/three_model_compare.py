# -*- coding: utf-8 -*-
"""
three_model_compare.py
三模型对照实验脚本：Official YOLOv8n vs Baseline vs CBAM+WIoU

评估指标（7 项）：
  1. mAP@0.5        — 基础检测精度
  2. mAP@0.5:0.95   — 严格定位精度
  3. Precision       — 查准率（整体）
  4. Recall          — 查全率（整体）
  5. F1 Score        — P/R 调和均值
  6. 每类 AP50       — defect_2 / defect_3 / defect_6 各自检测能力
  7. FPPI            — 负样本误报率（工业部署关键指标）

速度指标：
  - OpenVINO 平均推理耗时 (ms)
  - OpenVINO 纯推理 FPS

输出：控制台表格 + 报告文件
"""

import os
import time
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

# ============================================================
# 全局配置
# ============================================================

PROJECT_ROOT = r"D:\files_1\PythonProject\XinGuang"
VAL_IMG_DIR = os.path.join(PROJECT_ROOT, "datasets", "images", "val")
LABEL_DIR = os.path.join(PROJECT_ROOT, "datasets", "labels", "val")
DATA_YAML = os.path.join(PROJECT_ROOT, "configs", "xinguang.yaml")

CLASS_NAMES = {0: "defect_2", 1: "defect_3", 2: "defect_6"}

# 三模型配置
MODEL_CONFIGS = [
    {
        "name": "Official_YOLOv8n",
        "pt_path": os.path.join(PROJECT_ROOT, "models", "weights", "yolov8n.pt"),
    },
    {
        "name": "Baseline",
        "pt_path": os.path.join(PROJECT_ROOT, "runs", "detect", "baseline_train", "weights", "best.pt"),
    },
    {
        "name": "CBAM_WIoU",
        "pt_path": os.path.join(PROJECT_ROOT, "runs", "detect", "cbam_wiou_v1", "weights", "best.pt"),
    },
]

IMG_SIZE = 640
CONF_THRES = 0.25
IOU_THRES = 0.45
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

SAVE_DIR = os.path.join(PROJECT_ROOT, "runs", "detect", "three_model_compare")
REPORT_FILE = os.path.join(SAVE_DIR, "comparison_report.txt")


# ============================================================
# 工具函数
# ============================================================

def imread_unicode(path: str):
    """兼容中文路径读取图片"""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def collect_images(img_dir: str):
    """收集目录下所有图片路径"""
    if not os.path.isdir(img_dir):
        return []
    files = []
    for name in os.listdir(img_dir):
        if os.path.splitext(name)[1].lower() in IMG_EXTS:
            files.append(os.path.join(img_dir, name))
    files.sort()
    return files


def get_negative_images():
    """获取负样本图片列表（无标注文件的图片）"""
    label_stems = set()
    for fname in os.listdir(LABEL_DIR):
        if fname.endswith(".txt"):
            label_stems.add(os.path.splitext(fname)[0])

    negative_imgs = []
    for fname in os.listdir(VAL_IMG_DIR):
        stem, ext = os.path.splitext(fname)
        if ext.lower() in IMG_EXTS and stem not in label_stems:
            negative_imgs.append(os.path.join(VAL_IMG_DIR, fname))
    negative_imgs.sort()
    return negative_imgs


def compute_f1(precision, recall):
    """计算 F1 Score"""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ============================================================
# 评估函数
# ============================================================

def evaluate_accuracy(pt_path: str):
    """PyTorch 精度验证，返回各项指标"""
    model = YOLO(pt_path)
    metrics = model.val(
        data=DATA_YAML, imgsz=IMG_SIZE, device="cpu",
        plots=False, save_json=False, verbose=False
    )

    # 整体指标
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)

    # 每类 AP50
    ap50_per_class = metrics.box.ap50.tolist() if hasattr(metrics.box.ap50, 'tolist') else list(metrics.box.ap50)

    # 整体 Precision / Recall
    p_arr = metrics.box.p.tolist() if hasattr(metrics.box.p, 'tolist') else list(metrics.box.p)
    r_arr = metrics.box.r.tolist() if hasattr(metrics.box.r, 'tolist') else list(metrics.box.r)

    precision_overall = float(np.mean(p_arr)) if p_arr else 0.0
    recall_overall = float(np.mean(r_arr)) if r_arr else 0.0

    return {
        "map50": map50,
        "map50_95": map50_95,
        "precision": precision_overall,
        "recall": recall_overall,
        "f1": compute_f1(precision_overall, recall_overall),
        "ap50_per_class": {CLASS_NAMES[i]: ap50_per_class[i] for i in range(len(CLASS_NAMES))},
        "p_per_class": {CLASS_NAMES[i]: p_arr[i] for i in range(len(p_arr))},
        "r_per_class": {CLASS_NAMES[i]: r_arr[i] for i in range(len(r_arr))},
    }


def evaluate_fppi(pt_path: str, negative_imgs: list):
    """在负样本上评估 FPPI"""
    if not negative_imgs:
        return {"fppi": 0.0, "fp_images": 0, "total_neg": 0}

    model = YOLO(pt_path)
    total_fp = 0
    imgs_with_fp = 0

    for img_path in negative_imgs:
        img = imread_unicode(img_path)
        if img is None:
            continue
        results = model.predict(
            source=img, device="cpu", imgsz=IMG_SIZE,
            conf=CONF_THRES, iou=IOU_THRES, verbose=False
        )
        if results and len(results) > 0:
            n_det = len(results[0].boxes)
            total_fp += n_det
            if n_det > 0:
                imgs_with_fp += 1

    fppi = total_fp / len(negative_imgs)
    return {
        "fppi": fppi,
        "fp_images": imgs_with_fp,
        "total_neg": len(negative_imgs),
        "total_fp": total_fp,
    }


def evaluate_speed(pt_path: str, img_paths: list):
    """OpenVINO 推理速度测试"""
    # 导出 OpenVINO（如果已有则复用）
    ov_dir = str(Path(pt_path).parent / (Path(pt_path).stem + "_openvino_model"))
    if not (os.path.isdir(ov_dir) and len(os.listdir(ov_dir)) > 0):
        print(f"   [INFO] 导出 OpenVINO 模型...")
        model = YOLO(pt_path)
        exported = model.export(format="openvino", imgsz=IMG_SIZE, half=False)
        # export 返回导出路径
        if isinstance(exported, str) and os.path.isdir(exported):
            ov_dir = exported
    else:
        print(f"   [INFO] OpenVINO 模型已存在，复用：{ov_dir}")

    model = YOLO(ov_dir, task="detect")

    # 预热
    warmup_img = imread_unicode(img_paths[0])
    if warmup_img is not None:
        _ = model.predict(warmup_img, device="cpu", imgsz=IMG_SIZE,
                          conf=CONF_THRES, iou=IOU_THRES, verbose=False)

    infer_ms_list = []
    t_start = time.perf_counter()
    for img_path in tqdm(img_paths, desc="   测速中", ncols=100, unit="img"):
        img = imread_unicode(img_path)
        if img is None:
            continue
        try:
            results = model.predict(img, device="cpu", imgsz=IMG_SIZE,
                                    conf=CONF_THRES, iou=IOU_THRES, verbose=False)
        except Exception:
            continue
        if not results:
            continue
        infer_ms = float(results[0].speed.get("inference", 0.0))
        infer_ms_list.append(infer_ms)

    total_sec = time.perf_counter() - t_start
    avg_ms = np.mean(infer_ms_list) if infer_ms_list else 0.0
    fps_pure = 1000.0 / avg_ms if avg_ms > 0 else 0.0
    fps_e2e = len(infer_ms_list) / total_sec if total_sec > 0 else 0.0

    return {
        "avg_ms": float(avg_ms),
        "fps_pure": float(fps_pure),
        "fps_e2e": float(fps_e2e),
        "n_imgs": len(infer_ms_list),
    }


# ============================================================
# 报告生成
# ============================================================

def generate_report(all_results: list):
    """生成对照实验报告"""
    os.makedirs(SAVE_DIR, exist_ok=True)

    lines = []
    lines.append("=" * 90)
    lines.append("              三模型对照实验报告")
    lines.append("=" * 90)
    lines.append(f"验证数据集  : {DATA_YAML}")
    lines.append(f"图片尺寸    : {IMG_SIZE}")
    lines.append(f"置信度阈值  : {CONF_THRES}")
    lines.append(f"IoU 阈值    : {IOU_THRES}")
    lines.append("")

    # ---- 精度对比 ----
    lines.append("【一、精度指标对比】")
    lines.append("-" * 90)
    header = f"{'模型':<18} {'mAP50':>8} {'mAP50-95':>10} {'Precision':>10} {'Recall':>8} {'F1':>8}"
    lines.append(header)
    lines.append("-" * 90)
    for r in all_results:
        line = f"{r['name']:<18} {r['map50']:>8.4f} {r['map50_95']:>10.4f} {r['precision']:>10.4f} {r['recall']:>8.4f} {r['f1']:>8.4f}"
        lines.append(line)
    lines.append("-" * 90)
    lines.append("")

    # ---- 每类 AP50 ----
    lines.append("【二、每类 AP50 对比】")
    lines.append("-" * 90)
    class_header = f"{'模型':<18}"
    for cname in CLASS_NAMES.values():
        class_header += f" {cname:>12}"
    lines.append(class_header)
    lines.append("-" * 90)
    for r in all_results:
        line = f"{r['name']:<18}"
        for cname in CLASS_NAMES.values():
            val = r['ap50_per_class'].get(cname, 0.0)
            line += f" {val:>12.4f}"
        lines.append(line)
    lines.append("-" * 90)
    lines.append("")

    # ---- 每类 Recall ----
    lines.append("【三、每类 Recall 对比】")
    lines.append("-" * 90)
    rec_header = f"{'模型':<18}"
    for cname in CLASS_NAMES.values():
        rec_header += f" {cname:>12}"
    lines.append(rec_header)
    lines.append("-" * 90)
    for r in all_results:
        line = f"{r['name']:<18}"
        for cname in CLASS_NAMES.values():
            val = r['r_per_class'].get(cname, 0.0)
            line += f" {val:>12.4f}"
        lines.append(line)
    lines.append("-" * 90)
    lines.append("")

    # ---- 负样本误报率 ----
    lines.append("【四、负样本误报率 (FPPI)】")
    lines.append("-" * 90)
    lines.append(f"{'模型':<18} {'负样本数':>8} {'有假框图片':>10} {'假框总数':>8} {'FPPI':>8} {'状态':>6}")
    lines.append("-" * 90)
    for r in all_results:
        fppi = r['fppi']
        status = "PASS" if fppi < 0.1 else "WARN"
        line = (f"{r['name']:<18} {r['total_neg']:>8d} {r['fp_images']:>10d} "
                f"{r['total_fp']:>8d} {fppi:>8.4f} {status:>6}")
        lines.append(line)
    lines.append("-" * 90)
    lines.append("")

    # ---- 推理速度 ----
    lines.append("【五、OpenVINO 推理速度对比】")
    lines.append("-" * 90)
    lines.append(f"{'模型':<18} {'平均耗时(ms)':>12} {'纯推理FPS':>10} {'端到端FPS':>10} {'测试图片':>8}")
    lines.append("-" * 90)
    for r in all_results:
        spd = r['speed']
        line = (f"{r['name']:<18} {spd['avg_ms']:>12.2f} {spd['fps_pure']:>10.2f} "
                f"{spd['fps_e2e']:>10.2f} {spd['n_imgs']:>8d}")
        lines.append(line)
    lines.append("-" * 90)
    lines.append("")

    # ---- 总结 ----
    lines.append("【六、总结】")
    lines.append("-" * 90)

    # 找最优
    best_map50 = max(all_results, key=lambda x: x['map50'])
    best_f1 = max(all_results, key=lambda x: x['f1'])
    best_speed = max(all_results, key=lambda x: x['speed']['fps_pure'])
    best_fppi = min(all_results, key=lambda x: x['fppi'])

    lines.append(f"  mAP50 最高    : {best_map50['name']} ({best_map50['map50']:.4f})")
    lines.append(f"  F1 最高       : {best_f1['name']} ({best_f1['f1']:.4f})")
    lines.append(f"  速度最快      : {best_speed['name']} ({best_speed['speed']['fps_pure']:.2f} FPS)")
    lines.append(f"  误报最低      : {best_fppi['name']} (FPPI={best_fppi['fppi']:.4f})")
    lines.append("=" * 90)

    report_text = "\n".join(lines)

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report_text)

    return report_text


# ============================================================
# 主流程
# ============================================================

def main():
    print("=" * 70)
    print("         三模型对照实验：Official vs Baseline vs CBAM+WIoU")
    print("=" * 70)

    # 收集图片
    img_paths = collect_images(VAL_IMG_DIR)
    negative_imgs = get_negative_images()
    print(f"[INFO] 验证集图片: {len(img_paths)} 张, 负样本: {len(negative_imgs)} 张")

    all_results = []

    for cfg in MODEL_CONFIGS:
        name = cfg["name"]
        pt_path = cfg["pt_path"]

        print(f"\n{'='*50}")
        print(f"  模型: {name}")
        print(f"{'='*50}")

        if not os.path.isfile(pt_path):
            print(f"  [ERROR] 权重文件不存在: {pt_path}，跳过。")
            continue

        # 1. 精度验证
        print(f"  [1/3] 精度验证...")
        acc = evaluate_accuracy(pt_path)

        # 2. 负样本误报率
        print(f"  [2/3] 负样本误报率评估...")
        fppi_result = evaluate_fppi(pt_path, negative_imgs)

        # 3. 速度测试
        print(f"  [3/3] OpenVINO 速度测试...")
        speed = evaluate_speed(pt_path, img_paths)

        result = {
            "name": name,
            **acc,
            **fppi_result,
            "speed": speed,
        }
        all_results.append(result)

        # 打印单模型摘要
        print(f"\n  ┌─ {name} 结果摘要 ─────────────────────────")
        print(f"  │ mAP50={acc['map50']:.4f}  mAP50-95={acc['map50_95']:.4f}  F1={acc['f1']:.4f}")
        print(f"  │ P={acc['precision']:.4f}  R={acc['recall']:.4f}  FPPI={fppi_result['fppi']:.4f}")
        print(f"  │ 推理耗时={speed['avg_ms']:.2f}ms  FPS={speed['fps_pure']:.2f}")
        print(f"  └──────────────────────────────────────────")

    if not all_results:
        print("[ERROR] 没有成功评估的模型，退出。")
        return

    # 生成报告
    report_text = generate_report(all_results)
    print(f"\n{'='*70}")
    print(report_text)
    print(f"\n[DONE] 报告已保存至: {REPORT_FILE}")


if __name__ == "__main__":
    main()
