# -*- coding: utf-8 -*-
"""
val_detailed.py
详细的 Baseline 模型精度评估脚本。
输出：
  1. 每类 AP50、Precision、Recall
  2. 整体 mAP50、mAP50-95
  3. 负样本误报率 (FPPI)
  4. 训练/验证 loss（如有日志）
"""

import os
from ultralytics import YOLO

PROJECT_ROOT = r"D:/files_1/PythonProject/XinGuang"
MODEL_PATH = os.path.join(PROJECT_ROOT, "runs", "detect", "baseline_train", "weights", "best.pt")
DATA_YAML = os.path.join(PROJECT_ROOT, "configs", "xinguang.yaml")
CLASS_NAMES = {0: "defect_2", 1: "defect_3", 2: "defect_6"}


def run_validation():
    """运行完整验证，输出每类指标。"""
    print(f"[INFO] 加载模型: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    print(f"[INFO] 开始验证 (数据集: {DATA_YAML}) ...")
    metrics = model.val(
        data=DATA_YAML,
        imgsz=640,
        device="cpu",
        plots=False,
        save_json=False,
        verbose=False
    )

    # 提取每类指标
    ap50_per_class = metrics.box.ap50.tolist() if hasattr(metrics.box.ap50, 'tolist') else metrics.box.ap50
    ap50_95_per_class = metrics.box.ap.tolist() if hasattr(metrics.box.ap, 'tolist') else metrics.box.ap
    precision_per_class = metrics.box.p.tolist() if hasattr(metrics.box.p, 'tolist') else metrics.box.p
    recall_per_class = metrics.box.r.tolist() if hasattr(metrics.box.r, 'tolist') else metrics.box.r

    # 打印报告
    print(f"\n{'='*70}")
    print(f"  Baseline 模型详细精度评估报告")
    print(f"{'='*70}")
    print(f"  模型: {MODEL_PATH}")
    print(f"  数据集: {DATA_YAML}")
    print(f"{'-'*70}")
    print(f"  {'类别':<12} {'AP50':>8} {'AP50-95':>10} {'Precision':>10} {'Recall':>8}")
    print(f"{'-'*70}")

    for i in sorted(CLASS_NAMES.keys()):
        name = CLASS_NAMES[i]
        ap50 = ap50_per_class[i] if i < len(ap50_per_class) else 0
        ap50_95 = ap50_95_per_class[i] if i < len(ap50_95_per_class) else 0
        p = precision_per_class[i] if i < len(precision_per_class) else 0
        r = recall_per_class[i] if i < len(recall_per_class) else 0
        print(f"  {name:<12} {ap50:>8.4f} {ap50_95:>10.4f} {p:>10.4f} {r:>8.4f}")

    print(f"{'-'*70}")
    print(f"  {'整体':<12} {metrics.box.map50:>8.4f} {metrics.box.map:>10.4f}")
    print(f"{'='*70}")

    return metrics


def evaluate_false_positives(model_path, project_root):
    """在负样本上评估误报率 (FPPI)。"""
    print(f"\n[INFO] 评估负样本误报率...")

    # 负样本：图片存在但没有对应标注文件的样本
    img_dir = os.path.join(project_root, "datasets", "images", "val")
    label_dir = os.path.join(project_root, "datasets", "labels", "val")

    # 收集所有标注文件的 stem
    label_stems = set()
    for fname in os.listdir(label_dir):
        if fname.endswith(".txt"):
            label_stems.add(os.path.splitext(fname)[0])

    # 找没有标注文件的图片 = 负样本
    negative_imgs = []
    img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    for fname in os.listdir(img_dir):
        stem, ext = os.path.splitext(fname)
        if ext.lower() in img_exts and stem not in label_stems:
            negative_imgs.append(os.path.join(img_dir, fname))

    if not negative_imgs:
        print("[WARN] 未找到负样本，跳过误报率评估。")
        return

    print(f"[INFO] 找到 {len(negative_imgs)} 张负样本图片")

    model = YOLO(model_path)
    total_fp = 0
    imgs_with_fp = 0

    import cv2
    import numpy as np

    for img_path in negative_imgs:
        # 兼容中文路径
        data = np.fromfile(img_path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            continue

        results = model.predict(
            source=img, device="cpu", imgsz=640,
            conf=0.25, iou=0.45, verbose=False
        )

        if results and len(results) > 0:
            n_det = len(results[0].boxes)
            total_fp += n_det
            if n_det > 0:
                imgs_with_fp += 1

    fppi = total_fp / len(negative_imgs) if negative_imgs else 0
    fp_ratio = imgs_with_fp / len(negative_imgs) * 100 if negative_imgs else 0

    print(f"\n{'='*60}")
    print(f"  负样本误报率评估")
    print(f"{'='*60}")
    print(f"  负样本总数: {len(negative_imgs)}")
    print(f"  检测到假框的图片: {imgs_with_fp} ({fp_ratio:.1f}%)")
    print(f"  假框总数: {total_fp}")
    print(f"  FPPI (每张假框数): {fppi:.4f}")
    print(f"  目标: FPPI < 0.1")
    if fppi < 0.1:
        print(f"  状态: PASS")
    else:
        print(f"  状态: WARN - 误报率偏高")
    print(f"{'='*60}")


if __name__ == "__main__":
    metrics = run_validation()
    evaluate_false_positives(MODEL_PATH, PROJECT_ROOT)
    print("\n[DONE] 详细评估完成。")
