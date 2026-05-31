# -*- coding: utf-8 -*-
"""
analyze_dataset.py
统计训练集和验证集中各类别的标注框数量、每类出现的样本数、
defect_2 尺寸分布（中位数、分段统计）。
用于评估 defect_2 小目标的数据充足性。
"""

import os
import statistics
from collections import defaultdict

PROJECT_ROOT = r"D:/files_1/PythonProject/XinGuang"
LABEL_DIRS = {
    "train": os.path.join(PROJECT_ROOT, "datasets", "labels", "train"),
    "val": os.path.join(PROJECT_ROOT, "datasets", "labels", "val"),
}
CLASS_NAMES = {0: "defect_2", 1: "defect_3", 2: "defect_6"}


def analyze_split(split_name, label_dir):
    """分析单个数据集划分的标注统计。"""
    if not os.path.isdir(label_dir):
        print(f"[ERROR] 目录不存在: {label_dir}")
        return

    # 统计
    total_files = 0          # 总标注文件数（有标注的样本）
    empty_files = 0          # 空文件数（负样本）
    class_box_count = defaultdict(int)    # 每类的标注框总数
    class_sample_count = defaultdict(int) # 每类出现的样本数
    box_sizes = defaultdict(list)         # 每类标注框的宽高

    for fname in os.listdir(label_dir):
        if not fname.endswith(".txt"):
            continue
        total_files += 1
        fpath = os.path.join(label_dir, fname)

        with open(fpath, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if not lines:
            empty_files += 1
            continue

        classes_in_file = set()
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            w, h = float(parts[3]), float(parts[4])
            class_box_count[cls_id] += 1
            classes_in_file.add(cls_id)
            box_sizes[cls_id].append((w, h))

        for cls_id in classes_in_file:
            class_sample_count[cls_id] += 1

    # 打印报告
    print(f"\n{'='*60}")
    print(f"  {split_name} 数据集统计")
    print(f"{'='*60}")
    print(f"  标注文件总数: {total_files}")
    print(f"  有标注的样本: {total_files - empty_files}")
    print(f"  负样本（空标注）: {empty_files}")
    print(f"{'-'*60}")
    print(f"  {'类别':<12} {'框数':>8} {'样本数':>8} {'样本占比':>10}")
    print(f"{'-'*60}")

    positive_samples = total_files - empty_files
    for cls_id in sorted(CLASS_NAMES.keys()):
        name = CLASS_NAMES[cls_id]
        boxes = class_box_count.get(cls_id, 0)
        samples = class_sample_count.get(cls_id, 0)
        ratio = samples / positive_samples * 100 if positive_samples > 0 else 0
        print(f"  {name:<12} {boxes:>8} {samples:>8} {ratio:>9.1f}%")

    print(f"{'-'*60}")

    # defect_2 框尺寸分析
    if 0 in box_sizes and box_sizes[0]:
        sizes = box_sizes[0]
        widths = [s[0] for s in sizes]
        heights = [s[1] for s in sizes]
        areas = [w * h for w, h in sizes]
        # 取宽高的较大值作为目标尺寸
        max_dims = [max(w, h) for w, h in sizes]

        print(f"\n  defect_2 标注框尺寸分析 (归一化坐标):")
        print(f"    宽度  - min: {min(widths):.4f}, max: {max(widths):.4f}, avg: {sum(widths)/len(widths):.4f}, median: {statistics.median(widths):.4f}")
        print(f"    高度  - min: {min(heights):.4f}, max: {max(heights):.4f}, avg: {sum(heights)/len(heights):.4f}, median: {statistics.median(heights):.4f}")
        print(f"    面积  - min: {min(areas):.6f}, max: {max(areas):.6f}, avg: {sum(areas)/len(areas):.6f}, median: {statistics.median(areas):.6f}")

        # 在 640x640 图像中的实际像素尺寸
        print(f"\n  defect_2 在 640×640 图像中的实际像素尺寸:")
        print(f"    宽度  - min: {min(widths)*640:.1f}px, max: {max(widths)*640:.1f}px, avg: {sum(widths)/len(widths)*640:.1f}px, median: {statistics.median(widths)*640:.1f}px")
        print(f"    高度  - min: {min(heights)*640:.1f}px, max: {max(heights)*640:.1f}px, avg: {sum(heights)/len(heights)*640:.1f}px, median: {statistics.median(heights)*640:.1f}px")

        # 分段统计（按目标最大边长分）
        print(f"\n  defect_2 尺寸分段统计 (按较长边, 640px 坐标):")
        bins = [
            ("  <16px  (极小)", 0, 16),
            ("  16-32px (小目标)", 16, 32),
            ("  32-64px", 32, 64),
            ("  64-128px", 64, 128),
            ("  128-256px", 128, 256),
            ("  256-640px (大)", 256, 641),
        ]
        for label, lo, hi in bins:
            count = sum(1 for d in max_dims if lo <= d * 640 < hi)
            pct = count / len(max_dims) * 100
            print(f"    {label:<20}: {count:>5} ({pct:>5.1f}%)")
        print(f"    {'总计':<20}: {len(max_dims):>5}")

    print(f"{'='*60}")


if __name__ == "__main__":
    print("[INFO] 开始分析数据集...")
    for split_name, label_dir in LABEL_DIRS.items():
        analyze_split(split_name, label_dir)
    print("\n[DONE] 分析完成。")
