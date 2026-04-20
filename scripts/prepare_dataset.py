# -*- coding: utf-8 -*-
"""
prepare_dataset.py
工业板材瑕疵数据集预处理脚本：
将 LabelMe JSON 标注 + 超大 PNG 原图，通过滑窗裁剪转换为 YOLOv8 训练所需的数据格式。

作者：高级计算机视觉算法工程师
运行方式：python prepare_dataset.py
"""

import os
import cv2
import json
import random
import shutil
from pathlib import Path

# ============================================================
# 一、全局路径与超参数配置
# ============================================================

# 输入图像目录（PNG 原图，分辨率约 1326x5006）
IMAGES_DIR = r"D:\dataset_2\xinguang\defect_detection\dataset\defects\images"
# 输入标注目录（LabelMe 生成的 JSON 文件）
LABELS_DIR = r"D:\dataset_2\xinguang\defect_detection\dataset\defects\label"
# 输出项目根目录
PROJECT_ROOT = r"D:\files_1\PythonProject\XinGuang"

# 滑窗参数
WIN_SIZE = 640           # 裁剪窗口大小（640x640）
STRIDE = 500             # 滑动步长

# 交集保留阈值
IOU_RATIO_THRESH = 0.2   # 交集面积 / 原框面积 > 20% 即保留
ABS_AREA_THRESH = 200    # 或者交集绝对面积 > 200 像素也保留（兜底小缺陷）

# 负样本（无瑕疵裁剪图）保留概率
NEG_KEEP_PROB = 0.10

# 训练集比例
TRAIN_RATIO = 0.8

# 类别映射：原始 LabelMe 标签字符串 -> YOLO class_id
LABEL_MAP = {
    "2": 0,
    "3": 1,
    "6": 2,
}
# YOLO class_id -> 类别名（用于生成 yaml）
CLASS_NAMES = {0: "defect_2", 1: "defect_3", 2: "defect_6"}

# 随机种子，保证可复现
RANDOM_SEED = 42
random.seed(RANDOM_SEED)


# ============================================================
# 二、工具函数
# ============================================================

def ensure_dirs(root: str):
    """
    在项目根目录下创建 YOLOv8 标准的 datasets 子目录结构。
    datasets/
        images/train, images/val
        labels/train, labels/val
    如果已存在则清空重建，保证每次运行结果干净。
    """
    sub_dirs = [
        os.path.join(root, "datasets", "images", "train"),
        os.path.join(root, "datasets", "images", "val"),
        os.path.join(root, "datasets", "labels", "train"),
        os.path.join(root, "datasets", "labels", "val"),
    ]
    for d in sub_dirs:
        if os.path.exists(d):
            shutil.rmtree(d)  # 清空旧数据，避免脏数据污染
        os.makedirs(d, exist_ok=True)
    print(f"[INFO] 已创建输出目录结构于：{os.path.join(root, 'datasets')}")


def parse_labelme_json(json_path: str):
    """
    解析 LabelMe JSON 文件，提取 shapes 中的矩形框。
    返回： [(class_id, x1, y1, x2, y2), ...]
    只保留 LABEL_MAP 中定义的类别，其他类别直接跳过。
    """
    boxes = []
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[WARN] 解析 JSON 失败 {json_path}: {e}")
        return boxes

    for shape in data.get("shapes", []):
        label = str(shape.get("label", "")).strip()
        if label not in LABEL_MAP:
            # 未在映射表中的类别，忽略
            continue
        points = shape.get("points", [])
        if len(points) < 2:
            continue
        # LabelMe 的矩形框通常为左上 + 右下两个点
        (x1, y1), (x2, y2) = points[0], points[1]
        # 统一左上小、右下大，防止标注时点位颠倒
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        boxes.append((LABEL_MAP[label], float(x_min), float(y_min),
                      float(x_max), float(y_max)))
    return boxes


def compute_intersection(box, win):
    """
    计算原图标注框与裁剪窗口的交集。
    box: (x1, y1, x2, y2)  原图绝对坐标
    win: (wx1, wy1, wx2, wy2) 裁剪窗口在原图的绝对坐标
    返回交集矩形 (ix1, iy1, ix2, iy2) 以及交集面积；
    若无交集返回 None, 0。
    """
    x1, y1, x2, y2 = box
    wx1, wy1, wx2, wy2 = win
    ix1 = max(x1, wx1)
    iy1 = max(y1, wy1)
    ix2 = min(x2, wx2)
    iy2 = min(y2, wy2)
    if ix2 <= ix1 or iy2 <= iy1:
        return None, 0.0
    inter_area = (ix2 - ix1) * (iy2 - iy1)
    return (ix1, iy1, ix2, iy2), inter_area


def box_to_yolo(inter, win, win_size):
    """
    将交集框（原图绝对坐标）转换为 YOLO 归一化格式（相对裁剪小图）。
    inter: (ix1, iy1, ix2, iy2) 原图绝对坐标
    win:   (wx1, wy1, wx2, wy2) 窗口左上角 + 右下角
    win_size: 小图尺寸（640）
    返回 (x_center, y_center, w, h)，均已截断到 [0, 1]。
    """
    ix1, iy1, ix2, iy2 = inter
    wx1, wy1, _, _ = win
    # 平移到裁剪小图坐标系
    rx1 = ix1 - wx1
    ry1 = iy1 - wy1
    rx2 = ix2 - wx1
    ry2 = iy2 - wy1
    # 中心点 + 宽高 的归一化
    xc = (rx1 + rx2) / 2.0 / win_size
    yc = (ry1 + ry2) / 2.0 / win_size
    bw = (rx2 - rx1) / win_size
    bh = (ry2 - ry1) / win_size
    # 强制夹到 [0, 1]，防止浮点误差越界
    clip = lambda v: max(0.0, min(1.0, v))
    return clip(xc), clip(yc), clip(bw), clip(bh)


def generate_window_positions(img_w, img_h, win_size, stride):
    """
    生成滑窗左上角坐标列表。
    策略：按 stride 滑动，末端不足时强制贴边（起点回退到 W-win_size / H-win_size），
    保证图像边缘区域也能被完整覆盖。
    """
    xs, ys = [], []

    # 横向
    if img_w <= win_size:
        xs.append(0)
    else:
        x = 0
        while x + win_size < img_w:
            xs.append(x)
            x += stride
        xs.append(img_w - win_size)  # 贴边最后一窗
        xs = sorted(set(xs))

    # 纵向
    if img_h <= win_size:
        ys.append(0)
    else:
        y = 0
        while y + win_size < img_h:
            ys.append(y)
            y += stride
        ys.append(img_h - win_size)  # 贴边最后一窗
        ys = sorted(set(ys))

    return [(x, y) for y in ys for x in xs]


# ============================================================
# 三、主流程
# ============================================================

def main():
    # 1) 创建输出目录
    ensure_dirs(PROJECT_ROOT)

    # 2) 以 JSON 为主索引，按 stem 匹配图像（因为有标注的才需要处理）
    json_files = [f for f in os.listdir(LABELS_DIR) if f.lower().endswith(".json")]
    print(f"[INFO] 共发现 {len(json_files)} 个 JSON 标注文件")

    # 收集所有生成的样本（内存中先保存，最后再按 8:2 随机划分写盘）
    # 每个元素：(img_ndarray, yolo_lines_or_None, unique_stem)
    samples = []

    # 统计信息
    total_crops = 0
    pos_crops = 0          # 含瑕疵的裁剪
    neg_kept = 0           # 保留的负样本
    missing_images = 0     # 缺失原图的 JSON

    for json_name in json_files:
        stem = Path(json_name).stem  # 如 "img_001"
        json_path = os.path.join(LABELS_DIR, json_name)

        # 按 stem 精确匹配原图，支持 .png / .PNG
        img_path = None
        for ext in (".png", ".PNG", ".jpg", ".jpeg"):
            candidate = os.path.join(IMAGES_DIR, stem + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break
        if img_path is None:
            print(f"[WARN] JSON 找不到对应的图片，跳过：{json_name}")
            missing_images += 1
            continue

        # 读图（cv2.imread 在中文路径下可能失败，用 np.fromfile + imdecode 更稳）
        try:
            import numpy as np
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"[WARN] 读图失败，跳过：{img_path} ({e})")
            continue
        if img is None:
            print(f"[WARN] 读图返回 None，跳过：{img_path}")
            continue

        H, W = img.shape[:2]

        # 解析标注
        gt_boxes = parse_labelme_json(json_path)

        # 生成所有窗口位置
        positions = generate_window_positions(W, H, WIN_SIZE, STRIDE)

        for idx, (wx, wy) in enumerate(positions):
            win = (wx, wy, wx + WIN_SIZE, wy + WIN_SIZE)
            crop = img[wy:wy + WIN_SIZE, wx:wx + WIN_SIZE].copy()

            # 理论上 crop 一定是 640x640，但极端情况下图像小于窗口时需补齐
            ch, cw = crop.shape[:2]
            if ch != WIN_SIZE or cw != WIN_SIZE:
                # 右/下方补黑边，保证尺寸统一
                pad_bottom = WIN_SIZE - ch
                pad_right = WIN_SIZE - cw
                crop = cv2.copyMakeBorder(
                    crop, 0, max(0, pad_bottom), 0, max(0, pad_right),
                    borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
                )

            # 在当前窗口内寻找有效的标注框
            yolo_lines = []
            for (cls_id, x1, y1, x2, y2) in gt_boxes:
                inter, inter_area = compute_intersection((x1, y1, x2, y2), win)
                if inter is None:
                    continue
                orig_area = max(1e-6, (x2 - x1) * (y2 - y1))
                # 保留条件：相对占比 > 20% 或 绝对面积 > 阈值
                if (inter_area / orig_area) < IOU_RATIO_THRESH and inter_area < ABS_AREA_THRESH:
                    continue
                xc, yc, bw, bh = box_to_yolo(inter, win, WIN_SIZE)
                # 过滤掉宽高过小的退化框（可能是极细条瑕疵被切到边）
                if bw <= 0 or bh <= 0:
                    continue
                yolo_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

            unique_stem = f"{stem}_x{wx}_y{wy}"

            if len(yolo_lines) > 0:
                # 正样本：保存图 + txt
                samples.append((crop, yolo_lines, unique_stem))
                pos_crops += 1
            else:
                # 负样本：按 10% 概率保留，只存图不存 txt
                if random.random() < NEG_KEEP_PROB:
                    samples.append((crop, None, unique_stem))
                    neg_kept += 1
            total_crops += 1

    print(f"[INFO] 滑窗总数：{total_crops}，含瑕疵：{pos_crops}，"
          f"保留负样本：{neg_kept}，缺图 JSON：{missing_images}")
    print(f"[INFO] 最终待写入样本数：{len(samples)}")

    if len(samples) == 0:
        print("[ERROR] 没有任何可用样本，流程终止。请检查输入路径与标注格式。")
        return

    # 3) 按 8:2 划分训练/验证集
    random.shuffle(samples)
    split_idx = int(len(samples) * TRAIN_RATIO)
    train_set = samples[:split_idx]
    val_set = samples[split_idx:]
    print(f"[INFO] 训练集：{len(train_set)}，验证集：{len(val_set)}")

    # 4) 写盘
    def dump(subset, split):
        img_dir = os.path.join(PROJECT_ROOT, "datasets", "images", split)
        lbl_dir = os.path.join(PROJECT_ROOT, "datasets", "labels", split)
        for crop, lines, name in subset:
            img_save = os.path.join(img_dir, name + ".jpg")
            # 用 imencode + tofile，兼容中文路径
            import numpy as np
            ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if ok:
                buf.tofile(img_save)
            else:
                cv2.imwrite(img_save, crop)
            if lines is not None:
                with open(os.path.join(lbl_dir, name + ".txt"), "w", encoding="utf-8") as f:
                    f.write("\n".join(lines))

    dump(train_set, "train")
    dump(val_set, "val")
    print("[INFO] 图片与标签已写入完成。")

    # 5) 生成 xinguang.yaml
    yaml_path = os.path.join(PROJECT_ROOT, "configs", "xinguang.yaml")
    train_abs = os.path.join(PROJECT_ROOT, "datasets", "images", "train").replace("\\", "/")
    val_abs = os.path.join(PROJECT_ROOT, "datasets", "images", "val").replace("\\", "/")
    names_str = "\n".join([f"  {k}: {v}" for k, v in CLASS_NAMES.items()])

    yaml_content = (
        f"# YOLOv8 数据集配置文件（自动生成）\n"
        f"train: {train_abs}\n"
        f"val: {val_abs}\n\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names:\n{names_str}\n"
    )
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"[INFO] 已生成 YOLOv8 配置文件：{yaml_path}")
    print("[DONE] 全部流程完成，可直接使用 yolo detect train data=configs/xinguang.yaml ... 进行训练。")


if __name__ == "__main__":
    main()
