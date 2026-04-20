# -*- coding: utf-8 -*-
"""
infer_openvino.py
YOLOv8n + OpenVINO 全量验证集推理与性能基准测试脚本。

功能：
1. 检测并导出 yolov8n.pt 为 OpenVINO 格式
2. 使用 Intel cpu 推理 datasets/images/val/ 下全部图片
3. 带 tqdm 进度条，逐张保存可视化结果
4. 输出性能报告（总数 / 总耗时 / 平均耗时 / FPS）

作者：高级计算机视觉算法工程师
运行方式：python infer_openvino.py
"""

import os
import cv2
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

# ============================================================
# 一、全局配置
# ============================================================

# 项目根目录
PROJECT_ROOT = r"D:\files_1\PythonProject\XinGuang"

# 基线模型权重文件
PT_WEIGHTS = os.path.join(PROJECT_ROOT, "runs", "detect", "baseline_train", "weights", "best.pt")

# OpenVINO 导出后的模型目录（Ultralytics 默认命名：<stem>_openvino_model）
OV_MODEL_DIR = os.path.join(PROJECT_ROOT, "baseline_best_openvino_model")

# 验证集图片目录
VAL_IMG_DIR = os.path.join(PROJECT_ROOT, "datasets", "images", "val")

# 推理结果保存目录
SAVE_DIR = os.path.join(PROJECT_ROOT, "runs", "detect", "openvino_infer")

# 推理设备：OpenVINO 后端支持 'CPU' / 'cpu' / 'cpu.0' / 'cpu.1' / 'AUTO'
DEVICE = "cpu"

# 推理输入尺寸（与训练保持一致）
IMG_SIZE = 640

# 置信度与 IoU 阈值（可按需调整）
CONF_THRES = 0.25
IOU_THRES = 0.45

# 支持的图片后缀
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ============================================================
# 二、工具函数
# ============================================================

def ensure_dir(path: str):
    """确保目录存在，不存在则创建。"""
    os.makedirs(path, exist_ok=True)


def imread_unicode(path: str):
    """
    兼容中文路径的图片读取。
    cv2.imread 在 Windows 中文路径下会静默返回 None，改用 np.fromfile + imdecode。
    """
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite_unicode(path: str, img) -> bool:
    """兼容中文路径的图片写入。"""
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
    """收集目录下所有支持格式的图片路径，按文件名排序，结果稳定可复现。"""
    if not os.path.isdir(img_dir):
        return []
    files = []
    for name in os.listdir(img_dir):
        if os.path.splitext(name)[1].lower() in IMG_EXTS:
            files.append(os.path.join(img_dir, name))
    files.sort()
    return files


def export_openvino_if_needed(pt_path: str, ov_dir: str) -> str:
    """
    如果 OpenVINO 模型目录不存在，则通过 Ultralytics 导出；否则直接复用。
    返回可用于加载的模型路径（目录）。
    """
    if os.path.isdir(ov_dir) and len(os.listdir(ov_dir)) > 0:
        print(f"[INFO] 检测到已存在 OpenVINO 模型目录，直接复用：{ov_dir}")
        return ov_dir

    if not os.path.isfile(pt_path):
        raise FileNotFoundError(
            f"未找到基线权重文件：{pt_path}\n"
            f"请先将 yolov8n.pt 放置到项目根目录，或从 Ultralytics 官方仓库下载。"
        )

    print(f"[INFO] 开始将 {os.path.basename(pt_path)} 导出为 OpenVINO 格式 ...")
    tmp_model = YOLO(pt_path)
    # half=False：FP32 导出，兼容更多硬件；如需 FP16 提速可改为 True（Intel cpu 通常支持）
    exported = tmp_model.export(format="openvino", imgsz=IMG_SIZE, half=False)
    print(f"[INFO] 导出完成：{exported}")

    # Ultralytics 默认会把模型放在 pt 同级目录，名为 <stem>_openvino_model
    # 如果导出路径与预期不一致，做一次兜底确认
    if not os.path.isdir(ov_dir):
        # 某些版本 export 会返回字符串路径
        if isinstance(exported, (str, os.PathLike)) and os.path.isdir(str(exported)):
            return str(exported)
        raise RuntimeError(f"导出后未找到 OpenVINO 模型目录：{ov_dir}")
    return ov_dir


def format_report(n_imgs, total_sec, infer_ms_list):
    """
    生成一份美观的终端报告。
    - total_sec：端到端墙钟总耗时（含读图、前后处理、可视化保存）
    - infer_ms_list：每张图片的纯推理耗时（ms），来自 result.speed['inference']
    """
    infer_total_ms = sum(infer_ms_list) if infer_ms_list else 0.0
    avg_infer_ms = (infer_total_ms / len(infer_ms_list)) if infer_ms_list else 0.0
    # FPS 用"纯推理"口径计算，更能反映模型本身的性能
    fps_pure = (1000.0 / avg_infer_ms) if avg_infer_ms > 0 else 0.0
    # 端到端 FPS（含 IO / 可视化 / 保存）
    fps_e2e = (n_imgs / total_sec) if total_sec > 0 else 0.0

    line = "=" * 60
    report = f"""
{line}
              OpenVINO 推理性能测试报告 ({DEVICE})
{line}
  测试图片总数        : {n_imgs}
  端到端总耗时        : {total_sec:.2f} s
  纯推理总耗时        : {infer_total_ms / 1000.0:.2f} s
  平均单张推理耗时    : {avg_infer_ms:.2f} ms
  纯推理 FPS          : {fps_pure:.2f}
  端到端 FPS          : {fps_e2e:.2f}
  结果保存目录        : {SAVE_DIR}
{line}
"""
    return report


# ============================================================
# 三、主流程
# ============================================================

def main():
    # 1) 准备输出目录
    ensure_dir(SAVE_DIR)

    # 2) 确保 OpenVINO 模型就绪
    ov_path = export_openvino_if_needed(PT_WEIGHTS, OV_MODEL_DIR)

    # 3) 加载 OpenVINO 模型
    #    Ultralytics 会根据传入的目录自动识别为 OpenVINO 后端
    print(f"[INFO] 正在加载 OpenVINO 模型：{ov_path}")
    model = YOLO(ov_path, task="detect")

    # 4) 收集验证集图片
    img_paths = collect_images(VAL_IMG_DIR)
    if len(img_paths) == 0:
        print(f"[ERROR] 未在验证集目录中找到任何图片：{VAL_IMG_DIR}")
        return
    print(f"[INFO] 共发现 {len(img_paths)} 张验证集图片，开始在设备 [{DEVICE}] 上推理 ...")

    # 5) 预热（首次推理通常包含编译/图优化耗时，剔除后更能反映稳态性能）
    try:
        warmup_img = imread_unicode(img_paths[0])
        if warmup_img is not None:
            _ = model.predict(
                source=warmup_img, device=DEVICE, imgsz=IMG_SIZE,
                conf=CONF_THRES, iou=IOU_THRES, verbose=False
            )
            print("[INFO] 预热完成。")
    except Exception as e:
        # 典型错误：当前机器没有可用的 Intel cpu
        print(f"[ERROR] 预热推理失败，可能是设备 [{DEVICE}] 不可用：{e}")
        print("[HINT] 如果本机没有 Intel 核显/独显，请将 DEVICE 改为 'CPU' 后重试。")
        return

    # 6) 遍历推理
    infer_ms_list = []     # 每张图片的纯推理耗时（来自 result.speed['inference']）
    failed_cnt = 0         # 读图或推理失败计数

    t_start = time.perf_counter()

    pbar = tqdm(img_paths, desc="OpenVINO Inferring", ncols=100, unit="img")
    for img_path in pbar:
        img = imread_unicode(img_path)
        if img is None:
            failed_cnt += 1
            pbar.write(f"[WARN] 读图失败，跳过：{img_path}")
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
        except Exception as e:
            failed_cnt += 1
            pbar.write(f"[WARN] 推理失败，跳过 {os.path.basename(img_path)}：{e}")
            continue

        if not results:
            continue

        result = results[0]

        # 纯推理耗时（ms），Ultralytics 统一提供
        infer_ms = float(result.speed.get("inference", 0.0))
        infer_ms_list.append(infer_ms)

        # 绘制并保存可视化结果
        try:
            plotted = result.plot()  # 返回 BGR numpy
            save_name = Path(img_path).stem + ".jpg"
            save_path = os.path.join(SAVE_DIR, save_name)
            if not imwrite_unicode(save_path, plotted):
                pbar.write(f"[WARN] 结果保存失败：{save_path}")
        except Exception as e:
            pbar.write(f"[WARN] 可视化保存异常 {os.path.basename(img_path)}：{e}")

        # 实时在进度条上显示当前图片的推理耗时
        pbar.set_postfix_str(f"infer={infer_ms:.1f} ms")

    total_sec = time.perf_counter() - t_start

    # 7) 打印最终报告
    ok_cnt = len(infer_ms_list)
    print(format_report(ok_cnt, total_sec, infer_ms_list))
    if failed_cnt > 0:
        print(f"[WARN] 过程中共有 {failed_cnt} 张图片读图/推理失败，已跳过。")
    print("[DONE] 全量验证集推理完成。")


if __name__ == "__main__":
    main()
