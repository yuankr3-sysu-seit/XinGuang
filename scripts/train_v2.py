"""
train_v2.py
基于魔改配置 + last.pt 预训练权重，重新开始 100 轮训练。
使用绝对路径，输出到独立文件夹 cbam_wiou_v2，启用多线程 workers=6。
"""

from ultralytics import YOLO
import multiprocessing

if __name__ == '__main__':
    # Windows 多进程安全锁
    multiprocessing.freeze_support()

    # ==================== 绝对路径配置 ====================
    PROJECT_ROOT = "D:/files_1/PythonProject/XinGuang"
    
    # 魔改模型配置文件
    MODEL_CFG = f"{PROJECT_ROOT}/configs/yolov8n_cbam_wiou.yaml"
    
    # 预训练权重（使用 v1 的第 51 轮存档）
    PRETRAINED_WEIGHTS = f"{PROJECT_ROOT}/runs/detect/cbam_wiou_v1/weights/last.pt"
    
    # 数据集配置文件
    DATA_YAML = f"{PROJECT_ROOT}/configs/xinguang.yaml"
    
    # 输出目录配置
    PROJECT_DIR = f"{PROJECT_ROOT}/runs/detect"
    EXPERIMENT_NAME = "cbam_wiou_v2"
    # ====================================================

    print(f"[INFO] 正在加载魔改模型配置: {MODEL_CFG}")
    print(f"[INFO] 正在加载预训练权重: {PRETRAINED_WEIGHTS}")
    
    # 加载配置 + 预训练权重
    model = YOLO(MODEL_CFG).load(PRETRAINED_WEIGHTS)

    print("[INFO] 🚀 开始全新训练（基于 last.pt 迁移学习）...")
    print(f"[INFO] 输出目录: {PROJECT_DIR}/{EXPERIMENT_NAME}")
    print("[INFO] 多线程数据加载已启用 (workers=6)")

    # 开始训练
    results = model.train(
        data=DATA_YAML,
        epochs=100,
        imgsz=640,
        device="cpu",
        batch=8,
        workers=6,                       # 这次必定生效
        project=PROJECT_DIR,
        name=EXPERIMENT_NAME,
        plots=True,
        save=True,
        exist_ok=False                   # 防止误覆盖，如果文件夹已存在则报错
    )

    print(f"[DONE] 🎉 训练完成！结果保存在: {PROJECT_DIR}/{EXPERIMENT_NAME}")