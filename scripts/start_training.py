from ultralytics import YOLO
import multiprocessing

if __name__ == '__main__':
    # 【安全锁】防止 Windows 系统下 CPU 多进程报错，万无一失的保证
    multiprocessing.freeze_support()
    
    print("[INFO] 正在加载 YOLOv8n 基线模型...")
    model = YOLO('configs/yolov8n_cbam_wiou.yaml') 

    print("[INFO] 🚀 训练即将开始！请确保电脑已插电，散热良好，且未设置自动休眠。")
    print("[INFO] 进度条将在数据加载完成后出现，请耐心等待 1-2 分钟...")
    
    # 核心训练参数
    results = model.train(
        data='configs/xinguang.yaml',  
        epochs=100,             # 训练 100 轮
        imgsz=640,             
        device='cpu',          # 强制使用 CPU
        batch=8,               # 【安全锁】将 Batch 设为 8，确保 32GB 内存绝对不会溢出
        workers=4,             # 使用 4 个线程加载数据
        project=r'D:\files_1\PythonProject\XinGuang\runs\detect',
        name='cbam_wiou_v1',
        plots=True             # 【报告锁】强制生成各种图表和可视化报告
    )
    
    print("[DONE] 🎉 训练圆满结束！请前往 runs/detect/cbam_wiou_v1 目录查看报告！")
