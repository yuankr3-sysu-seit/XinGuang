from ultralytics import YOLO
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()
    
    print("[INFO] 正在加载 YOLOv8n-P2 纯净版模型...")
    model = YOLO('configs/yolov8n-p2.yaml').load('models/weights/yolov8n.pt')  # 加载官方预训练权重
    
    print("[INFO] 🚀 训练即将开始！请确保电脑已插电，散热良好，且未设置自动休眠。")
    print("[INFO] 进度条将在数据加载完成后出现，请耐心等待 1-2 分钟...")
    
    results = model.train(
        data='configs/xinguang.yaml',  
        epochs=50,                     # 50轮验证
        imgsz=640,             
        device='cpu',
        batch=8,
        workers=0,
        project=r'D:/files_1/PythonProject/XinGuang/runs/detect',
        name='baseline_p2_v1',         # 新名称，独立保存
        pretrained=True,               # 使用预训练权重
        plots=True
    )
    
    print("[DONE] 🎉 训练圆满结束！请前往 runs/detect/baseline_p2_v1 目录查看报告！")