from ultralytics import YOLO
import multiprocessing

if __name__ == '__main__':
    # 【安全锁】依然保留，防止 Windows 多进程报错
    multiprocessing.freeze_support()
    
    # 1. 精准定位到你昨晚中断时的“记忆存档”文件
    last_weight_path = r"D:/files_1/PythonProject/XinGuang/runs/detect/cbam_wiou_v1/weights/last.pt"
    
    print(f"[INFO] 正在加载中断的模型存档: {last_weight_path}")
    model = YOLO(last_weight_path)
    
    print("[INFO] 🚀 正在恢复训练！模型会自动读取之前的配置和进度...")
    
    # 2. 核心指令：只需要加一个 resume=True，其他参数全都不用写！
    # YOLO 会自动知道你要跑 100 轮，并且接着你中断的那一轮继续往下跑
    results = model.train(resume=True) 
    
    print("[DONE] 🎉 恢复训练圆满结束！")
