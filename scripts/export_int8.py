from ultralytics import YOLO

if __name__ == '__main__':
    # 1. 加载你刚刚训练出炉的魔改版模型
    # (注意：路径严格使用了你终端里显示的实际路径)
    model_path = r"D:/files_1/PythonProject/XinGuang/runs/detect/cbam_wiou_v1/weights/best.pt"
    model = YOLO(model_path)

    print("[INFO] 开始进行 OpenVINO INT8 极致量化导出...")
    print("[INFO] 正在调用 NNCF (Neural Network Compression Framework)...")
    print("[INFO] 注意：这需要几分钟时间，因为模型需要读取验证集图片来进行 INT8 校准（Calibration）...")

    # 2. 核心量化指令
    # int8=True 开启 INT8 量化
    # data='configs/xinguang.yaml' 必须提供，因为量化需要参考真实图片的数据分布
    model.export(
        format='openvino', 
        int8=True, 
        data='configs/xinguang.yaml'
    )

    print("[DONE] 🎉 INT8 量化模型导出彻底完成！")
    print("你可以去 weights 目录下查看生成的 best_openvino_model 文件夹了！")
