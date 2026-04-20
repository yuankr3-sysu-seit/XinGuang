from ultralytics import YOLO

# 1. 加载你昨晚辛辛苦苦训练出来的专属大脑
model = YOLO(r"D:\files_1\PythonProject\XinGuang\runs\detect\baseline_train\weights\best.pt")

print("[INFO] 正在将专属大脑转换为 OpenVINO 引擎格式...")
# 2. 这一步极其关键！将 .pt 转换为 OpenVINO 格式
model.export(format='openvino')

print("[INFO] 转换完成！开始极速推理测试...")
# 3. 加载转换后的 OpenVINO 模型（注意路径变成了文件夹）
ov_model = YOLO(r"D:\files_1\PythonProject\XinGuang\runs\detect\baseline_train\weights\best_openvino_model")

# 4. 在验证集上跑测试，找回你的 70 FPS！
results = ov_model.predict(
    source=r"D:\files_1\PythonProject\XinGuang\datasets\images\val", 
    device='cpu', # 这里调用的是 OpenVINO 的 CPU 插件
    save=True,    # 画出红绿框并保存图片
    project="runs/detect",
    name="baseline_openvino_infer"
)
