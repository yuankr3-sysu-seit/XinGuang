from ultralytics import YOLO

# 加载你的 Baseline 模型
model = YOLO('runs/detect/baseline_train/weights/best.pt')

# 运行验证，plots=True 会自动生成 PR 曲线等图表
metrics = model.val(
    data='configs/xinguang.yaml',
    imgsz=640,
    device='cpu',
    plots=True,
    save_json=False
)

# 打印关键指标
print("\n" + "=" * 50)
print("          Baseline 模型精度评估报告")
print("=" * 50)
print(f"mAP@0.5       : {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95  : {metrics.box.map:.4f}")
print(f"Precision (P) : {metrics.box.p[0]:.4f}")
print(f"Recall (R)    : {metrics.box.r[0]:.4f}")
print("=" * 50)