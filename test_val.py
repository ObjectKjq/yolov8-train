from ultralytics import YOLO

model = YOLO("./runs/detect/train/weights/best.pt")

# 评估test集（核心：split='test'）
results = model.val(
    data="./data/data.yaml",  # 替换为你的数据配置文件路径
    split="test",        # 明确指定评估test集（默认是val）
    save_json=False,     # 可选：保存评估结果为JSON文件（便于后续分析）
    plots=True           # 必选：开启绘图（生成P/R/F1曲线等）
)