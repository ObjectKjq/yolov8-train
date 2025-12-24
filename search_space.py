from ultralytics import YOLO

# 自动优化超参数

# Initialize the YOLO model
model = YOLO("yolov8n.pt")

# Define search space
# search_space = {
#     "lr0": (1e-5, 1e-1),
#     "degrees": (0.0, 45.0),
# }

# 调整超参数，训练 30 个周期
model.tune(
    data='./data/data.yaml',
    epochs=50,
    iterations=10, # 生成多少组参数优化
    optimizer="AdamW", # 优化器
    # space=search_space,
    # plots=False, # 是否生成超参数调优过程的可视化图表
    # save=False, # 是否保存每一组超参数训练后的模型权重
    # val=False, # 是否在验证集上进行评估
)