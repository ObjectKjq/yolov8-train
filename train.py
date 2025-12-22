from ultralytics import YOLO
import torch
print(torch.__version__, torch.version.cuda)  # 输出当前 PyTorch 和 CUDA 版本

model = YOLO('./yolov8n.pt')

model.train(
    data='./data/data.yaml',
    epochs=50,
    imgsz=640,
    # device='0', # '0'表示GPU
    device='CPU',
    batch=8,
    cls=True, # 类别权重自动平衡
)
