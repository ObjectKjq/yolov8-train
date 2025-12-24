from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# 分块推理测试，可以与正常推理最比较。

detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    # model_path='yolov8n.pt',
    model_path='runs/detect/train/weights/best.pt',
    confidence_threshold=0.5,
    device='cuda:0'
)

results = get_sliced_prediction(
    './images/PartA_00211.jpg',
    detection_model,
    slice_height=640,
    slice_width=640,
    # slice_height=256,
    # slice_width=256,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2
)

# 保存可视化结果
results.export_visuals(export_dir='runs/output/')
