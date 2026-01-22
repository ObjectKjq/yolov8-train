from ultralytics import YOLO
import os
from pathlib import Path

# 加载预训练模型（可替换为自己训练的模型）
model = YOLO("best.pt")  # 官方轻量模型，也可使用yolov8x.pt等更大模型

# 未标注图像所在文件夹
img_dir = "auto-data/images"
# 标注文件保存文件夹（需与图像文件夹结构对应）
label_dir = "auto-data/labels"
Path(label_dir).mkdir(parents=True, exist_ok=True)

# 遍历图像并推理
for img_file in os.listdir(img_dir):
    if img_file.endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(img_dir, img_file)
        # 推理（设置置信度阈值过滤低质量预测，如conf=0.5）
        results = model(img_path, conf=0.5, classes=[6])
        if img_file.lower().endswith(".jpg"):  # 先判断是否为.jpg后缀，避免出错
            img_file = img_file[:-4]  # 截取到倒数第4个字符（不含）
        else:
            img_file = img_file.split(".")[0]  # 非.jpg文件则保持原样
        # 生成YOLO格式标注文件（文件名与图像一致，后缀为.txt）
        label_file = os.path.join(label_dir, img_file + ".txt")
        with open(label_file, "a") as f:
        # with open(label_file, "w") as f:
            for result in results:
                # 遍历每个预测框
                for box in result.boxes:
                    cls = int(box.cls)  # 类别ID
                    conf = box.conf.item()  # 置信度（可选保留）
                    # 边界框转换为YOLO格式（归一化中心坐标+宽高）
                    x_center, y_center, width, height = box.xywhn[0].tolist()
                    # 写入格式：cls x_center y_center width height（保留6位小数）
                    # f.write(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                    f.write(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")