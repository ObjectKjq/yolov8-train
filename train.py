from ultralytics import YOLO
import torch
print(torch.__version__, torch.version.cuda)  # 输出当前 PyTorch 和 CUDA 版本

model = YOLO('./yolov8n.pt')

model.train(
    data='./data/data.yaml',
    epochs=50,
    
    imgsz=640,
    # device='0',
    device='CPU',
    batch=8,
    cls=True, # 类别权重自动平衡

    augment=True,  # 开启参数增强

    # 参数增强
    hsv_h=0.2,  # 色调扰动，默认值0.015
    hsv_s=0.7,  # 饱和度扰动，默认值0.7
    hsv_v=0.5,  # 亮度扰动，默认值0.4
    degrees=10.0,  # 随机旋转角度（±15°）,默认0.0
    translate=0.0, # 关闭随机平移
    scale=0.3,  # 随机缩放模拟远近,默认0.5
    shear=5.0,  # 形变30度,默认0.0
    # flipud=0.5,  # 上下翻转，默认0.0
    # fliplr=0.5,  # 左右翻转，默认0.5
    mosaic=1.0,  # 马赛克增强，默认1.0
    mixup=0.3,  # 图像混合，默认值0.0
    # cutmix=0.2, # 混合图像部分区域，默认值0.0
    copy_paste=0.2, # 复制粘贴增强，默认值0.0.和上面效果基本一样,之不过他是随机复制,上面是相同位置
)
