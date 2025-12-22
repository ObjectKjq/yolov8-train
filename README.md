# YOLOv8模型训练教程+优化思路

## 前言



## 基础环境

- Python 3.9.21
  - ultralytics==8.3.76
  - ffmpeg-python==0.2.0
  - opencv-python==4.6.0.66
  - numpy==1.26.4
- ffmpeg 7.0.2
- mediamtx_v1.11.2_windows_amd64

## 准备阶段

1.源代码准备

GitHub仓库：https://github.com/ObjectKjq/yolov8-train.git

下载源码：https://codeload.github.com/ObjectKjq/yolov8-train/zip/refs/heads/master

2.安装python环境

如果有conda，创建新的python环境

```sh
# 创建python环境
conda create -n yolov8-train python=3.9.21
```

根据自己显卡下载相应的PyTorch, https://pytorch.org/get-started/locally/

```sh
# 进入项目根目录打开控制台，进入当前conda的yolov8-train环境
conda activate yolov8-train
# 安装PyTorch，我这里使用CPU做测试
pip install torch torchvision
# 下载训练模型所需要的依赖
pip install -r requirements.txt
```

**提示**

如果没有conda也没关系，直接下载官方Python3.9.21即可[Windows 版 Python 版本 |Python.org](https://www.python.org/downloads/windows/)，后续安装依赖即可

3.目录说明

![项目目录结构](.\doc\目录结构.png)

## 训练模型

1. data目录里存放了训练模型所需要的数据，这里我没有加test测试集，因为是不影响训练模型。如果你需要测试集的测试数据是需要加这部分内容的。
   1. train训练集
   2. val验证集
   3. test测试集

2. data.yaml

```yml
# 修改为你自己的目录
train: C:\Users\dxzw-xm16\Desktop\yolov8-train\data\train\images
val: C:\Users\dxzw-xm16\Desktop\yolov8-train\data\train\images

# 类别数量
nc: 1

# 配置序号对应类别的名称
names:
  - 'person'
```

3. train.py配置参数说明

```python
from ultralytics import YOLO

model = YOLO('./yolov8n.pt') # 加载预训练模型

model.train(
    data='./data/data.yaml', # 配置文件路径
    epochs=50, # 训练轮次，根据需求更改
    imgsz=640,
    # device='0',
    device='CPU', # 训练模型用的设备类型。0表示GPU
    batch=8, # 训练模型的批次大小，根据自己设备和需求动态调整

    # 其它参数.....
    # cls=True, # 类别权重自动平衡
    # augment=True,  # 开启参数增强
    # hsv_h=0.2,  # 色调扰动，默认值0.015
    # hsv_s=0.7,  # 饱和度扰动，默认值0.7
    # hsv_v=0.5,  # 亮度扰动，默认值0.4
    # degrees=10.0,  # 随机旋转角度（±15°）,默认0.0
    # translate=0.0, # 关闭随机平移
    # scale=0.3,  # 随机缩放模拟远近,默认0.5
    # shear=5.0,  # 形变30度,默认0.0
    # flipud=0.5,  # 上下翻转，默认0.0
    # fliplr=0.5,  # 左右翻转，默认0.5
    # mosaic=1.0,  # 马赛克增强，默认1.0
    # mixup=0.3,  # 图像混合，默认值0.0
    # cutmix=0.2, # 混合图像部分区域，默认值0.0
    # copy_paste=0.2, # 复制粘贴增强，默认值0.0.和上面效果基本一样,之不过他是随机复制,上面是相同位置
)
```

运行程序开始训练

```sh
python ./train.py
```

![](.\doc\正在训练.png)

## 测试模型

2. test_img.py，找一张图片测试模型训练的怎么样

```python
# 加载模型
model = YOLO("./runs/detect/train/weights/best.pt")  
print(f"检测类别: {model.names}")
# 执行检测并保存结果（默认存到runs/detect/predict）
model.predict(source='./images/2008_002378.jpg', save=True, conf=0.6)
```

2. test_video.py，使用rtsp流检验模型，首先需要下载ffmpeg和mediamtx

修改视频源地址，0一般表示笔记本的摄像头，也可以是一个rtsp流

```python
cap = cv2.VideoCapture(0)
```

修改推流地址

```python
process = ffmpeg.input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{width}x{height}',
                 r=fps).output(
        "rtsp://localhost:8554/camera2" # 这里改成自己mediamtx地址
    ).overwrite_output().run_async(
        pipe_stdin=True
    )
```

启动程序

```python
python ./test_video.py
```

浏览器访问：http://localhost:8889/camera2/。mediamtx提供了很多访问接口，8889是webrtc的访问端口。

## 训练结果

![](.\doc\results.png)

这张图展示了训练过程中的各项指标



![](.\doc\confusion_matrix.png)

指标讲解

## 优化思路
