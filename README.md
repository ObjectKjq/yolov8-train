# YOLOv8模型训练教程+优化思路

## 前言

本教程将从基础环境搭建入手，逐步引导大家完成YOLOv8模型的训练、测试全流程，并对训练结果进行解读，最后分享实用的优化思路。本教程适合初学者或者是完全不懂的小白，如果操作方法得当是可以训练出工业上可以使用的模型。

## 基础环境

具体所需环境如下：

- Python环境：Python 3.9.21
- 核心依赖库：

```sh
# sahi==0.11.36，非必须，测试分块推理时需要
ultralytics==8.3.76
ffmpeg-python==0.2.0
opencv-python==4.6.0.66
numpy==1.26.4
```

- 视频处理工具(**非必须，当需要测试视频流监测时需要**)：ffmpeg 7.0.2（用于视频流的解码、编码，配合视频测试脚本使用）
- 流媒体服务工具(**非必须，当需要测试视频流监测时需要**)：mediamtx_v1.11.2_windows_amd64（Windows系统下的流媒体服务器）

## 准备阶段

在正式训练模型前，需完成源代码获取、Python环境配置以及项目目录熟悉。具体步骤如下：

### 代码准备

本教程使用的YOLOv8训练项目源码已托管至GitHub，可直接下载获取：

- GitHub仓库地址：https://github.com/ObjectKjq/yolov8-train.git
- 源码直接下载链接：https://codeload.github.com/ObjectKjq/yolov8-train/zip/refs/heads/master

下载完成后，将压缩包解压至本地指定目录。

### 安装环境

推荐使用conda创建独立的Python环境，避免与其他项目的依赖包冲突。若未安装conda，也可直接使用官方Python安装包配置环境。

1. 基于conda创建环境（推荐）

```sh
# 创建名为yolov8-train的Python环境，指定Python版本为3.9.21
conda create -n yolov8-train python=3.9.21
```

接下来需要安装PyTorch框架（YOLOv8模型训练的核心依赖），需根据自身显卡型号选择对应版本的PyTorch，具体可参考PyTorch官方指南：https://pytorch.org/get-started/locally/

```sh
# 进入项目根目录打开控制台，进入当前conda的yolov8-train环境
conda activate yolov8-train
# 安装PyTorch，我这里使用CPU做测试
pip install torch torchvision
# 下载训练模型所需要的依赖
pip install -r requirements.txt
```

2. 无conda环境配置

若未安装conda，可直接下载官方Python 3.9.21安装包：[Windows 版 Python 版本 | Python.org](https://www.python.org/downloads/windows/)，完成安装后，直接通过pip命令安装上述PyTorch及项目依赖包即可。

### 目录说明

项目解压后，各目录承担不同的功能，熟悉目录结构有助于后续数据准备、脚本运行等操作。项目目录结构如下：

![项目目录结构](.\doc\目录结构.png)

## 训练模型

### 数据准备

项目的data目录用于存放模型训练所需的数据集，默认包含训练集、验证集和测试集三个子目录，以及data.yaml配置文件，具体说明如下：

- train：训练集，存放用于模型训练的图像及对应的标签文件
- val：验证集，用于在训练过程中评估模型性能，指导模型参数优化
- test：测试集，用于**训练完成后**测试模型的泛化能力（测试集主要在训练完成后验证模型，在训练过程中不起作用）

注意：图像文件与标签文件需一一对应，标签格式需符合YOLO系列模型的要求，常见的标注工具有Label Studio、LabelImg、CVAT等等，详细请参见官方文档[计算机视觉的数据收集和标注策略](https://docs.ultralytics.com/zh/guides/data-collection-and-annotation/#techniques-of-annotation)。

### 修改配置文件

data.yaml文件用于指定数据集路径、类别数量及类别名称，是模型训练的核心配置文件之一，需根据实际数据集情况修改：

```yml
# 修改为你自己的目录
train: C:\Users\dxzw-xm16\Desktop\yolov8-train\data\train\images
val: C:\Users\dxzw-xm16\Desktop\yolov8-train\data\val\images
test: C:\Users\dxzw-xm16\Desktop\yolov8-train\data\test\images

# 类别数量
nc: 1

# 配置序号对应类别的名称
names:
  - 'person'
```

### 配置train.py训练参数并启动

train.py文件包含模型加载、训练参数设置及训练启动逻辑，可根据自身设备情况和训练需求调整参数：

```python
from ultralytics import YOLO

model = YOLO('./yolov8n.pt') # 加载预训练模型

model.train(
    data='./data/data.yaml', # 数据集配置文件路径
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

参数配置完成后，在项目根目录的控制台中执行以下命令启动训练：

```sh
python ./train.py
```

训练启动后，控制台会输出训练进度、损失值、评估指标等信息，同时在runs/detect/train目录下生成训练日志和模型文件。训练过程图示如下：

![](.\doc\正在训练.png)

## 简单测试模型

模型训练完成后，会在项**目根目录生成runs/detect/train文件夹**，需通过图像、视频流等多种方式测试模型性能，验证模型的检测效果。项目提供了3个测试脚本，分别适用于不同的测试场景，具体操作如下：

### 单图像测试（test_img.py）

```python
# 加载模型
model = YOLO("./runs/detect/train/weights/best.pt")  
print(f"检测类别: {model.names}")
# 执行检测并保存结果（默认存到runs/detect/predict）
model.predict(source='./images/2008_002378.jpg', save=True, conf=0.6)
```

运行脚本后，检测结果默认保存至runs/detect/predict目录，可查看检测图像，观察目标框的准确性。

![](.\doc\img.jpg)

### 视频流测试（test_video.py）

该脚本支持摄像头实时流或RTSP流测试，需配合ffmpeg和mediamtx工具使用，适用于动态目标检测场景：

```python
# 视频源地址配置（0表示笔记本内置摄像头，也可替换为RTSP流地址，如\"rtsp://xxx.xxx.xxx.xxx:554/stream\"）
cap = cv2.VideoCapture(0)

# 推流地址配置（修改为mediamtx服务的RTSP地址，默认本地地址为rtsp://localhost:8554/camera2）
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

启动后，可通过浏览器访问mediamtx的WebRTC接口查看检测结果：http://localhost:8889/camera2/（8889为mediamtx默认的WebRTC访问端口）。mediamtx支持多种访问接口，可根据需求选择。

### 分块推理测试（test_sahi.py）

该脚本采用SAHI分块推理技术，适用于大尺寸图像或小目标检测场景。**通过将图像分块后分别推理，再融合结果，可提升小目标的检测精度，但相应的会消耗更多GPU资源**，具体可以参考官方文档[Ultralytics 文档：将 YOLO11 与 SAHI 结合使用于切片推理](https://docs.ultralytics.com/zh/guides/sahi-tiled-inference/)。使用方式可参考脚本内部注释，直接运行即可：

```sh
python ./test_sahi.py
```

![](.\doc\sahi.png)

## 训练结果

### 核心性能指标

模型训练完成后，在runs/detect/train目录下会生成训练日志、性能指标曲线、混淆矩阵等结果文件。要正确解读训练结果，需先理解核心性能指标，再结合生成的图表分析模型性能。

- **精确率(Precision)**: 模型预测为“正例”的结果中，实际是正例的比例。用于评估模型避免假阳性（误报）的能力，准确率越高，误报越少。
- **召回率(Recall)**: 实际是正例的样本中，被模型正确预测的比例。用于评估模型避免假阴性（漏报）的能力，召回率越高，漏报越少。
- **置信度(Confidence)**：模型对预测结果的信任程度，置信度阈值可调整，用于筛选可靠的检测结果。
- **交并比 (IoU):** IoU 是一种量化预测边界框与真实边界框之间重叠程度的度量。
- **平均精度 (AP)：**AP 计算精度-召回曲线下的面积，提供一个单一值，概括了模型的精度和召回性能。
- **平均精度均值 (mAP):** mAP 通过计算多个对象类别的平均 AP 值来扩展 AP 的概念。这在多类对象 detect 场景中非常有用，可以提供对模型性能的全面评估。
- **mAP50**: 在交并比 (IoU) 阈值为 0.50 时计算的平均精度均值。它衡量了模型在仅考虑“简单”检测时的准确性。
- **mAP50-95**: 在 0.50 到 0.95 范围内的不同 IoU 阈值下计算的平均精度均值。它全面反映了模型在不同检测难度下的性能。
- **F1 分数：** F1 分数是精度和召回率的调和平均值，在考虑假正例和假负例的同时，对模型的性能进行均衡评估。

更多性能指标请参见官方文档[性能指标深入分析 - Ultralytics YOLO 文档](https://docs.ultralytics.com/zh/guides/yolo-performance-metrics/)

### 使用test测试集生成指标

为什么需要测试集生成性能指标，在runs/detect/train目录下的指标是通过验证集val来生成的，而val在训练过程中起到调节模型训练方向的作用，每个轮次结束都需要val来验证，从而调整下一个轮次的训练策略，所以模型在验证集上面的表现没有说服力，就需要测试集来验证模型。代码如下：

```python
from ultralytics import YOLO

model = YOLO("./runs/detect/train/weights/best.pt")

# 评估test集（核心：split='test'）
results = model.val(
    data="./data/data.yaml",  # 替换为你的数据配置文件路径
    split="test",        # 明确指定评估test集（默认是val）
    save_json=False,     # 可选：保存评估结果为JSON文件（便于后续分析）
    plots=True           # 必选：开启绘图（生成P/R/F1曲线等）
)
```

运行测试，最终会在runs/detect/val目录下生成指标。

```sh
python ./test_val.py
```

### 训练结果图表解读

#### 混淆矩阵

![](.\doc\confusion_matrix.png)

#### F1分数曲线

![](.\doc\F1_curve.png)

这张图里的F1 是 F1 分数，是评估模型检测性能的核心指标之一。
它是精确率（Precision）和召回率（Recall）的调和平均数，公式为：`F1=2× [(Precision+Recall)/(Precision×Recall)]`

精确率：模型预测为 “正例” 的结果中，实际是正例的比例（避免 “误报”）；
召回率：实际是正例的样本中，被模型正确预测的比例（避免 “漏报”）。
F1 分数的取值范围是0~1，越接近 1 说明模型在 “精确率” 和 “召回率” 之间的平衡越好。

#### 精确率曲线

![](.\doc\P_curve.png)

#### 召回率曲线

![](.\doc\R_curve.png)

#### 训练结果曲线集

![](.\doc\results.png)

## 优化思路

若模型训练结果未达到预期（如检测精度低、漏报误报多、推理速度慢等），可从超参数、数据集、模型结构、部署等多个维度进行优化。具体优化方向如下：

### 超参数优化

超参数直接影响模型的训练过程和性能，合理调整超参数可显著提升模型效果。超参数调优也要根据场景和数据集进行调优，比如我们要监测的物体永远在光照充足的场景下，就不必变换hsv_v的值。

建议参考Ultralytics官方超参数调整指南：[Ultralytics YOLO 超参数调整指南 - Ultralytics YOLO 文档](https://docs.ultralytics.com/zh/guides/hyperparameter-tuning/)，重点优化以下参数：

- 训练轮次（epochs）：若训练损失未收敛，可适当增加epochs；若出现过拟合，可减少epochs或增加早停机制。
- 批次大小（batch）：在设备性能允许的情况下，增大batch可提升训练稳定性和效率，同时需调整学习率（batch增大时可适当增大学习率）。
- 学习率（lr0、lrf）：学习率过大易导致训练震荡不收敛，过小则训练速度慢。可采用学习率预热、余弦退火等策略优化学习率变化。
- 数据增强参数：根据数据集特点调整增强强度，如小目标数据集可增大缩放比例、开启马赛克增强等，提升模型泛化能力。

**如果我们刚开始不知道怎么优化，使用 `model.tune()` 方法，使用AdamW优化器进行自动优化**

```python
from ultralytics import YOLO

# Initialize the YOLO model
model = YOLO("yolov8n.pt")

# 优化器会在这里指定的搜索空间中选值
# search_space = {
#     "lr0": (1e-5, 1e-1),
#     "degrees": (0.0, 45.0),
# }

# 调整超参数，训练 50 个周期。如果不指定space参数范围，会使用默认的https://docs.ultralytics.com/zh/guides/hyperparameter-tuning/#default-search-space-description
model.tune(
    data='./data/data.yaml',
    epochs=50,
    iterations=10, # 生成多少组参数优化
    optimizer="AdamW",
    # space=search_space,
)
```

最终生成优化结果，best_hyperparameters.yaml 文件包含在调整过程中找到的最佳性能超参数。您可以使用此文件来初始化具有这些优化设置的未来训练。详细情况请参考官方文档[Ultralytics YOLO 超参数调整指南 - Ultralytics YOLO 文档](https://docs.ultralytics.com/zh/guides/hyperparameter-tuning/#resuming-an-interrupted-hyperparameter-tuning-session)

```
runs/
└── detect/
    ├── train1/
    ├── train2/
    ├── ...
    └── tune/
        ├── best_hyperparameters.yaml
        ├── best_fitness.png
        ├── tune_results.csv
        ├── tune_scatter_plots.png
        └── weights/
            ├── last.pt
            └── best.pt
```

### 数据集优化

数据集是模型训练的基础，数据集的质量直接决定模型性能。优化方向如下：

- 数据扩充：增加数据集样本数量，覆盖更多场景（如不同光照、角度、背景），可通过数据增强、收集真实场景数据等方式实现。
- 数据清洗：删除模糊、标注错误、重复的样本，修正不准确的边界框标注，提升数据集质量。
- 类别平衡：若存在类别不平衡问题（部分类别样本过少），可采用过采样（增加少数类样本）、欠采样（减少多数类样本）、类别权重调整等方法解决。
- 添加测试集：补充独立的测试集，更准确地评估模型的泛化能力，避免因验证集过拟合导致的性能误判。

### 模型结构优化

根据实际应用场景选择合适的模型结构，平衡检测精度和速度：

- 选择轻量级模型：若需部署在边缘设备（如手机、嵌入式设备），可选择YOLOv8n、YOLOv8s等轻量级模型，牺牲少量精度换取更快的推理速度。
- 模型融合：将多个训练好的模型进行融合，提升检测精度（如投票法、加权融合等）。
- 迁移学习：若数据集较小，可使用更大的公开数据集（如COCO、VOC）预训练模型，再基于自有数据集进行微调，提升模型性能。

### 部署优化

若模型用于实际部署，需优化推理速度和资源占用：

- 模型导出优化：将PyTorch模型导出为ONNX、TensorRT等格式，利用TensorRT等工具进行量化（INT8量化）、加速推理。
- 推理参数优化：调整推理时的图像尺寸、置信度阈值、NMS阈值等，在满足精度要求的前提下提升推理速度。
- 硬件适配：根据部署硬件选择合适的模型和加速方案（如GPU部署用TensorRT，CPU部署用OpenVINO）。

更多选项请参考官方文档[YOLO11 部署选项的对比分析](https://docs.ultralytics.com/zh/guides/model-deployment-options/)

