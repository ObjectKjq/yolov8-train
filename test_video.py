import time

from ultralytics import YOLO
import ffmpeg
import cv2

model = YOLO("runs/detect/train/weights/best.pt")

cap = cv2.VideoCapture(0)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps is None or fps == 0:
    fps = 15

process = ffmpeg.input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{width}x{height}',
                 r=fps).output(
        "rtsp://localhost:8554/camera2",
        format='rtsp',
        rtsp_transport='tcp',  # 使用tcp传输
        vcodec='libx264',
        preset='superfast',  # 比 ultrafast 稍慢但压缩率更好
        # preset='ultrafast',  # 最快编码速度
        tune='zerolatency',
        reorder_queue_size='10000',  # 增加网络抖动缓冲
        crf=23,  # 控制质量（18-28，值越小质量越高）
        # 增加关键帧间隔
        g=fps,  # 每1秒一个关键帧
        keyint_min=fps,  # 最小关键帧间隔
        bufsize='4000k',  # 增大缓冲区
        maxrate='2000k',  # 限制最大码率
        threads='4',  # 多线程编码
        fflags='nobuffer',  # 减少输入缓冲
    ).overwrite_output().run_async(
        pipe_stdin=True,
        # quiet=True # 关闭日志打印
    )

ms = int(1000 / 1)  # 计算多少毫秒处理一帧
start_time = 0  # 初始化开始时间
results = None
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("摄像头离线！")
            break
        end_time = int(time.time() * 1000)
        if end_time - start_time > ms:
            start_time = end_time
            results = model(frame, conf=0.3, imgsz=640)  # 确保设置置信度阈值
        # 遍历每个检测结果
        for result in results:
            if result.boxes:  # 检查是否有检测到目标
                boxes = result.boxes
                cls_ids = boxes.cls.cpu().numpy().astype(int)  # 转为整数数组
                confidences = boxes.conf.cpu().numpy()  # 置信度
                xyxy_boxes = boxes.xyxy.cpu().numpy()  # 获取边界框坐标(x1,y1,x2,y2)

                # 遍历每个检测框
                for box, cls_id, conf in zip(xyxy_boxes, cls_ids, confidences):
                    x1, y1, x2, y2 = map(int, box)  # 转换为整数坐标

                    # 绘制边界框
                    color = (0, 255, 0)  # BGR格式，绿色
                    thickness = 2
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                    # 准备标签文本
                    label = f"{model.names[cls_id]}: {conf:.2f}"

                    # 计算文本大小
                    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                    # 绘制文本背景
                    cv2.rectangle(frame, (x1, y1 - text_height - 5), (x1 + text_width, y1), color, -1)

                    # 绘制文本
                    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                    # print(f"检测到类别: {model.names[cls_id]}, 置信度: {conf:.2f}")

        process.stdin.write(frame.tobytes())
except KeyboardInterrupt as e:
    print("流媒体被用户中断！" + str(e))
finally:
    cap.release()
    process.stdin.close()
    process.wait()