import os
import cv2
import ffmpeg
import glob
from datetime import datetime

# ======================== 配置区 (可根据需求修改，已按你的要求预设) ========================
# 视频所在文件夹路径
VIDEO_FOLDER = r"\\192.168.50.101\myshare\a68f8ba682f84502b58faebac77c993c"
# 截图保存根文件夹
SAVE_ROOT = "images"
# 指定的时间筛选范围 (字符串格式)
TIME_START = "2025-12-03_06-00-00"
TIME_END = "2025-12-03_20-00-00"
# 视频帧率（固定15帧/秒，你的硬性要求）
FPS = 15
# 截图间隔（分钟）
CAP_INTERVAL_MIN = 5
# ========================================================================================

# 计算：5分钟对应的帧数 15帧/s * 60s *5min = 4500帧
INTERVAL_FRAME = FPS * 60 * CAP_INTERVAL_MIN
# 时间格式化规则（匹配视频文件名的时间格式）
TIME_FORMAT = "%Y-%m-%d_%H-%M-%S"
# 转换筛选的开始/结束时间为datetime对象，用于对比
start_datetime = datetime.strptime(TIME_START, TIME_FORMAT)
end_datetime = datetime.strptime(TIME_END, TIME_FORMAT)


def get_video_file_datetime(video_filename):
    """
    从视频文件名中提取时间并转为datetime对象
    文件名格式：2025-08-27_08-30-50-496711.mp4
    """
    # 提取文件名前缀的时间部分 2025-08-27_08-30-50
    time_str = video_filename[:19]
    return datetime.strptime(time_str, TIME_FORMAT)


def mkdir_if_not_exist(dir_path):
    """文件夹不存在则创建"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"创建文件夹: {dir_path}")


def capture_frame_from_video(video_path, video_name_no_suffix):
    """
    对单个视频执行截帧逻辑
    :param video_path: 视频文件的完整路径
    :param video_name_no_suffix: 视频文件名（不含后缀）
    """
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 打开视频失败，跳过：{video_path}")
        return

    # 获取视频总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\n✅ 开始处理视频：{video_name_no_suffix}.mp4 | 总帧数：{total_frames}")

    # 初始化截帧序号（图片名的递增数字，从1开始）
    capture_index = 1
    # 初始化要截取的帧位置，起始位置为第0帧（视频开头）
    current_frame = 0

    while current_frame < total_frames:
        # 跳转到指定帧的位置
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        # 读取该帧的图像
        ret, frame = cap.read()

        if ret:
            # ✅ 核心修改：图片直接保存到images下，文件名=视频名-递增序号.jpg
            save_img_name = f"{video_name_no_suffix}-{capture_index}.jpg"
            save_img_path = os.path.join(SAVE_ROOT, save_img_name)
            # 保存图片（无损保存，画质最佳）
            cv2.imwrite(save_img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
            print(f"📸 截取帧[{current_frame}] -> 保存至：{save_img_path}")

            # 截帧序号+1
            capture_index += 1
            # 计算下一次要截取的帧位置：当前帧 + 间隔帧数
            current_frame += INTERVAL_FRAME
        else:
            # 读取帧失败，跳过该帧
            print(f"⚠️  读取帧[{current_frame}]失败，跳过")
            current_frame += INTERVAL_FRAME

    # 释放视频资源
    cap.release()
    print(f"✅ 视频处理完成：{video_name_no_suffix}.mp4")


if __name__ == "__main__":
    # 创建根保存文件夹
    mkdir_if_not_exist(SAVE_ROOT)
    # 获取视频文件夹下所有mp4格式文件
    video_file_list = glob.glob(os.path.join(VIDEO_FOLDER, "*.mp4"))
    print(f"\n📌 扫描到视频文件夹下共有 {len(video_file_list)} 个mp4视频文件")

    # 遍历所有视频文件
    for video_path in video_file_list:
        # 获取视频文件名（含后缀）
        video_filename = os.path.basename(video_path)
        # 获取视频文件名（不含后缀）
        video_name_no_suffix = os.path.splitext(video_filename)[0]

        try:
            # 提取视频文件的时间
            video_datetime = get_video_file_datetime(video_filename)
            # 判断视频是否在指定的时间范围内
            if start_datetime <= video_datetime <= end_datetime:
                print(f"\n=====================================================")
                print(f"符合时间筛选条件：{video_filename}")
                # 执行截帧
                capture_frame_from_video(video_path, video_name_no_suffix)
            else:
                print(f"⏭️  不在时间范围内，跳过：{video_filename}")
        except Exception as e:
            print(f"❌ 解析视频[{video_filename}]出错，跳过，错误信息：{str(e)}")

    print(f"\n\n🎉 所有视频处理完成！")