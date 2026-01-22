import os
import cv2
import ffmpeg


def extract_frames_with_opencv(video_path, output_dir, interval=5):
    """
    使用OpenCV每interval帧截取一张图片
    :param video_path: 输入视频路径
    :param output_dir: 输出图片目录
    :param interval: 截取间隔（每多少帧截取一次）
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return
    
    frame_count = 0  # 帧计数器
    save_count = 0   # 保存图片计数器
    
    while True:
        # 读取一帧
        ret, frame = cap.read()
        if not ret:
            break  # 视频读取完毕
        
        # 每interval帧保存一次
        if frame_count % interval == 0:
            # 生成保存路径
            save_path = os.path.join(output_dir, f"frame_{save_count:06d}.jpg")
            # 保存图片
            cv2.imwrite(save_path, frame)
            print(f"已保存: {save_path}")
            save_count += 1
        
        frame_count += 1
    
    # 释放资源
    cap.release()
    print(f"处理完成，共保存 {save_count} 张图片")


def extract_frames_with_ffmpeg(video_path, output_dir, interval=5):
    """
    使用ffmpeg每interval帧截取一张图片
    :param video_path: 输入视频路径
    :param output_dir: 输出图片目录
    :param interval: 截取间隔（每多少帧截取一次）
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 使用ffmpeg命令：每interval帧提取一张图片
        (
            ffmpeg
            .input(video_path)
            .filter('select', f'not(mod(n\,{interval}))')  # 每interval帧选择一帧
            .output(os.path.join(output_dir, 'frame_%06d.jpg'), vframes='v', qscale=2)
            # vframes='v' 表示根据输入视频自动计算输出帧数
            # qscale=2 表示图片质量（值越小质量越高，1-31）
            .run(capture_stdout=True, capture_stderr=True)
        )
        print(f"处理完成，图片已保存至: {output_dir}")
    except ffmpeg.Error as e:
        print(f"FFmpeg处理出错: {e.stderr.decode()}")


if __name__ == "__main__":
    # 配置参数
    video_file = "input.mp4"  # 输入视频文件路径
    output_folder = "images"  # 输出图片文件夹
    frame_interval = 5  # 每5帧截取一张
    
    # 检查视频文件是否存在
    if not os.path.exists(video_file):
        print(f"视频文件不存在: {video_file}")
    else:
        # 选择一种方法执行（取消注释对应行）
        print("使用OpenCV提取帧...")
        extract_frames_with_opencv(video_file, output_folder, frame_interval)
        
        # 或者使用ffmpeg提取
        # print("使用FFmpeg提取帧...")
        # extract_frames_with_ffmpeg(video_file, output_folder, frame_interval)