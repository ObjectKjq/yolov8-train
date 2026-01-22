import os
from PIL import Image
from PIL import UnidentifiedImageError

# ===================== 配置参数 =====================
# 输入目录（请确保这两个目录存在）
INPUT_IMAGES_DIR = "images"
INPUT_LABELS_DIR = "labels"

# 输出目录（脚本会自动创建）
OUTPUT_IMAGES_DIR = "images_1"
OUTPUT_LABELS_DIR = "labels_1"

# 720p目标分辨率 (宽, 高)，标准720p为1280×720
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

# 支持的图片格式
SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')


# ===================== 工具函数 =====================
def create_dir_if_not_exists(dir_path):
    """创建目录（如果不存在）"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"创建目录: {dir_path}")


def resize_image(image_path, output_path):
    """
    等比例缩放图片到720p分辨率（避免拉伸）
    :param image_path: 输入图片路径
    :param output_path: 输出图片路径
    :return: 是否成功
    """
    try:
        # 打开图片
        with Image.open(image_path) as img:
            # 获取原始尺寸
            original_width, original_height = img.size
            print(f"处理图片 {os.path.basename(image_path)}: 原始尺寸 {original_width}×{original_height}")

            # 计算等比例缩放的比例（取最小比例，避免超出目标分辨率）
            scale = min(TARGET_WIDTH / original_width, TARGET_HEIGHT / original_height)

            # 计算新尺寸（确保为整数）
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)

            # 等比例缩放图片（使用高质量缩放算法）
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # 保存缩放后的图片（保持原格式）
            resized_img.save(output_path)
            print(f"保存缩放后图片: {output_path}, 新尺寸 {new_width}×{new_height}")
            return True
    except UnidentifiedImageError:
        print(f"错误：无法识别的图片文件 {image_path}")
        return False
    except Exception as e:
        print(f"处理图片 {image_path} 时出错: {str(e)}")
        return False


def copy_label(label_path, output_label_path):
    """
    复制标签文件（YOLOv8归一化标签无需修改）
    :param label_path: 输入标签路径
    :param output_label_path: 输出标签路径
    """
    try:
        with open(label_path, 'r', encoding='utf-8') as f_in, open(output_label_path, 'w', encoding='utf-8') as f_out:
            # 逐行读取并写入（保持原格式）
            content = f_in.read()
            f_out.write(content)
        print(f"复制标签文件: {output_label_path}")
    except Exception as e:
        print(f"处理标签 {label_path} 时出错: {str(e)}")


# ===================== 主流程 =====================
def main():
    # 创建输出目录
    create_dir_if_not_exists(OUTPUT_IMAGES_DIR)
    create_dir_if_not_exists(OUTPUT_LABELS_DIR)

    # 遍历所有图片文件
    for filename in os.listdir(INPUT_IMAGES_DIR):
        # 过滤支持的图片格式
        if filename.lower().endswith(SUPPORTED_FORMATS):
            # 获取文件基本名（不带后缀），用于匹配标签文件
            file_basename = os.path.splitext(filename)[0]

            # 构建输入输出路径
            input_img_path = os.path.join(INPUT_IMAGES_DIR, filename)
            output_img_path = os.path.join(OUTPUT_IMAGES_DIR, filename)

            # 缩放图片
            if resize_image(input_img_path, output_img_path):
                # 处理对应的标签文件
                input_label_path = os.path.join(INPUT_LABELS_DIR, f"{file_basename}.txt")
                output_label_path = os.path.join(OUTPUT_LABELS_DIR, f"{file_basename}.txt")

                if os.path.exists(input_label_path):
                    copy_label(input_label_path, output_label_path)
                else:
                    print(f"警告：未找到 {file_basename} 对应的标签文件 {input_label_path}")
        else:
            print(f"跳过非图片文件: {filename}")

    print("\n===== 处理完成 =====")


if __name__ == "__main__":
    # 检查依赖
    try:
        import PIL
    except ImportError:
        print("请先安装Pillow库：pip install pillow")
        exit(1)

    # 检查输入目录是否存在
    if not os.path.exists(INPUT_IMAGES_DIR):
        print(f"错误：输入图片目录 {INPUT_IMAGES_DIR} 不存在！")
        exit(1)

    if not os.path.exists(INPUT_LABELS_DIR):
        print(f"错误：输入标签目录 {INPUT_LABELS_DIR} 不存在！")
        exit(1)

    # 执行主流程
    main()