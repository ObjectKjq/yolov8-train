import os
import cv2
import random
import numpy as np
import albumentations as A
from tqdm import tqdm  # 进度条

SRC_IMG_DIR = "target/images"
SRC_LABEL_DIR = "target/labels"
DST_IMG_DIR = "result/images"
DST_LABEL_DIR = "result/labels"
AUGMENT_NUM_PER_IMG = 10
# ROTATE_LIMIT = (-30, 30)  # 仅-30~30°旋转，无90°倍数
IMG_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]

# ====================== 定义增强管道（核心：移除RandomRotate90） ======================
def get_augmentation_pipeline(img_h, img_w):
    # 彻底修复裁剪尺寸错误（多重兜底）
    crop_h = int(img_h * 0.8) # 计算裁剪的高度和宽度：取原图尺寸的 80%，并转为整数（因为裁剪尺寸必须是整数）；
    crop_w = int(img_w * 0.8)
    # crop_h = min(crop_h, img_h)
    # crop_w = min(crop_w, img_w)
    # crop_p = 0.2 if (crop_h > 0 and crop_w > 0 and crop_h < img_h and crop_w < img_w) else 0.0
    crop_p = 0.2

    # 随机参数（确保每张图效果不同）
    random_shear = random.randint(5, 10) # 随机剪切角度
    random_shift_x = random.uniform(0.2, 0.6) # 随机平移比例
    random_shift_y = random.uniform(0.2, 0.6) # 随机平移比例
    random_scale = random.uniform(-0.1, 0.2) # 随机缩放比例
    random_brightness = random.uniform(0.05, 0.1) # 生成随机亮度 / 对比度 / 饱和度调整幅度
    random_hue = random.uniform(0.02, 0.05) # 生成随机色调调整幅度
    random_hue_shift = random.randint(5, 10) # 生成随机色调偏移量

    aug = A.Compose([
        # ========== 几何变换 ==========
        A.Rotate(limit=(-30, 30), border_mode=cv2.BORDER_CONSTANT, p=0.5),  # limit=(-30, 30)：仅允许在 - 30°（逆时针）~30°（顺时针）之间旋转；
        A.HorizontalFlip(p=0.5),  # 仅水平翻转，无上下翻转
        # A.RandomScale(scale_limit=random_scale, p=0.4), # 仅缩放，无旋转
        # 平移：仅平移，无旋转/缩放
        A.ShiftScaleRotate(
            shift_limit_x=random_shift_x, shift_limit_y=random_shift_y,
            scale_limit=0, rotate_limit=0,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.3
        ),
        A.Affine(shear=random_shear, border_mode=cv2.BORDER_CONSTANT, p=0.3), # 观点：应用透视扭曲（透视）。可以模拟从不同视角观察平面表面。
        A.RandomCrop(height=crop_h, width=crop_w, p=crop_p), # 随机裁剪操作
        A.Resize(height=img_h, width=img_w, p=1.0), # 强制 resize 操作：把裁剪 / 变换后的图片还原为原图尺寸（img_h/img_w）

        # ========== 色彩空间变换 ==========
        # 低强度色彩变换
        A.ColorJitter(
            brightness=(1.0, 1.2),
            contrast=(1.0, 1.2),
            saturation=(1.0, 1.2),
            hue=random_hue,
            p=0.5
        ),
        # 色调 / 饱和度 / 明度调整：
        A.HueSaturationValue(
            hue_shift_limit=random_hue_shift, # 5~10 的色调偏移；
            sat_shift_limit=15, # 饱和度偏移 ±15；
            val_shift_limit=15, # 明度偏移 ±15；
            p=0.4
        ),
        # 随机亮度 / 对比度调整
        A.RandomBrightnessContrast(
            brightness_limit=(0.01, 0.05), # 亮度 ±5%；
            contrast_limit=(-0.05, 0.05), # 对比度 ±5%；
            p=0.4
        ),
        # A.ToGray(p=0.1), # 转为灰度图操作：

        # ========== 噪声和模糊 ==========
        # 低强度噪声/模糊/擦除/天气
        A.GaussNoise(p=0.3), # 高斯噪声：20% 概率添加
        A.MotionBlur(p=0.2), # 运动模糊：20% 概率添加，模拟拍摄运动模糊，提升模型对模糊目标的识别能力。
        A.GaussianBlur(p=0.2), # 高斯模糊：20% 概率添加，低强度模糊，避免目标特征丢失。
        A.MedianBlur(blur_limit=5, p=0.1), # 中值模糊

        # ========== 擦除和遮挡 ==========
        A.CoarseDropout(p=0.2), # 粗擦除（CoarseDropout）：20% 概率随机遮挡图片局部区域
        A.CoarseDropout(p=0.1),

        # ========== 环境增强 ==========
        A.RandomRain(p=0.1), # 随机下雨效果：10% 概率添加，模拟雨天场景，提升模型环境适应性。
        A.RandomFog(p=0.05), # 随机雾天效果：5% 概率添加，低概率避免大量雾天图影响训练。
        A.RandomSnow(brightness_coeff=1.5, p=0.05), # 随机下雪效果：5% 概率添加，模拟雪天场景。
        # A.RandomSunFlare(p=0.05), # 随机阳光眩光：5% 概率添加，模拟强光场景。
        A.CoarseDropout(p=0.1),
        A.CLAHE(p=0.05), # 增强图片局部对比度，提升暗部目标的可见性。
        A.ImageCompression(
            # compression_type="webp",
            compression_type="jpeg",
            quality_range=[50, 50],
            p=1.0
        ),
    ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"], # 指定类别标签字段，保证类别和边界框同步处理；
            min_area=10, # 过滤掉变换后面积小于 10 像素的边界框（避免无效小框）；
            min_visibility=0.1 # 过滤掉可见度低于 10% 的边界框（比如被裁剪 / 遮挡大部分的目标）。
        ))
    return aug

# ====================== 标签处理 ======================
def read_yolo_label(label_path):
    bboxes = []
    class_labels = []
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f.readlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                class_id = int(parts[0])
                x, y, w, h = map(float, parts[1:])
                bboxes.append([x, y, w, h])
                class_labels.append(class_id)
    return bboxes, class_labels


def save_yolo_label(label_path, bboxes, class_labels):
    with open(label_path, "w") as f:
        for bbox, cls in zip(bboxes, class_labels):
            x = np.clip(bbox[0], 0.0, 1.0)
            y = np.clip(bbox[1], 0.0, 1.0)
            w = np.clip(bbox[2], 0.0, 1.0)
            h = np.clip(bbox[3], 0.0, 1.0)
            f.write(f"{int(cls)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


# ====================== 核心增强逻辑 ======================
def augment_dataset():
    img_files = [f for f in os.listdir(SRC_IMG_DIR) if any(f.lower().endswith(ext) for ext in IMG_EXTENSIONS)]
    if not img_files:
        print("错误：未找到任何图片文件！")
        return

    for img_file in tqdm(img_files, desc="数据集增强进度"):
        img_name = os.path.splitext(img_file)[0]
        img_path = os.path.join(SRC_IMG_DIR, img_file)
        label_path = os.path.join(SRC_LABEL_DIR, f"{img_name}.txt")

        img = cv2.imread(img_path)
        if img is None:
            print(f"警告：无法读取 {img_path}，跳过！")
            continue
        img_h, img_w = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        bboxes, class_labels = read_yolo_label(label_path)

        for idx in range(AUGMENT_NUM_PER_IMG):
            aug = get_augmentation_pipeline(img_h, img_w)
            try:
                augmented = aug(image=img, bboxes=bboxes, class_labels=class_labels)
            except Exception as e:
                print(f"警告：增强 {img_file}_{idx} 出错：{str(e)[:100]}，跳过！")
                continue

            # 修复图片格式
            aug_img = augmented["image"]
            if aug_img.dtype != np.uint8:
                aug_img = (aug_img * 255).astype(np.uint8) if aug_img.max() <= 1 else aug_img.astype(np.uint8)

            # 保存图片和标签
            aug_img_name = f"{img_name}_aug_{idx:03d}{os.path.splitext(img_file)[1]}"
            # cv2.imwrite(os.path.join(DST_IMG_DIR, aug_img_name), cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 9])
            cv2.imwrite(os.path.join(DST_IMG_DIR, aug_img_name), cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
            aug_label_name = f"{img_name}_aug_{idx:03d}.txt"
            save_yolo_label(os.path.join(DST_LABEL_DIR, aug_label_name), augmented["bboxes"], augmented["class_labels"])

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", message="Error fetching version info")
    warnings.filterwarnings("ignore", message="ShiftScaleRotate is a special case of Affine transform")

    print("开始数据集增强（无±90°/倒置图+低色彩强度+无裁剪报错）...")
    print(f"源目录：{SRC_IMG_DIR} | 输出目录：{DST_IMG_DIR}")
    print(f"每张图片生成 {AUGMENT_NUM_PER_IMG} 张增强图")
    print("=" * 50)

    augment_dataset()

    print("=" * 50)
    print("增强完成！")
    dst_imgs = [f for f in os.listdir(DST_IMG_DIR) if any(f.lower().endswith(ext) for ext in IMG_EXTENSIONS)]
    dst_labels = [f for f in os.listdir(DST_LABEL_DIR) if f.lower().endswith(".txt")]
    print(f"最终生成：图片 {len(dst_imgs)} 张 | 标签 {len(dst_labels)} 张")