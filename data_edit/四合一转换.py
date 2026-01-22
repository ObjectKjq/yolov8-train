import os
from PIL import Image

# ===================== 配置参数 =====================
# 源图片目录
src_img_dir = r"C:\Users\dxzw-xm16\Desktop\增强火花\挑选好的\images"
# 合并后图片保存目录
dst_img_dir = r"C:\Users\dxzw-xm16\Desktop\增强火花\挑选好的\asdfa"
# 标注文件目录
label_dir = r"C:\Users\dxzw-xm16\Desktop\增强火花\挑选好的\labels"
# 图片格式（支持多格式）
img_exts = [".jpg", ".png"]
# 拼接网格（固定2×2）
grid_rows, grid_cols = 2, 2
# 空白区域填充颜色（白色）
fill_color = (255, 255, 255)

# ===================== 工具函数 =====================
def create_dir_if_not_exist(dir_path):
    """创建目录（若不存在）"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def get_img_label_pairs(img_dir, label_dir, img_exts):
    """获取图片-标注文件对应关系，返回排序后的图片列表（支持多格式）"""
    img_files = []
    for f in os.listdir(img_dir):
        f_lower = f.lower()
        if any(f_lower.endswith(ext) for ext in img_exts):
            img_files.append(f)
    
    img_files.sort()  # 按文件名排序，保证分组顺序固定
    img_label_pairs = []
    for img_file in img_files:
        img_name = os.path.splitext(img_file)[0]
        label_file = os.path.join(label_dir, f"{img_name}.txt")
        img_label_pairs.append({
            "img_path": os.path.join(img_dir, img_file),
            "label_path": label_file if os.path.exists(label_file) else None,
            "img_name": img_name  # 保存文件名，方便调试
        })
    return img_label_pairs

def get_group_cell_size(img_pairs):
    """
    计算当前组4张图片的单元格尺寸（最大宽、最大高）
    :param img_pairs: 4个图片-标注字典的列表
    :return: 单元格尺寸 (cell_w, cell_h)
    """
    max_w, max_h = 0, 0
    for pair in img_pairs:
        with Image.open(pair["img_path"]) as img:
            w, h = img.size
            if w > max_w:
                max_w = w
            if h > max_h:
                max_h = h
    return (max_w, max_h)

def convert_label_coords(label_lines, img_size, cell_pos, cell_size, grid_cols):
    """
    转换标注坐标到合并后的大图（无缩放，仅偏移）
    :param label_lines: 原标注文件的行列表
    :param img_size: 单张图片的原始尺寸 (w, h)
    :param cell_pos: 图片所在单元格位置 (行索引, 列索引)
    :param cell_size: 单元格尺寸 (cell_w, cell_h)
    :param grid_cols: 网格列数（2）
    :return: 转换后的标注行列表
    """
    converted_lines = []
    img_w, img_h = img_size
    cell_w, cell_h = cell_size
    
    # 1. 计算单元格在大图中的偏移（左上角坐标）
    cell_offset_x = cell_pos[1] * cell_w  # 列索引 × 单元格宽
    cell_offset_y = cell_pos[0] * cell_h  # 行索引 × 单元格高
    
    # 2. 计算图片在单元格内的居中偏移（图片未缩放，仅居中）
    img_offset_x = (cell_w - img_w) // 2  # 水平居中偏移
    img_offset_y = (cell_h - img_h) // 2  # 垂直居中偏移
    
    # 3. 大图总尺寸
    big_w = cell_w * grid_cols
    big_h = cell_h * grid_rows
    
    # 4. 图片在大图中的实际偏移（单元格偏移 + 居中偏移）
    total_offset_x = cell_offset_x + img_offset_x
    total_offset_y = cell_offset_y + img_offset_y

    for line in label_lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue  # 跳过格式错误的行
        cls_id, cx_rel, cy_rel, w_rel, h_rel = parts
        cx_rel = float(cx_rel)
        cy_rel = float(cy_rel)
        w_rel = float(w_rel)
        h_rel = float(h_rel)

        # 原标注是相对坐标，先转绝对坐标（无缩放，直接×原图尺寸）
        cx_abs = cx_rel * img_w
        cy_abs = cy_rel * img_h
        w_abs = w_rel * img_w
        h_abs = h_rel * img_h

        # 转换为大图的绝对坐标（仅加偏移，无缩放）
        cx_big_abs = cx_abs + total_offset_x
        cy_big_abs = cy_abs + total_offset_y

        # 转大图的相对坐标（0-1）
        cx_big_rel = cx_big_abs / big_w
        cy_big_rel = cy_big_abs / big_h
        w_big_rel = w_abs / big_w  # 宽度相对大图的比例
        h_big_rel = h_abs / big_h  # 高度相对大图的比例

        # 保留6位小数，避免精度丢失
        converted_line = f"{cls_id} {cx_big_rel:.6f} {cy_big_rel:.6f} {w_big_rel:.6f} {h_big_rel:.6f}"
        converted_lines.append(converted_line)
    return converted_lines

def merge_four_imgs(img_pairs, dst_img_path, dst_label_path):
    """
    合并4张图片（保持原尺寸+居中）+ 转换标注
    :param img_pairs: 4个图片-标注字典的列表
    :param dst_img_path: 合并后图片保存路径
    :param dst_label_path: 合并后标注保存路径
    """
    # 1. 计算当前组的单元格尺寸（最大宽、最大高）
    cell_w, cell_h = get_group_cell_size(img_pairs)
    # 2. 大图尺寸（2×2网格）
    big_w = cell_w * grid_cols
    big_h = cell_h * grid_rows
    # 3. 创建大图画布（白色背景）
    big_img = Image.new("RGB", (big_w, big_h), fill_color)

    # 4. 网格位置：(行索引, 列索引) → 左上、右上、左下、右下
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    # 保存每张图片的原始尺寸（用于标注转换）
    img_sizes = []

    # 5. 逐个放置图片（保持原尺寸，居中）
    for idx, (row, col) in enumerate(positions):
        if idx >= len(img_pairs):
            img_sizes.append(None)
            continue
        
        # 读取图片（保持原尺寸，不缩放）
        img_path = img_pairs[idx]["img_path"]
        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size
        img_sizes.append((img_w, img_h))

        # 计算单元格左上角坐标
        cell_x = col * cell_w
        cell_y = row * cell_h

        # 计算图片在单元格内的居中坐标
        img_x = cell_x + (cell_w - img_w) // 2
        img_y = cell_y + (cell_h - img_h) // 2

        # 将图片粘贴到大图对应位置
        big_img.paste(img, (img_x, img_y))

    # 6. 保存合并后的图片（无压缩，保持清晰度）
    big_img.save(dst_img_path, quality=100)

    # 7. 转换并合并标注
    all_converted_labels = []
    for idx, pair in enumerate(img_pairs):
        if idx >= len(positions) or not pair["label_path"] or not os.path.exists(pair["label_path"]):
            continue
        
        # 读取原标注
        with open(pair["label_path"], "r", encoding="utf-8") as f:
            label_lines = f.readlines()
        
        # 获取当前图片的原始尺寸和单元格位置
        img_size = img_sizes[idx]
        cell_pos = positions[idx]
        
        # 转换标注坐标
        converted_labels = convert_label_coords(label_lines, img_size, cell_pos, (cell_w, cell_h), grid_cols)
        all_converted_labels.extend(converted_labels)
    
    # 8. 保存合并后的标注
    with open(dst_label_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_converted_labels))

# ===================== 主程序 =====================
if __name__ == "__main__":
    # 创建目标目录
    create_dir_if_not_exist(dst_img_dir)
    dst_label_dir = os.path.join(os.path.dirname(dst_img_dir), "大火花-四个合一个-合并完-labels")
    create_dir_if_not_exist(dst_label_dir)

    # 获取图片-标注对应关系
    img_label_pairs = get_img_label_pairs(src_img_dir, label_dir, img_exts)
    total_groups = len(img_label_pairs) // 4
    remaining_imgs = len(img_label_pairs) % 4

    # 按每4张分组处理
    for group_idx in range(total_groups):
        start_idx = group_idx * 4
        end_idx = start_idx + 4
        current_group = img_label_pairs[start_idx:end_idx]
        
        # 定义合并后文件名称
        group_name = f"merged_group_{group_idx + 1}"
        dst_img_path = os.path.join(dst_img_dir, f"{group_name}.jpg")
        dst_label_path = os.path.join(dst_label_dir, f"{group_name}.txt")
        
        # 合并图片+转换标注
        merge_four_imgs(current_group, dst_img_path, dst_label_path)
        print(f"已处理第{group_idx + 1}组：{dst_img_path} | 标注：{dst_label_path}")
    
    # 提示剩余图片
    if remaining_imgs > 0:
        remaining_names = [p["img_name"] for p in img_label_pairs[-remaining_imgs:]]
        print(f"注意：剩余{remaining_imgs}张图片未处理（不足4张）：{remaining_names}")
    
    print("所有分组处理完成！")