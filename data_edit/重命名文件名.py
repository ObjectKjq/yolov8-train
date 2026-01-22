import os

def rename_images(image_dir, start_number):
    """
    批量重命名指定目录下的图片文件
    
    Args:
        image_dir (str): 图片所在目录路径
        start_number (int): 重命名起始数字
    """
    # 定义支持的图片后缀（小写）
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'}
    
    # 检查目录是否存在
    if not os.path.exists(image_dir):
        print(f"错误：目录 '{image_dir}' 不存在！")
        return
    
    # 检查起始数字是否为有效整数
    if not isinstance(start_number, int) or start_number < 0:
        print("错误：起始数字必须是非负整数！")
        return
    
    # 获取目录下的所有文件，并筛选出图片文件
    image_files = []
    for filename in os.listdir(image_dir):
        # 获取文件完整路径
        file_path = os.path.join(image_dir, filename)
        # 只处理文件（排除子目录）
        if os.path.isfile(file_path):
            # 获取文件后缀（小写）
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext in image_extensions:
                image_files.append(filename)
    
    # 如果没有找到图片文件
    if not image_files:
        print(f"在目录 '{image_dir}' 中未找到任何图片文件！")
        return
    
    # 开始重命名
    current_number = start_number
    renamed_count = 0
    print(f"开始重命名，共找到 {len(image_files)} 个图片文件...")
    
    for old_filename in image_files:
        # 获取文件完整路径和后缀
        old_path = os.path.join(image_dir, old_filename)
        file_ext = os.path.splitext(old_filename)[1].lower()
        
        # 构建新文件名
        new_filename = f"ai_generate_{current_number}{file_ext}"
        new_path = os.path.join(image_dir, new_filename)
        
        # 避免文件名重复（如果已存在则跳过）
        if os.path.exists(new_path):
            print(f"跳过：新文件名 '{new_filename}' 已存在，无法重命名 '{old_filename}'")
            current_number += 1
            continue
        
        try:
            # 执行重命名
            os.rename(old_path, new_path)
            print(f"成功：{old_filename} → {new_filename}")
            renamed_count += 1
            current_number += 1
        except Exception as e:
            print(f"失败：重命名 '{old_filename}' 时出错 - {str(e)}")
    
    print(f"\n重命名完成！成功重命名 {renamed_count} 个文件，共处理 {len(image_files)} 个图片文件。")

# 主程序入口
if __name__ == "__main__":
    # 配置参数（你可以直接修改这里的参数）
    IMAGE_DIRECTORY = "images"  # 图片目录（相对路径/绝对路径都可以）
    START_NUMBER = 0            # 起始数字（比如从1开始）
    
    # 调用重命名函数
    rename_images(IMAGE_DIRECTORY, START_NUMBER)