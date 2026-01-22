import os
import shutil

# 设置源文件夹和目标文件夹路径
src_folder = 'labels'
dst_folder = 'labels_file'

# 确保目标文件夹存在
os.makedirs(dst_folder, exist_ok=True)

# 获取labels文件夹下所有文件，并按文件名排序
files = sorted(os.listdir(src_folder))

# 遍历文件列表（从索引0开始计数）
for idx, file_name in enumerate(files):
    # 如果索引满足 (9, 19, 29, ...) 即 (idx % 10 == 9)
    if (idx + 1) % 10 == 0:
        src_path = os.path.join(src_folder, file_name)
        dst_path = os.path.join(dst_folder, file_name)
        
        # 移动文件
        shutil.move(src_path, dst_path)
        print(f'Moved: {file_name}')

print("File movement complete!")