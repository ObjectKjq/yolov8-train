import os

def delete_unmatched_label_files(images_dir, labels_dir):
    # 获取images目录中所有文件名（不带扩展名）
    image_basenames = set()
    for filename in os.listdir(images_dir):
        # 处理带点的文件名
        basename = os.path.splitext(filename)[0]
        image_basenames.add(basename)
    
    # 遍历labels目录并删除不匹配的文件
    deleted_files = []
    for filename in os.listdir(labels_dir):
        filepath = os.path.join(labels_dir, filename)
        basename = os.path.splitext(filename)[0]
        
        # 如果基本名不在image集合中
        if basename not in image_basenames:
            os.remove(filepath)
            deleted_files.append(filename)
    
    # 输出结果报告
    print(f"共删除 {len(deleted_files)} 个文件:")
    for f in deleted_files:
        print(f"  - {f}")

if __name__ == "__main__":
    # 配置目录路径（根据实际情况修改）
    images_directory = "images"
    labels_directory = "labels"
    
    # 确保目录存在
    if not os.path.exists(images_directory):
        print(f"错误: {images_directory} 目录不存在")
        exit(1)
        
    if not os.path.exists(labels_directory):
        print(f"错误: {labels_directory} 目录不存在")
        exit(1)
    
    # 执行删除操作
    delete_unmatched_label_files(images_directory, labels_directory)