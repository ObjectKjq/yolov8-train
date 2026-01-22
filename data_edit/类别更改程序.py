import os
import glob

def convert_labels_with_custom_mapping(file_path, mapping_dict):
    """
    使用自定义映射字典转换标签文件
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    converted_count = 0
    
    for line in lines:
        if line.strip():  # 跳过空行
            parts = line.strip().split()
            if parts:  # 确保行不为空
                class_id = int(parts[0])
                
                # 检查是否需要转换
                if class_id in mapping_dict:
                    parts[0] = str(mapping_dict[class_id])
                    converted_count += 1
                
                new_lines.append(' '.join(parts) + '\n')
    
    # 将转换后的内容写回文件
    with open(file_path, 'w') as f:
        f.writelines(new_lines)
    
    return converted_count, len(new_lines)

def main():
    # 设置标签文件所在的目录
    labels_dir = 'labels'  # 根据实际情况修改路径
    
    # 定义类别映射关系
    # 格式: {原类别索引: 新类别索引}
    class_mapping = {
        1: 3,  # vest -> no-helmet
        3: 1,  # no-helmet -> vest
        2: 6,  # gloves -> person
        6: 2,  # person -> gloves
    }
    
    if not os.path.exists(labels_dir):
        print(f"错误：目录 '{labels_dir}' 不存在")
        return
    
    # 获取所有txt文件
    txt_files = glob.glob(os.path.join(labels_dir, '*.txt'))
    print(f"找到 {len(txt_files)} 个标签文件")
    
    if not txt_files:
        print("没有找到任何txt文件")
        return
    
    # 备份原文件（可选）
    backup_dir = os.path.join(labels_dir, 'backup')
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
        for file_path in txt_files:
            import shutil
            shutil.copy2(file_path, os.path.join(backup_dir, os.path.basename(file_path)))
        print(f"已创建备份到: {backup_dir}")
    
    # 处理所有文件
    total_converted = 0
    total_labels = 0
    
    for file_path in txt_files:
        converted_count, label_count = convert_labels_with_custom_mapping(file_path, class_mapping)
        total_converted += converted_count
        total_labels += label_count
        print(f"处理文件: {os.path.basename(file_path):20} - 标签: {label_count:4} 个, 转换: {converted_count:4} 个")
    
    print(f"\n处理完成！")
    print(f"总共处理文件: {len(txt_files)} 个")
    print(f"总共标签数: {total_labels} 个")
    print(f"转换标签数: {total_converted} 个")
    
    # 显示类别映射表
    print("\n类别映射表:")
    print(f"原类别 -> 新类别")
    for old_class, new_class in class_mapping.items():
        print(f"  {old_class:2} -> {new_class:2}")

if __name__ == "__main__":
    main()