import os

def remove_specific_class(labels_dir, class_id_to_remove):
    """
    删除labels目录下所有TXT文件中指定的类别ID，保留其他类别
    """
    # 确保输入的类别ID是整数
    class_id_to_remove = int(class_id_to_remove)
    
    for filename in os.listdir(labels_dir):
        if not filename.endswith('.txt'):
            continue
        
        file_path = os.path.join(labels_dir, filename)
        keep_lines = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue  # 跳过空行
                
                # 分割行内容（处理可能的多空格分隔）
                parts = line.split()
                if len(parts) != 5:
                    # 跳过格式错误的行（非5列）
                    print(f"警告：{filename} 中存在格式错误的行：{line}")
                    continue
                
                try:
                    # 提取当前行的类别ID（第一列）
                    current_id = int(parts[0])
                    # 只保留不是目标删除ID的行
                    if current_id != class_id_to_remove:
                        keep_lines.append(line)
                except ValueError:
                    # 跳过类别ID无法转换为整数的行
                    print(f"警告：{filename} 中类别ID格式错误：{parts[0]}")
                    continue
        
        # 写回过滤后的内容（保持原格式）
        with open(file_path, 'w', encoding='utf-8') as f:
            if keep_lines:  # 如果有内容，用换行连接
                f.write('\n'.join(keep_lines) + '\n')
            else:  # 如果无内容，保持空文件
                f.write('')
        
        print(f"已处理：{filename}")

if __name__ == "__main__":
    # 配置参数
    labels_directory = "labels"  # 标签目录
    class_id_to_remove = 0       # 要删除的类别ID（这里指定0）
    
    remove_specific_class(labels_directory, class_id_to_remove)
    print("所有文件处理完成！")