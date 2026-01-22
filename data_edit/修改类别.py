import os

def modify_labels_by_mapping(labels_dir, class_mapping):
    """
    批量修改YOLOv8标签文件中的类别，根据指定的类别映射关系替换
    
    参数:
        labels_dir: 标签文件(.txt)所在的文件夹路径
        class_mapping: 类别映射字典，格式如 {原类别: 目标类别}，例如 {2:1, 5:3}
    """
    # 遍历文件夹中的所有文件
    for filename in os.listdir(labels_dir):
        if not filename.endswith('.txt'):
            continue  # 只处理txt文件
        
        file_path = os.path.join(labels_dir, filename)
        
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 处理每行标签
        new_lines = []
        for line in lines:
            line = line.strip()
            if not line:  # 空行直接保留
                new_lines.append('\n')
                continue
            
            # 分割标签（类别 + 4个坐标）
            parts = line.split()
            if len(parts) != 5:  # 不符合YOLO格式的行，保留原样
                new_lines.append(line + '\n')
                continue
            
            # 替换类别（如果在映射中）
            original_class = parts[0]
            # 尝试将类别转为整数（YOLO类别通常是整数）
            try:
                original_class_int = int(original_class)
                # 检查是否需要替换
                if original_class_int in class_mapping:
                    parts[0] = str(class_mapping[original_class_int])
            except ValueError:
                # 如果类别不是整数（极少数情况），不处理
                pass
            
            # 重组行并添加到新内容
            new_line = ' '.join(parts) + '\n'
            new_lines.append(new_line)
        
        # 写回修改后的内容
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        
        print(f"已处理: {filename}")

if __name__ == "__main__":
    # -------------------------- 配置参数 --------------------------
    # 1. 标签文件夹路径（请替换为你的实际路径）
    labels_directory = "./labels"  # 例如: "./train/labels" 或 "./val/labels"
    
    # 2. 类别映射规则（ key: 原类别，value: 目标类别 ）
    # 示例：将类别2改为1，其他类别不变
    class_mapping = {
        2: 1
        # 可以添加更多映射，例如：3:2, 5:0 等
    }
    # --------------------------------------------------------------
    
    # 检查文件夹是否存在
    if not os.path.isdir(labels_directory):
        print(f"错误：文件夹 '{labels_directory}' 不存在，请检查路径！")
    else:
        modify_labels_by_mapping(labels_directory, class_mapping)
        print("所有标签文件处理完成！")