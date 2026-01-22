import os

# ===================== 核心配置（无需修改，直接运行） =====================
# 源标签文件夹路径，就是你存放所有txt的labels文件夹
source_dir = "labels"
# 目标文件夹前缀，会自动生成 labels0, labels1 ... labels8
target_prefix = "labels"
# 类别标签的取值范围，根据你的需求是0-8
class_nums = range(0, 9)

# ===================== 自动创建所有目标文件夹 =====================
for cls in class_nums:
    target_dir = f"{target_prefix}{cls}"
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"创建文件夹: {target_dir}")

# ===================== 遍历并拆分所有txt标签文件 =====================
# 遍历源文件夹下的所有文件
for file_name in os.listdir(source_dir):
    # 只处理txt格式的标签文件，跳过其他无关文件
    if file_name.endswith(".txt"):
        source_file_path = os.path.join(source_dir, file_name)
        
        # 读取当前txt文件的所有内容
        with open(source_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        # 遍历当前文件的每一行标注内容
        for line in lines:
            # 去除首尾空格/换行符，跳过空行
            line_strip = line.strip()
            if not line_strip:
                continue
            
            # 核心：拆分出开头的类别数字（YOLO格式 类别 x y w h）
            cls_id = int(line_strip.split()[0])
            
            # 拼接目标文件路径：对应类别文件夹 + 原文件名
            target_dir = f"{target_prefix}{cls_id}"
            target_file_path = os.path.join(target_dir, file_name)
            
            # 将该行内容追加写入目标文件（保持原格式不变）
            with open(target_file_path, "a", encoding="utf-8") as f:
                f.write(line)

print("✅ 标签拆分完成！所有文件已按类别存入对应labelsN文件夹")