import os

# ===================== 核心配置（无需修改，直接运行） =====================
# 目标合并后的文件夹，最终所有文件都放这里
target_dir = "labels"
# 需要合并的源文件夹前缀+类别范围（labels0 ~ labels8）
source_prefix = "labels"
class_nums = range(0, 9)

# ===================== 第一步：自动创建目标文件夹 =====================
if not os.path.exists(target_dir):
    os.makedirs(target_dir)
    print(f"创建目标文件夹: {target_dir}")

# ===================== 第二步：遍历所有源文件夹，合并同名文件 =====================
# 遍历 labels0 到 labels8 所有文件夹
for cls in class_nums:
    source_dir = f"{source_prefix}{cls}"
    # 如果当前类别文件夹不存在，直接跳过，不报错
    if not os.path.exists(source_dir):
        print(f"文件夹 {source_dir} 不存在，跳过")
        continue
    
    # 遍历当前类别文件夹下的所有文件
    for file_name in os.listdir(source_dir):
        # 只处理txt标签文件，跳过其他格式的无关文件
        if file_name.endswith(".txt"):
            # 拼接 源文件路径 + 目标文件路径
            source_file_path = os.path.join(source_dir, file_name)
            target_file_path = os.path.join(target_dir, file_name)
            
            # 读取源文件的所有内容
            with open(source_file_path, "r", encoding="utf-8") as f_read:
                lines = f_read.readlines()
            
            # 以【追加模式】写入目标文件，同名文件自动合并内容
            with open(target_file_path, "a", encoding="utf-8") as f_write:
                for line in lines:
                    # 去除首尾空白符，跳过空行，避免写入无效空内容
                    line_strip = line.strip()
                    if line_strip:
                        f_write.write(line)

print("✅ 所有标签文件合并完成！合并后的文件均在 labels 文件夹下")