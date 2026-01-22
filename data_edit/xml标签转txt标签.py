import os
import xml.etree.ElementTree as ET

def xml_to_yolo(xml_dir, txt_dir, classes):
    """
    将XML标签文件转换为YOLOv8格式的TXT文件
    :param xml_dir: XML文件所在目录（Annotations）
    :param txt_dir: 输出TXT文件的目录（建议命名为labels）
    :param classes: 类别名称列表（需与XML中的name对应）
    """
    # 创建输出目录（如果不存在）
    os.makedirs(txt_dir, exist_ok=True)
    
    # 遍历所有XML文件
    for xml_file in os.listdir(xml_dir):
        if not xml_file.endswith('.xml'):
            continue  # 跳过非XML文件
        
        # 解析XML
        xml_path = os.path.join(xml_dir, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # 获取图片宽高
        size = root.find('size')
        width = float(size.find('width').text)
        height = float(size.find('height').text)
        
        # 准备TXT内容
        txt_content = []
        for obj in root.iter('object'):
            # 获取类别名称并转换为ID
            cls_name = obj.find('name').text
            if cls_name not in classes:
                continue  # 跳过未定义的类别
            cls_id = classes.index(cls_name)
            
            # 获取边界框坐标
            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
            
            # 转换为YOLO格式（归一化坐标）
            x_center = (xmin + xmax) / (2 * width)
            y_center = (ymin + ymax) / (2 * height)
            w = (xmax - xmin) / width
            h = (ymax - ymin) / height
            
            # 确保坐标在0-1范围内
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            w = max(0, min(1, w))
            h = max(0, min(1, h))
            
            # 添加到内容（保留6位小数）
            txt_content.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
        
        # 写入TXT文件（与XML同名，替换扩展名为txt）
        txt_file = os.path.splitext(xml_file)[0] + '.txt'
        txt_path = os.path.join(txt_dir, txt_file)
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(txt_content))
        
        print(f"已转换: {xml_file} -> {txt_file}")

if __name__ == "__main__":
    # -------------------------- 配置参数 --------------------------
    # XML文件所在目录（Annotations）
    xml_directory = "Annotations"  
    # 输出TXT文件的目录（建议放在labels目录）
    txt_directory = "labels"       
    # 类别列表（需与XML中的<name>标签完全一致，顺序对应类别ID）
    class_names = ["smoke"]  # 示例中只有smoke类别，根据实际情况添加
    # --------------------------------------------------------------
    
    xml_to_yolo(xml_directory, txt_directory, class_names)
    print("所有文件转换完成！")