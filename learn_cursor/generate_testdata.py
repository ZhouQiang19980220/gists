import json
import xml.etree.ElementTree as ET
from pathlib import Path

# 创建测试数据目录
output_dir = Path("test_data")
output_dir.mkdir(exist_ok=True)

# 生成COCO格式数据
coco_data = {
    "images": [{"id": 1, "file_name": "image.jpg", "width": 640, "height": 480}],
    "annotations": [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": [100, 150, 200, 300],
            "area": 60000,
            "iscrowd": 0,
        }
    ],
    "categories": [{"id": 1, "name": "cat"}],
}

with open(output_dir / "annotations.json", "w", encoding="utf-8") as f:
    json.dump(coco_data, f, ensure_ascii=False, indent=4)

# 生成YOLO格式数据
with open(output_dir / "annotations.txt", "w", encoding="utf-8") as f:
    f.write("0 0.5 0.5 0.5 0.625\n")  # 类别0，中心点和宽高相对值

# 生成VOC格式数据
voc_root = ET.Element("annotation")
ET.SubElement(voc_root, "folder").text = "images"
ET.SubElement(voc_root, "filename").text = "image.jpg"
ET.SubElement(voc_root, "path").text = str(output_dir / "image.jpg")

size = ET.SubElement(voc_root, "size")
ET.SubElement(size, "width").text = "640"
ET.SubElement(size, "height").text = "480"
ET.SubElement(size, "depth").text = "3"

obj = ET.SubElement(voc_root, "object")
ET.SubElement(obj, "name").text = "cat"
ET.SubElement(obj, "pose").text = "Unspecified"
ET.SubElement(obj, "truncated").text = "0"
ET.SubElement(obj, "difficult").text = "0"

bbox = ET.SubElement(obj, "bndbox")
ET.SubElement(bbox, "xmin").text = "100"
ET.SubElement(bbox, "ymin").text = "150"
ET.SubElement(bbox, "xmax").text = "300"
ET.SubElement(bbox, "ymax").text = "450"

tree = ET.ElementTree(voc_root)
tree.write(output_dir / "annotations.xml")
