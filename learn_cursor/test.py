import json
import xml.etree.ElementTree as ET
from pathlib import Path
from main import AnnotationConverter


def test_annotation_converter():
    converter = AnnotationConverter()

    # 测试COCO格式转换
    coco_input = "test_data/annotations.json"
    yolo_output = "test_data/annotations_yolo.txt"
    voc_output = "test_data/annotations_voc.xml"

    # 从COCO转换到YOLO
    annotations = converter.read_coco(coco_input)
    converter.write_yolo(annotations, yolo_output)

    # 从COCO转换到VOC
    converter.write_voc(annotations, voc_output)

    # 验证YOLO格式
    with open(yolo_output, "r", encoding="utf-8") as f:
        yolo_data = f.readlines()
    assert len(yolo_data) == 1, "YOLO格式转换失败"
    assert "0" in yolo_data[0], "YOLO格式类别不正确"

    # 验证VOC格式
    tree = ET.parse(voc_output)
    root = tree.getroot()
    objects = root.findall("object")
    assert len(objects) == 1, "VOC格式转换失败"
    assert objects[0].find("name").text == "cat", "VOC格式类别不正确"

    print("所有测试通过！")


if __name__ == "__main__":
    test_annotation_converter()
