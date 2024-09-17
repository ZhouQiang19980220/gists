"""
用于转换不同格式的图像标注的类。
"""

# %%
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
import argparse

import cv2


# %%
class AnnotationConverter:
    """用于转换不同格式的图像标注的类。"""

    def __init__(self):
        self.formats = {
            "COCO": self.read_coco,
            "VOC": self.read_voc,
            "YOLO": self.read_yolo,
        }
        self.dataset_info = defaultdict(int)

    def read_coco(self, annotation_file: str) -> list:
        """读取COCO格式的标注文件。

        参数:
            annotation_file (str): COCO格式标注文件的路径。

        返回:
            list: 包含标注信息的字典列表，每个字典包含'bbox'和'category'。
        """
        with Path(annotation_file).open("r", encoding="utf-8") as f:
            data = json.load(f)
        annotations = []
        for ann in data["annotations"]:
            bbox = ann["bbox"]
            category = data["categories"][ann["category_id"] - 1]["name"]
            annotations.append({"bbox": bbox, "category": category})
            self.dataset_info[category] += 1
        return annotations

    def read_voc(self, annotation_file: str) -> list:
        """读取VOC格式的标注文件。

        参数:
            annotation_file (str): VOC格式标注文件的路径。

        返回:
            list: 包含标注信息的字典列表，每个字典包含'bbox'和'category'。
        """
        tree = ET.parse(Path(annotation_file))
        root = tree.getroot()
        annotations = []
        for obj in root.findall("object"):
            name_elem = obj.find("name")
            category = name_elem.text if name_elem is not None else "unknown"
            bbox = obj.find("bndbox")
            if bbox is not None:
                x1 = (
                    int(bbox.find("xmin").text)  # type: ignore
                    if bbox.find("xmin") is not None
                    and bbox.find("xmin").text.isdigit()  # type: ignore
                    else 0
                )
                y1 = (
                    int(bbox.find("ymin").text)  # type: ignore
                    if bbox.find("ymin") is not None
                    and bbox.find("ymin").text.isdigit()  # type: ignore
                    else 0
                )
                x2 = (
                    int(bbox.find("xmax").text)  # type: ignore
                    if bbox.find("xmax") is not None
                    and bbox.find("xmax").text.isdigit()  # type: ignore
                    else 0
                )
                y2 = (
                    int(bbox.find("ymax").text)  # type: ignore
                    if bbox.find("ymax") is not None
                    and bbox.find("ymax").text.isdigit()  # type: ignore
                    else 0
                )
                annotations.append(
                    {"bbox": [x1, y1, x2 - x1, y2 - y1], "category": category}
                )
                self.dataset_info[category] += 1
        return annotations

    def read_yolo(self, annotation_file: str, img_width: int, img_height: int) -> list:
        """读取YOLO格式的标注文件。

        参数:
            annotation_file (str): YOLO格式标注文件的路径。
            img_width (int): 图像的宽度。
            img_height (int): 图像的高度。

        返回:
            list: 包含标注信息的字典列表，每个字典包含'bbox'和'category'。
        """
        with Path(annotation_file).open("r", encoding="utf-8") as f:
            lines = f.readlines()
        annotations = []
        for line in lines:
            data = line.strip().split()
            category = int(data[0])
            x_center, y_center, width, height = map(float, data[1:])
            x1 = int((x_center - width / 2) * img_width)
            y1 = int((y_center - height / 2) * img_height)
            x2 = int((x_center + width / 2) * img_width)
            y2 = int((y_center + height / 2) * img_height)
            annotations.append(
                {"bbox": [x1, y1, x2 - x1, y2 - y1], "category": category}
            )
            self.dataset_info[category] += 1
        return annotations

    def convert(
        self, input_format: str, output_format: str, input_file: str, output_file: str
    ) -> None:
        """转换标注格式。

        参数:
            input_format (str): 输入标注格式。
            output_format (str): 输出标注格式。
            input_file (str): 输入文件路径。
            output_file (str): 输出文件路径。

        返回:
            None
        """
        annotations = self.formats[input_format](input_file)

        if output_format == "COCO":
            self.write_coco(annotations, output_file)
        elif output_format == "VOC":
            self.write_voc(annotations, output_file)
        elif output_format == "YOLO":
            self.write_yolo(annotations, output_file)
        else:
            raise ValueError(f"不支持的输出格式：{output_format}")

    def write_coco(self, annotations: list, output_file: str) -> None:
        """写入COCO格式的标注文件。

        参数:
            annotations (list): 包含标注信息的字典列表。
            output_file (str): 输出文件路径。

        返回:
            None
        """
        coco_data = {
            "images": [
                {"id": 0, "file_name": "image.jpg", "width": 640, "height": 480}
            ],
            "annotations": [],
            "categories": [],
        }
        category_id_map: dict[str, int] = {}

        for idx, ann in enumerate(annotations):
            if ann["category"] not in category_id_map:
                category_id = len(category_id_map) + 1
                category_id_map[ann["category"]] = category_id
                coco_data["categories"].append(
                    {"id": category_id, "name": ann["category"]}
                )

            coco_data["annotations"].append(
                {
                    "id": idx,
                    "image_id": 0,
                    "category_id": category_id_map[ann["category"]],
                    "bbox": ann["bbox"],
                    "area": ann["bbox"][2] * ann["bbox"][3],
                    "iscrowd": 0,
                }
            )

        with Path(output_file).open("w", encoding="utf-8") as f:
            json.dump(coco_data, f)

    def write_voc(self, annotations: list, output_file: str) -> None:
        """写入VOC格式的标注文件。

        参数:
            annotations (list): 包含标注信息的字典列表。
            output_file (str): 输出件路径。

        返回:
            None
        """
        root = ET.Element("annotation")
        ET.SubElement(root, "folder").text = "images"
        ET.SubElement(root, "filename").text = "image.jpg"
        ET.SubElement(root, "path").text = "/path/to/image.jpg"

        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = "640"
        ET.SubElement(size, "height").text = "480"
        ET.SubElement(size, "depth").text = "3"

        for ann in annotations:
            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = str(ann["category"])
            ET.SubElement(obj, "pose").text = "Unspecified"
            ET.SubElement(obj, "truncated").text = "0"
            ET.SubElement(obj, "difficult").text = "0"

            bbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bbox, "xmin").text = str(int(ann["bbox"][0]))
            ET.SubElement(bbox, "ymin").text = str(int(ann["bbox"][1]))
            ET.SubElement(bbox, "xmax").text = str(int(ann["bbox"][0] + ann["bbox"][2]))
            ET.SubElement(bbox, "ymax").text = str(int(ann["bbox"][1] + ann["bbox"][3]))

        tree = ET.ElementTree(root)
        tree.write(Path(output_file))

    def write_yolo(self, annotations: list, output_file: str) -> None:
        """写入YOLO格式的标注文件。

        参数:
            annotations (list): 包含标注信息的字典列表。
            output_file (str): 输出文件路径。

        返回:
            None
        """
        img_width, img_height = 640, 480  # 假设图像尺寸
        with Path(output_file).open("w", encoding="utf-8") as f:
            for ann in annotations:
                category_id = ann["category"] if isinstance(ann["category"], int) else 0
                x, y, w, h = ann["bbox"]
                x_center = (x + w / 2) / img_width
                y_center = (y + h / 2) / img_height
                width = w / img_width
                height = h / img_height
                f.write(
                    f"{category_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                )

    def visualize(self, image_file: str, annotations: list) -> None:
        """可视化标注的图像。

        参数:
            image_file (str): 图像文件的路径。
            annotations (list): 包含标注信息的字典列表。

        返回:
            None
        """
        image = cv2.imread(str(Path(image_file)))
        for ann in annotations:
            x, y, w, h = map(int, ann["bbox"])
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                image,
                ann["category"],
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )
        cv2.imshow("Annotated Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_dataset_info(self) -> None:
        """获取数据集信息。

        返回:
            None
        """
        print("数据集信息：")
        print(f"类别数：{len(self.dataset_info)}")
        for category, count in self.dataset_info.items():
            print(f"{category}: {count}个实例")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="转换图像标注格式")
    parser.add_argument(
        "input_format", type=str, help="输入标注格式（如 COCO, VOC, YOLO）"
    )
    parser.add_argument(
        "output_format", type=str, help="输出标注格式（如 COCO, VOC, YOLO）"
    )
    parser.add_argument("input_file", type=str, help="输入文件路径")
    parser.add_argument(
        "output_file",
        type=str,
        nargs="?",
        default="output.json",
        help="输出文件路径（默认为 output.json）",
    )

    args = parser.parse_args()

    converter = AnnotationConverter()
    converter.convert(
        args.input_format, args.output_format, args.input_file, args.output_file
    )
    print("转换完成！")
