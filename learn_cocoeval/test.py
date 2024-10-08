# %%
from pathlib import Path
from collections import defaultdict
from pycocotools.coco import COCO


# %%
ROOT = Path("/Users/zq/Documents/gists/learn_cocoeval")
# %%
# 加载标注文件和预测文件
# ignore
annFile = ROOT / "annotations/instances_val2014.json"
cocoGt = COCO(str(annFile.resolve()))
resFile = ROOT / "instances_val2014_fakebbox100_results.json"
cocoDt = cocoGt.loadRes(str(resFile.resolve()))

# %%
imgIds = sorted(cocoGt.getImgIds())
imgIds = imgIds[0:1]
gts = cocoGt.loadAnns(cocoGt.getAnnIds(imgIds=imgIds))
dts = cocoDt.loadAnns(cocoDt.getAnnIds(imgIds=imgIds))
_gts = defaultdict(list)
for gt in gts:
    gt["ignore"] = gt.get("ignore", 0)
    _gts[gt["image_id"], gt["category_id"]].append(gt)
# %%
print("hello")


# %%
# 不要将函数的默认参数设置为可变对象
def f(a=[]):
    a.append(1)
    return a


print(f())
print(f())

# %%


# (x1, y1, x2, y2)
gt_box = [295.55, 93.96, 313.97, 152.79]
dt_box = [464.08, 105.09, 495.74, 146.99]
inter_x1 = max(dt_box[0], dt_box[0])
inter_y1 = max(dt_box[1], gt_box[1])
inter_x2 = min(dt_box[2], gt_box[2])
inter_y2 = min(dt_box[3], gt_box[3])
inter_w = inter_x2 - inter_x1
inter_h = inter_y2 - inter_y1

print()

# %%
import torch
from torchvision.ops import box_iou


def test_box_iou():
    n = 100
    boxes1 = xywh2xyxy(torch.rand(n, 4))
    boxes2 = xywh2xyxy(torch.rand(n, 4))
    iou = box_iou(boxes1, boxes2)
    my_iou = my_box_iou(boxes1, boxes2)
    assert torch.allclose(iou, my_iou, atol=1e-5)


def xywh2xyxy(boxes):
    """
    input: Xmin, Ymin, Width, Height
    output: Xmin, Ymin, Xmax, Ymax
    """
    xyxy = torch.zeros_like(boxes)
    xyxy[:, 0] = boxes[:, 0]
    xyxy[:, 1] = boxes[:, 1]
    xyxy[:, 2] = boxes[:, 2] + boxes[:, 0]
    xyxy[:, 3] = boxes[:, 3] + boxes[:, 1]
    return xyxy


def my_box_iou(boxes1, boxes2):
    ious = torch.zeros(boxes1.size(0), boxes2.size(0))
    for i, box1 in enumerate(boxes1):
        for j, box2 in enumerate(boxes2):
            iou = calc_iou(box1, box2)
            ious[i, j] = iou
    return ious


def calc_iou(box1, box2):
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    area1, area2 = w1 * h1, w2 * h2

    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)

    inter_area = inter_w * inter_h
    union_area = area1 + area2 - inter_area
    iou = inter_area / union_area
    return iou


test_box_iou()

# %%
