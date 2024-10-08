# %%
from pathlib import Path
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import pylab
import numpy as np
from objprint import objprint


ROOT = Path("/Users/zq/Documents/gists/learn_cocoeval")
# %%
# 加载标注文件和预测文件
annFile = ROOT / "annotations/instances_val2014.json"
cocoGt = COCO(str(annFile.resolve()))
resFile = ROOT / "instances_val2014_fakebbox100_results.json"
cocoDt = cocoGt.loadRes(str(resFile.resolve()))
# %%
# 创建评估器对象
annType = "bbox"
cocoEval = COCOeval(cocoGt, cocoDt, annType)
# %%
# 选取前ground truth的前100个预测结果进行评估
imgIds = sorted(cocoGt.getImgIds())
imgIds = imgIds[0:100]
cocoEval.params.imgIds = imgIds
objprint(cocoEval.params)

# %%
# 开始评估
cocoEval.evaluate()
# 汇总评估结果
cocoEval.accumulate()
# 打印评估结果
cocoEval.summarize()


# %%
# %%


import numpy as np
from collections import defaultdict


class COCOevalSimplified:
    def __init__(self, cocoGt=None, cocoDt=None, iouType="bbox", iouThrs=[0.5]):
        """
        初始化 COCOevalSimplified 类，仅用于指定 IoU 阈值下的目标检测结果评估
        :param cocoGt: COCO ground truth 对象
        :param cocoDt: COCO detection 结果对象
        :param iouType: IoU 类型，默认为 'bbox'
        :param iouThrs: 指定的 IoU 阈值列表，例如 [0.5]
        """
        self.cocoGt = cocoGt
        self.cocoDt = cocoDt
        self.params = Params(iouType=iouType)
        self.params.iouThrs = np.array(iouThrs)
        self.evalImgs = []
        self.eval = {}
        self._gts = defaultdict(list)
        self._dts = defaultdict(list)
        self.ious = {}
        if cocoGt is not None:
            self.params.imgIds = sorted(cocoGt.getImgIds())
            self.params.catIds = sorted(cocoGt.getCatIds())

    def _prepare(self):
        """
        准备评估所需的数据
        """
        p = self.params
        gts = self.cocoGt.loadAnns(
            self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds)
        )
        dts = self.cocoDt.loadAnns(
            self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds)
        )

        for gt in gts:
            gt["ignore"] = gt.get("ignore", 0)
            self._gts[gt["image_id"], gt["category_id"]].append(gt)
        for dt in dts:
            self._dts[dt["image_id"], dt["category_id"]].append(dt)

    def computeIoU(self, imgId, catId):
        """
        计算单张图片、单个类别下的 IoU
        """
        p = self.params
        gt = self._gts[imgId, catId]
        dt = self._dts[imgId, catId]
        if len(gt) == 0 or len(dt) == 0:
            return []

        dt = sorted(dt, key=lambda x: -x["score"])
        dt = dt[: p.maxDets[-1]]

        g = [g["bbox"] for g in gt]
        d = [d["bbox"] for d in dt]

        # 计算 IoU
        iscrowd = [int(o.get("iscrowd", 0)) for o in gt]
        ious = self.iou(d, g, iscrowd)
        return ious

    def iou(self, dts, gts, iscrowd):
        """
        计算 IoU 矩阵
        """
        ious = np.zeros((len(dts), len(gts)))
        for i, dt in enumerate(dts):
            dt_area = dt[2] * dt[3]
            dt_box = [dt[0], dt[1], dt[0] + dt[2], dt[1] + dt[3]]
            for j, gt in enumerate(gts):
                gt_area = gt[2] * gt[3]
                gt_box = [gt[0], gt[1], gt[0] + gt[2], gt[1] + gt[3]]

                # 计算交集
                inter_x1 = max(dt_box[0], gt_box[0])
                inter_y1 = max(dt_box[1], gt_box[1])
                inter_x2 = min(dt_box[2], gt_box[2])
                inter_y2 = min(dt_box[3], gt_box[3])
                inter_w = max(0, inter_x2 - inter_x1)
                inter_h = max(0, inter_y2 - inter_y1)
                inter_area = inter_w * inter_h

                union_area = dt_area + gt_area - inter_area
                iou = inter_area / union_area if union_area > 0 else 0
                ious[i, j] = iou
        return ious

    def evaluate(self):
        """
        运行评估过程
        """
        self._prepare()
        p = self.params
        catIds = p.catIds
        imgIds = p.imgIds

        self.ious = {
            (imgId, catId): self.computeIoU(imgId, catId)
            for imgId in imgIds
            for catId in catIds
        }

        self.evalImgs = [
            self.evaluateImg(imgId, catId) for catId in catIds for imgId in imgIds
        ]

    def evaluateImg(self, imgId, catId):
        """
        对单张图片、单个类别进行评估
        """
        p = self.params
        gt = self._gts[imgId, catId]
        dt = self._dts[imgId, catId]
        if len(gt) == 0 and len(dt) == 0:
            return None

        dt = sorted(dt, key=lambda x: -x["score"])
        dt = dt[: p.maxDets[-1]]

        gt_ignore = np.array([g.get("ignore", 0) for g in gt])
        dt_scores = np.array([d["score"] for d in dt])

        ious = self.ious[imgId, catId]
        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        dt_matches = np.zeros((T, D))
        dt_ignore = np.zeros((T, D))
        gt_matches = np.zeros((T, G))

        if len(ious) > 0:
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    iou = min([t, 1 - 1e-10])
                    m = -1
                    for gind, g in enumerate(gt):
                        if gt_matches[tind, gind] > 0:
                            continue
                        if ious[dind, gind] < iou:
                            continue
                        iou = ious[dind, gind]
                        m = gind
                    if m == -1:
                        continue
                    dt_ignore[tind, dind] = gt_ignore[m]
                    dt_matches[tind, dind] = 1
                    gt_matches[tind, m] = 1

        return {
            "dtMatches": dt_matches,
            "dtScores": dt_scores,
            "dtIgnore": dt_ignore,
            "gtIgnore": gt_ignore,
            "image_id": imgId,
            "category_id": catId,
        }

    def accumulate(self):
        """
        汇总评估结果，计算指定 IoU 阈值下的 AP
        """
        print("Accumulating evaluation results...")
        p = self.params
        T = len(p.iouThrs)
        R = len(p.recThrs)
        K = len(p.catIds)
        precision = -np.ones((T, R, K))
        recall = -np.ones((T, K))

        for k, catId in enumerate(p.catIds):
            E = [
                e for e in self.evalImgs if e is not None and e["category_id"] == catId
            ]
            if len(E) == 0:
                continue
            dtScores = np.concatenate([e["dtScores"] for e in E])
            inds = np.argsort(-dtScores, kind="mergesort")
            dtScoresSorted = dtScores[inds]

            dtm = np.concatenate([e["dtMatches"] for e in E], axis=1)[:, inds]
            dtIg = np.concatenate([e["dtIgnore"] for e in E], axis=1)[:, inds]
            gtIg = np.concatenate([e["gtIgnore"] for e in E])

            npig = np.count_nonzero(gtIg == 0)
            if npig == 0:
                continue

            tps = np.logical_and(dtm, np.logical_not(dtIg))
            fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg))

            tp_sum = np.cumsum(tps, axis=1).astype(dtype=float)
            fp_sum = np.cumsum(fps, axis=1).astype(dtype=float)

            for t in range(T):
                tp = tp_sum[t]
                fp = fp_sum[t]
                nd = len(tp)
                rc = tp / npig
                pr = tp / (fp + tp + np.spacing(1))

                recall[t, k] = rc[-1] if nd else 0
                q = np.zeros((R,))
                if nd:
                    for i in range(nd - 1, 0, -1):
                        if pr[i] > pr[i - 1]:
                            pr[i - 1] = pr[i]
                    inds = np.searchsorted(rc, p.recThrs, side="left")
                    try:
                        for ri, pi in enumerate(inds):
                            q[ri] = pr[pi]
                    except:
                        pass
                precision[t, :, k] = q

        self.eval = {
            "params": p,
            "precision": precision,
            "recall": recall,
        }
        print("DONE")

    def summarize(self):
        """
        计算并显示指定 IoU 阈值下的 AP
        """
        p = self.params
        precision = self.eval["precision"]
        recall = self.eval["recall"]

        for t, iouThr in enumerate(p.iouThrs):
            ap = np.mean(precision[t, :, :][precision[t, :, :] > -1])
            print(
                "Average Precision (AP) @[ IoU={:<9} | area=   all | maxDets={:>3d} ] = {:.3f}".format(
                    "{:0.2f}".format(iouThr), p.maxDets[-1], ap
                )
            )


class Params:
    """
    参数类，仅保留与目标检测相关的部分
    """

    def __init__(self, iouType="bbox"):
        self.imgIds = []
        self.catIds = []
        self.iouType = iouType
        self.iouThrs = np.array([0.5])
        self.recThrs = np.linspace(0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01)) + 1)
        self.maxDets = [1, 10, 100]


iouThr = 0.50
cocoEvalSimplified_50 = COCOevalSimplified(
    cocoGt, cocoDt, iouType="bbox", iouThrs=[iouThr]
)
cocoEvalSimplified_50.params.imgIds = imgIds
cocoEvalSimplified_50.params.catIds = cocoGt.getCatIds()
cocoEvalSimplified_50.evaluate()
cocoEvalSimplified_50.accumulate()
print("\n简化版评估器结果（IoU=0.50）：")
cocoEvalSimplified_50.summarize()

# 使用 IoU=0.75 进行评估
iouThr = 0.75
cocoEvalSimplified_75 = COCOevalSimplified(
    cocoGt, cocoDt, iouType="bbox", iouThrs=[iouThr]
)
cocoEvalSimplified_75.params.imgIds = imgIds
cocoEvalSimplified_75.params.catIds = cocoGt.getCatIds()
cocoEvalSimplified_75.evaluate()
cocoEvalSimplified_75.accumulate()
print("\n简化版评估器结果（IoU=0.75）：")
cocoEvalSimplified_75.summarize()

# %%
