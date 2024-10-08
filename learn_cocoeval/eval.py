"""
简化版 cocoEval 类
"""

from typing import Optional
from collections import defaultdict
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from loguru import logger


class cocoEvalSimplified:  # pylint: disable=invalid-name
    """
    简化版 cocoEval 类
    """

    def __init__(
        self, cocoGt=None, cocoDt=None, iouType="bbox", iouThrs=None
    ):  # pylint: disable=invalid-name
        """
        初始化函数
        """
        logger.info("Initializing cocoEvalSimplified...")
        self.cocoGt: Optional[COCO] = cocoGt
        self.cocoDt = cocoDt
        self.params = Params()
        if iouThrs is None:
            iouThrs = [0.5]
        self.params.iouThrs = np.array(iouThrs)
        self.params.iouType = iouType

        self.eval = {}
        self.evalImgs = []
        self._gts = defaultdict(list)  # 默认返回空列表
        self._dts = defaultdict(list)

        if self.cocoGt is not None:
            self.params.imgIds = sorted(self.cocoGt.getImgIds())
            self.params.catIds = sorted(self.cocoGt.getCatIds())

    def _prepare(self):
        """
        按照imgId和catId分组
        """
        logger.info("Preparing data...")
        p = self.params
        gts = self.cocoGt.loadAnns(
            self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds)
        )
        dts = self.cocoDt.loadAnns(
            self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds)
        )

        for gt in gts:
            gt["ignore"] = gt.get("ignore", 0)  # 为 gt 的 ignore 字段设置默认值 0
            self._gts[gt["image_id"], gt["category_id"]].append(gt)
        for dt in dts:
            self._dts[dt["image_id"], dt["category_id"]].append(dt)
        # self.print_gt_dt()

    def print_gt_dt(self):
        """
        打印标注和检测结果
        """
        print("Ground Truth:")
        for k, v in self._gts.items():
            print(k, len(v))
        print("Detection Results:")
        for k, v in self._dts.items():
            print(k, len(v))

    def iou(self, dts, gts):
        """
        计算 IoU
        dts: (N, 4) ndarray of float
        gts: (K, 4) ndarray of float
        (x, y, w, h)
        """
        ious = np.zeros((len(dts), len(gts)))
        for i, dt in enumerate(dts):
            dt_area = dt[2] * dt[3]
            # convert to x1y1x2y2
            dt_box = [dt[0], dt[1], dt[0] + dt[2], dt[1] + dt[3]]
            # dt[2] = dt[0] + dt[2]
            # dt[3] = dt[1] + dt[3]
            for j, gt in enumerate(gts):
                gt_area = gt[2] * gt[3]
                # convert to x1y1x2y2
                gt_box = [gt[0], gt[1], gt[0] + gt[2], gt[1] + gt[3]]
                # gt[2] = gt[0] + gt[2]
                # gt[3] = gt[1] + gt[3]

                # calculate the area of intersection rectangle
                inter_x1 = max(dt_box[0], gt_box[0])
                inter_y1 = max(dt_box[1], gt_box[1])
                inter_x2 = min(dt_box[2], gt_box[2])
                inter_y2 = min(dt_box[3], gt_box[3])
                inter_w = inter_x2 - inter_x1
                inter_h = inter_y2 - inter_y1
                # if inter_w > 0 and inter_h > 0:
                #     inter_area = inter_w * inter_h
                # else:
                #     inter_area = 0
                inter_area = inter_w * inter_h
                union_area = dt_area + gt_area - inter_area
                iou = inter_area / union_area if union_area > 0 else 0
                ious[i, j] = iou

        return ious

    # def iou(self, dts, gts, iscrowd=None):
    #     """
    #     计算 IoU 矩阵
    #     """
    #     ious = np.zeros((len(dts), len(gts)))
    #     for i, dt in enumerate(dts):
    #         dt_area = dt[2] * dt[3]
    #         dt_box = [dt[0], dt[1], dt[0] + dt[2], dt[1] + dt[3]]
    #         for j, gt in enumerate(gts):
    #             gt_area = gt[2] * gt[3]
    #             gt_box = [gt[0], gt[1], gt[0] + gt[2], gt[1] + gt[3]]

    #             # 计算交集
    #             inter_x1 = max(dt_box[0], gt_box[0])
    #             inter_y1 = max(dt_box[1], gt_box[1])
    #             inter_x2 = min(dt_box[2], gt_box[2])
    #             inter_y2 = min(dt_box[3], gt_box[3])
    #             inter_w = max(0, inter_x2 - inter_x1)
    #             inter_h = max(0, inter_y2 - inter_y1)
    #             inter_area = inter_w * inter_h

    #             union_area = dt_area + gt_area - inter_area
    #             iou = inter_area / union_area if union_area > 0 else 0
    #             ious[i, j] = iou
    #     return ious

    def computeIoU(self, imgId, catId):
        """
        calculate IoU for a single image and category
        """
        p = self.params
        gt = self._gts[imgId, catId]
        dt = self._dts[imgId, catId]
        if len(gt) == 0 or len(dt) == 0:
            return []

        # sort dt by confidence
        dt = sorted(dt, key=lambda x: -x["score"])
        # keep top maxDets
        dt = dt[: p.maxDets[-1]]

        # get bbox
        g = [g["bbox"] for g in gt]
        d = [d["bbox"] for d in dt]

        # calculate IoU
        ious = self.iou(d, g)
        return ious

    def evaluateImg(self, imgId, catId):
        """
        match dt and gt in a single image and category
        """
        p = self.params
        gt = self._gts[imgId, catId]
        dt = self._dts[imgId, catId]

        if len(gt) == 0 and len(dt) == 0:
            return None

        dt = sorted(dt, key=lambda x: -x["score"])
        dt = dt[: p.maxDets[-1]]

        gt_ignore = [g.get("ignore", 0) for g in gt]
        dt_scores = np.array([d["score"] for d in dt])

        ious = self.computeIoU(imgId, catId)
        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)

        # dt_matches[i, j] = 1 if dt j is matched at iou threshold T[i]
        dt_matches = np.zeros((T, D))
        dt_ignore = np.zeros((T, D))
        gt_matches = np.zeros((T, G))

        if len(ious) > 0:
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    iou = min([t, 1 - 1e-10])
                    # find the matched gt, if not found, m = -1
                    m = -1
                    for gind, g in enumerate(gt):
                        # ignore matched gt
                        if gt_matches[tind, gind] > 0:
                            continue
                        # ignore iou < t
                        if ious[dind, gind] < iou:
                            continue
                        iou = ious[dind, gind]
                        m = gind
                    if m == -1:
                        continue

                    # TODO: 这里的缩进存疑，待确认。gt_matches的赋值是不是应该在 for gind内？要不要在找到匹配的gt后break？
                    # if gt is matched, set gt_matches = 1
                    gt_matches[tind, m] = 1
                    # if dt is matched, set dt_matches = 1
                    dt_matches[tind, dind] = 1
                    dt_ignore[tind, dind] = gt_ignore[m]

        return {
            "dtMatches": dt_matches,
            "dtScores": dt_scores,
            "dtIgnore": dt_ignore,
            "gtIgnore": gt_ignore,
            "image_id": imgId,
            "category_id": catId,
        }

    def evaluate(self):
        """
        evaluate all images and categories
        """
        self._prepare()
        p = self.params
        catIds = p.catIds
        imgIds = p.imgIds

        # compute IoU for all images and categories
        self.ious = {
            (imgId, catId): self.computeIoU(imgId, catId)
            for imgId in imgIds
            for catId in catIds
        }

        self.evalImgs = [
            self.evaluateImg(imgId, catId) for imgId in imgIds for catId in catIds
        ]

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


# %%
# 简化版参数类
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


def main():
    # 读取标注文件和预测文件
    annFile = "annotations/instances_val2014.json"
    cocoGt = COCO(annFile)
    resFile = "instances_val2014_fakebbox100_results.json"
    cocoDt = cocoGt.loadRes(resFile)

    # 筛选部分图片进行评估
    imgIds = sorted(cocoGt.getImgIds())
    imgIds = imgIds[0:100]

    # cocoEval
    cocoEval = COCOeval(cocoGt, cocoDt, iouType="bbox")
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    # 简化版 cocoEval
    myCocoEval_05 = cocoEvalSimplified(cocoGt, cocoDt, iouType="bbox", iouThrs=[0.5])
    myCocoEval_05.params.imgIds = imgIds
    myCocoEval_05.params.catIds = cocoGt.getCatIds()
    myCocoEval_05.evaluate()
    myCocoEval_05.accumulate()
    print("\n简化版评估器结果（IoU=0.50）：")
    myCocoEval_05.summarize()

    myCocoEval_075 = cocoEvalSimplified(cocoGt, cocoDt, iouType="bbox", iouThrs=[0.75])
    myCocoEval_075.params.imgIds = imgIds
    myCocoEval_075.params.catIds = cocoGt.getCatIds()
    myCocoEval_075.evaluate()
    myCocoEval_075.accumulate()
    print("\n简化版评估器结果（IoU=0.75）：")
    myCocoEval_075.summarize()


# %%
if __name__ == "__main__":
    main()
