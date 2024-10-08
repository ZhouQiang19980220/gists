# %%
import os
from collections import defaultdict
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# %%
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


# %%
class cocoEvalSimplified:
    """
    简化版 COCOeval 类
    """

    def __init__(self, cocoGT, cocoDt, iouType=None, iouThrs=None):
        """
        构造函数，初始化 COCOevalSimplified 类
        """
        self.cocoGT = cocoGT
        self.cocoDt = cocoDt

        if iouType is None:
            iouType = "bbox"
        self.params = Params(iouType=iouType)

        if iouThrs is None:
            iouThrs = [0.5]
        self.params.iouThrs = np.array(iouThrs)

        self.evalImgs = []
        self.eval = {}

        self._gts = defaultdict(list)
        self._dts = defaultdict(list)
        self.ious = {}

        if cocoGT is not None:
            self.params.imgIds = sorted(cocoGT.getImgIds())
            self.params.catIds = sorted(cocoGT.getCatIds())

    def _prepare(self):
        """
        准备评估数据，按照imgIds和catIds加载ground truth和detection结果
        """
        p = self.params
        # gts: [dict]
        # gts[0].keys() = dict_keys(['image_id', 'bbox', 'category_id', 'ignore', ...])
        # gts[0]['bbox'] = [x, y, w, h]
        gts = self.cocoGT.loadAnns(
            self.cocoGT.getAnnIds(imgIds=p.imgIds, catIds=p.catIds)
        )
        dts = self.cocoDt.loadAnns(
            self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds)
        )

        # 将ground truth和detection结果按照image id进行分组
        for gt in gts:
            key = (gt["image_id"], gt["category_id"])
            gt["ignore"] = gt.get("ignore", 0)
            self._gts[key].append(gt)

        for dt in dts:
            key = (dt["image_id"], dt["category_id"])
            self._dts[key].append(dt)

    def iou(self, dts, gts, iscrowd=None):
        """
        计算IoU
        dts: [x, y, w, h]
        gts: [x, y, w, h]
        """
        # iou[i, j] = iou of dts[i] and gts[j]
        ious = np.zeros((len(dts), len(gts)))
        for dind, dt in enumerate(dts):
            dt_area = dt[2] * dt[3]
            dt_box = [dt[0], dt[1], dt[0] + dt[2], dt[1] + dt[3]]
            for gind, gt in enumerate(gts):
                gt_area = gt[2] * gt[3]
                gt_box = [gt[0], gt[1], gt[0] + gt[2], gt[1] + gt[3]]

                # 计算交集
                inter_x1 = max(dt_box[0], gt_box[0])
                inter_y1 = max(dt_box[1], gt_box[1])
                inter_x2 = min(dt_box[2], gt_box[2])
                inter_y2 = min(dt_box[3], gt_box[3])

                # 计算交集面积
                inter_w = max(inter_x2 - inter_x1, 0)
                inter_h = max(inter_y2 - inter_y1, 0)
                inter_area = inter_w * inter_h

                # 计算并集面积
                union_area = dt_area + gt_area - inter_area
                if union_area == 0:
                    iou = 0
                else:
                    iou = inter_area / union_area
                ious[dind, gind] = iou
        return ious

    def computeIoU(self, imgId, catId):
        """
        计算IoU, 以ImgId和CatId分类
        """
        p = self.params
        gt = self._gts[imgId, catId]
        dt = self._dts[imgId, catId]
        if len(gt) == 0 or len(dt) == 0:
            return []

        # 按照置信度对detection结果进行排序
        dt = sorted(dt, key=lambda x: -x["score"])
        # 保留置信度最高的maxDets个结果
        dt = dt[: p.maxDets[-1]]

        # 获取bbox
        g = [g["bbox"] for g in gt]
        d = [d["bbox"] for d in dt]
        return self.iou(d, g)

    def evaluateImg(self, imgId, catId):
        """
        match dt and gt in a single image and category
        """
        p = self.params
        gt = self._gts[imgId, catId]
        dt = self._dts[imgId, catId]

        if len(gt) == 0 and len(dt) == 0:
            return None

        # 保留置信度最高的maxDets个结果
        dt = sorted(dt, key=lambda x: -x["score"])
        dt = dt[: p.maxDets[-1]]

        gt_ignore = [g.get("ignore", 0) for g in gt]
        dt_scores = np.array([d["score"] for d in dt])
        ious = self.computeIoU(imgId, catId)
        T = len(p.iouThrs)  # the number of IoU thresholds
        G = len(gt)  # the number of ground truth
        D = len(dt)  # the number of detection results

        dt_matches = np.zeros((T, D))
        dt_ignore = np.zeros((T, D))
        gt_matches = np.zeros((T, G))

        for tind, t in enumerate(p.iouThrs):
            for dind, d in enumerate(dt):   # iterate over dt sorted by score in descending order
                iou = min([t, 1 - 1e-10])   # iou threshold, to avoid numerical instability
                m = -1  # find the matched gt, if not found, m = -1
                for gind, g in enumerate(gt):
                    # ignore matched gt
                    if m != -1:
                        continue
                    # ignore matched gt
                    if gt_ignore[gind] > 0:
                        continue
                    # ignore iou < t
                    if ious[dind, gind] < iou:
                        continue
                    iou = ious[dind, gind]
                    m = gind # find the matched gt

                    # TODO
                    
                

    # def evaluateImg(self, imgId, catId):
    #     """
    #     match dt and gt in a single image and category
    #     """
    #     p = self.params
    #     gt = self._gts[imgId, catId]
    #     dt = self._dts[imgId, catId]

    #     if len(gt) == 0 and len(dt) == 0:
    #         return None

    #     dt = sorted(dt, key=lambda x: -x["score"])
    #     dt = dt[: p.maxDets[-1]]

    #     gt_ignore = [g.get("ignore", 0) for g in gt]
    #     dt_scores = np.array([d["score"] for d in dt])

    #     ious = self.computeIoU(imgId, catId)
    #     T = len(p.iouThrs)
    #     G = len(gt)
    #     D = len(dt)

    #     # dt_matches[i, j] = 1 if dt j is matched at iou threshold T[i]
    #     dt_matches = np.zeros((T, D))
    #     dt_ignore = np.zeros((T, D))
    #     gt_matches = np.zeros((T, G))

    #     if len(ious) > 0:
    #         for tind, t in enumerate(p.iouThrs):
    #             for dind, d in enumerate(dt):
    #                 iou = min([t, 1 - 1e-10])
    #                 # find the matched gt, if not found, m = -1
    #                 m = -1
    #                 for gind, g in enumerate(gt):
    #                     # ignore matched gt
    #                     if gt_matches[tind, gind] > 0:
    #                         continue
    #                     # ignore iou < t
    #                     if ious[dind, gind] < iou:
    #                         continue
    #                     iou = ious[dind, gind]
    #                     m = gind
    #                 if m == -1:
    #                     continue

    #                 # TODO: 这里的缩进存疑，待确认。gt_matches的赋值是不是应该在 for gind内？要不要在找到匹配的gt后break？
    #                 # if gt is matched, set gt_matches = 1
    #                 gt_matches[tind, m] = 1
    #                 # if dt is matched, set dt_matches = 1
    #                 dt_matches[tind, dind] = 1
    #                 dt_ignore[tind, dind] = gt_ignore[m]

    #     return {
    #         "dtMatches": dt_matches,
    #         "dtScores": dt_scores,
    #         "dtIgnore": dt_ignore,
    #         "gtIgnore": gt_ignore,
    #         "image_id": imgId,
    #         "category_id": catId,
    #     }

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

#%%
