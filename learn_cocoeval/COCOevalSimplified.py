from collections import defaultdict
import numpy as np


class COCOevalSimplified:

    def __init__(self, cocoGt, cocoDt, iouType="bbox", iouThrs=[0.5]) -> None:
        """
        简化版 COCOeval: 评估指定 IoU 阈值下的目标检测mAP
        :param cocoGt: COCO ground truth 对象
        :param cocoDt: COCO detection 结果对象
        :param iouType: IoU 类型，默认为 'bbox'
        :param iouThrs: 指定的 IoU 阈值列表，例如 [0.5]
        TP: IoU > 阈值，且类别正确
        FP: 要么与全部的 ground truth 的 IoU 都小于阈值，要么类别错误，要么重复检测
        FN: 未被检测到的 ground truth
        """
        self.cocoGt = cocoGt  # COCO ground
        self.cocoDt = cocoDt  # COCO detection
        # 参数对象，iouType为"bbox"表示目标检测任务
        self.params = Params(iouType=iouType)
        self.params.iouThrs = np.array(iouThrs)  # IoU 阈值

        self.evalImgs = []  # 用来存储每张图片的评估结果，即预测框和真实框的匹配情况
        self.eval = {}  # 用来存储最终的评估结果
        self._gts = defaultdict(list)  # 用来存储 ground truth, key=(imgId, catId)
        self._dts = defaultdict(list)  # 用来存储 detection, key=(imgId, catId)
        self.ious = {}  # 用来存储 IoU 矩阵, key=(imgId, catId)

        if cocoGt is not None:  # 对 imgIds 排序
            self.params.imgIds = sorted(cocoGt.getImgIds())
            self.params.catIds = sorted(cocoGt.getCatIds())

    def _prepare(self):
        """
        准备数据：在self._gts和self._dts中存储ground truth和detection，key=(imgId, catId)
        """
        p = self.params
        # gts[0].keys = ['segmentation', 'area', 'iscrowd', 'image_id', 'bbox', 'category_id', 'id']
        gts = self.cocoGt.loadAnns(
            self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds)
        )
        dts = self.cocoDt.loadAnns(
            self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds)
        )

        # self._gts[(imgId, catId)] = [gt1, gt2, ...]
        # 根据imgId和catId将gt和dt分组
        for gt in gts:
            gt["ignore"] = gt.get("ignore", 0)
            self._gts[gt["image_id"], gt["category_id"]].append(gt)
        for dt in dts:
            self._dts[dt["image_id"], dt["category_id"]].append(dt)

    def computeIoU(self, imgId, catId):
        """
        计算单张图片，单个类别下的 IoU
        """
        p = self.params
        # 根据imgId和catId获取gt和dt
        # 换言之，gt和dt是同一张图片，同一个类别下的真实框和预测框
        # 考虑某个类别时，暂时忽略其他类别
        gt = self._gts[imgId, catId]  # list of gt
        dt = self._dts[imgId, catId]  # list of dt
        if len(gt) == 0 or len(dt) == 0:
            return []

        # 根据置信度降序排列，优先匹配置信度高的预测框
        dt = sorted(dt, key=lambda x: -x["score"])
        # 截取最大检测数
        dt = dt[: p.maxDets[-1]]

        # 解析bbox: [x, y, w, h]
        g = [g["bbox"] for g in gt]
        d = [d["bbox"] for d in dt]

        # 计算IoU
        iscrowd = [int(o.get("iscrowd", 0)) for o in gt]
        ious = self.iou(d, g, iscrowd)
        return ious

    def iou(self, dts, gts, iscrowd):
        """
        计算IoU矩阵
        dts中有n1个bbox，gts中有n2个bbox，返回一个n1*n2的矩阵，矩阵中的元素为IoU
        dts: list of dt bbox
        gts: list of gt bbox
        bbox: [x, y, w, h]
        """
        ious = np.zeros((len(dts), len(gts)))
        for i, dt in enumerate(dts):
            dt_area = dt[2] * dt[3]
            # dt_box: [x1, y1, x2, y2]
            dt_box = [dt[0], dt[1], dt[0] + dt[2], dt[1] + dt[3]]
            for j, gt in enumerate(gts):
                gt_area = gt[2] * gt[3]
                # gt_box: [x1, y1, x2, y2]
                gt_box = [gt[0], gt[1], gt[0] + gt[2], gt[1] + gt[3]]

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

        # 遍历所有的imgId和catId，计算IoU
        # 牢记，如果预测框和真实框的imgId和catId都相同，才会计算IoU
        self.ious = {
            (imgId, catId): self.computeIoU(imgId, catId)
            for imgId in imgIds
            for catId in catIds
        }

        # 遍历所有的imgId和catId，计算评估结果
        self.evalImgs = [
            self.evaluateImg(imgId, catId) for catId in catIds for imgId in imgIds
        ]

    def evaluateImg(self, imgId, catId):
        """
        指定imgId和catId，计算评估结果
        """
        p = self.params
        gt = self._gts[imgId, catId]
        dt = self._dts[imgId, catId]

        if len(gt) == 0 and len(dt) == 0:
            return None

        # 根据置信度降序排列，优先匹配置信度高的预测框
        dt = sorted(dt, key=lambda x: -x["score"])
        # 截取最大检测数
        dt = dt[: p.maxDets[-1]]

        # 获取真实框的ignore属性和预测框的置信度
        gt_ignore = np.array([g.get("ignore", 0) for g in gt])
        dt_scores = np.array([d["score"] for d in dt])

        # 获取同imgId，同catId下的全部IoU
        ious = self.ious[imgId, catId]

        # 暂时没搞懂这些变量的含义
        T = len(p.iouThrs)  # IoU阈值数
        G = len(gt)  # 真实框数
        D = len(dt)  # 预测框数
        # 记录预测框匹配情况，如果预测框匹配过，则为1，否则为0
        dt_matches = np.zeros((T, D))
        # 记录预测框的ignore属性，如果忽略，则为1，否则为0
        dt_ignore = np.zeros((T, D))
        # 记录真实框匹配情况，如果真实框匹配过，则为1，否则为0
        gt_matches = np.zeros((T, G))

        # 这里计算预测框和真实框的匹配情况
        if len(ious) > 0:  # len(ious)和 len(dt)的值相等
            for tind, t in enumerate(p.iouThrs):  # 遍历IoU阈值
                for dind, d in enumerate(dt):  # 遍历预测框，置信度降序
                    iou = min([t, 1 - 1e-10])  # 确保iou阈值在0-1之间
                    m = -1
                    for gind, g in enumerate(gt):  # 遍历真实框
                        # 如果真实框已经匹配过，则跳过
                        if gt_matches[tind, gind] > 0:
                            continue
                        # 如果 IoU 小于阈值，则跳过
                        if ious[dind, gind] < iou:
                            continue
                        iou = ious[dind, gind]
                        m = gind  # 记录匹配的真实框索引
                    if (
                        m == -1
                    ):  # 这里如果m=-1，说明当前预测框和所有的ground truth 的IoU都小于阈值
                        continue

                    # 将真实框的ignore属性赋值给预测框
                    dt_ignore[tind, dind] = gt_ignore[m]
                    # 记录预测框匹配情况，方便快速找到被匹配上的真实框
                    dt_matches[tind, dind] = 1
                    # 记录真实框匹配情况，确保每个真实框只匹配一次
                    gt_matches[tind, m] = 1

        # 返回结果中并没有指明某个预测框匹配上了哪个真实框
        # 因为计算 AP 时，只需要知道某个预测框是否匹配上了真实框，或者某个真实框是否被匹配上了
        # TP：预测框中的某个框匹配上了真实框
        # FP：预测框中的某个框没有匹配上真实框
        return {
            "dtMatches": dt_matches,  # dt_matches[t, d]=1表示第t个IoU阈值下的第d个预测框匹配上了真实框
            "dtScores": dt_scores,  # 预测框的置信度, shape=(D,), D为预测框数, 降序排列
            "dtIgnore": dt_ignore,  # 预测框的ignore属性, shape=(T, D), T为IoU阈值, D为预测框数, 1表示忽略, 0表示不忽略,来源于真实框
            "gtIgnore": gt_ignore,  # 真实框的ignore属性, shape=(G,), G为真实框数, 1表示忽略, 0表示不忽略
            "image_id": imgId,  # 图像ID
            "category_id": catId,  # 类别ID
        }

    def accumulate(self):
        """
        汇总评估结果
        """
        print("Accumulating evaluation results...")
        p = self.params
        # IoU阈值数
        T = len(p.iouThrs)
        # 召回率阈值数，用于计算 AP。因为 AP 就是在不同召回率下的精度的平均值
        R = len(p.recThrs)
        # 类别数
        K = len(p.catIds)
        # shape=(T, R, K), 用于记录每个类别、每个IoU阈值、每个召回率阈值下的精度
        # 这里为什么多出来类别这个维度？因为要先每个类别单独评估，然后再求平均值
        precision = -np.ones((T, R, K))
        # shape=(T, K), 用于记录每个类别、每个IoU阈值下的召回率
        recall = -np.ones((T, K))

        for k, catId in enumerate(p.catIds):
            # E：包含当前类别下的所有评估结果
            E = [
                # e是通过evaluateImg计算出来的评估结果，是一个字典
                # 包含了dtMatches, dtScores, dtIgnore, gtIgnore，image_id, category_id
                e
                for e in self.evalImgs
                if e is not None and e["category_id"] == catId
            ]
            # 如果没有评估结果，则跳过
            if len(E) == 0:
                continue

            # 这个类别下的所有预测框的置信度分数
            dtScores = np.concatenate([e["dtScores"] for e in E])
            # inds是dtScores的降序排列索引。换言之, inds[0]是dtScores中的最大值的索引, inds[-1]是最小值的索引
            inds = np.argsort(-dtScores, kind="mergesort")
            # 根据inds对dtScores进行降序排列
            dtScoresSorted = dtScores[inds]
            # dtm是dtMatches的拼接，shape=(T, D), D为预测框数，T为IoU阈值数。
            # dtm[t, d]=1表示第t个IoU阈值下的第d个预测框匹配上了真实框
            # inds是dtScores的降序排列索引，所以dtm的列也是按照dtScores的降序排列
            # dtm[t, inds[0]]表示第t个IoU阈值下置信度最高的预测框是否匹配
            dtm = np.concatenate([e["dtMatches"] for e in E], axis=1)[:, inds]
            # 与dtm类似，dtIg[t, d]=1表示第t个IoU阈值下的第d个预测框的ignore属性为1
            # dtIg[t, inds[0]]表示第t个IoU阈值下置信度最高的预测框的ignore属性
            dtIg = np.concatenate([e["dtIgnore"] for e in E], axis=1)[:, inds]
            gtIg = np.concatenate([e["gtIgnore"] for e in E])

            # npig是非忽略的真实框数
            npig = np.count_nonzero(gtIg == 0)

            if npig == 0:
                continue

            # np.logical_and表示对数组逐元素进行逻辑与操作
            # 这里筛选出非忽略且匹配上的预测框，其实就是 TP
            tps = np.logical_and(dtm, np.logical_not(dtIg))

            # 这里筛选出非忽略但未匹配上的预测框，其实就是 FP
            fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg))

            # 按行累加，计算TP 和 FP 的累积值。tp_sum[0]表示第0个IoU阈值下的TP累积值
            tp_sum = np.cumsum(tps, axis=1).astype(dtype=float)
            fp_sum = np.cumsum(fps, axis=1).astype(dtype=float)

            for t in range(T):
                tp = tp_sum[t]
                fp = fp_sum[t]
                nd = len(tp)  # 预测框数

                # 计算召回率
                rc = tp / npig
                # 计算精度：分母中防止除0
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


class Params:
    """
    参数类：用于 COCOevalSimplified 类的参数设置
    """

    def __init__(self, iouType="bbox"):
        self.imgIds = []  # 评估的图像 ID
        self.catIds = []  # 评估的类别 ID
        self.iouType = iouType  # IoU 类型
        self.iouThrs = np.array([0.5])  # IoU 阈值
        self.recThrs = np.linspace(
            0.0, 1.00, np.floor((1.00 - 0.0) / 0.01) + 1, endpoint=True
        )  # 召回率阈值
        self.maxDets = [1, 10, 100]  # 最大检测数
