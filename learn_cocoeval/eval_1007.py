# %%
import os
from collections import defaultdict
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# %%
class Pramas:

    def __init__(self, iouType="bbox"):
        self.img_ids = None
        self.cat_ids = None
        self.iouType = iouType
        self.iouThrs = np.array([0.5])
        self.recThrs = np.linspace(0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01)) + 1)
        self.maxDets = [1, 10, 100]

class COCOevalSimplified:

    def __init__(self, cocoGT, cocoDT, iouType="bbox", iouThrs=None):
        self.cocoGT = cocoGT
        self.cocoDT = cocoDT
        self.params = Pramas(iouType)

        if iouThrs is None:
            iouThrs = np.array([0.5])
        self.params.iouThrs = iouThrs

        self.evalImgs = []
        self.eval = {}
        self._gts = defaultdict(list)
        self._dts = defaultdict(list)
        self.ious = {}

        if cocoGT is not None:
            self.params.img_ids = sorted(cocoGT.getImgIds())
            self.params.cat_ids = sorted(cocoGT.getCatIds())

    
    def _prepare(self):
        """
        数据预处理 将ground truth和detection结果按照img_id和cat_id进行分组
        """
        p = self.params

        gts = self.cocoGT.loadAnns(
            self.cocoGT.getAnnIds(imgIds=p.imgIds, catIds=p.catIds)
        )
        dts = self.cocoDT.loadAnns(
            self.cocoDT.getAnnIds(imgIds=p.imgIds, catIds=P.catIds)
        )

        for gt in gts:
            

# %%
ann_file = "annotations/instances_val2014.json"
rst_file = "results/instances_val2014_fakebbox100_results.json"
coco = COCO(ann_file)
cocoRes = coco.loadRes(rst_file)
img_ids = coco.getImgIds()
img_ids = sorted(img_ids)
img_ids = img_ids[:100]

cocoEval = COCOeval(coco, cocoRes, "bbox")
cocoEval.params.imgIds = img_ids
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
#%%

myCocoEval = COCOevalSimplified(coco, cocoRes, "bbox")

#%%