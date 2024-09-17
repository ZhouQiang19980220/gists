"""
测试类BinaryConfusionMatrix
"""

# %%
from unittest import TestCase
from learn_confusion_matrix.my_confusion_matrix import BinaryConfusionMatrix, utils
import numpy as np
from sklearn.metrics import confusion_matrix


# %%
class TestBinaryConfusionMatrix(TestCase):
    """
    测试类BinaryConfusionMatrix
    """

    n = 100

    generate_data = utils.generate_binary_data

    def test_update(self):
        """
        测试更新函数
        """
        for i in range(100):
            cm = BinaryConfusionMatrix()
            y_trues, y_preds = self.generate_data(self.n)
            cm.update(y_trues, y_preds)

            # 使用sklearn的混淆矩阵验证
            y_preds = (y_preds >= cm.threshold).astype(int)
            sk_cm = confusion_matrix(y_trues, y_preds)
            self.assertTrue(np.allclose(cm.matrix, sk_cm))
