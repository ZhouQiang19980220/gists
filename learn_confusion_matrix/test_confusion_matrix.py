"""
测试类BinaryConfusionMatrix
"""

# %%
from unittest import TestCase
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from learn_confusion_matrix.my_confusion_matrix import BinaryConfusionMatrix, utils
from learn_confusion_matrix.my_confusion_matrix import MultiConfusionMatrix


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
        for _ in range(100):
            cm = BinaryConfusionMatrix()
            y_trues, y_preds = self.generate_data(self.n)
            cm.update(y_trues, y_preds)

            # 使用sklearn的混淆矩阵验证
            y_preds = (y_preds >= cm.threshold).astype(int)
            sk_cm = confusion_matrix(y_trues, y_preds)
            self.assertTrue(np.array_equal(cm.matrix, sk_cm))


# %%
class TestMultiConfusionMatrix(TestCase):
    """
    测试多分类混淆矩阵
    """

    n = 100
    num_classes = 3

    generate_data = utils.generate_multi_data

    def test_update(self):
        """
        测试混淆矩阵的计算是否正确
        """
        for _ in range(100):
            cm = MultiConfusionMatrix(num_classes=3)
            y_trues, y_preds = self.generate_data(self.n, self.num_classes)
            y_preds = np.argmax(y_preds, axis=1)
            cm.update(y_trues, y_preds)

            # 使用sklearn的混淆矩阵验证
            sk_cm = confusion_matrix(y_trues, y_preds)
            self.assertTrue(np.array_equal(cm.matrix, sk_cm))

            # 使用sklearn的classification_report验证
            sk_report_dict = classification_report(
                y_trues,
                y_preds,
                output_dict=True,
                labels=list(range(self.num_classes)),
            )
            report_dict = cm.get_report_dict()

            keys = [str(i) for i in range(self.num_classes)]
            keys.append("macro avg")
            for k in keys:
                if k.isdigit():
                    k = int(k)
                sub_keys = ["precision", "recall", "f1-score", "support"]
                for sub_k in sub_keys:
                    self.assertAlmostEqual(
                        sk_report_dict[str(k)][sub_k], report_dict[k][sub_k]  # type: ignore
                    )
            self.assertAlmostEqual(
                sk_report_dict["accuracy"], sk_report_dict["accuracy"]
            )
