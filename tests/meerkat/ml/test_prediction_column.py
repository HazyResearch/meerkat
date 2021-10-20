import unittest

import torch
from torch.distributions.categorical import Categorical

from meerkat.ml.prediction_column import ClassificationOutputColumn

logits = torch.as_tensor(
    [
        [-100, -2, -50, 0, 1],
        [0, 3, -1, 5, 4],
        [100, 0, 0, -1, 5],
        [-100, -2, -50, 0, 1],
    ]
).type(torch.float32)

expected_preds = torch.as_tensor([4, 3, 0, 4])
expected_multilabel_preds_t50 = torch.as_tensor(
    [
        [0, 0, 0, 1, 1],
        [1, 1, 0, 1, 1],
        [1, 1, 1, 0, 1],
        [0, 0, 0, 1, 1],
    ]
)


class TestClassificationOutputColumn(unittest.TestCase):
    def test_init(self):
        logit_col = ClassificationOutputColumn(logits=logits)

        assert (logit_col.logits() == logits).all()
        assert (logit_col.probabilities() == torch.softmax(logits, dim=1)).all()
        assert logit_col.predictions() == expected_preds

        prob_col = ClassificationOutputColumn(probs=logit_col.probabilities())
        with self.assertRaises(ValueError):
            prob_col.logits()
        assert (prob_col.predictions() == logit_col.predictions()).all()

        pred_col = ClassificationOutputColumn(preds=expected_preds)
        with self.assertRaises(ValueError):
            pred_col.logits()
        with self.assertRaises(ValueError):
            pred_col.probabilities()

        assert logit_col.num_classes == prob_col.num_classes
        assert logit_col.num_classes == pred_col.num_classes

    def test_bincount(self):
        col = ClassificationOutputColumn(logits=logits)
        expected_bincount = torch.as_tensor([1, 0, 0, 1, 2])
        assert (col.bincount() == expected_bincount).all()

        col = ClassificationOutputColumn(logits=logits, num_classes=10)
        expected_bincount = torch.as_tensor([1, 0, 0, 1, 2, 0, 0, 0, 0, 0])
        assert (col.bincount() == expected_bincount).all()

        col = ClassificationOutputColumn(logits=logits, multi_label=True)
        expected_bincount = torch.as_tensor([2, 2, 1, 3, 4])
        assert (col.bincount() == expected_bincount).all()

        col = ClassificationOutputColumn(
            logits=logits, multi_label=True, num_classes=10
        )
        expected_bincount = torch.as_tensor([2, 2, 1, 3, 4, 0, 0, 0, 0, 0])
        assert (col.bincount() == expected_bincount).all()

    def test_mode(self):
        col = ClassificationOutputColumn(logits=logits)
        assert col.mode() == 4

    def test_entropy(self):
        logit_col = ClassificationOutputColumn(logits=logits)
        probs = logit_col.probabilities()
        expected_entropy = Categorical(probs=probs.data).entropy()
        assert (logit_col.entropy() == expected_entropy).all()

        logit_col = ClassificationOutputColumn(logits=logits, multi_label=True)
        probs = logit_col.probabilities()
        expected_entropy = Categorical(
            probs=torch.stack([probs.data, 1 - probs.data], dim=-1)
        ).entropy()
        assert (logit_col.entropy() == expected_entropy).all()
