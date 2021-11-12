import pytest
import torch

from meerkat.columns.tensor_column import TensorColumn
from meerkat.ml.segmentation_column import SegmentationOutputColumn

logits_multi = torch.tensor(
    [
        [
            [
                [1.1579, 0.7951, 1.2094],
                [-1.8820, 0.5593, 1.4548],
                [0.6161, 2.7659, -0.3512],
            ],
            [
                [-0.1776, 0.6782, -0.2487],
                [0.8022, 0.0941, -1.2005],
                [-1.1943, 0.3030, 1.0630],
            ],
            [
                [0.5057, 2.2815, -0.5638],
                [1.1235, 0.0428, 0.0373],
                [0.0714, 1.4041, -0.3600],
            ],
        ]
    ]
    * 2
)

expected_preds_multi = torch.tensor([[[0, 2, 0], [2, 0, 0], [0, 0, 1]]] * 2)

expected_mask_multi = torch.tensor([[[0, 1, 0], [1, 0, 0], [0, 0, 0]]] * 2)

logits_binary = torch.tensor(
    [
        [
            [
                [1.1579, 0.7951, 1.2094],
                [-1.8820, 0.5593, 1.4548],
                [0.6161, 2.7659, -0.3512],
            ],
            [
                [-0.1776, 0.6782, -0.2487],
                [0.8022, 0.0941, -1.2005],
                [-1.1943, 0.3030, 1.0630],
            ],
        ]
    ]
    * 2
)

expected_preds_binary = torch.tensor([[[0, 0, 0], [1, 0, 0], [0, 0, 1]]] * 2)

expected_mask_binary = expected_preds_binary


@pytest.mark.parametrize(
    "logits, expected_preds",
    [(logits_binary, expected_preds_binary), (logits_multi, expected_preds_multi)],
)
def test_init(logits, expected_preds):
    logit_col = SegmentationOutputColumn(logits=logits)

    assert (logit_col.logits() == logits).all()
    assert (logit_col.probabilities() == torch.softmax(logits, dim=1)).all()
    assert (logit_col.predictions() == expected_preds).all()


@pytest.mark.parametrize(
    "logits, expected_mask, class_index",
    [
        (logits_binary, expected_mask_binary, None),
        (logits_multi, expected_mask_multi, 2),
    ],
)
def test_binarymask(logits, expected_mask, class_index):
    logit_col = SegmentationOutputColumn(logits=logits)
    mask = logit_col.binarymask(class_index)

    assert isinstance(mask, TensorColumn)

    assert (mask == expected_mask).all()
    assert (
        SegmentationOutputColumn(probs=logit_col.probabilities().data).binarymask(
            class_index
        )
        == expected_mask
    ).all()
    assert (
        SegmentationOutputColumn(preds=logit_col.predictions().data).binarymask(
            class_index
        )
        == expected_mask
    ).all()
