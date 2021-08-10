from __future__ import annotations

import itertools
from typing import List, Sequence, Union

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from meerkat.columns.tensor_column import TensorColumn
from meerkat.datapanel import DataPanel
from meerkat.nn.prediction_column import (
    ClassificationOutputColumn,
    _ClassifierOutputType,
)
from meerkat.tools.lazy_loader import LazyLoader

cv2 = LazyLoader("cv2")

Columnable = Union[Sequence, np.ndarray, pd.Series, torch.Tensor]


# TODO(Priya): Make probs, preds method return SegmentationOutputColumn type
class SegmentationOutputColumn(ClassificationOutputColumn):
    def __init__(
        self,
        logits: Columnable = None,
        probs: Columnable = None,
        preds: Columnable = None,
        num_classes: int = None,
        *args,
        **kwargs,
    ):
        super(SegmentationOutputColumn, self).__init__(
            logits=logits,
            probs=probs,
            preds=preds,
            num_classes=num_classes,
            *args,
            **kwargs,
        )

    def binarymask(
        self, class_index: int
    ) -> TensorColumn:  # TODO(Priya): Check column type

        if self.num_classes > 2 and class_index is None:
            raise ValueError("Provide class_index in case of multi-class segmentation")

        if self.num_classes == 2:
            # Binary mask is same as predictions
            mask = TensorColumn(
                self.data
                if self._ctype == _ClassifierOutputType.PREDICTION
                else self.predictions().data
            )
        else:
            # Convert to predictions if required
            preds = (
                self.data
                if self._ctype == _ClassifierOutputType.PREDICTION
                else self.predictions().data
            )
            mask = TensorColumn(torch.where(preds == class_index, 1, 0))

        return mask

    @staticmethod
    def rle2mask(
        dataset: DataPanel,
        input_columns: List[str],  # TODO(Priya): Support multiple RLE columns?
        orig_dim,
        resize_dim=None,
        to_nan: bool = False,
        batch_size: int = 32,
    ) -> TensorColumn:

        masks = []

        for batch in tqdm(
            dataset[input_columns].batch(batch_size),
            total=(len(dataset) // batch_size + int(len(dataset) % batch_size != 0)),
        ):

            batch_masks = _convert_rle2mask(
                batch, input_columns, orig_dim, resize_dim, to_nan
            )

            masks = list(itertools.chain(masks, batch_masks))

        masks_col = TensorColumn(masks)
        dataset.add_column(f"Binary Mask (from {input_columns[0]})", masks_col)

        return masks_col


def _convert_rle2mask(
    batch: DataPanel,
    input_columns: List[str],
    orig_dim,
    resize_dim=None,
    to_nan: bool = False,
):

    """Convert run length encoding (RLE) to 2D binary mask.

    Args:
    batch (DataPanel): DataPanel.
    input_columns: List of columns containing Run Length Encodings
    orig_dim (Tuple[int]): Shape of the image.
    resize_dim (Tuple[int]): Shape to resize to.
      Resizing is done with cubic interporlation.
    to_nan (bool, optional): Convert 0s to np.nan.

    Returns:
    List[np.ndarray]: List of np.ndarray containing binary mask
    """

    # TODO(Priya): Support for multiple input_columns?
    rle_data = batch[input_columns[0]].data
    height, width = orig_dim
    masks = []

    for rle in rle_data:
        mask = np.zeros(width * height)
        if rle != "-1":
            array = np.asarray([int(x) for x in rle.split()])
            starts = array[0::2]
            lengths = array[1::2]
            current_position = 0

            for index, start in enumerate(starts):
                current_position += start
                mask[current_position : current_position + lengths[index]] = 1
                current_position += lengths[index]
            mask = mask.reshape(width, height)

        if resize_dim is not None:
            mask = cv2.resize(mask, resize_dim, interpolation=cv2.INTER_CUBIC)
        if to_nan:
            mask[mask == 0] = np.nan

        masks.append(mask)

    return masks
