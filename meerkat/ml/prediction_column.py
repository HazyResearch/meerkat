from __future__ import annotations

from enum import Enum
from typing import Sequence, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from meerkat.columns.tensor_column import TensorColumn

Columnable = Union[Sequence, np.ndarray, pd.Series, torch.Tensor]


class _ClassifierOutputType(Enum):
    LOGIT = 1, ("logit", "logits")
    PROBABILITY = 2, ("probability", "probabilities", "prob", "probs")
    PREDICTION = 3, ("prediction", "predictions", "pred", "preds")

    def __new__(cls, key_code, names):
        """
        Args:
            key_code (int): Enum value.
            extensions (tuple[str]): Extensions supported by format.
        """
        obj = object.__new__(cls)
        obj._value_ = key_code
        obj.names = names
        return obj

    @classmethod
    def get_ctype(cls, ctype):
        if isinstance(ctype, cls):
            return ctype
        if not isinstance(ctype, (int, str)):
            raise TypeError(
                f"Cannot get {cls} from ctype of type {type(ctype)}. Must be int or str"
            )

        for _ctype in cls:
            if (isinstance(ctype, int) and ctype == _ctype.value) or (
                ctype in _ctype.names
            ):
                return _ctype

        raise ValueError(f"Invalid ctype - no {cls} corresponding to `ctype={ctype}`")

    def readable_name(self):
        return self.names[0]


class ClassificationOutputColumn(TensorColumn):
    def __init__(
        self,
        logits: Columnable = None,
        probs: Columnable = None,
        preds: Columnable = None,
        num_classes: int = None,
        multi_label: bool = False,
        one_hot: bool = None,
        threshold=0.5,
        *args,
        **kwargs,
    ):
        """Classification output handler initialized by one (and only one) of
        ``logits``, ``probs``, or ``preds``.

        Args:
            logits (array-like | tensor-like): Event probabilities (unnormalized).
            probs (array-like | tensor-like): Event probabilities.
            preds (array-like | tensor-like): Predictions. Can either be one-hot
                encoded (set ``one_hot=True``) or array-like of categorical labels.
            num_classes (int): The number of categories.
            multi_label (bool): If ``True``, examples can have multiple categories.
            one_hot (bool): If ``True``, ``preds`` (if specified) is considered to
                be a one-hot encoded matrix. If ``True`` and ``multi_label=False``,
                ``preds`` will be auto-converted into categorical labels.
            threshold (float): The threshold for binarizing multi-label probabilities
                to predictions. This is not relevant if ``multi_label=False``.
        """
        valid_data = [
            (x, ctype)
            for x, ctype in [
                (logits, _ClassifierOutputType.LOGIT),
                (probs, _ClassifierOutputType.PROBABILITY),
                (preds, _ClassifierOutputType.PREDICTION),
            ]
            if x is not None
        ]

        if len(valid_data) == 0:
            raise ValueError("Must specify one of `logits`, `probs`, or `preds`")

        if len(valid_data) > 1:
            raise ValueError(
                "Only one of `logits`, `probs`, or `preds` should be specified"
            )

        data, ctype = valid_data[0]

        if isinstance(data, TensorColumn):
            data = data.data  # unwrap tensor out of TensorColumn

        if ctype == _ClassifierOutputType.PREDICTION and not multi_label and one_hot:
            # Convert one-hot single-label into categorical labels
            data = torch.argmax(data)
            one_hot = False

        self._ctype = _ClassifierOutputType.get_ctype(ctype)
        self.num_classes = num_classes
        self.multi_label = multi_label
        self.one_hot = one_hot
        self.threshold = threshold

        super().__init__(data=data, *args, **kwargs)

    def _set_data(self, data):

        if self.num_classes is None:
            if self._ctype in (
                _ClassifierOutputType.LOGIT,
                _ClassifierOutputType.PROBABILITY,
            ):
                self.num_classes = data.shape[1]
            if self._ctype == _ClassifierOutputType.PREDICTION:
                if self.multi_label:
                    if self.one_hot is False or data.ndim < 2 or not _is_binary(data):
                        raise ValueError(
                            "Multi-label predictions must be one-hot encoded "
                            "and have at least 2 dimensions - (N, C, ...)"
                        )
                    self.one_hot = True
                self.num_classes = (
                    data.shape[1] if self.multi_label else torch.max(data) + 1
                )
        super(ClassificationOutputColumn, self)._set_data(data)

    def logits(self) -> ClassificationOutputColumn:
        if self._ctype in (
            _ClassifierOutputType.PROBABILITY,
            _ClassifierOutputType.PREDICTION,
        ):
            raise ValueError(
                f"Cannot convert from {self._ctype.readable_name()} to logits"
            )

        return self.view()

    def probabilities(self) -> ClassificationOutputColumn:
        if self._ctype == _ClassifierOutputType.LOGIT:
            probs = (
                torch.sigmoid(self.data)
                if self.multi_label
                else torch.softmax(self.data, dim=1)
            )
            return ClassificationOutputColumn(
                probs=probs,
                num_classes=self.num_classes,
                multi_label=self.multi_label,
                threshold=self.threshold,
            )

        if self._ctype == _ClassifierOutputType.PROBABILITY:
            return self.view()

        raise ValueError(f"Cannot convert from {self._ctype.readable_name()} to logits")

    probs = probabilities

    def predictions(self) -> ClassificationOutputColumn:
        """Compute predictions."""
        if self._ctype == _ClassifierOutputType.PREDICTION:
            return self.view()

        probs = self.probabilities().data
        preds = (
            (probs >= self.threshold).type(torch.uint8)
            if self.multi_label
            else torch.argmax(probs, dim=1)
        )
        return ClassificationOutputColumn(
            preds=preds,
            num_classes=self.num_classes,
            multi_label=self.multi_label,
            threshold=self.threshold,
            # multi-label predictions are always one hot encoded.
            one_hot=self.multi_label,
        )

    preds = predictions

    def bincount(self) -> TensorColumn:
        """Compute the count (cardinality) for each category.

        Categories which are not available will have a count of 0.

        If ``self.multi_label=True``, the bincount will include the
        total number of times the category is seen. If an example is marked
        as 2 categories, the bincount will increase the count for both categories.
        Note, this means the sum of the number of classes can be more than
        the number of examples ``N``.

        Returns:
            torch.Tensor: A 1D tensor of length ``self.num_classes``.
        """
        preds = self.predictions().data
        if (self.multi_label and preds.ndim > 2) or (
            not self.multi_label and preds.ndim > 1
        ):
            raise ValueError(
                "mode for multi-dimensional tensors is not currently supported"
            )
        if preds.ndim == 1:
            return TensorColumn(torch.bincount(preds, minlength=self.num_classes))
        else:
            out = preds.sum(dim=0)
            if len(out) < self.num_classes:
                pad = (0,) * (2 * (out.ndim - 1)) + (0, self.num_classes - len(out))
                out = F.pad(out, pad=pad)
            return TensorColumn(out)

    def mode(self):
        count = self.bincount()
        return torch.argmax(count.data)

    def entropy(self) -> TensorColumn:
        """Compute the entropy for each example.

        If ``self.multi_label`` is True, each category is treated as a binary
        classification problem. There will be an entropy calculation for each category
        as well. For example, if the probabilities are of shape ``(N, C)``, there will
        be ``NxC`` entropy values.

        In the multi-dimensional case, this returns the entropy for each element.
        For example, if the probabilities are of shape ``(N, C, A, B)``, there will
        be ``NxAxB`` entropy values.

        Returns:
            TensorColumn: Tensor of entropies
        """
        probs = self.probabilities().data
        if self.multi_label:
            probs = torch.stack([probs, 1 - probs], dim=-1)
        elif probs.ndim > 2:
            # make channels last
            probs = probs.transpose((0,) + tuple(range(2, probs.ndim)) + (1,))
        return TensorColumn(Categorical(probs=probs).entropy())

    @classmethod
    def _state_keys(cls) -> set:
        """List of attributes that describe the state of the object."""
        return super()._state_keys() | {
            "_ctype",
            "num_classes",
            "multi_label",
            "one_hot",
            "threshold",
        }


def _is_binary(tensor: torch.Tensor):
    return torch.all(
        (tensor == tensor.type(torch.uint8)) & (tensor >= 0) & (tensor <= 0)
    )
