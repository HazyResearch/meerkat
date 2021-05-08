from __future__ import annotations

import logging
from collections.abc import Sequence
from types import SimpleNamespace

import numpy as np
import torch

from mosaic.columns.tensor_column import TensorColumn
from mosaic.tools.lazy_loader import LazyLoader

faiss = LazyLoader("faiss")
umap = LazyLoader("umap")
umap_plot = LazyLoader("umap.plot")

logger = logging.getLogger(__name__)


class EmbeddingColumn(TensorColumn):
    faiss_index = None

    def __init__(
        self,
        data: Sequence = None,
        *args,
        **kwargs,
    ):
        super(EmbeddingColumn, self).__init__(data=data, *args, **kwargs)

        # Cast to float32
        self._data = self._data.type(torch.FloatTensor)

    def build_faiss_index(self, index):
        if self.ndim < 2:
            raise ValueError("Building an index requires `ndim` >= 2.")

        # Create the faiss index
        self.faiss_index = index(self.shape[1])

        # Add the data: must be np.ndarray
        self.faiss_index.add(self.numpy())

    def search(self, query, k):
        if isinstance(query, np.ndarray):
            query = query.astype("float32")
            return self.faiss_index.search(query, k)
        return NotImplemented("`query` be np.ndarray.")

    def umap(self):
        reducer = umap.UMAP()
        return SimpleNamespace(
            embeddings=reducer.fit_transform(self.numpy()),
            reducer=reducer,
        )

    def visualize_umap(self):
        p = umap_plot.interactive(
            self.umap().reducer,
            # labels=list(output['first-person']),
            # hover_data={'article': output['short:article']},
            point_size=4,
        )
        umap_plot.show(p)
