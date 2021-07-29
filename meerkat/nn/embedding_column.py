from __future__ import annotations

import logging
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Union

import numpy as np
import pandas as pd
import torch

from meerkat.columns.tensor_column import TensorColumn
from meerkat.tools.lazy_loader import LazyLoader

faiss = LazyLoader("faiss")
umap = LazyLoader("umap")
umap_plot = LazyLoader("umap.plot")
sklearn_decom = LazyLoader("sklearn.decomposition")

Columnable = Union[Sequence, np.ndarray, pd.Series, torch.Tensor]

logger = logging.getLogger(__name__)


class EmbeddingColumn(TensorColumn):
    def __init__(
        self,
        data: Sequence = None,
        *args,
        **kwargs,
    ):
        if data is not None and isinstance(data, TensorColumn):
            data = data._data
        super(EmbeddingColumn, self).__init__(data=data, *args, **kwargs)

        # Cast to float32
        self._data = self._data.type(torch.FloatTensor)

        self.faiss_index = None

    def build_faiss_index(self, index=None, overwrite=False):
        if self.ndim < 2:
            raise ValueError("Building an index requires `ndim` >= 2.")

        if self.faiss_index is not None and not overwrite:
            return

        if index is None:
            index = faiss.IndexFlatL2

        # Create the faiss index
        self.faiss_index = index(self.shape[1])

        # Add the data: must be np.ndarray
        self.faiss_index.add(self.numpy())

    def search(self, query, k: int):
        assert (
            self.faiss_index is not None
        ), "You must call ``build_faiss_index`` first."
        if isinstance(query, np.ndarray):
            query = query.astype("float32")
            return self.faiss_index.search(query, k)
        return NotImplemented("`query` be np.ndarray.")

    def pca(self, n_components=2):
        pca = sklearn_decom.PCA(n_components)
        return SimpleNamespace(
            embeddings=pca.fit_transform(self.numpy()),
            reducer=pca,
        )

    def umap(self, n_neighbors=15, n_components=2):
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
        )
        return SimpleNamespace(
            embeddings=reducer.fit_transform(self.numpy()),
            reducer=reducer,
        )

    def visualize_umap(self, n_neighbors=15, n_components=2, point_size=4):
        p = umap_plot.interactive(
            self.umap(n_neighbors, n_components).reducer,
            point_size=point_size,
        )
        umap_plot.show(p)
