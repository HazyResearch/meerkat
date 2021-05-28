"""Entity Class"""
import logging
from typing import Any, Dict, List, Union

import numpy as np
import torch

from mosaic import DataPanel, EmbeddingColumn
from mosaic.tools.identifier import Identifier

logger = logging.getLogger(__name__)


class Entity(DataPanel):
    def __init__(
        self,
        *args,
        identifier: Identifier = None,
        column_names: List[str] = None,
        embedding_columns: List[str] = None,
        index_column: str = None,
        **kwargs,
    ):
        super().__init__(
            *args,
            identifier=identifier,
            column_names=column_names,
            info=None,
            split=None,
            **kwargs,
        )
        if len(self.column_names) > 0:
            self.embedding_columns = embedding_columns if embedding_columns else []

            self._check_columns_unique(self.embedding_columns)
            self._check_columns_exist(self.embedding_columns)
            for c in self.embedding_columns:
                self._cast_to_embedding(c)

            self.index_column = index_column if index_column else None
            if not self.index_column:
                self._add_ent_index()
            self._check_columns_exist([self.index_column])
            # TODO (Laurel): This will break when changing rows in every way.
            self._index_to_rowid = {idx: i for i, idx in enumerate(self.index)}
        else:  # Initializing empty Entity DP
            self.embedding_columns = []
            self.index_column = None
            self._index_to_rowid = {}

    @classmethod
    def from_datapanel(
        cls,
        datapanel: DataPanel,
        embedding_columns: List[str] = None,
        index_column: str = None,
    ):
        return cls(
            datapanel._data,
            embedding_columns=embedding_columns,
            index_column=index_column,
        )

    @property
    def index(self):
        return self[self.index_column]

    def iget(self, idx: Any):
        if not isinstance(idx, type(next(iter(self._index_to_rowid.keys())))):
            raise ValueError(
                "Query must be the same type as the index column of the data"
            )
        assert idx in self._index_to_rowid, f"{idx} not in index set"
        row_idx = self._index_to_rowid[idx]
        if self.visible_rows is not None:
            # Map from original index into visible row index (inverse of remap index)
            try:
                row_idx = int(np.where(self.visible_rows == row_idx)[0][0])
            except IndexError:
                raise IndexError(f"{idx} not in data")
        return self[row_idx]

    def _get(self, index, materialize: bool = False):
        """When _get returns a DataPanel with same columns, cast back to Entity"""
        ret = super(Entity, self)._get(index, materialize)
        if isinstance(ret, DataPanel) and ret.column_names == self.column_names:
            return Entity.from_datapanel(
                ret,
                embedding_columns=self.embedding_columns,
                index_column=self.index_column,
            )
        return ret

    def _add_ent_index(self):
        """Add an integer index to the dataset. Not using index from DataPanel
        as ids need to be integers to serve as embedding row indices"""
        # TODO: Why do you have a string index...feels weird and expensive.
        # with row indexes due to filtering.
        self.add_column("_ent_index", [i for i in range(len(self))])
        self.index_column = "_ent_index"

    def _cast_to_embedding(self, name):
        """Cast column to Embedding column if not already"""
        if not isinstance(self._data[name], EmbeddingColumn):
            self._data[name] = EmbeddingColumn(self._data[name])

    def add_embedding_column(
        self,
        name: str,
        embedding: Union[np.ndarray, torch.tensor, EmbeddingColumn],
        index_to_rowid: Dict[Any, int] = None,
        overwrite: bool = False,
    ):
        """Adds embedding column to data. If index_to_rowid provided,
        maps DP index column to rowid of embedding."""
        if index_to_rowid is not None:
            permutation = [index_to_rowid[i] for i in self.index]
            embedding = embedding[permutation]
        assert len(embedding) == len(
            self
        ), "Length of embedding needs to be the same as data"
        self.add_column(name, embedding, overwrite)
        self._cast_to_embedding(name)
        # If name is duplicate, overwrite must be true
        # otherwise ``add_column`` would fail
        if name not in self.embedding_columns:
            self.embedding_columns.append(name)

    def most_similar(
        self,
        query: Any,
        k: int,
        query_embedding_column=None,
        search_embedding_column=None,
    ):
        """Returns most similar entities distinct from query"""
        if query_embedding_column is None:
            query_embedding_column = self.embedding_columns[0]
        self._check_columns_exist([query_embedding_column])
        if search_embedding_column is None:
            search_embedding_column = query_embedding_column
        else:
            self._check_columns_exist([search_embedding_column])
            assert (
                self[query_embedding_column].shape[-1]
                == self[search_embedding_column].shape[-1]
            ), "Length of search embedding needs to match query embedding"
        # Will noop if already exists
        self[search_embedding_column].build_faiss_index(overwrite=False)

        emb_query = self.iget(query)[query_embedding_column].numpy().reshape(1, -1)
        dist, sims = self[search_embedding_column].search(emb_query, k + 1)
        # May return the emb_query. If the embeddings are not unique,
        # we must selectively remove the query in the answer
        sims = sims[0][sims[0] != self._index_to_rowid[query]]
        return self[sims]
