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
        self.embedding_columns = embedding_columns if embedding_columns else []

        self._check_columns_unique(self.embedding_columns)
        self._check_columns_exist(self.embedding_columns)
        for c in self.embedding_columns:
            self._cast_to_embedding(c)

        self.index_column = index_column if index_column else None
        if not self.index_column:
            self._add_ent_index()
        self._check_columns_exist([self.index_column])
        # TODO (Laurel): This will break when filtering.
        self._index_to_rowid = {idx: i for i, idx in enumerate(self.index)}

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
        return self[self._index_to_rowid[idx]]

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
        #  My guess is it's to avoid confusion
        # TODO: How do we handle filering of entities
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
    ):
        """Adds embedding column to data. If index_to_rowid provided,
        maps DP index column to rowid of embedding."""
        if index_to_rowid is not None:
            permutation = [index_to_rowid[i] for i in self.index]
            embedding = embedding[permutation]
        assert len(embedding) == len(
            self
        ), "Length of embedding needs to be the same as data"
        self.add_column(name, embedding)
        self._cast_to_embedding(name)
        self.embedding_columns.append(name)

    def most_similar(self, query: Any, k: int, embedding_column=None):
        """Returns most similar entities to the given query index"""
        if embedding_column is None:
            embedding_column = self.embedding_columns[0]
        self._check_columns_exist([embedding_column])
        # Will noop if already exists
        self[embedding_column].build_faiss_index(overwrite=False)

        emb_query = self.iget(query)[embedding_column].numpy().reshape(1, -1)
        dist, sims = self[embedding_column].search(emb_query, k + 1)
        # Will return the emb_query. If the embeddings are not unique,
        # we must selectively remove the query in the answer
        sims = sims[0][sims[0] != self._index_to_rowid[query]]
        return self[sims]
