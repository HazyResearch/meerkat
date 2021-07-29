"""EntityDataPanel Class."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence, Tuple, Union

import datasets
import numpy as np
import torch

from meerkat import DataPanel, ListColumn, NumpyArrayColumn, TensorColumn
from meerkat.nn import EmbeddingColumn
from meerkat.tools.identifier import Identifier

logger = logging.getLogger(__name__)


class EntityDataPanel(DataPanel):
    def __init__(
        self,
        data: Union[dict, list, datasets.Dataset] = None,
        identifier: Identifier = None,
        column_names: List[str] = None,
        embedding_columns: List[str] = None,
        index_column: str = None,
        **kwargs,
    ):
        """
        EntityDataPanel: A data panel with "atomic" unit semantics.
        This means each row represents a unique "atom" or thing.
        Each row has an index column that is used in merges.
        Further, each row can get one or more embedding columns
        'associated with it. This allows for Embedding
        operations such as nearest neighbor search.

        Args:
            identifier: identifier
            column_names: all column names
            embedding_columns: embedding columns in all columns
            index_column: index column
        """
        super().__init__(
            data=data,
            identifier=identifier,
            column_names=column_names,
            info=None,
            split=None,
            **kwargs,
        )
        if len(self.column_names) > 0:
            self._embedding_columns = embedding_columns if embedding_columns else []

            self._check_columns_unique(self._embedding_columns)
            self._check_columns_exist(self._embedding_columns)
            for c in self._embedding_columns:
                self._cast_to_embedding(c)

            self._index_column = index_column if index_column else None
            if not self._index_column:
                self._add_ent_index()
            else:
                self._check_index_unique()
            self._check_columns_exist([self._index_column])
            self._index_to_rowid = {idx: i for i, idx in enumerate(self.index)}
        else:  # Initializing empty EntityDataPanel DP - needed when loading
            self._embedding_columns = []
            self._index_column = None
            self._index_to_rowid = {}

    @classmethod
    def from_datapanel(
        cls,
        datapanel: DataPanel,
        embedding_columns: List[str] = None,
        index_column: str = None,
    ):
        """Returns EntityDataPanel DP from standard DP."""
        return cls(
            data=datapanel.data,
            embedding_columns=embedding_columns,
            index_column=index_column,
        )

    def to_datapanel(self, klass: type = None):
        """Casts to a DataPanel.

        Args:
            klass (type): If specified, casts to this class.
                This class must be a subclass of :ref:`meerkat.datapanel.DataPanel`.
                If not specified, defaults to :ref:`meerkat.datapanel.DataPanel`.

        Returns:
            DataPanel: The datapanel.
        """
        if klass is None:
            klass = DataPanel
        elif not issubclass(klass, DataPanel):
            raise ValueError("`klass` must be a subclass of DataPanel")
        return klass.from_batch({k: self[k] for k in self.visible_columns})

    @property
    def index(self):
        """Returns index column."""
        return self[self.index_column]

    @property
    def embedding_columns(self):
        """Returns _visible_ embedding columns."""
        return [e for e in self._embedding_columns if e in self.visible_columns]

    @property
    def index_column(self):
        """Returns index column."""
        return self._index_column

    def _check_index_unique(self):
        assert len(self.index) == len(
            set(self.index)
        ), "Index must be unique and hashable"

    def icontain(self, idx: Any):
        """Checks if idx in the index column or not."""
        return idx in self._index_to_rowid

    def iget(self, idx: Any):
        """Gets the row given the entity index."""
        idx_col_type = type(next(iter(self._index_to_rowid.keys())))
        if not isinstance(idx, idx_col_type):
            raise ValueError(
                f"Query ({type(idx)}) must be the same type as the "
                f"index column ({idx_col_type}) of the data"
            )
        assert idx in self._index_to_rowid, f"{idx} not in index set"
        row_idx = self._index_to_rowid[idx]
        if self.index.visible_rows is not None:
            # Map from original index into visible row index (inverse of remap index)
            try:
                row_idx = int(np.where(self.index.visible_rows == row_idx)[0][0])
            except IndexError:
                raise IndexError(f"{idx} not in data")
        return self[row_idx]

    def _add_ent_index(self):
        """Add an integer index to the dataset.

        Not using index from DataPanel as ids need to be integers to
        serve as embedding row indices
        """
        # TODO: Why do you have a string index...feels weird and expensive.
        # with row indexes due to filtering.
        self.add_column("_ent_index", [i for i in range(len(self))])
        self._index_column = "_ent_index"

    def _cast_to_embedding(self, name):
        """Cast column to Embedding column if not already."""
        if not isinstance(self._data[name], EmbeddingColumn):
            self._data[name] = EmbeddingColumn(self._data[name])

    def add_embedding_column(
        self,
        name: str,
        embedding: Union[np.ndarray, torch.tensor, EmbeddingColumn],
        index_to_rowid: Dict[Any, int] = None,
        overwrite: bool = False,
    ):
        """Adds embedding column to data.

        If index_to_rowid provided, maps DP index column to rowid of
        embedding.
        """
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
        if name not in self._embedding_columns:
            self._embedding_columns.append(name)

    def remove_column(self, column: str) -> None:
        """Remove column.

        Assert index column is not removed
        """
        assert column != self.index_column, "Can't remove an index column"
        super(EntityDataPanel, self).remove_column(column)
        if column in self._embedding_columns:
            self._embedding_columns.remove(column)

    def append(
        self,
        dp: EntityDataPanel,
        axis: Union[str, int] = "rows",
        suffixes: Tuple[str] = None,
        overwrite: bool = False,
    ) -> EntityDataPanel:
        """Append an EntityDataPanel."""
        if axis == 0 or axis == "rows":
            # append new rows
            return super(EntityDataPanel, self).append(dp, axis, suffixes, overwrite)
        elif axis == 1 or axis == "columns":
            # append new columns; must make sure the entities are in the same order
            # data property takes `visible_rows` into account
            if self.index.data != dp.index.data:
                raise ValueError(
                    "Can only append along axis 1 (columns) if the entity indexes match"
                )
            new_embedding_cols = self._get_merged_embedding_columns(
                dp, overwrite, suffixes
            )
            # Save the new index column for saving EntityDataPanel
            new_index_column = self.index_column
            if self.index_column in dp.column_names and not overwrite:
                new_index_column += suffixes[0]
            ret = super(EntityDataPanel, self).append(dp, axis, suffixes, overwrite)
            ret._embedding_columns = new_embedding_cols
            ret._index_column = new_index_column
            return ret
        else:
            raise ValueError("DataPanel `axis` must be either 0 or 1.")

    def merge(
        self,
        right: "EntityDataPanel",
        how: str = "inner",
        left_on: str = None,
        sort: bool = False,
        suffixes: Sequence[str] = ("_x", "_y"),
        validate=None,
        keep_indexes: bool = False,
    ):
        """Perform merge of two EntityDPs.

        By default we merge both on their index columns. We do allow for
        changing the join column of the left side
        """
        left_on = left_on if left_on is not None else self.index_column
        new_embedding_cols = self._get_merged_embedding_columns(
            right, overwrite=False, suffixes=suffixes
        )
        # If the left index column is in the right set of columns,
        # and the left index column won't stay the same because it's the
        # the joining column, then the index column will change to
        # have a suffix
        new_index_column = self.index_column
        if self.index_column in right.column_names:
            # Column will stay the same if it's the joining
            # column of both left and right
            if not (
                self.index_column == left_on and self.index_column == right.index_column
            ):
                assert suffixes is not None and len(suffixes) == 2, (
                    "The index column is the same as rights columns. "
                    "Must provide suffixes"
                )
                new_index_column += suffixes[0]
        # Merge calls append and EntityDataPanel append ensures
        # indexes are aligned. When merging, we adopt the index of
        # the left, making it okay that indexes are unaligned. To avoid
        # this, we cast to DataPanel.
        ret = self.to_datapanel().merge(
            right.to_datapanel(),
            how=how,
            left_on=left_on,
            right_on=right.index_column,
            sort=sort,
            suffixes=suffixes,
            validate=validate,
            keep_indexes=False,
        )
        return EntityDataPanel.from_datapanel(
            datapanel=ret,
            embedding_columns=new_embedding_cols,
            index_column=new_index_column,
        )

    def _get_merged_embedding_columns(self, dp, overwrite, suffixes):
        """In order to create EntityDataPanel post merge/append, we need to
        have the embedding columns.

        This calculates the new EntityDataPanel embedding columns in the
        case of suffix changes.
        """
        new_embedding_cols = self.embedding_columns + dp.embedding_columns
        # Capture the new embedding columns before merging
        if len(set(self.embedding_columns).intersection(dp.embedding_columns)) > 0:
            if overwrite:
                new_embedding_cols = self.embedding_columns + [
                    c for c in dp.embedding_columns if c not in self.embedding_columns
                ]
            else:
                assert (
                    suffixes is not None and len(suffixes) == 2
                ), "Suffixes must be tuple of len 2 when columns share names"
                new_embedding_cols = (
                    [c for c in self.embedding_columns if c not in dp.embedding_columns]
                    + [
                        c
                        for c in dp.embedding_columns
                        if c not in self.embedding_columns
                    ]
                    + [
                        c + suffixes[0]
                        for c in self.embedding_columns
                        if c in dp.embedding_columns
                    ]
                    + [
                        c + suffixes[1]
                        for c in dp.embedding_columns
                        if c in self.embedding_columns
                    ]
                )
        return new_embedding_cols

    def convert_entities_to_ids(
        self, column: Union[ListColumn, TensorColumn, NumpyArrayColumn]
    ):
        """Maps column of entity idx to their row ids for the embeddings.

        Used in data prep before training.
        """

        def recursive_map(seq):
            if isinstance(seq, (np.ndarray, torch.Tensor, list)):
                return [recursive_map(item) for item in seq]
            else:
                # TODO: handle UNK entity ids
                return self._index_to_rowid[seq]

        assert isinstance(column, (ListColumn, TensorColumn, NumpyArrayColumn)), (
            "We only support DataPanel list column types "
            "(ListColumn, TensorColumn, NumpyArrayColumn)"
        )
        return column.map(lambda x: recursive_map(x))

    def most_similar(
        self,
        query: Any,
        k: int,
        query_embedding_column=None,
        search_embedding_column=None,
    ):
        """Returns most similar entities distinct from query."""
        assert k < len(self), "k must be less than the total number of entities"
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
        # May or may not return the emb_query. If the embeddings are not unique,
        # we must selectively remove the query in the answer
        sims = sims[0][sims[0] != self._index_to_rowid[query]]
        sims = sims[:k]
        return self[sims]

    @classmethod
    def _state_keys(cls) -> set:
        """List of attributes that describe the state of the object."""
        return {
            "_visible_columns",
            "_identifier",
            "_embedding_columns",
            "_index_column",
        }

    def _clone_kwargs(self) -> Dict[str, Any]:
        default_kwargs = super()._clone_kwargs()
        default_kwargs.update({"index_column": self.index_column})
        return default_kwargs
