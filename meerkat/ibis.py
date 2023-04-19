from functools import lru_cache
import ibis

from meerkat.tools.lazy_loader import LazyLoader

import cytoolz as tz
import dill
import numpy as np
import pandas as pd
import pyarrow as pa
from pandas._libs import lib

import meerkat
from meerkat.block.manager import BlockManager
from meerkat.columns.abstract import Column
from meerkat.columns.scalar.abstract import ScalarColumn
from meerkat.columns.scalar.arrow import ArrowScalarColumn
from meerkat.errors import ConversionError
from meerkat.interactive.graph.marking import is_unmarked_context, unmarked
from meerkat.interactive.graph.reactivity import reactive
from meerkat.interactive.graph.store import Store
from meerkat.interactive.modification import DataFrameModification
from meerkat.interactive.node import NodeMixin
from meerkat.mixins.cloneable import CloneableMixin
from meerkat.mixins.deferable import DeferrableMixin
from meerkat.mixins.identifiable import IdentifiableMixin
from meerkat.mixins.indexing import IndexerMixin, MaterializationMixin
from meerkat.mixins.inspect_fn import FunctionInspectorMixin
from meerkat.mixins.reactifiable import ReactifiableMixin
from meerkat.provenance import ProvenanceMixin
from meerkat.row import Row
from meerkat.tools.lazy_loader import LazyLoader
from meerkat.tools.utils import convert_to_batch_fn, dump_yaml, load_yaml


from .dataframe import DataFrame


torch = LazyLoader("torch")


class IbisDataFrame(DataFrame):
    def __init__(
        self,
        expr: ibis.Expr,
    ):
        self.expr = expr

        super().__init__(data=expr, primary_key="id")

    @property
    def columns(self):
        return self.expr.columns

    def __len__(self) -> int:
        out = self.expr.count().execute()
        return out

    @property
    def nrows(self):
        """Number of rows in the DataFrame."""
        if self.ncols == 0:
            return 0
        return self.expr.count().execute()

    @property
    def ncols(self):
        """Number of rows in the DataFrame."""
        return len(self.data.columns)
    
    def _get_loc(self, keyidx, materialize: bool = False):
        if self.primary_key_name is None:
            raise ValueError(
                "Cannot use `loc` without a primary key. Set a primary key using "
                "`set_primary_key`."
            )

        if isinstance(
            keyidx, (np.ndarray, list, tuple, pd.Series, torch.Tensor, Column)
        ):
            return IbisDataFrame(
                self.expr[self.expr[self.primary_key_name].isin(keyidx)].execute(),
            )

        else:
            posidx = self.primary_key._keyidx_to_posidx(keyidx)
            row = self.data.apply("_get", index=posidx, materialize=materialize)
            return {k: row[k] for k in self.columns}

    def _get(self, posidx, materialize: bool = False):
        if isinstance(posidx, str):
            # str index => column selection (AbstractColumn)
            if posidx in self.columns:
    
                return IbisColumn(self.expr[[posidx]])
            raise KeyError(f"Column `{posidx}` does not exist.")

        elif isinstance(posidx, int):
            # int index => single row (dict)
            out = self.expr.limit(n=1, offset=posidx).to_pandas().iloc[0].to_dict()

            return out

        # cases where `index` returns a dataframe
        index_type = None
        if isinstance(posidx, slice):
            # slice index => multiple row selection (DataFrame)
            index_type = "row"
            if posidx.step is not None:
                raise ValueError("Slice step is not supported.")
            start = posidx.start if posidx.start is not None else 0
            stop = posidx.stop if posidx.stop is not None else len(self)

            return DataFrame.from_arrow(
                self.expr.limit(
                    n=max(0, stop - start), offset=start
                ).to_pyarrow()
            )

        elif (isinstance(posidx, tuple) or isinstance(posidx, list)) and len(posidx):
            # tuple or list index => multiple row selection (DataFrame)
            if isinstance(posidx[0], str):
                index_type = "column"
            else:
                index_type = "row"
                return DataFrame([
                    self._get(i, materialize=materialize)
                    for i in posidx
                ])

        elif isinstance(posidx, np.ndarray):
            if len(posidx.shape) != 1:
                raise ValueError(
                    "Index must have 1 axis, not {}".format(len(posidx.shape))
                )
            # numpy array index => multiple row selection (DataFrame)
            index_type = "row"

        elif torch.is_tensor(posidx):
            if len(posidx.shape) != 1:
                raise ValueError(
                    "Index must have 1 axis, not {}".format(len(posidx.shape))
                )
            # torch tensor index => multiple row selection (DataFrame)
            index_type = "row"

        elif isinstance(posidx, pd.Series):
            index_type = "row"

        elif isinstance(posidx, Column):
            # column index => multiple row selection (DataFrame)
            index_type = "row"

        else:
            raise TypeError("Invalid index type: {}".format(type(posidx)))

        if index_type == "column":
            if not set(posidx).issubset(self.columns):
                missing_cols = set(posidx) - set(self.columns)
                raise KeyError(f"DataFrame does not have columns {missing_cols}")
            df = IbisDataFrame(self.expr[posidx])
            return df
        elif index_type == "row":  # pragma: no cover
            raise NotImplementedError("TODO: implement row selection")
    
    def _clone(self, data: object = None, **kwargs):
        state = self._get_state(clone=True)
        state.update(kwargs)

        obj = self.__class__.__new__(self.__class__)
        obj._set_state(state)
        obj._set_data(data)

        if isinstance(self, ProvenanceMixin):
            # need to create a node for the object
            obj._init_node()

        if isinstance(self, IdentifiableMixin):
            obj._set_id()

        if isinstance(self, NodeMixin):
            obj._set_inode()
            # obj._set_children()

        from meerkat.dataframe import DataFrame

        if isinstance(obj, DataFrame):
            if obj.primary_key_name not in obj:
                # need to reset the primary key if we remove the column
                obj.set_primary_key(None, inplace=True)
        obj.expr = data
        return obj
    
    def _set_data(self, value: ibis.expr = None):
        self.expr = value
        self._data = value



class IbisColumn(Column):

    def __init__(self, expr: ibis.Expr):
        self.expr = expr
        super().__init__(data=expr)
    
    def _set_data(self, value: ibis.expr = None):
        self.expr = value
        self._data = value
    
    @lru_cache(maxsize=1)
    def full_length(self):
        return self.expr.count().execute()

    def _get_default_formatters(self):
        # can't implement this as a class level property because then it will treat
        # the formatter as a method
        from meerkat.interactive.formatter import (
            BooleanFormatterGroup,
            NumberFormatterGroup,
            TextFormatterGroup,
        )

        dtype = self.expr.schema().types[0]
        if isinstance(dtype, ibis.expr.datatypes.String):
            return TextFormatterGroup()
        elif isinstance(dtype, ibis.expr.datatypes.Boolean):
            return BooleanFormatterGroup()
        elif isinstance(dtype, ibis.expr.datatypes.Integer):
            return NumberFormatterGroup(dtype="int")
        elif isinstance(dtype, ibis.expr.datatypes.Floating):
            return NumberFormatterGroup(dtype="float")
        
        return super()._get_default_formatters() 
    
    def _get(self, index, materialize: bool = True):
        if self._is_batch_index(index):
            # only create a numpy array column
            if isinstance(index, slice):
                start = index.start if index.start is not None else 0
                stop = index.stop if index.stop is not None else len(self)
                data = self.expr.limit(
                    n=max(0, stop - start), offset=start
                ).to_pandas()[self.expr.columns[0]]
                return ScalarColumn(data=data)
            raise NotImplementedError("TODO: implement batch index")
        else:
            limited = self.expr.limit(n=1, offset=index).to_pandas()
            return limited[limited.columns[0]][0]
