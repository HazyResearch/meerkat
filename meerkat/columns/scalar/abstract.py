from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, List, Set, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow as pa
from pandas.core.accessor import CachedAccessor

from meerkat.block.abstract import BlockView
from meerkat.block.arrow_block import ArrowBlock
from meerkat.block.pandas_block import PandasBlock
from meerkat.columns.tensor.abstract import TensorColumn
from meerkat.tools.lazy_loader import LazyLoader

from ..abstract import Column

torch = LazyLoader("torch")

if TYPE_CHECKING:
    import torch

    from meerkat.dataframe import DataFrame

ScalarColumnTypes = Union[np.ndarray, "torch.TensorType", pd.Series, List]


class StringMethods:
    def __init__(self, data: Column):
        self.column = data

    def len(self, **kwargs) -> ScalarColumn:
        return self.column._dispatch_unary_function("len", _namespace="str", **kwargs)

    # predicate str methods ScalarColumn of bools
    def isalnum(self, **kwargs) -> ScalarColumn:
        return self.column._dispatch_unary_function(
            "isalnum", _namespace="str", **kwargs
        )

    def isalpha(self, **kwargs) -> ScalarColumn:
        return self.column._dispatch_unary_function(
            "isalpha", _namespace="str", **kwargs
        )

    def isdecimal(self, **kwargs) -> ScalarColumn:
        return self.column._dispatch_unary_function(
            "isdecimal", _namespace="str", **kwargs
        )

    def isdigit(self, **kwargs) -> ScalarColumn:
        return self.column._dispatch_unary_function(
            "isdigit", _namespace="str", **kwargs
        )

    def islower(self, **kwargs) -> ScalarColumn:
        return self.column._dispatch_unary_function(
            "islower", _namespace="str", **kwargs
        )

    def isupper(self, **kwargs) -> ScalarColumn:
        return self.column._dispatch_unary_function(
            "isupper", _namespace="str", **kwargs
        )

    def isnumeric(self, **kwargs) -> ScalarColumn:
        return self.column._dispatch_unary_function(
            "isnumeric", _namespace="str", **kwargs
        )

    def isspace(self, **kwargs) -> ScalarColumn:
        return self.column._dispatch_unary_function(
            "isspace", _namespace="str", **kwargs
        )

    def istitle(self, **kwargs) -> ScalarColumn:
        return self.column._dispatch_unary_function(
            "istitle", _namespace="str", **kwargs
        )

    def center(self, width: int, fillchar: str = " ", **kwargs) -> ScalarColumn:
        return self.column._dispatch_unary_function(
            "center", _namespace="str", width=width, fillchar=fillchar, **kwargs
        )

    # transform str methods
    def capitalize(self, **kwargs) -> ScalarColumn:
        return self.column._dispatch_unary_function(
            "capitalize", _namespace="str", **kwargs
        )

    def lower(self, **kwargs) -> ScalarColumn:
        return self.column._dispatch_unary_function("lower", _namespace="str", **kwargs)

    def upper(self, **kwargs) -> ScalarColumn:
        return self.column._dispatch_unary_function("upper", _namespace="str", **kwargs)

    def swapcase(self, **kwargs) -> ScalarColumn:
        return self.column._dispatch_unary_function(
            "swapcase", _namespace="str", **kwargs
        )

    def strip(self, to_strip: str = None, **kwargs) -> ScalarColumn:
        return self.column._dispatch_unary_function(
            "strip", _namespace="str", to_strip=to_strip, **kwargs
        )

    def lstrip(self, to_strip: str = None, **kwargs) -> ScalarColumn:
        return self.column._dispatch_unary_function(
            "lstrip", _namespace="str", to_strip=to_strip, **kwargs
        )

    def rstrip(self, to_strip: str = None, **kwargs) -> ScalarColumn:
        return self.column._dispatch_unary_function(
            "rstrip", _namespace="str", to_strip=to_strip, **kwargs
        )

    def replace(
        self, pat: str, repl: str, n: int = -1, regex: bool = False, **kwargs
    ) -> ScalarColumn:
        return self.column._dispatch_unary_function(
            "replace", _namespace="str", pat=pat, repl=repl, n=n, regex=regex, **kwargs
        )

    def title(self, **kwargs) -> ScalarColumn:
        return self.column._dispatch_unary_function("title", _namespace="str", **kwargs)

    def split(
        self, pat: str = None, n: int = -1, regex: bool = False, **kwargs
    ) -> "DataFrame":
        raise NotImplementedError()

    def rsplit(
        self, pat: str = None, n: int = -1, regex: bool = False, **kwargs
    ) -> "DataFrame":
        raise NotImplementedError()

    def startswith(self, pat: str, **kwargs) -> ScalarColumn:
        return self.column._dispatch_unary_function(
            "startswith", _namespace="str", pat=pat, **kwargs
        )

    def contains(self, pat: str, case: bool = True, regex: bool = True) -> ScalarColumn:
        return self.column._dispatch_unary_function(
            "contains", _namespace="str", pat=pat, case=case, regex=regex
        )

    def extract(self, pat: str, **kwargs) -> "DataFrame":
        return self.column._dispatch_unary_function(
            "extract", _namespace="str", pat=pat, **kwargs
        )


class ScalarColumn(Column):
    str = CachedAccessor("str", StringMethods)

    def __new__(cls, data: ScalarColumnTypes = None, backend: str = None):
        from .arrow import ArrowScalarColumn
        from .pandas import PandasScalarColumn

        if (cls is not ScalarColumn) or (data is None):
            return super().__new__(cls)

        backends = {"arrow": ArrowScalarColumn, "pandas": PandasScalarColumn}
        if backend is not None:
            if backend not in backends:
                raise ValueError(
                    f"Cannot create `ScalarColumn` with backend '{backend}'. "
                    f"Expected one of {list(backends.keys())}"
                )
            else:
                return super().__new__(backends[backend])

        if isinstance(data, BlockView):
            if isinstance(data.block, PandasBlock):
                return super().__new__(PandasScalarColumn)
            elif isinstance(data.block, ArrowBlock):
                return super().__new__(ArrowScalarColumn)
            else:
                raise ValueError(
                    f"Cannot create `ScalarColumn` from object of type {type(data)}."
                )

        if isinstance(data, (np.ndarray, torch.TensorType, pd.Series, List, Tuple)):
            return super().__new__(PandasScalarColumn)
        elif isinstance(data, pa.Array):
            return super().__new__(ArrowScalarColumn)
        elif isinstance(data, TensorColumn) and len(data.shape) == 1:
            return super().__new__(PandasScalarColumn)
        elif isinstance(data, ScalarColumn):
            return data
        else:
            raise ValueError(
                f"Cannot create `ScalarColumn` from object of type {type(data)}."
            )

    def _dispatch_unary_function(
        self, compute_fn: str, _namespace: str = None, **kwargs
    ):
        raise NotImplementedError()

    @property
    def dtype(self, **kwargs) -> Union[pa.DataType, np.dtype]:
        raise NotImplementedError()

    # aggregation functions
    @abstractmethod
    def _dispatch_aggregation_function(self, compute_fn: str, **kwargs):
        raise NotImplementedError()

    def mean(self, skipna: bool = True, **kwargs) -> float:
        return self._dispatch_aggregation_function("mean", skipna=skipna, **kwargs)

    def median(self, skipna: bool = True, **kwargs) -> Any:
        return self._dispatch_aggregation_function("median", skipna=skipna, **kwargs)

    def mode(self, **kwargs) -> ScalarColumn:
        return self._dispatch_aggregation_function("mode", **kwargs)

    def var(self, ddof: int = 1, **kwargs) -> ScalarColumn:
        return self._dispatch_aggregation_function("var", ddof=ddof, **kwargs)

    def std(self, ddof: int = 1, **kwargs) -> ScalarColumn:
        return self._dispatch_aggregation_function("std", ddof=ddof, **kwargs)

    def min(self, skipna: bool = True, **kwargs) -> ScalarColumn:
        return self._dispatch_aggregation_function("min", skipna=skipna, **kwargs)

    def max(self, skipna: bool = True, **kwargs) -> ScalarColumn:
        return self._dispatch_aggregation_function("max", skipna=skipna, **kwargs)

    def sum(self, skipna: bool = True, **kwargs) -> Any:
        return self._dispatch_aggregation_function("sum", skipna=skipna, **kwargs)

    def product(self, skipna: bool = True, **kwargs) -> Any:
        return self._dispatch_aggregation_function("product", skipna=skipna, **kwargs)

    def any(self, skipna: bool = True, **kwargs) -> Any:
        return self._dispatch_aggregation_function("any", skipna=skipna, **kwargs)

    def all(self, skipna: bool = True, **kwargs) -> Any:
        return self._dispatch_aggregation_function("all", skipna=skipna, **kwargs)

    def unique(self, **kwargs) -> ScalarColumn:
        return self._dispatch_unary_function("unique", **kwargs)

    # arithmetic functions
    def _dispatch_arithmetic_function(
        self, other, compute_fn: str, right: bool, **kwargs
    ):
        raise NotImplementedError()

    def __add__(self, other: ScalarColumn):
        return self._dispatch_arithmetic_function(other, "add", right=False)

    def __radd__(self, other: ScalarColumn):
        return self._dispatch_arithmetic_function(other, "add", right=True)

    def __sub__(self, other: ScalarColumn):
        return self._dispatch_arithmetic_function(other, "sub", right=False)

    def __rsub__(self, other: ScalarColumn):
        return self._dispatch_arithmetic_function(other, "sub", right=True)

    def __mul__(self, other: ScalarColumn):
        return self._dispatch_arithmetic_function(other, "mul", right=False)

    def __rmul__(self, other: ScalarColumn):
        return self._dispatch_arithmetic_function(other, "mul", right=True)

    def __truediv__(self, other: ScalarColumn):
        return self._dispatch_arithmetic_function(other, "truediv", right=False)

    def __rtruediv__(self, other: ScalarColumn):
        return self._dispatch_arithmetic_function(other, "truediv", right=True)

    def __floordiv__(self, other: ScalarColumn):
        return self._dispatch_arithmetic_function(other, "floordiv", right=False)

    def __rfloordiv__(self, other: ScalarColumn):
        return self._dispatch_arithmetic_function(other, "floordiv", right=True)

    def __mod__(self, other: ScalarColumn):
        return self._dispatch_arithmetic_function(other, "mod", right=False)

    def __rmod__(self, other: ScalarColumn):
        return self._dispatch_arithmetic_function(other, "mod", right=True)

    def __pow__(self, other: ScalarColumn):
        return self._dispatch_arithmetic_function(other, "pow", right=False)

    def __rpow__(self, other: ScalarColumn):
        return self._dispatch_arithmetic_function(other, "pow", right=True)

    # comparison functions
    def _dispatch_comparison_function(self, other, compute_fn: str, **kwargs):
        raise NotImplementedError()

    def __eq__(self, other: ScalarColumn):
        return self._dispatch_comparison_function(other, "eq")

    def __ne__(self, other: ScalarColumn):
        return self._dispatch_comparison_function(other, "ne")

    def __lt__(self, other: ScalarColumn):
        return self._dispatch_comparison_function(other, "lt")

    def __le__(self, other: ScalarColumn):
        return self._dispatch_comparison_function(other, "le")

    def __gt__(self, other: ScalarColumn):
        return self._dispatch_comparison_function(other, "gt")

    def __ge__(self, other: ScalarColumn):
        return self._dispatch_comparison_function(other, "ge")

    # logical functions
    def _dispatch_logical_function(self, other, compute_fn: str, **kwargs):
        raise NotImplementedError()

    def __and__(self, other: ScalarColumn):
        return self._dispatch_logical_function(other, "and")

    def __or__(self, other: ScalarColumn):
        return self._dispatch_logical_function(other, "or")

    def __invert__(self):
        return self._dispatch_logical_function(None, "invert")

    def __xor__(self, other: ScalarColumn):
        return self._dispatch_logical_function(other, "xor")

    # containment functions
    def isin(self, values: Union[List, Set], **kwargs) -> ScalarColumn:
        raise NotImplementedError()

    def isna(self, **kwargs) -> ScalarColumn:
        return self._dispatch_unary_function("isna", **kwargs)

    def isnull(self, **kwargs) -> ScalarColumn:
        return self._dispatch_unary_function("isnull", **kwargs)
