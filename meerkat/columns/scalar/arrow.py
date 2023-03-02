from __future__ import annotations

import os
import re
import warnings
from typing import TYPE_CHECKING, Any, List, Sequence, Set, Union

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from pandas.core.accessor import CachedAccessor

from meerkat.block.abstract import BlockView
from meerkat.block.arrow_block import ArrowBlock
from meerkat.errors import ImmutableError
from meerkat.tools.lazy_loader import LazyLoader

from ..abstract import Column
from .abstract import ScalarColumn, StringMethods

if TYPE_CHECKING:
    from meerkat import DataFrame
    from meerkat.interactive.formatter.base import BaseFormatter


torch = LazyLoader("torch")


class ArrowStringMethods(StringMethods):
    def center(self, width: int, fillchar: str = " ", **kwargs) -> ScalarColumn:
        return self.column._dispatch_unary_function(
            "utf8_center", width=width, padding=fillchar, **kwargs
        )

    def extract(self, pat: str, **kwargs) -> "DataFrame":
        from meerkat import DataFrame

        # Pandas raises a value error if the pattern does not include a group
        # but pyarrow does not. We check for this case and raise a value error.
        if not re.search(r"\(\?P<\w+>", pat):
            raise ValueError(
                "Pattern does not contain capture group. Use '(?P<name>...)' instead"
            )

        struct_array = pc.extract_regex(self.column.data, pattern=pat, **kwargs)

        result = {}
        for field_index in range(struct_array.type.num_fields):
            field = struct_array.type.field(field_index)
            result[field.name] = self.column._clone(
                pc.struct_field(struct_array, field.name)
            )

        return DataFrame(result)

    def _split(
        self, pat=None, n=-1, reverse: bool = False, regex: bool = False, **kwargs
    ) -> "DataFrame":
        from meerkat import DataFrame

        fn = pc.split_pattern_regex if regex else pc.split_pattern
        list_array = fn(
            self.column.data,
            pattern=pat,
            max_splits=n if n != -1 else None,
            reverse=reverse,
            **kwargs,
        )

        # need to find the max length of the list array
        if n == -1:
            n = pc.max(pc.list_value_length(list_array)).as_py() - 1

        return DataFrame(
            {
                str(i): self.column._clone(
                    data=pc.list_flatten(
                        pc.list_slice(
                            list_array, start=i, stop=i + 1, return_fixed_size_list=True
                        )
                    )
                )
                for i in range(n + 1)
            }
        )

    def split(
        self, pat: str = None, n: int = -1, regex: bool = False, **kwargs
    ) -> "DataFrame":
        return self._split(pat=pat, n=n, reverse=False, regex=regex, **kwargs)

    def rsplit(
        self, pat: str = None, n: int = -1, regex: bool = False, **kwargs
    ) -> "DataFrame":
        return self._split(pat=pat, n=n, reverse=True, regex=regex, **kwargs)

    def startswith(self, pat: str, **kwargs) -> ScalarColumn:
        return self.column._dispatch_unary_function(
            "starts_with", pattern=pat, **kwargs
        )

    def strip(self, to_strip: str = None, **kwargs) -> ScalarColumn:
        if to_strip is None:
            return self.column._dispatch_unary_function(
                "utf8_trim_whitespace", **kwargs
            )
        else:
            return self.column._dispatch_unary_function(
                "utf8_strip", characters=to_strip, **kwargs
            )

    def lstrip(self, to_strip: str = None, **kwargs) -> ScalarColumn:
        if to_strip is None:
            return self.column._dispatch_unary_function(
                "utf8_ltrim_whitespace", **kwargs
            )
        else:
            return self.column._dispatch_unary_function(
                "utf8_lstrip", characters=to_strip, **kwargs
            )

    def rstrip(self, to_strip: str = None, **kwargs) -> ScalarColumn:
        if to_strip is None:
            return self.column._dispatch_unary_function(
                "utf8_rtrim_whitespace", **kwargs
            )
        else:
            return self.column._dispatch_unary_function(
                "utf8_rstrip", characters=to_strip, **kwargs
            )

    def replace(
        self, pat: str, repl: str, n: int = -1, regex: bool = False, **kwargs
    ) -> ScalarColumn:

        fn = pc.replace_substring_regex if regex else pc.replace_substring
        return self.column._clone(
            fn(
                self.column.data,
                pattern=pat,
                replacement=repl,
                max_replacements=n if n != -1 else None,
                **kwargs,
            )
        )

    def contains(self, pat: str, case: bool = True, regex: bool = True) -> ScalarColumn:
        fn = pc.match_substring_regex if regex else pc.match_substring
        return self.column._clone(
            fn(
                self.column.data,
                pattern=pat,
                ignore_case=not case,
            )
        )


class ArrowScalarColumn(ScalarColumn):
    block_class: type = ArrowBlock

    str = CachedAccessor("str", ArrowStringMethods)

    def __init__(
        self,
        data: Sequence,
        *args,
        **kwargs,
    ):
        if isinstance(data, BlockView):
            if not isinstance(data.block, ArrowBlock):
                raise ValueError(
                    "ArrowArrayColumn can only be initialized with ArrowBlock."
                )
        elif not isinstance(data, (pa.Array, pa.ChunkedArray)):
            # Arrow cannot construct an array from a torch.Tensor.
            if isinstance(data, torch.Tensor):
                data = data.numpy()
            data = pa.array(data)

        super(ArrowScalarColumn, self).__init__(data=data, *args, **kwargs)

    def _get(self, index, materialize: bool = True):
        index = ArrowBlock._convert_index(index)

        if isinstance(index, slice) or isinstance(index, int):
            data = self._data[index]
        elif index.dtype == bool:
            data = self._data.filter(pa.array(index))
        else:
            data = self._data.take(index)

        if self._is_batch_index(index):
            return self._clone(data=data)
        else:
            # Convert to Python object for consistency with other ScalarColumn
            # implementations.
            return data.as_py()

    def _set(self, index, value):
        raise ImmutableError("ArrowArrayColumn is immutable.")

    def _is_valid_primary_key(self):
        try:
            return len(self.unique()) == len(self)
        except Exception as e:
            warnings.warn(f"Unable to check if column is a valid primary key: {e}")
            return False

    def _keyidx_to_posidx(self, keyidx: Any) -> int:
        """Get the posidx of the first occurrence of the given keyidx. Raise a
        key error if the keyidx is not found.

        Args:
            keyidx: The keyidx to search for.

        Returns:
            The posidx of the first occurrence of the given keyidx.
        """
        posidx = pc.index(self.data, keyidx)
        if posidx == -1:
            raise KeyError(f"keyidx {keyidx} not found in column.")
        return posidx.as_py()

    def _keyidxs_to_posidxs(self, keyidxs: Sequence[Any]) -> np.ndarray:
        # FIXME: this implementation is very slow. This should be done with indices
        return np.array([self._keyidx_to_posidx(keyidx) for keyidx in keyidxs])

    def _repr_cell(self, index) -> object:
        return self.data[index]

    def _get_default_formatters(self) -> BaseFormatter:
        # can't implement this as a class level property because then it will treat
        # the formatter as a method
        from meerkat.interactive.formatter import (
            NumberFormatterGroup,
            TextFormatterGroup,
        )

        if len(self) == 0:
            return super()._get_default_formatters()

        if self.data.type == pa.string():
            return TextFormatterGroup()

        cell = self[0]
        return NumberFormatterGroup(dtype=type(cell).__name__)

    def is_equal(self, other: Column) -> bool:
        if other.__class__ != self.__class__:
            return False
        return pc.all(pc.equal(self.data, other.data)).as_py()

    @classmethod
    def _state_keys(cls) -> Set:
        return super()._state_keys()

    def _write_data(self, path):
        table = pa.Table.from_arrays([self.data], names=["0"])
        ArrowBlock._write_table(os.path.join(path, "data.arrow"), table)

    @staticmethod
    def _read_data(path, mmap=False):
        table = ArrowBlock._read_table(os.path.join(path, "data.arrow"), mmap=mmap)
        return table["0"]

    @classmethod
    def concat(cls, columns: Sequence[ArrowScalarColumn]):
        arrays = []
        for c in columns:
            if isinstance(c.data, pa.Array):
                arrays.append(c.data)
            elif isinstance(c.data, pa.ChunkedArray):
                arrays.extend(c.data.chunks)
            else:
                raise ValueError(f"Unexpected type {type(c.data)}")
        data = pa.concat_arrays(arrays)
        return columns[0]._clone(data=data)

    def to_numpy(self):
        return self.data.to_numpy()

    def to_tensor(self):
        return torch.tensor(self.data.to_numpy())

    def to_pandas(self, allow_objects: bool = False):
        return self.data.to_pandas()

    def to_arrow(self) -> pa.Array:
        return self.data

    def equals(self, other: Column) -> bool:
        if other.__class__ != self.__class__:
            return False
        return pc.all(pc.equal(self.data, other.data)).as_py()

    @property
    def dtype(self) -> pa.DataType:
        return self.data.type

    KWARG_MAPPING = {"skipna": "skip_nulls"}
    COMPUTE_FN_MAPPING = {
        "var": "variance",
        "std": "stddev",
        "sub": "subtract",
        "mul": "multiply",
        "truediv": "divide",
        "pow": "power",
        "eq": "equal",
        "ne": "not_equal",
        "lt": "less",
        "gt": "greater",
        "le": "less_equal",
        "ge": "greater_equal",
        "isna": "is_nan",
        "capitalize": "utf8_capitalize",
        "center": "utf8_center",
        "isalnum": "utf8_is_alnum",
        "isalpha": "utf8_is_alpha",
        "isdecimal": "utf8_is_decimal",
        "isdigit": "utf8_is_digit",
        "islower": "utf8_is_lower",
        "isnumeric": "utf8_is_numeric",
        "isspace": "utf8_is_space",
        "istitle": "utf8_is_title",
        "isupper": "utf8_is_upper",
        "lower": "utf8_lower",
        "upper": "utf8_upper",
        "len": "utf8_length",
        "lstrip": "utf8_ltrim",
        "rstrip": "utf8_rtrim",
        "strip": "utf8_trim",
        "swapcase": "utf8_swapcase",
        "title": "utf8_title",
    }

    def _dispatch_aggregation_function(self, compute_fn: str, **kwargs):
        kwargs = {self.KWARG_MAPPING.get(k, k): v for k, v in kwargs.items()}
        out = getattr(pc, self.COMPUTE_FN_MAPPING.get(compute_fn, compute_fn))(
            self.data, **kwargs
        )
        return out.as_py()

    def mode(self, **kwargs) -> ScalarColumn:
        if "n" in "kwargs":
            raise ValueError(
                "Meerkat does not support passing `n` to `mode` when "
                "backend is Arrow."
            )

        # matching behavior of Pandas, get all counts, but only return top modes
        struct_array = pc.mode(self.data, n=len(self), **kwargs)
        modes = []
        count = struct_array[0]["count"]
        for mode in struct_array:
            if count != mode["count"]:
                break
            modes.append(mode["mode"].as_py())
        return ArrowScalarColumn(modes)

    def median(self, skipna: bool = True, **kwargs) -> any:
        warnings.warn("Arrow backend computes an approximate median.")
        return pc.approximate_median(self.data, skip_nulls=skipna).as_py()

    def _dispatch_arithmetic_function(
        self, other: ScalarColumn, compute_fn: str, right: bool, *args, **kwargs
    ):
        if isinstance(other, Column):
            assert isinstance(other, ArrowScalarColumn)
            other = other.data

        compute_fn = self.COMPUTE_FN_MAPPING.get(compute_fn, compute_fn)
        if right:
            out = self._clone(
                data=getattr(pc, compute_fn)(other, self.data, *args, **kwargs)
            )
            return out
        else:
            return self._clone(
                data=getattr(pc, compute_fn)(self.data, other, *args, **kwargs)
            )

    def _true_div(self, other, right: bool = False, **kwargs) -> ScalarColumn:
        if isinstance(other, Column):
            assert isinstance(other, ArrowScalarColumn)
            other = other.data

        # convert other to float if it is an integer
        if isinstance(other, pa.ChunkedArray) or isinstance(other, pa.Array):
            if other.type == pa.int64():
                other = other.cast(pa.float64())
        else:
            other = pa.scalar(other, type=pa.float64())

        if right:
            return self._clone(pc.divide(other, self.data), **kwargs)
        else:
            return self._clone(pc.divide(self.data, other), **kwargs)

    def __add__(self, other: ScalarColumn):
        if self.dtype == pa.string():
            # pyarrow expects a final str used as the spearator
            return self._dispatch_arithmetic_function(
                other, "binary_join_element_wise", False, ""
            )

        return self._dispatch_arithmetic_function(other, "add", right=False)

    def __radd__(self, other: ScalarColumn):
        if self.dtype == pa.string():
            return self._dispatch_arithmetic_function(
                other, "binary_join_element_wise", True, ""
            )

        return self._dispatch_arithmetic_function(other, "add", right=False)

    def __truediv__(self, other: ScalarColumn):
        return self._true_div(other, right=False)

    def __rtruediv__(self, other: ScalarColumn):
        return self._true_div(other, right=True)

    def _floor_div(self, other, right: bool = False, **kwargs) -> ScalarColumn:
        _true_div = self._true_div(other, right=right, **kwargs)
        return _true_div._clone(data=pc.floor(_true_div.data))

    def __floordiv__(self, other: ScalarColumn):
        return self._floor_div(other, right=False)

    def __rfloordiv__(self, other: ScalarColumn):
        return self._floor_div(other, right=True)

    def __mod__(self, other: ScalarColumn):
        raise NotImplementedError("Modulo is not supported by Arrow backend.")

    def __rmod__(self, other: ScalarColumn):
        raise NotImplementedError("Modulo is not supported by Arrow backend.")

    def _dispatch_comparison_function(
        self, other: ScalarColumn, compute_fn: str, **kwargs
    ):
        if isinstance(other, Column):
            assert isinstance(other, ArrowScalarColumn)
            other = other.data

        compute_fn = self.COMPUTE_FN_MAPPING.get(compute_fn, compute_fn)
        return self._clone(data=getattr(pc, compute_fn)(self.data, other, **kwargs))

    def _dispatch_logical_function(
        self, other: ScalarColumn, compute_fn: str, **kwargs
    ):
        if isinstance(other, Column):
            assert isinstance(other, ArrowScalarColumn)
            other = other.data

        compute_fn = self.COMPUTE_FN_MAPPING.get(compute_fn, compute_fn)

        if other is None:
            return self._clone(data=getattr(pc, compute_fn)(self.data, **kwargs))
        return self._clone(data=getattr(pc, compute_fn)(self.data, other, **kwargs))

    def isin(self, values: Union[List, Set], **kwargs) -> ScalarColumn:
        return self._clone(data=pc.is_in(self.data, pa.array(values), **kwargs))

    def _dispatch_unary_function(
        self, compute_fn: str, _namespace: str = None, **kwargs
    ):
        compute_fn = self.COMPUTE_FN_MAPPING.get(compute_fn, compute_fn)
        return self._clone(data=getattr(pc, compute_fn)(self.data, **kwargs))

    def isnull(self, **kwargs) -> ScalarColumn:
        return self._clone(data=pc.is_null(self.data, nan_is_null=True, **kwargs))
