from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from multiprocessing.sharedctypes import Value
from typing import Dict, Hashable, List, Sequence, Tuple, Union

import numpy as np
from cytoolz import merge_with

import meerkat as mk
from meerkat.block.ref import BlockRef
from meerkat.columns.abstract import AbstractColumn
from meerkat.tools.utils import translate_index

from .abstract import AbstractBlock, BlockIndex, BlockView


@dataclass
class LambdaCellOp:
    args: List[any]
    kwargs: Dict[str, any]
    fn: callable
    is_batched_fn: bool
    return_index: Union[str, int] = None

    @staticmethod
    def prepare_arg(arg):
        from ..columns.lambda_column import AbstractCell

        if isinstance(arg, AbstractCell):
            return arg.get()
        elif isinstance(arg, AbstractColumn):
            return arg[:]
        return arg

    def _get(self):

        args = [self.prepare_arg(arg) for arg in self.args]
        kwargs = {kw: self.prepare_arg(arg) for kw, arg in self.kwargs.items()}
        out = self.fn(*args, **kwargs)

        if self.return_index is not None:
            return out[self.return_index]

        if self.is_batched_fn:
            return out[0]

        return out

    def with_return_index(self, index: Union[str, int]):
        op = copy(self)
        op.return_index = index
        return op


@dataclass
class LambdaOp:
    args: List[mk.AbstractColumn]
    kwargs: Dict[str, mk.AbstractColumn]
    fn: callable
    is_batched_fn: bool
    batch_size: int
    return_format: type = None
    return_index: Union[str, int] = None

    def _get(
        self,
        index: Union[int, np.ndarray],
        indexed_inputs: dict = None,
        materialize: bool = True,
    ):
        if indexed_inputs is None:
            indexed_inputs = {}

        # if function is batched, but the index is singular, we need to turn the
        # single index into a batch index, and then later unpack the result
        single_on_batched = self.is_batched_fn and isinstance(index, int)
        if single_on_batched:
            index = np.array([index])

        # we pass results from other columns
        # prepare inputs
        kwargs = {
            # if column has already been indexed
            kwarg: indexed_inputs.get(
                id(column), column._get(index, materialize=materialize)
            )
            for kwarg, column in self.kwargs.items()
        }

        args = [
            indexed_inputs.get(id(column), column._get(index, materialize=materialize))
            for column in self.args
        ]

        if isinstance(index, int):
            if materialize:
                output = self.fn(*args, **kwargs)
                if self.return_index is not None:
                    output = output[self.return_index]
                return output
            else:
                return LambdaCellOp(
                    fn=self.fn,
                    args=args,
                    kwargs=kwargs,
                    is_batched_fn=self.is_batched_fn,
                    return_index=self.return_index,
                )

        elif isinstance(index, np.ndarray):
            if materialize:
                if self.is_batched_fn:
                    output = self.fn(*args, **kwargs)

                    if self.return_index is not None:
                        output = output[self.return_index]

                    if single_on_batched:
                        if (self.return_format is dict) and (self.return_index is None):
                            return {k: v[0] for k, v in output.items()}
                        elif (self.return_format is tuple) and (
                            self.return_index is None
                        ):
                            return [v[0] for v in output]
                        else:
                            return output[0]
                    return output

                else:
                    outputs = []
                    for i in range(len(index)):
                        output = self.fn(
                            *[arg[i] for arg in args],
                            **{kwarg: column[i] for kwarg, column in kwargs.items()}
                        )

                        if self.return_index is not None:
                            output = output[self.return_index]
                        outputs.append(output)

                    if (self.return_format is dict) and (self.return_index is None):
                        return merge_with(list, outputs)
                    elif (self.return_format is tuple) and (self.return_index is None):
                        return tuple(zip(*outputs))
                    else:
                        return outputs

            else:
                if single_on_batched:
                    return LambdaCellOp(
                        fn=self.fn,
                        args=args,
                        kwargs=kwargs,
                        is_batched_fn=self.is_batched_fn,
                        return_index=self.return_index,
                    )
                return LambdaOp(
                    fn=self.fn,
                    args=args,
                    kwargs=kwargs,
                    is_batched_fn=self.is_batched_fn,
                    batch_size=self.batch_size,
                )

    def __len__(self):
        if len(self.args) > 0:
            return len(self.args[0])
        else:
            for col in self.kwargs.values():
                return len(col)
        return 0

    def with_return_index(self, index: Union[str, int]):
        """Return a copy of the operation with a new return index."""
        op: LambdaOp = copy(self)
        op.return_index = index
        return op


class LambdaBlock(AbstractBlock):
    @dataclass(eq=True, frozen=True)
    class Signature:
        n_rows: int
        klass: type
        fn: callable
        # dicts are not hashable, so inputs should be a sorted tuple of tuples
        inputs: Tuple[Tuple[Union[str, int], int]]

    @property
    def signature(self) -> Hashable:
        return self.Signature(
            klass=LambdaBlock,
            fn=self.data.fn,
            inputs=tuple(sorted((k, id(v)) for k, v in self.data.inputs.items())),
            nrows=self.data.shape[0],
        )

    def __init__(self, data: LambdaOp):

        self.data = data

    @classmethod
    def from_column_data(cls, data: LambdaOp) -> Tuple[LambdaBlock, BlockView]:
        block_index = data.return_index
        data = data.with_return_index(None)
        block = cls(data=data)
        return BlockView(block=block, block_index=block_index)

    @classmethod
    def from_block_data(cls, data: LambdaOp) -> Tuple[AbstractBlock, BlockView]:
        return cls(data=data)

    @classmethod
    def _consolidate(cls, block_refs: Sequence[BlockRef]) -> BlockRef:
        pass

    def _convert_index(self, index):
        return translate_index(index, length=len(self.data))  # TODO

    def _get(
        self,
        index,
        block_ref: BlockRef,
        indexed_inputs: dict = None,
        materialize: bool = True,
    ) -> Union[BlockRef, dict]:
        if indexed_inputs is None:
            indexed_inputs = {}
        index = self._convert_index(index)

        outputs = self.data._get(
            index=index, indexed_inputs=indexed_inputs, materialize=materialize
        )

        # convert raw outputs into columns
        if isinstance(index, int):
            if materialize:
                return {
                    name: outputs
                    if (col._block_index is None)
                    else outputs[col._block_index]
                    for name, col in block_ref.columns.items()
                }
            else:
                # outputs is a
                return {
                    name: col._create_cell(outputs.with_return_index(col._block_index))
                    for name, col in block_ref.columns.items()
                }

        else:
            if materialize:
                outputs = {
                    name: col.from_data(
                        col.collate(
                            outputs
                            if (col._block_index is None)
                            else outputs[col._block_index]
                        )
                    )
                    for name, col in block_ref.columns.items()
                }
                return [
                    BlockRef(columns={name: col}, block=col._block)
                    for name, col in outputs.items()
                ]
            else:
                block = self.from_block_data(outputs)
                columns = {
                    name: col._clone(
                        data=BlockView(block=block, block_index=col._block_index)
                    )
                    for name, col in block_ref.columns.items()
                }
                return BlockRef(block=block, columns=columns)

    def _get_data(self, index: BlockIndex) -> object:
        return self.data.with_return_index(index)

    def _write_data(self, path: str, *args, **kwargs):
        return super()._write_data(path, *args, **kwargs)

    @staticmethod
    def _read_data(path: str, *args, **kwargs) -> object:
        return super()._read_data(path, *args, **kwargs)
