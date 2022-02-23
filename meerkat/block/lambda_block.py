from __future__ import annotations

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

    def get(self):
        from ..columns.lambda_column import AbstractCell

        args = [
            arg.get() if isinstance(arg, AbstractCell) else arg for arg in self.args
        ]
        kwargs = {
            kw: arg.get() if isinstance(arg, AbstractCell) else arg
            for kw, arg in self.kwargs.items()
        }
        return self.fn(*args, **kwargs)


@dataclass
class LambdaOp:
    args: List[mk.AbstractColumn]
    kwargs: Dict[str, mk.AbstractColumn]
    fn: callable
    is_batched_fn: bool
    batch_size: int
    return_format: type = None

    def _get(
        self,
        index: Union[int, np.ndarray],
        indexed_inputs: dict = None,
        materialize: bool = True,
    ):

        # TODO: support batching
        if indexed_inputs is None:
            indexed_inputs = {}

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
                return self.fn(*args, **kwargs)
            else:
                return LambdaCellOp(
                    fn=self.fn,
                    args=args,
                    kwargs=kwargs,
                    is_batched_fn=self.is_batched_fn,
                )

        elif isinstance(index, np.ndarray):
            if materialize:
                outputs = [
                    (
                        self.fn(
                            *[arg[i] for arg in args],
                            **{kwarg: column[i] for kwarg, column in kwargs.items()}
                        )
                    )
                    for i in range(len(index))
                ]

                if self.return_format is dict:
                    return merge_with(list, outputs)
                elif self.return_format is tuple:
                    return tuple(zip(*outputs))
                else:
                    return outputs

            else:
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
        block = cls(data=data)
        return BlockView(block=block, block_index=None)

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
                pass

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
        # TODO: consider returning a special version with the output index
        return self.data

    def _write_data(self, path: str, *args, **kwargs):
        return super()._write_data(path, *args, **kwargs)

    @staticmethod
    def _read_data(path: str, *args, **kwargs) -> object:
        return super()._read_data(path, *args, **kwargs)
