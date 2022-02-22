from __future__ import annotations
from dataclasses import dataclass
from multiprocessing.sharedctypes import Value
from typing import Dict, Hashable, List, Sequence, Tuple, Union
import numpy as np

from cytoolz import merge_with

from meerkat.block.ref import BlockRef
import meerkat as mk
from meerkat.columns.abstract import AbstractColumn
from meerkat.tools.utils import translate_index
from .abstract import AbstractBlock, BlockIndex, BlockView


#


@dataclass
class LambdaOp:
    inputs: Dict[Union[str, int], mk.AbstractColumn]
    fn: callable
    is_batched_fn: bool

    def _get(self, index: Union[int, np.ndarray], indexed_inputs: dict):

        # TODO: support batchuing

        # we pass results from other columns
        # prepare inputs
        if isinstance(self.inputs, dict):
            # multiple inputs into the function
            use_kwargs = True
            kwargs = {
                # if column has already been indexed
                kwarg: indexed_inputs.get(
                    id(column), column._get(index, materialize=True)
                )
                for kwarg, column in self.inputs.items()
            }
        elif isinstance(self.inputs, AbstractColumn):
            # single input into the function
            use_kwargs = False
            arg = indexed_inputs.get(
                id(self.inputs), self.inputs._get(index, materialize=True)
            )
        else:
            raise ValueError

        #
        if isinstance(index, int):
            outputs = self.fn(**kwargs) if use_kwargs else self.fn(arg)
        elif isinstance(index, np.ndarray):
            outputs = [
                (
                    self.fn(**{kwarg: column[i] for kwarg, column in kwargs.items()})
                    if use_kwargs
                    else self.fn(arg[i])
                )
                for i in range(len(index))
            ]


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

    def __init__(self, data):
        data = None

    @classmethod
    def from_column_data(cls, data: LambdaOp) -> Tuple[LambdaBlock, BlockView]:

        pass

    @classmethod
    def from_block_data(cls, data: LambdaOp) -> Tuple[AbstractBlock, BlockView]:

        return super().from_block_data(data)

    @classmethod
    def _consolidate(cls, block_refs: Sequence[BlockRef]) -> BlockRef:
        pass

    def _convert_index(self, index):
        return translate_index(index, length=len())  # TODO

    def _get(self, index, block_ref: BlockRef, inputs: dict) -> Union[BlockRef, dict]:

        index = self._convert_index()
        if not materialize:
            pass
            return

        self.data._get(index=index, inputs=inputs)

        # we pass results from other columns
        # prepare inputs
        if isinstance(data.inputs, dict):
            # multiple inputs into the function
            use_kwargs = True
            kwargs = {
                # if column has already been indexed
                kwarg: results.get(id(column), self._data._get(index, materialize=True))
                for kwarg, column in op.inputs.items()
            }
        elif isinstance(op.inputs, AbstractColumn):
            # single input into the function
            use_kwargs = False
            arg = results.get(id(op.inputs), self._data._get(index, materialize=True))
        else:
            raise ValueError

        #
        if isinstance(index, int):
            outputs = op.fn(**kwargs) if use_kwargs else op.fn(arg)
        elif isinstance(index, np.ndarray):
            outputs = [
                (
                    op.fn(**{kwarg: column[i] for kwarg, column in kwargs.items()})
                    if use_kwargs
                    else op.fn(arg[i])
                )
                for i in range(len(index))
            ]
            if isinstance(op.outputs, dict):
                outputs = merge_with(list, outputs)

                # TODO: apply output type
                # apply correct collate and convert to columns
                results = {
                    op.outputs[kwout].from_data(op.outputs[kwout].collate(out))
                    for kwout, out in outputs.items()
                }

            elif isinstance(op.outputs, AbstractColumn):
                outputs = op.outputs.from_data(op.outputs.collate(outputs))
                results.update({id(op.outputs): outputs})
            else:
                raise ValueError

        return

    def _get_data(self, index: BlockIndex) -> object:
        pass

    def _write_data(self, path: str, *args, **kwargs):
        return super()._write_data(path, *args, **kwargs)

    @staticmethod
    def _read_data(path: str, *args, **kwargs) -> object:
        return super()._read_data(path, *args, **kwargs)
