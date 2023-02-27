from __future__ import annotations

import os
from copy import copy
from dataclasses import dataclass
from typing import Dict, Hashable, List, Sequence, Tuple, Union

import numpy as np
from cytoolz import merge_with

import meerkat as mk
from meerkat.block.ref import BlockRef
from meerkat.columns.abstract import Column
from meerkat.tools.utils import dump_yaml, load_yaml, meerkat_dill_load, translate_index

from .abstract import AbstractBlock, BlockIndex, BlockView


@dataclass
class DeferredCellOp:
    args: List[any]
    kwargs: Dict[str, any]
    fn: callable
    is_batched_fn: bool
    return_index: Union[str, int] = None

    @staticmethod
    def prepare_arg(arg):
        from ..columns.deferred.base import AbstractCell, DeferredColumn

        if isinstance(arg, AbstractCell):
            return arg.get()
        elif isinstance(arg, DeferredColumn):
            return arg()
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

    def __len__(self):
        if len(self.args) > 0:
            return len(self.args[0])
        else:
            for col in self.kwargs.values():
                return len(col)
        return 0

    def is_equal(self, other: Column):
        if (
            self.fn != other.fn
            or self.is_batched_fn != other.is_batched_fn
            or self.return_index != other.return_index
        ):
            return False

        for arg, other_arg in zip(self.args, other.args):
            if arg != other_arg:
                return False

        if set(self.kwargs.keys()) != set(other.kwargs.keys()):
            return False

        for key in self.kwargs:
            if self.kwargs[key] != other.kwargs[key]:
                return False
        return True


@dataclass
class DeferredOp:
    args: List[mk.Column]
    kwargs: Dict[str, mk.Column]
    fn: callable
    is_batched_fn: bool
    batch_size: int
    return_format: type = None
    return_index: Union[str, int] = None
    materialize_inputs: bool = True

    @staticmethod
    def concat(ops: Sequence[DeferredOp]):
        """Concatenate a sequence of operations."""
        if len(ops) == 0:
            raise ValueError("Cannot concatenate empty sequence of LambdaOp.")

        if len(ops) == 1:
            return ops[0]

        # going to use the `fn` etc. of the first op
        op = copy(ops[0])

        op.args = [mk.concat([op.args[i] for op in ops]) for i in range(len(op.args))]
        op.kwargs = {
            kwarg: mk.concat([op.kwargs[kwarg] for op in ops])
            for kwarg in op.kwargs.keys()
        }
        return op

    def is_equal(self, other: Column):
        if (
            self.fn != other.fn
            or self.is_batched_fn != other.is_batched_fn
            or self.return_format != other.return_format
            or self.return_index != other.return_index
        ):
            return False

        for arg, other_arg in zip(self.args, other.args):
            if not arg.is_equal(other_arg):
                return False

        if set(self.kwargs.keys()) != set(other.kwargs.keys()):
            return False

        for key in self.kwargs:
            if not self.kwargs[key].is_equal(other.kwargs[key]):
                return False
        return True

    def write(self, path: str, written_inputs: Dict[int, str] = None):
        """_summary_

        Args:
            path (str): _description_
            written_inputs (dict, optional): _description_. Defaults to None.
        """
        # Make all the directories to the path
        os.makedirs(path, exist_ok=True)

        if written_inputs is None:
            written_inputs = {}
        state = {
            "fn": self.fn,
            "return_index": self.return_index,
            "return_format": self.return_format,
            "is_batched_fn": self.is_batched_fn,
            "batch_size": self.batch_size,
            "materialize_inputs": self.materialize_inputs,
        }
        # state_path = os.path.join(path, "state.dill")
        # dill.dump(state, open(state_path, "wb"))

        meta = {"args": [], "kwargs": {}, "state": state}

        args_dir = os.path.join(path, "args")
        os.makedirs(args_dir, exist_ok=True)
        for idx, arg in enumerate(self.args):
            if id(arg) in written_inputs:
                meta["args"].append(written_inputs[id(arg)])
            else:
                col_path = os.path.join(args_dir, f"{idx}.col")
                arg.write(col_path)
                meta["args"].append(os.path.relpath(col_path, path))

        kwargs_dir = os.path.join(path, "kwargs")
        os.makedirs(kwargs_dir, exist_ok=True)
        for key, arg in self.kwargs.items():
            if id(arg) in written_inputs:
                meta["kwargs"][key] = written_inputs[id(arg)]
            else:
                col_path = os.path.join(kwargs_dir, f"{key}.col")
                arg.write(col_path)
                meta["kwargs"][key] = os.path.relpath(col_path, path)

        # Save the metadata as a yaml file
        meta_path = os.path.join(path, "meta.yaml")
        dump_yaml(meta, meta_path)

    @classmethod
    def read(cls, path, read_inputs: Dict[str, Column] = None):
        if read_inputs is None:
            read_inputs = {}

        # Assert that the path exists
        assert os.path.exists(path), f"`path` {path} does not exist."

        meta = dict(load_yaml(os.path.join(path, "meta.yaml")))

        args = [
            read_inputs[arg_path]
            if arg_path in read_inputs
            else Column.read(os.path.join(path, arg_path))
            for arg_path in meta["args"]
        ]
        kwargs = {
            key: read_inputs[kwarg_path]
            if kwarg_path in read_inputs
            else Column.read(os.path.join(path, kwarg_path))
            for key, kwarg_path in meta["kwargs"]
        }

        if "state" in meta:
            state = meta["state"]
        else:
            state = meerkat_dill_load(os.path.join(path, "state.dill"))

        return cls(args=args, kwargs=kwargs, **state)

    def _get(
        self,
        index: Union[int, np.ndarray],
        indexed_inputs: Dict[int, Column] = None,
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
            kwarg: indexed_inputs[id(column)]
            if id(column) in indexed_inputs
            else column._get(index, materialize=self.materialize_inputs)
            for kwarg, column in self.kwargs.items()
        }

        args = [
            indexed_inputs[id(column)]
            if id(column) in indexed_inputs
            else column._get(index, materialize=self.materialize_inputs)
            for column in self.args
        ]

        if isinstance(index, int):
            if materialize:
                output = self.fn(*args, **kwargs)
                if self.return_index is not None:
                    output = output[self.return_index]
                return output
            else:
                return DeferredCellOp(
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
                        if (
                            (self.return_format is None or self.return_format is dict)
                            and isinstance(output, Dict)
                            and (self.return_index is None)
                        ):
                            return {k: v[0] for k, v in output.items()}
                        elif (
                            (self.return_format is None or self.return_format is tuple)
                            and isinstance(output, Tuple)
                            and (self.return_index is None)
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
                            **{kwarg: column[i] for kwarg, column in kwargs.items()},
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
                    return DeferredCellOp(
                        fn=self.fn,
                        args=args,
                        kwargs=kwargs,
                        is_batched_fn=self.is_batched_fn,
                        return_index=self.return_index,
                    )
                return DeferredOp(
                    fn=self.fn,
                    args=args,
                    kwargs=kwargs,
                    is_batched_fn=self.is_batched_fn,
                    batch_size=self.batch_size,
                    return_format=self.return_format,
                    return_index=self.return_index,
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
        op: DeferredOp = copy(self)
        op.return_index = index
        return op


class DeferredBlock(AbstractBlock):
    @dataclass(eq=True, frozen=True)
    class Signature:
        klass: type
        fn: callable
        args: Tuple[int]
        # dicts are not hashable, so inputs should be a sorted tuple of tuples
        kwargs: Tuple[Tuple[Union[str, int], int]]

    @property
    def signature(self) -> Hashable:
        return self.Signature(
            klass=DeferredBlock,
            fn=self.data.fn,
            args=tuple(map(id, self.data.args)),
            kwargs=tuple(sorted((k, id(v)) for k, v in self.data.kwargs.items())),
        )

    def __init__(self, data: DeferredOp):
        self.data = data

    @classmethod
    def from_column_data(cls, data: DeferredOp) -> Tuple[DeferredBlock, BlockView]:
        block_index = data.return_index
        data = data.with_return_index(None)
        block = cls(data=data)
        return BlockView(block=block, block_index=block_index)

    @classmethod
    def from_block_data(cls, data: DeferredOp) -> Tuple[AbstractBlock, BlockView]:
        return cls(data=data)

    @classmethod
    def _consolidate(
        cls,
        block_refs: Sequence[BlockRef],
        consolidated_inputs: Dict[int, Column] = None,
    ) -> BlockRef:
        if consolidated_inputs is None:
            consolidated_inputs = {}

        # if the input column has been consolidated, we need to update the inputs
        # (i.e. args and kwargs) of the data
        op = block_refs[0].block.data.with_return_index(None)
        op.args = [consolidated_inputs.get(id(arg), arg) for arg in op.args]
        op.kwargs = {
            kwarg: consolidated_inputs.get(id(column), column)
            for kwarg, column in op.kwargs.items()
        }

        block = DeferredBlock.from_block_data(op)

        columns = {
            name: col._clone(data=block[col._block_index])
            for ref in block_refs
            for name, col in ref.items()
        }

        return BlockRef(block=block, columns=columns)

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
                    name: col.convert_to_output_type(
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
                    if col.is_blockable()  # may return a non-blockable type
                    else (name, col)
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

    def _write_data(self, path: str, written_inputs: Dict[int, str] = None):
        path = os.path.join(path, "data.op")
        return self.data.write(path, written_inputs=written_inputs)

    @staticmethod
    def _read_data(
        path: str, mmap: bool = False, read_inputs: Dict[str, Column] = None
    ) -> object:
        path = os.path.join(path, "data.op")
        return DeferredOp.read(path, read_inputs=read_inputs)
