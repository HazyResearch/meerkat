from __future__ import annotations

import os
import shutil
from collections import defaultdict
from collections.abc import MutableMapping
from typing import Dict, Mapping, Sequence, Union

import numpy as np
import pandas as pd
import yaml

import meerkat.config
from meerkat.block.abstract import AbstractBlock, BlockIndex
from meerkat.columns.abstract import AbstractColumn
from meerkat.tools.utils import MeerkatLoader

from .ref import BlockRef


class BlockManager(MutableMapping):
    """Manages all blocks in a DataPanel."""

    def __init__(self) -> None:
        self._columns: Dict[str, AbstractColumn] = {}  # ordered as of 3.7
        self._column_to_block_id: Dict[str, int] = {}
        self._block_refs: Dict[int, BlockRef] = {}

    def update(self, block_ref: BlockRef):
        """data (): a single blockable object, potentially contains multiple
        columns."""
        for name in block_ref:
            if name in self:
                self.remove(name)

        # although we can't have the same column living in multiple managers
        # we don't view here because it can lead to multiple calls to clone
        self._columns.update(block_ref)

        block_id = id(block_ref.block)
        # check if there already is a block_ref in the manager for this block
        if block_id in self._block_refs:
            self._block_refs[block_id].update(block_ref)
        else:
            self._block_refs[block_id] = block_ref

        self._column_to_block_id.update({name: block_id for name in block_ref.keys()})

    def apply(self, method_name: str = "_get", *args, **kwargs) -> BlockManager:
        """[summary]

        Args:
            fn (str): a function that is applied to a block and column_spec and
                returns a new block and column_spec.
        Returns:
            [type]: [description]
        """
        results = None
        for block_ref in self._block_refs.values():
            result = block_ref.apply(method_name=method_name, *args, **kwargs)
            if results is None:
                results = BlockManager() if isinstance(result, BlockRef) else {}
            results.update(result)

        # apply method to columns not stored in block
        for name, col in self._columns.items():
            if results is not None and name in results:
                continue

            result = getattr(col, method_name)(*args, **kwargs)

            if results is None:
                results = BlockManager() if isinstance(result, AbstractColumn) else {}

            results[name] = result

        if isinstance(results, BlockManager):
            results.reorder(self.keys())
        return results

    def consolidate(self):
        column_order = list(
            self._columns.keys()
        )  # need to maintain order after consolidate

        block_ref_groups = defaultdict(list)
        for block_ref in self._block_refs.values():
            block_ref_groups[block_ref.block.signature].append(block_ref)

        for block_refs in block_ref_groups.values():
            if len(block_refs) == 1:
                # if there is only one block ref in the group, do not consolidate
                continue

            # consolidate group
            block_class = block_refs[0].block.__class__
            block_ref = block_class.consolidate(block_refs)
            self.update(block_ref)

        self.reorder(column_order)

    def remove(self, name):
        if name not in self._columns:
            raise ValueError(f"Remove failed: no column '{name}' in BlockManager.")

        self._columns.pop(name)

        if name in self._column_to_block_id:
            # column is blockable
            block_ref = self._block_refs[self._column_to_block_id[name]]
            del block_ref[name]

            if len(block_ref) == 0:
                self._block_refs.pop(self._column_to_block_id[name])

            self._column_to_block_id.pop(name)

    def reorder(self, order: Sequence[str]):
        if set(order) != set(self._columns):
            raise ValueError("Must include all columns when reordering a BlockManager.")
        self._columns = {name: self._columns[name] for name in order}

    def __getitem__(
        self, index: Union[str, Sequence[str]]
    ) -> Union[AbstractColumn, BlockManager]:
        if isinstance(index, str):
            return self._columns[index]
        elif isinstance(index, Sequence):
            mgr = BlockManager()
            block_id_to_names = defaultdict(list)
            for name in index:
                if name not in self._column_to_block_id:
                    if name in self:
                        # non-blockable column
                        mgr.add_column(col=self._columns[name], name=name)
                    else:
                        raise ValueError(
                            f"`BlockManager` does not contain column '{name}'."
                        )
                else:
                    # group blockable columns by block
                    block_id_to_names[self._column_to_block_id[name]].append(name)

            # block refs for blockable columns
            for block_id, names in block_id_to_names.items():
                block_ref = self._block_refs[block_id]
                mgr.update(block_ref[names])
            mgr.reorder(order=index)
            return mgr
        else:
            raise ValueError(
                f"Unsupported index of type `{type(index)}` passed to `BlockManager`."
            )

    def __setitem__(self, index: str, data: Union[str, Sequence[str]]):
        if isinstance(data, AbstractColumn):
            self.add_column(data, name=index)
        else:
            raise ValueError(
                f"Cannot set item with object of type `{type(data)}` on `BlockManager`."
            )

    def __delitem__(self, key):
        self.remove(key)

    def __len__(self):
        return len(self._columns)

    @property
    def nrows(self):
        return 0 if len(self) == 0 else len(next(iter(self._columns.values())))

    @property
    def ncols(self):
        return len(self)

    def __contains__(self, value):
        return value in self._columns

    def __iter__(self):
        return iter(self._columns)

    def get_block_ref(self, name: str):
        return self._block_refs[self._column_to_block_id[name]]

    def add_column(self, col: AbstractColumn, name: str):
        """Convert data to a meerkat column using the appropriate Column
        type."""
        if len(self) > 0 and len(col) != self.nrows:
            raise ValueError(
                f"Cannot add column '{name}' with length {len(col)} to `BlockManager` "
                f" with length {self.nrows} columns."
            )

        if not col.is_blockable():
            col = col.view()
            self._columns[name] = col

        else:
            col = col.view()
            self.update(BlockRef(columns={name: col}, block=col._block))

    @classmethod
    def from_dict(cls, data: Mapping[str, object]):
        mgr = cls()
        for name, data in data.items():
            col = AbstractColumn.from_data(data)
            mgr.add_column(col=col, name=name)
        return mgr

    def write(self, path: str):
        meta = {
            "dtype": BlockManager,
            "columns": {},
            "_column_order": list(self.keys()),
        }

        # prepare directories
        columns_dir = os.path.join(path, "columns")
        blocks_dir = os.path.join(path, "blocks")
        meta_path = os.path.join(path, "meta.yaml")
        if os.path.isdir(path):
            if (
                os.path.exists(meta_path)
                and os.path.exists(columns_dir)
                and os.path.exists(blocks_dir)
            ):
                # if overwriting, ensure that old columns are removed
                shutil.rmtree(columns_dir)
                shutil.rmtree(blocks_dir)
            else:
                # if path already points to a dir that wasn't previously holding a
                # block manager, do not overwrite it. We'd like to protect against
                # situation in which user accidentally puts in an important directory
                raise IsADirectoryError(
                    f"Cannot write `BlockManager`. {path} is a directory."
                )

        os.makedirs(path, exist_ok=True)
        os.makedirs(blocks_dir)
        os.makedirs(columns_dir)

        # consolidate before writing
        self.consolidate()
        for block_id, block_ref in self._block_refs.items():
            block: AbstractBlock = block_ref.block
            block_dir = os.path.join(blocks_dir, str(block_id))
            block.write(block_dir)

            for name, column in block_ref.items():
                column_dir = os.path.join(columns_dir, name)
                os.makedirs(column_dir, exist_ok=True)

                # don't write the data, reference the block
                meta["columns"][name] = {
                    **column._get_meta(),
                    "block": {
                        "block_dir": os.path.relpath(block_dir, path),
                        "block_index": _serialize_block_index(column._block_index),
                        "mmap": block.is_mmap,
                    },
                }
                column._write_state(column_dir)

        # write columns not in a block
        for name, column in self._columns.items():
            if name in meta["columns"]:
                continue
            meta["columns"][name] = column._get_meta()
            column.write(os.path.join(columns_dir, name))

        # Save the metadata as a yaml file
        yaml.dump(meta, open(meta_path, "w"))

    @classmethod
    def read(
        cls,
        path: str,
        columns: Sequence[str] = None,
        *args,
        **kwargs,
    ) -> BlockManager:
        """Load a DataPanel stored on disk."""

        # Load the metadata
        meta = dict(
            yaml.load(open(os.path.join(path, "meta.yaml")), Loader=MeerkatLoader)
        )

        blocks = {}
        mgr = cls()
        for name, col_meta in meta["columns"].items():
            column_dir = os.path.join(path, "columns", name)
            # load a subset of columns
            if columns is not None and name not in columns:
                continue

            if "block" in col_meta:
                # read block or fetch it from `blocks` if it's already been read
                block_meta = col_meta["block"]
                if block_meta["block_dir"] not in blocks:
                    blocks[block_meta["block_dir"]] = AbstractBlock.read(
                        os.path.join(path, block_meta["block_dir"]),
                        mmap=block_meta.get("mmap", False),
                    )
                block = blocks[block_meta["block_dir"]]

                # read column, passing in a block_view
                col = col_meta["dtype"].read(
                    column_dir,
                    _data=block[_deserialize_block_index(block_meta["block_index"])],
                    _meta=col_meta,
                    **kwargs,
                )
                mgr.add_column(col, name)
            else:
                mgr.add_column(
                    col_meta["dtype"].read(path=column_dir, _meta=col_meta, **kwargs),
                    name,
                )
        mgr.reorder(meta["_column_order"])
        return mgr

    def _repr_pandas_(self, max_rows: int = None):
        if max_rows is None:
            max_rows = meerkat.config.DisplayOptions.max_rows
        cols = {}
        formatters = {}
        for name, column in self._columns.items():
            cols[name], formatters[name] = column._repr_pandas_(max_rows=max_rows)
        if self.nrows > max_rows:
            pd_index = np.concatenate(
                (
                    np.arange(max_rows // 2),
                    np.zeros(1),
                    np.arange(self.nrows - max_rows // 2, self.nrows),
                ),
            )
        else:
            pd_index = np.arange(self.nrows)

        df = pd.DataFrame(cols)
        df = df.set_index(pd_index.astype(int))
        return df, formatters

    def view(self):
        mgr = BlockManager()
        for name, col in self.items():
            mgr.add_column(col.view(), name)
        return mgr

    def copy(self):
        mgr = BlockManager()
        for name, col in self.items():
            mgr.add_column(col.copy(), name)
        return mgr


def _serialize_block_index(index: BlockIndex) -> Union[Dict, str, int]:
    if not isinstance(index, (int, str, slice)):
        raise ValueError("Can only serialize `BlockIndex` objects.")
    elif isinstance(index, slice):
        return {"start": index.start, "stop": index.stop, "step": index.step}
    return index


def _deserialize_block_index(index: Union[Dict, int, str]) -> BlockIndex:
    if isinstance(index, Dict):
        return slice(index["start"], index["stop"], index["step"])
    return index
