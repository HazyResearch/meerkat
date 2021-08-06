from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import MutableMapping
from typing import Dict, Sequence, Union

import pandas as pd
import yaml

from meerkat.block.abstract import AbstractBlock, BlockIndex
from meerkat.columns.abstract import AbstractColumn

from .ref import BlockRef


class BlockManager(MutableMapping):
    """This manager manages all blocks."""

    def __init__(self) -> None:
        self._columns: Dict[str, AbstractColumn] = {}
        self._column_to_block_id: Dict[str, int] = {}
        self._block_refs: Dict[int, BlockRef] = {}

    def update(self, block_ref: BlockRef):
        """data (): a single blockable object, potentially contains multiple
        columns."""
        # although we can't have the same column living in multiple managers
        # we don't view here because it can lead to multiple calls to clone
        self._columns.update({name: column for name, column in block_ref.items()})

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

        return results

    def consolidate(self):
        block_ref_groups = defaultdict(list)
        for block_ref in self._block_refs.values():
            block_ref_groups[block_ref.block.signature].append(block_ref)

        for block_refs in block_ref_groups.values():
            if len(block_refs) == 1:
                # if there is only one block ref in the group, do not consolidate
                continue

            # remove old block_refs
            for old_ref in block_refs:
                self._block_refs.pop(id(old_ref.block))

            # consolidate group
            block_class = block_refs[0].block.__class__
            block_ref = block_class.consolidate(block_refs)
            self.update(block_ref)

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

    def __contains__(self, value):
        return value in self._columns

    def __iter__(self):
        return iter(self._columns)

    def get_block_ref(self, name: str):
        return self._block_refs[self._column_to_block_id[name]]

    def add_column(self, col: AbstractColumn, name: str):
        """Convert data to a meerkat column using the appropriate Column
        type."""
        if name in self._columns:
            self.remove(name)

        if not col.is_blockable():
            col = col.view()
            self._columns[name] = col

        else:
            col = col.view()
            self.update(BlockRef(columns={name: col}, block=col._block))

    @classmethod
    def from_dict(cls, data):
        mgr = cls()
        for name, data in data.items():
            col = AbstractColumn.from_data(data)
            mgr.add_column(col=col, name=name)
        return mgr

    def write(self, path: str):
        meta = {"dtype": BlockManager, "columns": {}}

        # prepare directories
        os.makedirs(path, exist_ok=True)
        block_dirs = os.path.join(path, "blocks")
        os.makedirs(block_dirs)
        columns_dir = os.path.join(path, "columns")
        os.makedirs(columns_dir)

        # consolidate before writing
        self.consolidate()
        for block_id, block_ref in self._block_refs.items():
            block: AbstractBlock = block_ref.block
            block_dir = os.path.join(block_dirs, str(block_id))
            block.write(block_dir)

            for name, column in block_ref.items():
                column_dir = os.path.join(columns_dir, name)
                os.makedirs(column_dir, exist_ok=True)

                # don't write the data, reference the block
                meta["columns"][name] = {
                    **column._get_meta(),
                    "block": {
                        "block_dir": block_dir,
                        "block_index": _serialize_block_index(column._block_index),
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
        meta_path = os.path.join(path, "meta.yaml")
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
            yaml.load(open(os.path.join(path, "meta.yaml")), Loader=yaml.FullLoader)
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
                        block_meta["block_dir"]
                    )
                block = blocks[block_meta["block_dir"]]

                # read column, passing in a block_view
                col = col_meta["dtype"].read(
                    column_dir,
                    _data=block[_deserialize_block_index(block_meta["block_index"])],
                    _meta=col_meta,
                )
                mgr.add_column(col, name)
            else:
                mgr.add_column(
                    col_meta["dtype"].read(path=column_dir, _meta=col_meta), name
                )

        return mgr

    def _repr_pandas_(self):
        dfs = []
        cols = set(self._columns.keys())
        for _, block_ref in self._block_refs.items():
            if hasattr(block_ref.block, "_repr_pandas_"):
                dfs.append(block_ref.block._repr_pandas_(block_ref))
                cols -= set(block_ref.keys())
        dfs.append(pd.DataFrame({k: self[k]._repr_pandas_() for k in cols}))
        return pd.concat(objs=dfs, axis=1)

    def view(self):
        mgr = BlockManager()
        for name, col in self._columns.items():
            mgr.add_column(col.view(), name)
        return mgr

    def copy(self):
        mgr = BlockManager()
        for name, col in self._columns.items():
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
