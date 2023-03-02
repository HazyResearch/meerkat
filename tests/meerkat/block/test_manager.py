import os
from itertools import product

import numpy as np
import pytest
import torch

import meerkat as mk
from meerkat.block.manager import BlockManager
from meerkat.tools.utils import load_yaml

from ...utils import product_parametrize


def test_consolidate_no_op():
    mgr = BlockManager()
    col1 = mk.TensorColumn(data=np.arange(10))
    mgr.add_column(col1, "a")
    col2 = mk.TensorColumn(np.arange(10) * 2)
    mgr.add_column(col2, "b")
    col2 = mk.TensorColumn(np.arange(10, dtype=float) * 2)
    mgr.add_column(col2, "c")
    block_ref = mgr.get_block_ref("c")

    assert len(mgr._block_refs) == 3
    mgr.consolidate()
    assert len(mgr._block_refs) == 2

    # assert that the block_ref hasn't changed for the isolated block ref
    assert mgr.get_block_ref("c") is block_ref


def test_consolidate():
    mgr = BlockManager()

    col1 = mk.TensorColumn(data=np.arange(10))
    mgr.add_column(col1, "col1")
    col2 = mk.TensorColumn(np.arange(10) * 2)
    mgr.add_column(col2, "col2")
    col3 = mk.ScalarColumn(np.arange(10) * 3)
    mgr.add_column(col3, "col3")
    col4 = mk.ScalarColumn(np.arange(10) * 4)
    mgr.add_column(col4, "col4")
    col5 = mk.TensorColumn(torch.arange(10) * 5)
    mgr.add_column(col5, "col5")
    col6 = mk.TensorColumn(torch.arange(10) * 6)
    mgr.add_column(col6, "col6")
    col9 = mk.TensorColumn(torch.ones(10, 5).to(int) * 9)
    mgr.add_column(col9, "col9")

    assert len(mgr._block_refs) == 7
    mgr.consolidate()
    assert len(mgr._block_refs) == 3

    # check that the same object backs both the block and the column
    for name, col in [("col1", col1), ("col2", col2)]:
        assert mgr[name].data.base is mgr.get_block_ref(name).block.data
        assert (mgr[name] == col).all()

    # check that the same object backs both the block and the column
    for name, col in [("col3", col3), ("col4", col4)]:
        assert mgr[name].data is mgr.get_block_ref(name).block.data[name]
        assert (mgr[name] == col).all()

    # check that the same object backs both the bock
    for name, col in [("col5", col5), ("col6", col6)]:
        # TODO (sabri): Figure out a way to check this for tensors
        assert (mgr[name] == col).all()


def test_consolidate_multiple_types():
    mgr = BlockManager()

    for dtype in [int, float]:
        for idx in range(3):
            col = mk.TensorColumn(np.arange(10, dtype=dtype))
            mgr.add_column(col, f"col{idx}_{dtype}")
    mgr.add_column(mk.ScalarColumn(np.arange(10) * 4), "col4_pandas")
    mgr.add_column(mk.ScalarColumn(np.arange(10) * 5), "col5_pandas")

    assert len(mgr._block_refs) == 8
    mgr.consolidate()
    assert len(mgr._block_refs) == 3


def test_consolidate_preserves_order():
    mgr = BlockManager()

    col1 = mk.TensorColumn(data=np.arange(10))
    mgr.add_column(col1, "col1")
    col2 = mk.TensorColumn(np.arange(10) * 2)
    mgr.add_column(col2, "col2")
    col3 = mk.ScalarColumn(np.arange(10) * 3)
    mgr.add_column(col3, "col3")

    order = ["col2", "col3", "col1"]
    mgr.reorder(order)
    assert list(mgr.keys()) == order
    mgr.consolidate()
    assert list(mgr.keys()) == order


@pytest.mark.parametrize(
    "num_blocks, consolidated",
    product([1, 2, 3], [True, False]),
)
def test_apply_get_multiple(num_blocks, consolidated):
    mgr = BlockManager()

    for dtype in [int, float]:
        for idx in range(num_blocks):
            col = mk.TensorColumn(np.arange(10, dtype=dtype) * idx)
            mgr.add_column(col, f"col{idx}_{dtype}")
    if consolidated:
        mgr.consolidate()

    for slc in [
        slice(2, 6, 1),
        slice(0, 1, 1),
        slice(2, 8, 3),
        np.array([1, 4, 6]),
        np.array([True, False] * 5),
    ]:
        new_mgr = mgr.apply(method_name="_get", index=slc)
        assert isinstance(new_mgr, BlockManager)
        for dtype in [int, float]:
            for idx in range(num_blocks):
                # check it's equivalent to applying the slice to each column in turn
                assert (
                    new_mgr[f"col{idx}_{dtype}"].data
                    == mgr[f"col{idx}_{dtype}"][slc].data
                ).all()

                # check that the base is the same (since we're just slicing)
                assert (
                    new_mgr[f"col{idx}_{dtype}"].data.base
                    is mgr[f"col{idx}_{dtype}"][slc].data.base
                ) == isinstance(slc, slice)


@pytest.mark.parametrize(
    "num_blocks, consolidated",
    product([1, 2, 3], [True, False]),
)
def test_apply_get_single(num_blocks, consolidated):
    mgr = BlockManager()

    for dtype in [int, float]:
        for idx in range(num_blocks):
            col = mk.TensorColumn(np.arange(10, dtype=dtype) * idx)
            mgr.add_column(col, f"col{idx}_{dtype}")
    if consolidated:
        mgr.consolidate()

    for slc in [0, 8]:
        result_dict = mgr.apply(method_name="_get", index=slc)
        assert isinstance(result_dict, dict)
        for dtype in [int, float]:
            for idx in range(num_blocks):
                # check it's equivalent to applying the slice to each column in turn
                assert result_dict[f"col{idx}_{dtype}"] == mgr[f"col{idx}_{dtype}"][slc]


@pytest.fixture()
def call_count(monkeypatch):
    from meerkat import TensorColumn
    from meerkat.block.numpy_block import NumPyBlock

    calls = {"count": 0}

    block_get = NumPyBlock._get

    def patched_get(self, *args, **kwargs):
        nonlocal calls
        calls["count"] += 1
        return block_get(self, *args, **kwargs)

    monkeypatch.setattr(NumPyBlock, "_get", patched_get)

    col_get = TensorColumn._get

    def patched_get_col(self, *args, **kwargs):
        nonlocal calls
        calls["count"] += 1
        return col_get(self, *args, **kwargs)

    monkeypatch.setattr(TensorColumn, "_get", patched_get_col)

    return calls


@product_parametrize({"consolidated": [True, False]})
def test_apply_get_single_lambda(call_count, consolidated):
    mgr = BlockManager()
    base_col = mk.TensorColumn(np.arange(10))
    mgr.add_column(base_col, "a")
    # lambda_column = base_col.defer(lambda x: x + 2)
    # mgr.add_column(lambda_column, "b")

    if consolidated:
        mgr.consolidate()

    mgr.apply(method_name="_get", index=1, materialize=True)

    # we should only call NumpyBlock._get once
    assert call_count["count"] == 1


@product_parametrize({"consolidated": [True, False]})
def test_apply_get_multiple_lambda(call_count, consolidated):
    mgr = BlockManager()

    base_col = mk.TensorColumn(np.arange(10))
    mgr.add_column(base_col, "a")
    lambda_column = base_col.defer(lambda x: x + 2)
    mgr.add_column(lambda_column, "b")

    if consolidated:
        mgr.consolidate()

    new_mgr = mgr.apply(method_name="_get", index=[1, 3, 5], materialize=False)

    # the columns should stil be linked after an index
    assert new_mgr["b"].data.args[0] is new_mgr["a"]

    # we should only call NumpyBlock._get once
    assert call_count["count"] == 1


@pytest.mark.parametrize(
    "consolidated",
    [True, False],
)
def test_remove(consolidated):
    mgr = BlockManager()
    col = mk.TensorColumn(np.arange(10))
    mgr.add_column(col, "a")
    col = mk.TensorColumn(np.arange(10))
    mgr.add_column(col, "b")

    if consolidated:
        mgr.consolidate()

    assert len(mgr) == 2
    assert len(mgr._block_refs) == 1 if consolidated else 2
    mgr.remove("a")
    assert len(mgr) == 1
    assert list(mgr.keys()) == ["b"]
    assert len(mgr._block_refs) == 1

    with pytest.raises(
        expected_exception=ValueError,
        match="Remove failed: no column 'c' in BlockManager.",
    ):
        mgr.remove("c")


def test_getitem():
    mgr = BlockManager()
    a = mk.TensorColumn(np.arange(10))
    mgr.add_column(a, "a")
    b = mk.TensorColumn(np.arange(10))
    mgr.add_column(b, "b")

    # check that manager holds coreference of original column, but returns a coreference
    assert mgr["a"] is a
    assert mgr["a"] is mgr["a"]

    with pytest.raises(
        ValueError,
        match="Unsupported index of type `<class 'int'>` passed to `BlockManager`.",
    ):
        mgr[0]

    out = mgr[["a", "b"]]
    assert isinstance(out, BlockManager)
    # check that manager holds reference of original column, and returns a coreference
    assert mgr["a"].data.base is out["a"].data.base
    assert mgr["a"] is out["a"]
    assert out["a"] is out["a"]

    with pytest.raises(ValueError, match="`BlockManager` does not contain column 'c'."):
        mgr[["a", "c"]]


def test_setitem():
    mgr = BlockManager()
    a = mk.TensorColumn(np.arange(10))
    mgr["a"] = a
    b = mk.TensorColumn(np.arange(10)) * 2
    mgr["b"] = b

    # check that manager holds coreference of original column, and returns a coreference
    assert mgr["a"] is a
    assert mgr["a"] is mgr["a"]

    with pytest.raises(
        ValueError,
        match="Cannot set item with object of type `<class 'int'>` on `BlockManager`.",
    ):
        mgr["a"] = 1


def test_contains():
    mgr = BlockManager()
    col = mk.TensorColumn(np.arange(10))
    mgr.add_column(col, "a")
    col = mk.TensorColumn(np.arange(10))
    mgr.add_column(col, "b")

    assert "a" in mgr
    assert "b" in mgr
    assert "c" not in mgr


@pytest.mark.parametrize(
    "num_blocks, consolidated",
    product([1, 2, 3], [True, False]),
)
def test_len(num_blocks, consolidated):
    mgr = BlockManager()
    for dtype in [int, float]:
        for idx in range(num_blocks):
            col = mk.TensorColumn(np.arange(10, dtype=dtype) * idx)
            mgr.add_column(col, f"col{idx}_{dtype}")

    if consolidated:
        mgr.consolidate()

    assert len(mgr) == num_blocks * 2


def test_io(tmpdir):
    tmpdir = os.path.join(tmpdir, "test")
    mgr = BlockManager()

    col1 = mk.TensorColumn(data=np.arange(10))
    mgr.add_column(col1, "col1")
    col2 = mk.TensorColumn(np.arange(10) * 2)
    mgr.add_column(col2, "col2")
    col3 = mk.ScalarColumn(np.arange(10) * 3)
    mgr.add_column(col3, "col3")
    col4 = mk.ScalarColumn(np.arange(10) * 4)
    mgr.add_column(col4, "col4")
    col5 = mk.TensorColumn(torch.arange(10) * 5)
    mgr.add_column(col5, "col5")
    col6 = mk.TensorColumn(torch.arange(10) * 6)
    mgr.add_column(col6, "col6")
    col7 = mk.ObjectColumn(list(range(10)))
    mgr.add_column(col7, "col7")
    col8 = mk.ObjectColumn(list(range(10)))
    mgr.add_column(col8, "col8")
    col9 = mk.TensorColumn(torch.ones(10, 5).to(int) * 9)
    mgr.add_column(col9, "col9")

    assert len(mgr._block_refs) == 7
    mgr.write(tmpdir)
    new_mgr = BlockManager.read(tmpdir)
    assert len(new_mgr._block_refs) == 3

    for idx in range(1, 7):
        assert (mgr[f"col{idx}"] == new_mgr[f"col{idx}"]).all()

    for idx in range(7, 8):
        assert mgr[f"col{idx}"].data == new_mgr[f"col{idx}"].data

    # test overwriting
    col1 = mk.TensorColumn(data=np.arange(10) * 100)
    mgr.add_column(col1, "col1")
    mgr.remove("col9")
    assert "col9" in load_yaml(os.path.join(tmpdir, "meta.yaml"))["columns"]
    mgr.write(tmpdir)
    # make sure the old column was removed
    assert "col9" not in load_yaml(os.path.join(tmpdir, "meta.yaml"))["columns"]
    new_mgr = BlockManager.read(tmpdir)
    assert len(new_mgr._block_refs) == 3

    for idx in range(1, 7):
        assert (mgr[f"col{idx}"] == new_mgr[f"col{idx}"]).all()

    for idx in range(7, 8):
        assert mgr[f"col{idx}"].data == new_mgr[f"col{idx}"].data


def test_io_no_overwrite(tmpdir):
    new_dir = os.path.join(tmpdir, "test")
    os.mkdir(new_dir)
    mgr = BlockManager()

    with pytest.raises(
        IsADirectoryError,
        match=f"Cannot write `BlockManager`. {new_dir} is a directory.",
    ):
        mgr.write(new_dir)


@product_parametrize(
    {
        "column_type": [
            mk.TensorColumn,
            mk.PandasScalarColumn,
            mk.ArrowScalarColumn,
            mk.TensorColumn,
        ],
        "column_order": [("z", "a"), ("a", "z")],
    }
)
def test_io_lambda_args(tmpdir, column_type, column_order):
    mgr = BlockManager()
    base_col_name, col_name = column_order
    base_col = column_type(np.arange(16))
    mgr.add_column(base_col, base_col_name)  # want to order backwards
    lambda_column = base_col.defer(lambda x: x + 2)
    mgr.add_column(lambda_column, col_name)
    mgr.write(os.path.join(tmpdir, "test"))
    new_mgr = BlockManager.read(os.path.join(tmpdir, "test"))

    # ensure that in the loaded df, the lambda column points to the same
    # underlying data as the base column
    assert new_mgr[col_name].data.args[0] is new_mgr[base_col_name]

    # ensure that the the base column was not written twice
    # check that dir is empty
    block_id = mgr._column_to_block_id[col_name]
    assert not os.listdir(
        os.path.join(tmpdir, "test", f"blocks/{block_id}", "data.op/args")
    )
    assert not os.listdir(
        os.path.join(tmpdir, "test", f"blocks/{block_id}", "data.op/kwargs")
    )


@product_parametrize(
    {
        "column_type": [
            mk.NumPyTensorColumn,
            mk.PandasScalarColumn,
            mk.ArrowScalarColumn,
            mk.TorchTensorColumn,
        ]
    }
)
def test_io_chained_lambda_args(tmpdir, column_type):
    mgr = BlockManager()
    base_col = column_type(np.arange(16))
    mgr.add_column(base_col, "a")
    lambda_column = base_col.defer(lambda x: x + 2)
    mgr.add_column(lambda_column, "b")
    second_lambda_column = lambda_column.defer(lambda x: x + 2)
    mgr.add_column(second_lambda_column, "c")
    mgr.write(os.path.join(tmpdir, "test"))
    new_mgr = BlockManager.read(os.path.join(tmpdir, "test"))

    # ensure that in the loaded df, the lambda column points to the same
    # underlying data as the base column
    # TODO: this should work once we get topological sort correct
    assert new_mgr["c"].data.args[0] is new_mgr["b"]

    # ensure that the the base column was not written twice
    # check that dir is empty
    block_id = mgr._column_to_block_id["c"]
    assert not os.listdir(
        os.path.join(tmpdir, "test", f"blocks/{block_id}", "data.op/args")
    )
    assert not os.listdir(
        os.path.join(tmpdir, "test", f"blocks/{block_id}", "data.op/kwargs")
    )


def test_topological_block_refs():
    mgr = BlockManager()
    base_col = mk.TensorColumn(np.arange(16))

    lambda_columns = []
    expected_order = [id(base_col._block)]
    curr_col = base_col
    for _ in range(10):
        curr_col = curr_col.defer(lambda x: x + 2)
        expected_order.append(id(curr_col._block))
        lambda_columns.append(curr_col)

    # add to manager in reversed order
    for i, col in enumerate(lambda_columns[::-1]):
        mgr.add_column(col, f"lambda_{i}")
    mgr.add_column(base_col, "base")

    sorted_block_refs = list(list(zip(*mgr.topological_block_refs()))[0])

    assert sorted_block_refs == expected_order


def test_topological_block_refs_w_gap():
    mgr = BlockManager()
    base_col = mk.TensorColumn(np.arange(16))

    lambda_columns = []
    curr_col = base_col
    for _ in range(10):
        curr_col = curr_col.defer(lambda x: x + 2)
        lambda_columns.append(curr_col)

    mgr.add_column(lambda_columns[0], "first")
    mgr.add_column(lambda_columns[-2], "second_to_last")
    mgr.add_column(lambda_columns[-1], "last")
    mgr.add_column(base_col, "base")

    expected_order = [
        id(base_col._block),
        id(lambda_columns[0]._block),
        id(lambda_columns[-2]._block),
        id(lambda_columns[-1]._block),
    ]

    sorted_block_refs = list(list(zip(*mgr.topological_block_refs()))[0])

    # because there is a gap, we cannot guarantee the global order of the blocks
    # at some point, we may want to support this, but for the time being we don't
    # need to support this
    assert sorted_block_refs.index(id(base_col._block)) < sorted_block_refs.index(
        id(lambda_columns[0]._block)
    )
    assert sorted_block_refs.index(
        id(lambda_columns[-2]._block)
    ) < sorted_block_refs.index(id(lambda_columns[-1]._block))
    assert len(sorted_block_refs) == len(expected_order)
