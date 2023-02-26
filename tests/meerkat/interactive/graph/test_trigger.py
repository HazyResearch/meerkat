import numpy as np
import pytest

import meerkat as mk
from meerkat.interactive import endpoint
from meerkat.interactive.graph.reactivity import reactive


@reactive()
def binary_op(df_1: mk.DataFrame, df_2: mk.DataFrame) -> mk.DataFrame:
    return mk.DataFrame({"a": df_1["a"] + df_2["a"]})


@reactive()
def unary_op(df_1) -> mk.DataFrame:
    return mk.DataFrame({"a": df_1["a"] * 3})


@endpoint()
def update_df(df: mk.DataFrame, col: str, value: np.ndarray) -> mk.DataFrame:
    df[col] = value
    return df


@reactive()
def _add(a, b):
    return a + b


# @endpoint()
# def _set_store(store: mk.Store, value, _check_equality=False):
#     # We have to explicitly check ifthe value is the same.
#     if _check_equality and store == value:
#         return
#     store.set(value)


@endpoint()
def _set_store(store: mk.Store, value):
    # We have to explicitly check ifthe value is the same.
    store.set(value)


def test_trigger():
    df_1 = mk.DataFrame({"a": np.arange(10)}).mark()
    df_2 = mk.DataFrame({"a": np.arange(10)}).mark()

    derived_1 = binary_op(df_1, df_2)
    derived_2 = unary_op(derived_1)
    derived_3 = binary_op(derived_1, derived_2)
    derived_4 = binary_op(derived_3, df_2)

    # Unpack the node from the output dataframes.
    derived_1_node = derived_1.inode
    derived_2_node = derived_2.inode
    derived_3_node = derived_3.inode
    derived_4_node = derived_4.inode

    # Update the values of df_1 and df_2.
    update_df(df_1, "a", np.arange(10, 20))
    update_df(df_2, "a", np.arange(10, 20))

    # The node is attached to different dataframes on trigger.
    # So we need to fetch the updated dataframe associated with the node.
    derived_1 = derived_1_node.obj
    derived_2 = derived_2_node.obj
    derived_3 = derived_3_node.obj
    derived_4 = derived_4_node.obj

    # assert len(modifications) == 6
    assert (derived_1["a"] == np.arange(10, 20) * 2).all()
    assert (derived_2["a"] == derived_1["a"] * 3).all()
    assert (derived_3["a"] == derived_2["a"] + derived_1["a"]).all()
    assert (derived_4["a"] == derived_3["a"] + np.arange(10, 20)).all()


# TODO: fix the test when we resolve endpoint partialing issue.
@pytest.mark.parametrize("check_equality", [False])
@pytest.mark.parametrize("toggle_mark", [None, "a", "b"])
def test_trigger_hybrid_marked_unmarked_inputs(check_equality: bool, toggle_mark: str):
    """Test trigger functionality when some inputs are marked and some are
    not."""

    a = mk.Store(1)
    b = mk.Store(2).unmark()

    c = _add(a, b)

    if toggle_mark == "a":
        a.unmark()
    elif toggle_mark == "b":
        b.mark()

    assert c == 3

    assert a.inode is not None
    assert b.inode is not None

    assert len(a.inode.trigger_children) == 1
    assert len(b.inode.trigger_children) == 0

    op_inode = a.inode.trigger_children[0]
    assert op_inode.obj.fn.__name__ == "_add"
    assert list(op_inode.obj.args) == [a.inode, b.inode]

    # a was marked on operation construction.
    # changing it should trigger the operation.
    _set_store(a, 2)
    assert c == 4

    # b was not marked on operation construction.
    # changing it should not trigger the operation.
    _set_store(b, 10)
    assert c == 4

    # Changing a will retrigger the operation.
    # But the value of b was updated right before, so the operation
    # should use the new value of b.
    _set_store(a, 3)
    assert c == 13

    # Changing b will not retrigger the operation.
    _set_store(b, 0)
    assert c == 13

    # If the endpoint does not issue a modification (i.e. store.set is not called),
    # then the operation should not be triggered.
    # When check_equality is True, .set is only called when the new value is different
    # from the old value.
    _set_store(a, a.value)
    if check_equality:
        assert c == 13
    else:
        assert c == 3


def test_trigger_unmarked_inputs():
    a = mk.Store(1).unmark()
    b = mk.Store(2).unmark()

    c = _add(a, b)
    assert c == 3

    # When all inputs are unmarked, we shouldn't create nodes
    # unnecessarily.
    assert a.inode is None
    assert b.inode is None
