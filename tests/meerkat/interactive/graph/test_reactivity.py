from typing import List, Mapping

import numpy as np
import pandas as pd
import pytest

import meerkat as mk
from meerkat.interactive.graph import is_unmarked_context, reactive, trigger
from meerkat.interactive.graph.magic import magic
from meerkat.interactive.graph.store import _unpack_stores_from_object
from meerkat.interactive.modification import DataFrameModification
from meerkat.mixins.reactifiable import MarkableMixin
from meerkat.state import state


def _create_dummy_reactive_df() -> mk.DataFrame:
    df = pd.DataFrame({"a": np.arange(10), "b": np.arange(10) + 10})
    return mk.DataFrame.from_pandas(df).mark()


@reactive()
def _add_to_list(_keys: List[str], new_key: str):
    return _keys + [new_key]


def test_react_basic():
    df = _create_dummy_reactive_df()

    keys_reactive = df.keys()
    out = _add_to_list(keys_reactive, "c")

    assert out.inode is not None
    assert isinstance(keys_reactive, mk.Store)
    assert keys_reactive.inode.has_trigger_children()
    op_node = keys_reactive.inode.trigger_children[0]
    assert op_node.obj.fn.__name__ == "_add_to_list"
    assert len(op_node.trigger_children) == 1
    assert op_node.trigger_children[0] is out.inode

    # Outside of context manager.
    with mk.unmarked():
        keys = df.keys()
    assert not isinstance(keys, mk.Store)


def test_unmarked_context_manager():
    df = _create_dummy_reactive_df()

    assert not is_unmarked_context()
    keys_reactive = df.keys()
    with mk.unmarked():
        assert is_unmarked_context()
        keys = df.keys()

    assert isinstance(keys_reactive, mk.Store)
    assert isinstance(keys, List)


def test_trigger_instance_method():
    rng = np.random.RandomState(0)

    # TODO: Why is this decorator affecting the return type?
    @reactive()
    def _subselect_df(df: mk.DataFrame) -> mk.DataFrame:
        cols = list(rng.choice(df.columns, 3))
        return df[cols]

    df = mk.DataFrame({str(k): [k] for k in range(10)})
    df = df.mark()

    df_sub = _subselect_df(df)
    keys_reactive = df_sub.keys()
    keys0 = keys_reactive.__wrapped__

    state.modification_queue.queue = [DataFrameModification(id=df.id, scope=[])]
    modifications1 = trigger()
    keys1 = modifications1[-1].value

    state.modification_queue.queue = [DataFrameModification(id=df.id, scope=[])]
    modifications2 = trigger()
    keys2 = modifications2[-1].value

    assert keys0 != keys1
    assert keys1 != keys2


@pytest.mark.parametrize("is_unmarked", [False, True])
def test_unmarked_on_reactive_fn(is_unmarked: bool):
    @mk.gui.reactive()
    def add(a, b):
        return a + b

    a = mk.Store(1)
    b = mk.Store(2)
    if is_unmarked:
        with mk.unmarked():
            c = add(a, b)
    else:
        c = add(a, b)

    expected_type = int if is_unmarked else mk.Store
    assert isinstance(c, expected_type)

    if not is_unmarked:
        assert a.inode.has_trigger_children() and b.inode.has_trigger_children()
    else:
        assert a.inode is None and b.inode is None


def test_default_nested_return():
    """By default, nested return is None."""

    @reactive()
    def _return_tuple(_s):
        return ("a", "b")

    @reactive()
    def _return_list(_s):
        return ["a", "b"]

    _s = mk.Store("")
    with magic():
        out = _return_tuple(_s)
    with mk.unmarked():
        a, b = out
    assert isinstance(out, mk.Store)
    assert not isinstance(a, mk.Store)
    assert not isinstance(b, mk.Store)

    with magic():
        out = _return_list(_s)
    with mk.unmarked():
        a, b = out
    # Lists are not unpacked by default.
    assert isinstance(out, mk.Store)
    assert not isinstance(a, mk.Store)
    assert not isinstance(b, mk.Store)


def test_nested_reactive_fns():
    """When reactive functions are executed, only the outer function should be
    added as a child to the input stores.

    In simpler language, a reactive function run inside another reactive
    function will not add things to the graph.
    """

    @mk.gui.reactive()
    def _inner(x):
        return ["a", "b", x]

    @mk.gui.reactive()
    def _outer(x):
        return ["example"] + _inner(x)

    x = mk.Store("c")
    _outer(x)

    assert x.inode.has_trigger_children()
    assert len(x.inode.trigger_children) == 1
    # Compare the names because the memory addresses will be different
    # when the function is wrapped in reactive.
    assert x.inode.trigger_children[0].obj.fn.__name__ == "_outer"


@pytest.mark.parametrize(
    "x",
    [
        # Primitives.
        1,
        "foo",
        [1, 2],
        (1, 4),
        {"a": 1, "b": 2},
        # Basic types.
        mk.Store(1),
        # TODO: Determine why the initialization below is causing problems.
        # mk.Store("foo"),
        mk.Store([1, 2]),
        mk.Store((1, 4)),
        mk.Store({"a": 1, "b": 2}),
        # # Stores in non-reactive containers.
        {"a": 1, "b": mk.Store(2)},
        [1, mk.Store(2)],
        (mk.Store(1), 2),
        {"a": {"b": mk.Store(1)}},
        # # Nested stores.
        mk.Store([mk.Store(1), 2]),
    ],
)
@pytest.mark.parametrize("use_kwargs", [False, True])
def test_unpacking(x, use_kwargs):
    """Test that all stores are unpacked correctly."""

    def _are_equal(x, y):
        if isinstance(x, Mapping):
            return _are_equal(x.keys(), y.keys()) and all(
                _are_equal(x[k], y[k]) for k in x.keys()
            )
        else:
            return x == y

    if use_kwargs:
        inputs = {"wrapped": x}
        unpacked_kwargs, _ = _unpack_stores_from_object(inputs)
        assert len(unpacked_kwargs) == 1
        outputs = unpacked_kwargs
    else:
        inputs = [x]
        unpacked_args, _ = _unpack_stores_from_object(inputs)
        assert len(unpacked_args) == 1
        outputs = unpacked_args

    # Recursively check for equality.
    assert _are_equal(inputs, outputs)


def test_instance_methods():
    """Test that instance methods get reactified correctly."""

    class Foo:
        def __init__(self, x):
            self.x = x

        @reactive()
        def add(self, y):
            return self.x + y

        @reactive()
        def __eq__(self, __o: int) -> bool:
            return self.x == __o

    foo = Foo(1)
    val = mk.Store(2)
    out_add = foo.add(val)
    out_eq = foo == val
    assert isinstance(out_add, mk.Store)
    assert isinstance(out_eq, mk.Store)

    assert len(val.inode.trigger_children) == 2
    assert val.inode.trigger_children[0].obj.fn.__name__ == "add"
    assert val.inode.trigger_children[0].trigger_children[0] is out_add.inode
    assert val.inode.trigger_children[1].obj.fn.__name__ == "__eq__"
    assert val.inode.trigger_children[1].trigger_children[0] is out_eq.inode

    # Trigger the functions.
    @mk.endpoint()
    def set_val(val: mk.Store):
        val.set(0)

    set_val(val)
    assert out_add == 1
    assert not out_eq


@pytest.mark.parametrize(
    "x",
    [
        [1, 2, 3, 4],
        mk.Store([1, 2, 3, 4]),
        mk.DataFrame({"a": [1, 2, 3, 4]}),
    ],
)
@pytest.mark.parametrize("mark", [True, False])
def test_slicing(x, mark: bool):
    @mk.endpoint()
    def update_store(store: mk.Store, value: int):
        store.set(value)

    @mk.unmarked()
    def _compare_objs(x_sl, expected):
        if isinstance(x_sl, mk.DataFrame):
            assert x_sl.columns == expected.columns
            for col in x_sl.columns:
                assert np.all(x_sl[col] == expected[col])
        else:
            assert x_sl == expected

    if mark:
        if not isinstance(x, MarkableMixin):
            x = mk.Store(x)
        x.mark()
    elif isinstance(x, MarkableMixin):
        x.unmark()

    start = mk.Store(0)
    stop = mk.Store(4)
    step = mk.Store(1)

    # Using store slices with non-markable objects should raise an error.
    # This is because __index__ is reactive, which raises an error.
    if not isinstance(x, MarkableMixin):
        with pytest.raises(TypeError):
            x_sl = x[start:stop]
        return

    x_sl = x[start:stop:step]
    _compare_objs(x_sl, x[0:4])

    if not x.marked:
        # Even if x is not marked, an inode should still be created.
        assert x.inode is not None
        return

    inode = x_sl.inode

    # Update the start value.
    update_store.run(start, 1)
    _compare_objs(inode.obj, x[1:4])

    # Update the stop value.
    update_store.run(stop, 3)
    _compare_objs(inode.obj, x[1:3])

    # Update the step value.
    update_store.run(start, 0)
    update_store.run(stop, 4)
    update_store.run(step, 2)
    _compare_objs(inode.obj, x[0:4:2])
