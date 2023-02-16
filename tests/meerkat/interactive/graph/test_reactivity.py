from typing import List, Mapping

import numpy as np
import pandas as pd
import pytest

import meerkat as mk
from meerkat.interactive.graph import is_unmarked_context, reactive, trigger
from meerkat.interactive.graph.magic import magic
from meerkat.interactive.graph.store import _unpack_stores_from_object
from meerkat.interactive.modification import DataFrameModification
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
    assert isinstance(keys_reactive, mk.gui.Store)
    assert keys_reactive.inode.has_trigger_children()
    op_node = keys_reactive.inode.trigger_children[0]
    assert op_node.obj.fn.__name__ == "_add_to_list"
    assert len(op_node.trigger_children) == 1
    assert op_node.trigger_children[0] is out.inode

    # Outside of context manager.
    with mk.unmarked():
        keys = df.keys()
    assert not isinstance(keys, mk.gui.Store)


def test_unmarked_context_manager():
    df = _create_dummy_reactive_df()

    assert not is_unmarked_context()
    keys_reactive = df.keys()
    with mk.gui.unmarked():
        assert is_unmarked_context()
        keys = df.keys()

    assert isinstance(keys_reactive, mk.gui.Store)
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

    a = mk.gui.Store(1)
    b = mk.gui.Store(2)
    if is_unmarked:
        with mk.gui.unmarked():
            c = add(a, b)
    else:
        c = add(a, b)

    expected_type = int if is_unmarked else mk.gui.Store
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

    _s = mk.gui.Store("")
    with magic():
        out = _return_tuple(_s)
    a, b = out
    assert not isinstance(out, mk.gui.Store)
    assert isinstance(a, mk.gui.Store)
    assert isinstance(b, mk.gui.Store)

    with magic():
        out = _return_list(_s)
    a, b = out
    # Lists are not unpacked by default.
    assert isinstance(out, mk.gui.Store)
    assert not isinstance(a, mk.gui.Store)
    assert not isinstance(b, mk.gui.Store)


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

    x = mk.gui.Store("c")
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
        mk.gui.Store(1),
        # TODO: Determine why the initialization below is causing problems.
        # mk.gui.Store("foo"),
        mk.gui.Store([1, 2]),
        mk.gui.Store((1, 4)),
        mk.gui.Store({"a": 1, "b": 2}),
        # # Stores in non-reactive containers.
        {"a": 1, "b": mk.gui.Store(2)},
        [1, mk.gui.Store(2)],
        (mk.gui.Store(1), 2),
        {"a": {"b": mk.gui.Store(1)}},
        # # Nested stores.
        mk.gui.Store([mk.gui.Store(1), 2]),
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
    val = mk.gui.Store(2)
    out_add = foo.add(val)
    out_eq = foo == val
    assert isinstance(out_add, mk.gui.Store)
    assert isinstance(out_eq, mk.gui.Store)

    assert len(val.inode.trigger_children) == 2
    assert val.inode.trigger_children[0].obj.fn.__name__ == "add"
    assert val.inode.trigger_children[0].trigger_children[0] is out_add.inode
    assert val.inode.trigger_children[1].obj.fn.__name__ == "__eq__"
    assert val.inode.trigger_children[1].trigger_children[0] is out_eq.inode

    # Trigger the functions.
    @mk.endpoint()
    def set_val(val: mk.gui.Store):
        val.set(0)

    set_val(val)
    assert out_add == 1
    assert not out_eq
