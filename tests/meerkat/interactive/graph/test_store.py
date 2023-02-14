from typing import Iterator, Tuple

import pytest

import meerkat as mk
from meerkat.interactive.graph.magic import magic


@pytest.mark.parametrize("is_magic", [False, True])
def test_store_reactive_math(is_magic: bool):
    """Test basic math methods are reactive.

    A method is reactive if it:
        1. Returns a Store
        2. Creates a connection based on the op.
    """
    store = mk.gui.Store(1)

    expected = {
        "add": 2,
        "sub": 0,
        "mul": 1,
        "truediv": 1,
        "floordiv": 1,
        "mod": 0,
        "divmod": (1, 0),
        "pow": 1,
        "neg": -1,
        "pos": 1,
        "abs": 1,
        "lt": False,
        "le": True,
        "eq": True,
        "ne": False,
        "gt": False,
        "ge": True,
    }

    out = {}
    with magic(magic=is_magic):
        out["add"] = store + 1
        out["sub"] = store - 1
        out["mul"] = store * 1
        out["truediv"] = store.__truediv__(1)
        out["floordiv"] = store // 1
        out["mod"] = store % 1
        out["divmod"] = divmod(store, 1)
        out["pow"] = store**1
        out["neg"] = -store
        out["pos"] = +store
        out["abs"] = abs(store)
        out["lt"] = store < 1
        out["le"] = store <= 1
        out["eq"] = store == 1
        out["ne"] = store != 1
        out["gt"] = store > 1
        out["ge"] = store >= 1

    op_children = {}
    if is_magic:
        assert len(store.inode.trigger_children) == len(expected)
        op_children = {c.obj.fn.__name__: c for c in store.inode.trigger_children}
    else:
        assert store.inode is None

    for k, v in out.items():
        if is_magic:
            assert isinstance(v, mk.gui.Store)
            op_name = f"__{k}__"
            op_inode = op_children[op_name]
            assert len(op_inode.trigger_children) == 1
            assert op_inode.trigger_children[0] is v.inode
        else:
            # If magic is not on, the output should still be a marked Store.
            # However nothing should be on the graph.
            assert isinstance(v, mk.gui.Store)
            assert v.inode is None

        assert v == expected[k]


@pytest.mark.parametrize("other", [1, 2])
def test_store_imethod(other):
    """Test traditional inplace methods are reactive, but return different
    stores."""
    store = original = mk.gui.Store(1)

    with pytest.warns(UserWarning):
        expected = {
            "__iadd__": store + other,
            "__isub__": store - other,
            "__imul__": store * other,
            "__itruediv__": store.__itruediv__(other),
            "__ifloordiv__": store // other,
            "__imod__": store % other,
            "__ipow__": store**other,
            "__ilshift__": store << other,
            "__irshift__": store >> other,
            "__iand__": store & other,
            "__ixor__": store ^ other,
            "__ior__": store | other,
        }

    out = {}
    with mk.gui.reactive():
        for k in expected:
            with pytest.warns(UserWarning):
                out[k] = getattr(store, k)(other)

    for k, v in out.items():
        assert isinstance(v, mk.gui.Store), f"{k} did not return a Store."
        assert id(v) != id(original), f"{k} did not return a new Store."
        assert v == expected[k], f"{k} did not return the correct value."


@pytest.mark.parametrize("react", [False, True])
def test_store_as_iterator(react: bool):
    store = mk.gui.Store((1, 2))

    with mk.gui.reactive(react):
        store_iter = iter(store)
    assert isinstance(store_iter, mk.gui.Store if react else Iterator)

    with mk.gui.reactive(react):
        for x in store_iter:
            assert isinstance(x, mk.gui.Store if react else int)


@pytest.mark.parametrize("react", [False, True])
def test_tuple_unpack(react: bool):
    store = mk.gui.Store((1, 2))

    with mk.gui.reactive(react):
        a, b = store

    if react:
        assert isinstance(a, mk.gui.Store)
        assert isinstance(b, mk.gui.Store)
    else:
        assert not isinstance(a, mk.gui.Store) and isinstance(a, int)
        assert not isinstance(b, mk.gui.Store) and isinstance(b, int)


@pytest.mark.parametrize("react", [False, True])
def test_tuple_unpack_return_value(react: bool):
    @mk.gui.reactive(nested_return=False)
    def add(seq: Tuple[int]):
        return tuple(x + 1 for x in seq)

    store = mk.gui.Store((1, 2))
    # We need to use the `react` decorator here because tuple unpacking
    # happens outside of the function `add`. Without the decorator, the
    # tuple unpacking will not be reactive.
    with mk.gui.reactive(react):
        a, b = add(store)

    if react:
        assert isinstance(a, mk.gui.Store)
        assert isinstance(b, mk.gui.Store)
    else:
        assert not isinstance(a, mk.gui.Store) and isinstance(a, int)
        assert not isinstance(b, mk.gui.Store) and isinstance(b, int)


@pytest.mark.parametrize("react", [False, True])
def test_bool(react: bool):
    store = mk.gui.Store(0)
    with mk.gui.reactive(react):
        if react:
            with pytest.warns(UserWarning):
                out_bool = bool(store)
            with pytest.warns(UserWarning):
                out_not = not store
        else:
            out_bool = bool(store)
            out_not = not store

    # Store.__bool__ is not reactive.
    assert not isinstance(out_bool, mk.gui.Store)
    assert not isinstance(out_not, mk.gui.Store)
