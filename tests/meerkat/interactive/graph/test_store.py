from typing import Iterator, Tuple

import pytest

import meerkat as mk
from meerkat.interactive.graph.magic import magic
from meerkat.interactive.graph.operation import Operation
from meerkat.interactive.graph.store import _IteratorStore


def _is_out_magiced(
    out,
    input_store: mk.gui.Store,
    *,
    op_name: str = None,
    op_num_children: int = None,
):
    """Check if the output is magiced.

    A store is magiced if:
        1. It is a Store
        2. It is marked
        3. out.inode is not None
        4. The input_store is the grandparent of out

    Args:
        out: The output store.
        input_store: The input store.
        op_name: The name of the operation. If None, all operations are checked.
        op_num_children: The number of children each operation should have.
    """
    assert isinstance(out, mk.gui.Store)
    assert out.marked
    assert out.inode is not None

    # These should always be operations.
    children = input_store.inode.trigger_children
    assert all(isinstance(c.obj, Operation) for c in children)
    if op_name is not None:
        children = {c.obj.fn.__name__: c for c in children}
        children = [children[op_name]]
    if op_num_children is not None:
        for op_inode in children:
            assert len(op_inode.trigger_children) == op_num_children, (
                f"{op_inode.obj.fn.__name__} has {len(op_inode.trigger_children)} "
                f"children, expected {op_num_children}"
            )
    # One of the children should be the op.
    grandchildren = [gc for c in children for gc in c.trigger_children]
    assert out.inode in grandchildren


def _is_out_unmagiced(out, input_store: mk.gui.Store):
    """Check if the output is unmagiced.

    A store is unmagiced if:
        1. It is a Store
        2. It is marked
        3. out.inode is None
    """
    assert not isinstance(out, mk.gui.Store)


@mk.endpoint()
def _set_store(store: mk.gui.Store, value):
    store.set(value)


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

    if is_magic:
        assert len(store.inode.trigger_children) == len(expected)
    else:
        assert store.inode is None

    for k, v in out.items():
        if is_magic:
            _is_out_magiced(v, store, op_name=f"__{k}__", op_num_children=1)
        else:
            _is_out_unmagiced(v, store)

        assert v == expected[k]


@pytest.mark.parametrize("other", [1, 2])
@pytest.mark.parametrize("is_magic", [False, True])
def test_store_imethod(other: int, is_magic: bool):
    """Test traditional inplace methods are reactive, but return different
    stores."""

    def _get_expected():
        # TODO: Have a variable that chooses which one of these to run.
        # So we can test each one separately.
        return {
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

    store = mk.gui.Store(1)
    original = store

    if is_magic:
        with pytest.warns(UserWarning):
            expected = _get_expected()
    else:
        expected = _get_expected()

    out = {}
    with magic():
        for k in expected:
            with pytest.warns(UserWarning):
                out[k] = getattr(store, k)(other)

    for k, v in out.items():
        assert not isinstance(
            expected[k], mk.gui.Store
        ), f"Expected: {k} returned a Store."
        assert isinstance(v, mk.gui.Store), f"{k} did not return a Store."
        assert v.marked
        assert id(v) != id(original), f"{k} did not return a new Store."
        assert v == expected[k], f"{k} did not return the correct value."


@pytest.mark.parametrize("is_magic", [False, True])
def test_store_as_iterator(is_magic: bool):
    store = mk.gui.Store((1, 2))

    with magic(is_magic):
        store_iter = iter(store)

    # Regardless of if magic is on, the iterator should be a Store.
    # However, only when magic is on, should the store be added to the graph.
    if is_magic:
        assert isinstance(store_iter, _IteratorStore)
        _is_out_magiced(store_iter, store, op_name="__iter__", op_num_children=1)
    else:
        assert isinstance(store_iter, Iterator)
        _is_out_unmagiced(store_iter, store)

    # When we fetch things from the iterator, they should be stores.
    # Similar to the above, only when magic is on, should the store be added
    # to the graph.
    with magic(is_magic):
        values = [v for v in store_iter]

    for v in values:
        if is_magic:
            isinstance(v, mk.gui.Store)
        else:
            not isinstance(v, mk.gui.Store)

    if not is_magic:
        return

    # Test the nodes get updated properly
    assert len(values) == 2
    with magic():
        inode1 = values[0].inode
        inode2 = values[1].inode

    _set_store(store, [10, 11])
    assert inode1.obj == 10
    assert inode2.obj == 11


@pytest.mark.parametrize("is_magic", [False, True])
def test_tuple_unpack(is_magic: bool):
    store = mk.gui.Store((1, 2))

    with magic(is_magic):
        a, b = store

    if is_magic:
        assert isinstance(a, mk.gui.Store)
        assert isinstance(b, mk.gui.Store)
    else:
        assert not isinstance(a, mk.gui.Store)
        assert not isinstance(b, mk.gui.Store)

    if not is_magic:
        return

    # Test the nodes get updated properly
    a_inode = a.inode
    b_inode = b.inode

    _set_store(store, [10, 11])
    assert a_inode.obj == 10
    assert b_inode.obj == 11


@pytest.mark.parametrize("is_magic", [False, True])
def test_tuple_unpack_return_value(is_magic: bool):
    @mk.gui.reactive(nested_return=False)
    def add(seq: Tuple[int]):
        return tuple(x + 1 for x in seq)

    store = mk.gui.Store((1, 2))
    # We need to use the `magic` decorator here because tuple unpacking
    # happens outside of the function `add`. Without the decorator, the
    # tuple unpacking will not be added to the graph.
    with magic(is_magic):
        a, b = add(store)
    assert a == 2
    assert b == 3

    if is_magic:
        assert isinstance(a, mk.gui.Store)
        assert isinstance(b, mk.gui.Store)
    else:
        assert not isinstance(a, mk.gui.Store)
        assert not isinstance(b, mk.gui.Store)

    if not is_magic:
        return

    # Test the nodes get updated properly
    a_inode = a.inode
    b_inode = b.inode

    _set_store(store, [10, 11])
    assert a_inode.obj == 11
    assert b_inode.obj == 12


@pytest.mark.parametrize("is_magic", [False, True])
def test_bool(is_magic: bool):
    store = mk.gui.Store(0)
    with magic(is_magic):
        if is_magic:
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
