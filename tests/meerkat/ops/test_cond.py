import pytest

import meerkat as mk
from meerkat.columns.abstract import Column
from meerkat.dataframe import DataFrame
from meerkat.interactive.node import NodeMixin


@pytest.mark.parametrize("x", [False, True, -1, 0, 1, 2, 4.3])
@pytest.mark.parametrize("y", [False, True, -1, 0, 1, 2, 4.3])
@pytest.mark.parametrize("react", [False, True])
@pytest.mark.parametrize("comp", [mk.cand, mk.cor])
def test_boolean_operators_multiple_arguments(x, y, react, comp):
    x_store = mk.Store(x)
    y_store = mk.Store(y)

    if comp == mk.cand:
        expected = x and y
    elif comp == mk.cor:
        expected = x or y

    out = comp(x_store, y_store)

    assert out == expected
    if react:
        assert isinstance(out, mk.Store)
    assert isinstance(out, type(expected))


@pytest.mark.parametrize("x", [False, True, -1, 0, 1, 2, 4.3])
@pytest.mark.parametrize("react", [False, True])
@pytest.mark.parametrize("comp", [mk.bool, mk.cnot])
def test_boolean_operators_single_operator(x, react, comp):
    x_store = mk.Store(x)

    if comp == mk.bool:
        expected = bool(x)
    elif comp == mk.cnot:
        expected = not x

    out = comp(x_store)

    assert out == expected
    if react:
        assert isinstance(out, mk.Store)
    assert isinstance(out, type(expected))


def _invoker_helper(x, *, mk_func, base_func, should_warn: bool = True):
    if isinstance(x, NodeMixin):
        x = mk.mark(x)
        # All custom classes that support __len__ should raise a warning
        # when invoked with `len(obj)`. Because NodeMixin classes are
        # custom classes in Meerkat, this is a check that we enforce.
        if should_warn:
            with pytest.warns(UserWarning):
                expected = base_func(x)
        else:
            expected = base_func(x)
    else:
        expected = base_func(x)
        x = mk.Store(x)

    out = mk_func(x)
    assert out == expected
    assert isinstance(out, mk.Store)
    assert isinstance(out, type(expected))

    # Check the graph is created.
    assert x.inode is not None
    assert len(x.inode.trigger_children) == 1
    op_node = x.inode.trigger_children[0]
    assert op_node.obj.fn.__name__ == mk_func.__name__
    assert len(op_node.trigger_children) == 1
    assert op_node.trigger_children[0] == out.inode


@pytest.mark.parametrize("x", [(), (1,), (1, 2), (0, 1, 2)])
def test_all(x):
    """Test mk.all works identically to all."""
    _invoker_helper(x, mk_func=mk.all, base_func=all)


@pytest.mark.parametrize("x", [(), (1,), (1, 2), (0, 1, 2)])
def test_any(x):
    """Test mk.any works identically to any."""
    _invoker_helper(x, mk_func=mk.any, base_func=any)


@pytest.mark.parametrize("x", [False, True, -1, 0, 1.0, 1.0 + 1j, "1", "1+1j"])
def test_bool(x):
    """Test mk.bool works identically to bool."""
    _invoker_helper(x, mk_func=mk.bool, base_func=bool)


@pytest.mark.parametrize("x", [False, True, -1, 0, 1.0, 1.0 + 1j, "1", "1+1j"])
def test_complex(x):
    """Test mk.complex works identically to complex."""
    _invoker_helper(x, mk_func=mk.complex, base_func=complex)


@pytest.mark.parametrize("x", [False, True, -1, 0, 1.0, "10"])
def test_int(x):
    """Test mk.int works identically to int."""
    _invoker_helper(x, mk_func=mk.int, base_func=int)


@pytest.mark.parametrize("x", [False, True, -1, 0, 1.0, "1.0"])
def test_float(x):
    """Test mk.float works identically to float."""
    _invoker_helper(x, mk_func=mk.float, base_func=float)


@pytest.mark.parametrize(
    "x",
    [
        [1, 2, 3],
        "hello world",
        (1, 2, 3),
        mk.DataFrame({"a": [1, 2, 3]}),
        mk.TensorColumn([1, 2, 3]),
    ],
)
def test_len(x):
    """Test mk.len works identically to len."""
    should_warn = not isinstance(x, Column)
    _invoker_helper(x, mk_func=mk.len, base_func=len, should_warn=should_warn)


@pytest.mark.parametrize("x", [False, True, -1, 0])
def test_hex(x):
    """Test mk.complex works identically to complex."""
    _invoker_helper(x, mk_func=mk.hex, base_func=hex)


@pytest.mark.parametrize("x", [False, True, -1, 0])
def test_oct(x):
    """Test mk.oct works identically to oct."""
    _invoker_helper(x, mk_func=mk.oct, base_func=oct)


@pytest.mark.parametrize(
    "x",
    [
        False,
        True,
        0,
        1,
        1.0,
        mk.DataFrame({"a": [1, 2, 3]}),
        mk.TensorColumn([1, 2, 3]),
    ],
)
def test_str(x):
    """Test mk.str works identically to str."""
    should_warn = not isinstance(x, (DataFrame, Column))
    _invoker_helper(x, mk_func=mk.str, base_func=str, should_warn=should_warn)


@pytest.mark.parametrize("x", [[3, 2, 1], ("hello", "world")])
def test_min_max(x):
    """Test mk.min and mk.max work identically to min and max."""
    _invoker_helper(x, mk_func=mk.min, base_func=min)
    _invoker_helper(x, mk_func=mk.max, base_func=max)
