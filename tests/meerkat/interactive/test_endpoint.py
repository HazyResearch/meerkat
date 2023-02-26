from typing import List, Union

import numpy as np
import pytest

import meerkat as mk
from meerkat.interactive.endpoint import _is_annotation_store
from meerkat.interactive.graph.store import _IteratorStore


@pytest.mark.parametrize("fn_decorator", [mk.gui.reactive])
def test_endpoint_wrapping_reactive_fn(fn_decorator):
    """When an endpoint wraps a reactive function, reactivity should be
    disabled to prevent adding anything to the graph.

    Note, we can only do this with methods decorated with @reactive. If
    a method decorated with `@mk.gui.react()` is called from an
    endpoint, the graph will be built because `@mk.gui.react()`
    activates reactivity prior to the method being called.
    """
    fn = fn_decorator(lambda store: store + 3)

    @mk.endpoint()
    def fn_endpoint(store: mk.gui.Store):
        store.set(fn(store))

    # Test with @reactive decorator.
    x = mk.gui.Store(1)
    assert not mk.gui.is_unmarked_context()  # Verify we are in a reactive context
    fn_endpoint(x)
    assert x == 4  # Verify the endpoint works
    assert x.inode is None  # Graph should be empty


@pytest.mark.parametrize(
    "x",
    [
        # TODO: Uncomment when we can issue column modifications.
        # mk.ScalarColumn([1, 2, 3, 4, 5]),
        mk.DataFrame({"a": [1, 2, 3, 4, 5]}),
        mk.Store(np.array([1, 2, 3, 4, 5])),
    ],
)
def test_endpoint_with_reactive_output(
    x,
):
    """
    Test that we can add endpoints to reactive outputs.

    The graph for this test looks like

        df -> view -> df_view -> view -> df_view2
       ^ |              ^   |
       | v              |   v
       add_one     add_one (endpoint)
    """

    def _get_value(_x):
        if isinstance(_x, mk.DataFrame):
            return _x["a"]
        else:
            return _x

    x.mark()

    @mk.reactive()
    def view(_x):
        if isinstance(_x, (mk.DataFrame, mk.Column)):
            return _x.view()
        else:
            return _x

    # TODO: Change the type hint to Union when unions are supported.
    @mk.endpoint()
    def add_one(_x: mk.Store):
        if isinstance(_x, mk.DataFrame):
            _x["a"] = _x["a"] + 1
            _x.set(_x)
        else:
            out = _x + 1
            _x.set(out)

    endpoint_df = add_one.partial(_x=x)

    df_view = view(x)
    assert df_view.inode is not None
    assert x.inode is not None
    df_view_inode = df_view.inode

    endpoint_df_view = add_one.partial(_x=df_view)
    assert df_view.inode is df_view_inode

    df_view2 = view(df_view)
    df_view2_inode = df_view2.inode

    # Run the endpoint on the original input.
    # This should trigger both views to update.
    endpoint_df.run()
    assert all(_get_value(x) == [2, 3, 4, 5, 6])
    assert all(_get_value(df_view_inode.obj) == [2, 3, 4, 5, 6])
    assert all(_get_value(df_view2_inode.obj) == [2, 3, 4, 5, 6])

    # Run the endpoint on the first view.
    # This should trigger the second view to update.
    endpoint_df_view.run()
    assert all(_get_value(x) == [2, 3, 4, 5, 6])
    assert all(_get_value(df_view_inode.obj) == [3, 4, 5, 6, 7])
    assert all(_get_value(df_view2_inode.obj) == [3, 4, 5, 6, 7])


@pytest.mark.parametrize("endpoint_id", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("partial", [True, False])
def test_endpoint_type_hints(endpoint_id: int, partial: bool):
    """
    Test that endpoints with different type hints will work.
    """

    @mk.endpoint()
    def endpoint1(x: mk.Store):
        assert isinstance(x, mk.Store)

    @mk.endpoint()
    def endpoint2(x: mk.Store[int]):
        assert isinstance(x, mk.Store)

    @mk.endpoint()
    def endpoint3(x: mk.Store[List[int]]):
        assert isinstance(x, mk.Store)

    @mk.endpoint()
    def endpoint4(x: Union[mk.Store, int]):
        assert isinstance(x, mk.Store)

    @mk.endpoint()
    def endpoint5(x: Union[mk.Store[int], int]):
        assert isinstance(x, mk.Store)

    endpoint = {
        1: endpoint1,
        2: endpoint2,
        3: endpoint3,
        4: endpoint4,
        5: endpoint5,
    }[endpoint_id]

    store = mk.Store(1)

    if partial:
        _endpoint = endpoint.partial(x=store)
        _endpoint.run()
    else:
        endpoint.run(store)


@pytest.mark.parametrize("x_input", ["a", mk.Store("a"), mk.Store(1)])
@pytest.mark.parametrize("endpoint_id", [1, 2, 3])
@pytest.mark.parametrize("partial", [True, False])
def test_endpoint_with_string(x_input, endpoint_id: int, partial: bool):
    """
    Endpoints resolve variables based on their ids, which are strings.
    This may cause problems when the input is actually as string.
    These tests are to check that endpoints can work properly with non-id strings.
    """

    @mk.endpoint()
    def endpoint1(x: str):
        # The type hint is `str`, so the input should never be a store.
        assert not isinstance(x, mk.Store)

    @mk.endpoint()
    def endpoint2(x: mk.Store[str]):
        # The type hint is `Store[str]`, so the input should be a store.
        # Type hints should never be strict constraints in Python.
        # So even if the user passes in some other type, we should still
        # be able to handle it.
        if isinstance(x_input, mk.Store):
            assert isinstance(x, mk.Store)
        else:
            assert not isinstance(x, mk.Store)

    @mk.endpoint()
    def endpoint3(x: Union[mk.Store, str]):
        # The type hint is `Union[Store, str]`, so the input should be a store
        # if a store was passed in. If a store wasn't passed in, then we
        if isinstance(x_input, mk.Store):
            assert isinstance(x, mk.Store)
        else:
            assert not isinstance(x, mk.Store)

    endpoint = {
        1: endpoint1,
        2: endpoint2,
        3: endpoint3,
    }[endpoint_id]

    if partial:
        _endpoint = endpoint.partial(x=x_input)
        _endpoint.run()
    else:
        endpoint.run(x_input)


@pytest.mark.parametrize(
    "type_hint",
    [
        mk.Store,
        mk.Store[int],
        mk.Store[List[int]],
        # subclass of Store
        _IteratorStore,
        # Union with non-generic store
        Union[mk.Store, int],
        # Union with generic store
        Union[mk.Store[int], int],
        # Nested stores
        Union[Union[mk.Store[int], int], int],
    ],
)
def test_is_annotation_store_true(type_hint):
    assert _is_annotation_store(type_hint)


@pytest.mark.parametrize("type_hint", [mk.DataFrame, mk.Column])
def test_is_annotation_store_false(type_hint):
    assert not _is_annotation_store(type_hint)
