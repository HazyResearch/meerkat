import pytest

import meerkat as mk


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

    @mk.gui.endpoint
    def fn_endpoint(store):
        store.set(fn(store))

    # Test with @reactive decorator.
    x = mk.gui.Store(1)
    with mk.gui.reactive():  # Turn on react context
        assert not mk.gui.is_unmarked_context()  # Verify we are in a reactive context
        fn_endpoint(x)
    assert x == 4  # Verify the endpoint works
    assert x.inode is None  # Graph should be empty
