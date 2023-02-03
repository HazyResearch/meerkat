import pydantic
import pytest

import meerkat as mk
from meerkat.interactive.event import EventInterface


def test_component_creation_fails_with_bad_endpoint():
    """
    Raise an error if an Endpoint with a mismatched function signature
    is passed to a Component.
    """

    class TestEventInterface(EventInterface):
        arg_1: int
        arg_2: int

    class Test(mk.gui.Component):
        on_click: mk.gui.Endpoint[TestEventInterface]

    @mk.gui.endpoint
    def test_endpoint_1(arg_1, arg_2, arg_3):
        """Extra argument."""
        pass

    @mk.gui.endpoint
    def test_endpoint_2(arg_1, arg_2):
        """Correct signature."""
        pass

    @mk.gui.endpoint
    def test_endpoint_3(arg_1):
        """Missing argument."""
        pass

    @mk.gui.endpoint
    def test_endpoint_4(**kwargs):
        """Keyword arguments are okay."""
        pass

    @mk.gui.endpoint
    def test_endpoint_5(arg_1, arg_2, arg_3=3):
        """Extra default arguments are okay."""
        pass

    with pytest.raises(pydantic.ValidationError):
        Test(on_click=test_endpoint_1)

    with pytest.raises(pydantic.ValidationError):
        Test(on_click=test_endpoint_3)

    Test(on_click=test_endpoint_2)
    Test(on_click=test_endpoint_4)
    Test(on_click=test_endpoint_5)

    # Partial functions are okay.

    @mk.gui.endpoint
    def test_endpoint_6(arg_1, arg_2, arg_3):
        pass

    @mk.gui.endpoint
    def test_endpoint_7(arg_0, arg_1, arg_2):
        pass

    @mk.gui.endpoint
    def test_endpoint_8(arg_0, arg_1, arg_2, arg_3):
        pass
    
    Test(on_click=test_endpoint_6.partial(arg_3=3))
    Test(on_click=test_endpoint_7.partial(3))
    Test(on_click=test_endpoint_8.partial(3, arg_3=3))


def test_endpoint_warning_on_component_creation():
    """Raise a warning if an Endpoint's generic type is not specified."""

    @mk.gui.endpoint
    def test_endpoint():
        pass

    class Test(mk.gui.Component):
        on_click: mk.gui.Endpoint

    with pytest.warns(UserWarning):
        Test(on_click=test_endpoint)
