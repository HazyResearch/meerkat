# import pytest

# import meerkat as mk


# TODO(Arjun): how can we get these to pass?
# @pytest.mark.parametrize("x", [False, True, -1, 0, 1, 2, 4.3])
# @pytest.mark.parametrize("y", [False, True, -1, 0, 1, 2, 4.3])
# @pytest.mark.parametrize("react", [False, True])
# @pytest.mark.parametrize("comp", [mk.cand, mk.cor])
# def test_boolean_operators_multiple_arguments(x, y, react, comp):
#     x_store = mk.gui.Store(x)
#     y_store = mk.gui.Store(y)

#     if comp == mk.cand:
#         expected = x and y
#     elif comp == mk.cor:
#         expected = x or y

#     with mk.gui._react():
#         out = comp(x_store, y_store)

#     assert out == expected
#     if react:
#         assert isinstance(out, mk.gui.Store)
#     assert isinstance(out, type(expected))


# @pytest.mark.parametrize("x", [False, True, -1, 0, 1, 2, 4.3])
# @pytest.mark.parametrize("react", [False, True])
# @pytest.mark.parametrize("comp", [mk.to_bool, mk.cnot])
# def test_boolean_operators_single_operator(x, react, comp):
#     x_store = mk.gui.Store(x)

#     if comp == mk.to_bool:
#         expected = bool(x)
#     elif comp == mk.cnot:
#         expected = not x

#     with mk.gui._react():
#         out = comp(x_store)

#     assert out == expected
#     if react:
#         assert isinstance(out, mk.gui.Store)
#     assert isinstance(out, type(expected))
