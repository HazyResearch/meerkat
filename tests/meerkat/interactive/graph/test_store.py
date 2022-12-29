import pytest

import meerkat as mk


@pytest.mark.parametrize("react", [False, True])
def test_store_reactive_methods(react: bool):
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
        "div": 1,
        "mod": 0,
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
    with mk.gui.react(reactive=react):
        out["add"] = store + 1
        out["sub"] = store - 1
        out["mul"] = store * 1
        out["div"] = store / 1
        out["mod"] = store % 1
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

    for k, v in out.items():
        if react:
            assert isinstance(v, mk.gui.Store)
            assert store.inode.has_trigger_children()
            # TODO: Check the parent of the current child.
            assert v == expected[k]
        else:
            assert not isinstance(v, mk.gui.Store)
            assert store.inode is None
