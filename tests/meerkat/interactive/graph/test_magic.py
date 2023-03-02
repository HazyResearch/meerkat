import pytest

import meerkat as mk


class FooToWrap:
    """Wrap this in a store to test for magic."""

    def __init__(self, x):
        self.fn = lambda y: x + y
        self.x = x

    def add(self, y):
        return self.x + y

    def __getitem__(self, i):
        return self.x

    def __len__(self):
        return self.x


@pytest.mark.parametrize("is_magic", [True, False])
@pytest.mark.parametrize("name", ["x", "fn"])
def test_magic_attribute_accessor(is_magic: bool, name: str):
    foo = mk.Store(FooToWrap(1))
    assert foo.inode is None

    with mk.magic(is_magic):
        if name == "x":
            out = foo.x
            expected = 1
        elif name == "fn":
            out = foo.fn
            with mk.magic(False):
                expected = foo.fn

    assert out == expected
    if is_magic:
        assert foo.inode is not None
        assert out.inode is not None
    else:
        assert foo.inode is None
        assert not isinstance(out, mk.Store)


@pytest.mark.parametrize("is_magic", [True, False])
def test_magic_getitem(is_magic: bool):
    foo = mk.Store(FooToWrap(1))
    assert foo.inode is None

    with mk.magic(is_magic):
        out = foo[0]

    assert out == 1
    if is_magic:
        assert foo.inode is not None
        assert out.inode is not None
    else:
        # getitem is reactive, so a node will always be created.
        assert foo.inode is not None
        assert isinstance(out, mk.Store)


@pytest.mark.parametrize("is_magic", [True, False])
def test_magic_instance_method(is_magic: bool):
    foo = mk.Store(FooToWrap(1))
    assert foo.inode is None

    with mk.magic(is_magic):
        fn = foo.add
        out_add: mk.Store = fn(1)

    assert out_add == 2
    if is_magic:
        assert foo.inode is not None
        assert out_add.inode is not None
    else:
        assert foo.inode is None
        assert isinstance(out_add, int) and not isinstance(out_add, mk.Store)
