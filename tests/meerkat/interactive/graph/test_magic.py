import pytest

import meerkat as mk


class FooToWrap:
    """Wrap this in a store to test for magic."""

    def __init__(self, x):
        self.x = x

    def add(self, y):
        return self.x + y

    def __getitem__(self, i):
        return self.x

    def __len__(self):
        return self.x


@pytest.mark.parametrize("is_magic", [True, False])
def test_footowrap_attribute_accessor(is_magic: bool):
    foo = mk.Store(FooToWrap(1))
    assert foo.inode is None

    with mk.magic():
        out_x: mk.Store = foo.x

    assert out_x == 1
    assert foo.inode is not None
    assert out_x.inode is not None
