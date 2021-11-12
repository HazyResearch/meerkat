import meerkat as mk


def test_names():
    names = mk.datasets.names
    assert isinstance(names, list)
    assert len(names) > 0


def test_catalog():
    catalog = mk.datasets.catalog
    assert isinstance(catalog, mk.DataPanel)
    assert len(catalog) > 0


def test_repr():
    out = repr(mk.datasets)
    isinstance(out, str)
