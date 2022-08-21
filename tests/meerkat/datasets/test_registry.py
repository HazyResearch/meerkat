from typing import List

import meerkat as mk


def test_versions():
    versions = mk.datasets.versions("imagenette")
    assert isinstance(versions, List)
    assert len(versions) > 0


def test_catalog():
    catalog = mk.datasets.catalog
    assert isinstance(catalog, mk.DataPanel)
    assert len(catalog) > 0


def test_repr():
    out = repr(mk.datasets)
    isinstance(out, str)
