import os

import dill
import numpy as np
from PIL import Image

from meerkat.columns.file_column import Downloader


def test_downloader(monkeypatch, tmpdir):
    import urllib

    ims = []

    def patched_urlretrieve(url, filename):
        img_array = np.ones((4, 4, 3)).astype(np.uint8)
        im = Image.fromarray(img_array)
        ims.append(im)
        im.save(filename)

    monkeypatch.setattr(urllib.request, "urlretrieve", patched_urlretrieve)

    downloader = Downloader(cache_dir=os.path.join(tmpdir, "cache"))

    out = downloader("https://test.com/dir/2.jpg")

    assert os.path.exists(os.path.join(tmpdir, "cache", "test.com/dir/2.jpg"))
    assert (np.array(out) == np.array(ims[0])).all()

    out = downloader("https://test.com/dir/2.jpg")
    assert len(ims) == 1

    out = downloader("https://test.com/dir/3.jpg")
    assert len(ims) == 2

    dill.dump(downloader, open(os.path.join(tmpdir, "cache", "downloader.pkl"), "wb"))
    downloader = dill.load(open(os.path.join(tmpdir, "cache", "downloader.pkl"), "rb"))

    # reload
    downloader = Downloader(cache_dir=os.path.join(tmpdir, "cache"))


def test_unsuccessful_download(monkeypatch, tmpdir):
    import urllib

    def patched_urlretrieve(url, filename):
        raise urllib.error.HTTPError(url, 404, "Not found", None, None)

    monkeypatch.setattr(urllib.request, "urlretrieve", patched_urlretrieve)

    downloader = Downloader(cache_dir=os.path.join(tmpdir, "cache"))

    out = downloader("https://test.com/dir/2.jpg")
    assert out is None


def test_serialize_downloader(tmpdir):
    downloader = Downloader(cache_dir="cache")

    dill.dump(downloader, open(os.path.join(tmpdir, "downloader.pkl"), "wb"))

    downloader = dill.load(open(os.path.join(tmpdir, "downloader.pkl"), "rb"))

    assert downloader.cache_dir == "cache"
