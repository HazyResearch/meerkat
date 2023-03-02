import json
import os
from typing import Union

import dill
import numpy as np
import pytest
from PIL import Image

import meerkat as mk
from meerkat.block.deferred_block import DeferredCellOp, DeferredOp
from meerkat.columns.deferred.base import DeferredCell
from meerkat.columns.deferred.file import FILE_TYPES, FileCell, FileColumn, FileLoader
from meerkat.columns.scalar import ScalarColumn
from tests.meerkat.columns.abstract import AbstractColumnTestBed, column_parametrize
from tests.utils import product_parametrize


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def add_one(data):
    return data + 1


class FileColumnTestBed(AbstractColumnTestBed):
    DEFAULT_CONFIG = {
        "use_base_dir": [True, False],
    }

    marks = pytest.mark.file_col

    def __init__(
        self,
        tmpdir: str,
        length: int = 16,
        use_base_dir: bool = False,
        seed: int = 123,
    ):
        self.paths = []
        self.data = []

        self.tmp_dir = tmpdir
        self.files_dir = os.path.join(tmpdir, "files")
        os.makedirs(self.files_dir, exist_ok=True)
        self.base_dir = self.files_dir if use_base_dir else None

        for i in range(0, length):
            # write a simple json file to disk
            filename = "file_{}.json".format(i)
            path = os.path.join(self.files_dir, filename)
            with open(path, "w") as f:
                json.dump(i, f)
            self.data.append(i)

            if use_base_dir:
                self.paths.append(filename)
            else:
                self.paths.append(os.path.join(self.files_dir, filename))

        self.data = np.arange(length)

        self.col = mk.files(
            self.paths,
            loader=load_json,
            base_dir=self.base_dir,
        )

    def get_data(self, index, materialize: bool = True):
        if materialize:
            return self.data[index]
        else:
            if isinstance(index, int):
                return FileCell(
                    DeferredCellOp(
                        args=[self.paths[index]],
                        kwargs={},
                        fn=self.col.fn,
                        is_batched_fn=False,
                        return_index=None,
                    )
                )

            index = np.arange(len(self.data))[index]
            col = ScalarColumn([self.paths[idx] for idx in index])
            return DeferredOp(
                args=[col], kwargs={}, fn=self.col.fn, is_batched_fn=False, batch_size=1
            )

    @staticmethod
    def assert_data_equal(
        data1: Union[np.ndarray, DeferredCell, DeferredOp],
        data2: Union[np.ndarray, DeferredCell, DeferredOp],
    ):
        if isinstance(data1, (int, np.int64)):
            assert data1 == data2
        elif isinstance(data1, np.ndarray):
            assert (data1 == data2).all()
        elif isinstance(data1, DeferredCell):
            assert data1 == data2
        elif isinstance(data1, DeferredOp):
            assert data1.is_equal(data2)
        else:
            raise ValueError(
                "Cannot assert data equal between objects type:"
                f" {type(data1), type(data2)}"
            )


@pytest.fixture(**column_parametrize([FileColumnTestBed]))
def testbed(request, tmpdir):
    testbed_class, config = request.param
    return testbed_class(**config, tmpdir=tmpdir)


@FileColumnTestBed.parametrize(
    config={
        "use_base_dir": [True],
    }
)
def test_change_base_dir(testbed):
    assert testbed.base_dir is not None
    col = testbed.col

    new_dir = os.path.join(testbed.tmp_dir, "new_files")
    os.rename(testbed.files_dir, new_dir)
    col.base_dir = new_dir

    assert (col[[1, 3, 5]]().data == testbed.get_data([1, 3, 5])).all()


BUCKET_URL = "https://storage.googleapis.com/meerkat-ml/tests/file_types"

TEST_FILES = {
    "image": [
        os.path.join(BUCKET_URL, "image/test-img-01.jpg"),
        os.path.join(BUCKET_URL, "image/test-img-02.jpg"),
        os.path.join(BUCKET_URL, "image/test-img-01.png"),
        os.path.join(BUCKET_URL, "image/test-img-02.png"),
        os.path.join(BUCKET_URL, "image/test-img-01.jpeg"),
        os.path.join(BUCKET_URL, "image/test-img-02.jpeg"),
    ],
    "pdf": [
        os.path.join(BUCKET_URL, "pdf/test-pdf-01.pdf"),
        os.path.join(BUCKET_URL, "pdf/test-pdf-02.pdf"),
        os.path.join(BUCKET_URL, "pdf/test-pdf-03.pdf"),
    ],
    "text": [
        os.path.join(BUCKET_URL, "text/test-txt-01.txt"),
        os.path.join(BUCKET_URL, "text/test-txt-02.txt"),
        os.path.join(BUCKET_URL, "text/test-txt-03.txt"),
    ],
    "html": [
        os.path.join(BUCKET_URL, "html/test-html-01.html"),
        os.path.join(BUCKET_URL, "html/test-html-02.html"),
        os.path.join(BUCKET_URL, "html/test-html-03.html"),
    ],
    "code": [
        os.path.join(BUCKET_URL, "code/test-code-01.py"),
        os.path.join(BUCKET_URL, "code/test-code-02.py"),
        os.path.join(BUCKET_URL, "code/test-code-03.py"),
    ],
}


@product_parametrize({"file_type": list(TEST_FILES.keys())})
def test_file_types(file_type: str, tmpdir):
    files = TEST_FILES[file_type]

    col = mk.files(files)

    assert isinstance(col, FileColumn)

    assert isinstance(col.formatters, FILE_TYPES[file_type]["formatters"])

    col.formatters["base"].encode(col[0])


def test_downloader(monkeypatch, tmpdir):
    import urllib

    ims = []

    def patched_urlretrieve(url, filename):
        img_array = np.ones((4, 4, 3)).astype(np.uint8)
        im = Image.fromarray(img_array)
        ims.append(im)
        im.save(filename)

    monkeypatch.setattr(urllib.request, "urlretrieve", patched_urlretrieve)

    downloader = FileLoader(
        loader=Image.open, downloader="url", cache_dir=os.path.join(tmpdir, "cache")
    )

    out = downloader("https://test.com/dir/2.jpg")

    assert os.path.exists(os.path.join(tmpdir, "cache", "test.com/dir/2.jpg"))
    assert (np.array(out) == np.array(ims[0])).all()

    out = downloader("https://test.com/dir/2.jpg")
    assert len(ims) == 1

    out = downloader("https://test.com/dir/3.jpg")
    assert len(ims) == 2


def test_fallback_download(monkeypatch, tmpdir):
    import urllib

    def patched_urlretrieve(url, filename):
        raise urllib.error.HTTPError(url, 404, "Not found", None, None)

    monkeypatch.setattr(urllib.request, "urlretrieve", patched_urlretrieve)

    ims = []

    def fallback(filename):
        img_array = np.ones((4, 4, 3)).astype(np.uint8)
        im = Image.fromarray(img_array)
        ims.append(im)
        im.save(filename)

    downloader = FileLoader(
        loader=Image.open,
        downloader="url",
        fallback_downloader=fallback,
        cache_dir=os.path.join(tmpdir, "cache"),
    )
    with pytest.warns(UserWarning):
        out = downloader("https://test.com/dir/2.jpg")

    assert os.path.exists(os.path.join(tmpdir, "cache", "test.com/dir/2.jpg"))
    assert (np.array(out) == np.array(ims[0])).all()

    out = downloader("https://test.com/dir/2.jpg")
    assert len(ims) == 1

    with pytest.warns(UserWarning):
        out = downloader("https://test.com/dir/3.jpg")
    assert len(ims) == 2


def test_serialize_downloader(tmpdir):
    downloader = FileLoader(
        loader=Image.open,
        downloader="url",
        cache_dir=os.path.join(tmpdir, "cache"),
    )

    dill.dump(downloader, open(os.path.join(tmpdir, "downloader.pkl"), "wb"))

    downloader = dill.load(open(os.path.join(tmpdir, "downloader.pkl"), "rb"))

    assert downloader.cache_dir == os.path.join(tmpdir, "cache")
