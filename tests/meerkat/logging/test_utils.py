import glob
import os
import tempfile
from pathlib import Path

import pytest

from meerkat import initialize_logging


def test_initialize_logging():
    initialize_logging()


@pytest.fixture
def unreadable_dir(tmpdir):
    unread_dir = tmpdir / "unreadable"
    os.makedirs(unread_dir)
    unread_dir.chmod(0)
    if os.access(str(unread_dir), os.R_OK):
        # Docker container or similar
        pytest.skip("File was still readable")

    yield unread_dir

    unread_dir.chmod(0o755)


def test_initialize_logging_permission_denied(monkeypatch, unreadable_dir):
    def mock_no_access_dir():
        return unreadable_dir

    monkeypatch.setattr(Path, "home", mock_no_access_dir)
    monkeypatch.setattr(tempfile, "gettempdir", mock_no_access_dir)

    with pytest.raises(
        PermissionError,
        match="Permission denied in all of Meerkat's default logging directories. "
        "Set environment variable `MEERKAT_LOG_DIR` to specify a directory for "
        "Meerkat logging.",
    ):
        initialize_logging()


def test_initialize_logging_environment_variable(monkeypatch, tmpdir):
    monkeypatch.setattr(
        os, "environ", {"MEERKAT_LOG_DIR": os.path.join(tmpdir, "env_dir")}
    )
    initialize_logging()
    out = list(glob.glob(str(tmpdir) + "/**/meerkat.log", recursive=True))
    assert len(out) != 0


def test_initialize_logging_arg(tmpdir):
    initialize_logging(log_dir=os.path.join(tmpdir, "env_dir"))
    out = list(glob.glob(str(tmpdir) + "/**/meerkat.log", recursive=True))
    assert len(out) != 0
