import logging
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Union

from rich.logging import RichHandler

logger = logging.getLogger(__name__)


def initialize_logging(
    log_dir: str = None,
    log_name: str = "meerkat.log",
    format: str = "[%(funcName)s()] [%(name)s: %(lineno)s] :: %(message)s",
    level: int = os.environ.get("MEERKAT_LOGGING_LEVEL", logging.WARNING),
) -> None:
    """Initialize logging for Meerkat."""

    # Generate a new directory using the log_dir, if it doesn't exist
    date = datetime.now().strftime("%Y_%m_%d")
    time = datetime.now().strftime("%H_%M_%S")
    uid = str(uuid.uuid4())[:8]

    if log_dir is None:
        log_dir = os.environ.get("MEERKAT_LOG_DIR")

    if log_dir is None:
        success = False
        # try potential logging directories until we find one with adequate permissions
        for log_dir in [
            tempfile.gettempdir(),
            os.path.join(Path.home(), ".meerkat"),
        ]:
            try:
                log_path = os.path.join(log_dir, "log", date, time, uid)
                os.makedirs(log_path, exist_ok=True)
                success = True
            except PermissionError:
                pass
        if not success:
            raise PermissionError(
                "Permission denied in all of Meerkat's default logging directories. "
                "Set environment variable `MEERKAT_LOG_DIR` to specify a directory for "
                "Meerkat logging."
            )

    else:
        log_path = os.path.join(log_dir, "log", date, time, uid)
        # Make the logdir
        os.makedirs(log_path, exist_ok=True)

    # Initialize logging
    logging.basicConfig(
        format=format,
        level=level,
        handlers=[
            logging.FileHandler(os.path.join(log_path, log_name)),
            # logging.StreamHandler(),
            RichHandler(rich_tracebacks=True),
        ],
    )

    # Set logging levels for dependencies
    set_logging_level_for_imports()
    logger.info("Logging initialized.")


def set_logging_level_for_imports(level: int = logging.WARNING) -> None:
    """Set logging levels for dependencies."""
    # Set levels for imports
    logging.getLogger("tensorflow").setLevel(level)
    logging.getLogger("matplotlib").setLevel(level)
    logging.getLogger("textattack").setLevel(level)
    logging.getLogger("filelock").setLevel(level)
    logging.getLogger("sse_starlette").setLevel(level)


def set_logging_level(level: Union[int, str] = logging.INFO):
    """Set logging level for Meerkat."""
    # Set the top-level logger
    if isinstance(level, int):
        logging.getLogger("meerkat").setLevel(level)
    elif isinstance(level, str):
        logging.getLogger("meerkat").setLevel(
            {
                "debug": logging.DEBUG,
                "info": logging.INFO,
                "warning": logging.WARNING,
                "critical": logging.CRITICAL,
                "fatal": logging.FATAL,
            }[level]
        )
    else:
        raise NotImplementedError(f"Level `{level}` not recognized.")
