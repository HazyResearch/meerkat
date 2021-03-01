import logging
import os
import tempfile
import uuid
from datetime import datetime
from typing import Union

logger = logging.getLogger(__name__)


def initialize_logging(
    log_dir: str = tempfile.gettempdir(),
    log_name: str = "robustnessgym.log",
    format: str = "[%(asctime)s][%(levelname)s][%(name)s:%(lineno)s] :: %(message)s",
    level: int = logging.INFO,
) -> None:
    """Initialize logging for Robustness Gym."""

    # Generate a new directory using the log_dir, if it doesn't exist
    date = datetime.now().strftime("%Y_%m_%d")
    time = datetime.now().strftime("%H_%M_%S")
    uid = str(uuid.uuid4())[:8]
    log_path = os.path.join(log_dir, date, time, uid)

    # Make the logdir
    os.makedirs(log_path, exist_ok=True)

    # Initialize logging
    logging.basicConfig(
        format=format,
        level=level,
        handlers=[
            logging.FileHandler(os.path.join(log_path, log_name)),
            logging.StreamHandler(),
        ],
    )

    # Set logging levels for dependencies
    set_logging_level_for_imports()

    logging.info("Logging initialized.")


def set_logging_level_for_imports(level: int = logging.WARNING) -> None:
    """Set logging levels for dependencies."""
    # Set levels for imports
    logging.getLogger("tensorflow").setLevel(level)
    logging.getLogger("matplotlib").setLevel(level)
    logging.getLogger("textattack").setLevel(level)
    logging.getLogger("filelock").setLevel(level)


def set_logging_level(level: Union[int, str] = logging.INFO):
    """Set logging level for Robustness Gym."""
    # Set the top-level logger
    if isinstance(level, int):
        logging.getLogger("robustnessgym").setLevel(level)
    elif isinstance(level, str):
        logging.getLogger("robustnessgym").setLevel(
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
