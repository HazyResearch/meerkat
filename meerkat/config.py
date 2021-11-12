import os
from pathlib import Path


class DisplayOptions:
    max_rows: int = 10
    show_images: bool = True

    max_image_height: int = 128
    max_image_width: int = 128


class ContribOptions:
    download_dir: str = os.path.join(Path.home(), ".meerkat/datasets")
