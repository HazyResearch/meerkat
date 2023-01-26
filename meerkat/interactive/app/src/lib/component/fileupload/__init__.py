from typing import Optional
from meerkat.interactive.app.src.lib.component.abstract import AutoComponent
from meerkat.interactive.endpoint import Endpoint


class FileUpload(AutoComponent):

    files: list = []
    filenames: list = []
    contents: list = []
    classes: str = None

    webkitdirectory: bool = False
    directory: bool = False
    multiple: bool = False

    on_upload: Endpoint = None
