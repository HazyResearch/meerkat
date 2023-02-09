from typing import Optional

from meerkat.interactive.app.src.lib.component.abstract import Component
from meerkat.interactive.endpoint import Endpoint
from meerkat.interactive.event import EventInterface


class OnUploadFileUpload(EventInterface):
    pass


class FileUpload(Component):
    files: list = []
    filenames: list = []
    contents: list = []
    classes: Optional[str] = None

    webkitdirectory: bool = False
    directory: bool = False
    multiple: bool = False

    on_upload: Endpoint[OnUploadFileUpload] = None
