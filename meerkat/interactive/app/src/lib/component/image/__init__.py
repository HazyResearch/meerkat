import base64
from io import BytesIO
from meerkat.columns.deferred.base import DeferredCell
from meerkat.interactive.app.src.lib.component.abstract import AutoComponent
from meerkat.interactive.formatter.base import Formatter, Variant

class Image(AutoComponent):

    data: str
    classes: str = ""

class ImageFormatter(Formatter):

    component_class: type = Image
    data_prop: str = "data"

    variants: dict = {
        "small": Variant(
            props={},
            encode_kwargs={"thumbnail": True},
        )
    }

    def __init__(self, classes: str = ""):
        super().__init__(classes=classes)
    
    def _encode(self, image: Image, thumbnail: bool = False) -> str:
        with BytesIO() as buffer:
            if thumbnail:
                image.thumbnail((256, 256))
            image.save(buffer, "jpeg")
            return "data:image/jpeg;base64,{im_base_64}".format(
                im_base_64=base64.b64encode(buffer.getvalue()).decode()
            )


    def html(self, cell: Image) -> str:
        encoded = self.encode(cell, thumbnail=True)
        return f'<img src="{encoded}">'

class DeferredImageFormatter(ImageFormatter):

    component_class: type = Image
    data_prop: str = "data"

    def _encode(self, image: DeferredCell, thumbnail: bool = False) -> str:
        if image.absolute_path.startswith("http"):
            return image.absolute_path
        else:
            image = image()
            return super()._encode(image, thumbnail=thumbnail)