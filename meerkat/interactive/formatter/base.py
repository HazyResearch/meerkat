"""Desiderata for formatters:

1.
"""
from abc import ABC
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Type

from meerkat.columns.deferred.base import DeferredCell

if TYPE_CHECKING:
    from meerkat.interactive.app.src.lib.component.abstract import Component


@dataclass
class Variant:
    props: Dict[str, Any]
    encode_kwargs: Dict[str, Any]


class Formatter(ABC):

    component_class: Type["Component"]
    data_prop: str = "data"
    variants: Dict[str, Variant] = {}

    def __init__(self, encode: str = None, **kwargs):
        if encode is not None:
            self._encode = encode

        default_props, required_props = self._get_props()
        if not all(k in kwargs for k in required_props):
            raise ValueError(
                f"Missing required properties {required_props} for {self.__class__.__name__}"
            )

        default_props.update(kwargs)
        self._props = default_props

    def encode(self, cell: Any, variants: List[str] = None, **kwargs):
        """Encode the cell on the backend before sending it to the
        frontend.

        The cell is lazily loaded, so when used on a LambdaColumn,
        ``cell`` will be a ``LambdaCell``. This is important for
        displays that don't actually need to apply the lambda in order
        to display the value.
        """
        if variants is not None:
            for name in variants:
                if name in self.variants:
                    kwargs.update(self.variants[name].encode_kwargs)
                    break
        if self._encode is not None:
            return self._encode(cell, **kwargs)
        return cell

    @property
    def props(self):
        return self._props

    def get_props(self, variants: str = None):
        if variants is None:
            return self.props

        props = self.props.copy()
        for name in variants:
            if name in self.variants:
                props.update(self.variants[name].props)
                break
        return props

    @classmethod
    def _get_props(cls):
        default_props = {}
        required_props = []
        for k, v in cls.component_class.__fields__.items():
            if k == cls.data_prop:
                continue
            if v.required:
                required_props.append(k)
            else:
                default_props[k] = v.default
        return default_props, required_props

    def html(self, cell: Any):
        """When not in interactive mode, objects are visualized using static
        html.

        This method should produce that static html for the cell.
        """
        return str(cell)


class DeferredFormatter(Formatter):
    def __init__(self, formatter: Formatter):
        self.wrapped = formatter

    def encode(self, cell: DeferredCell, variants: List[str] = None, **kwargs):
        return self.wrapped.encode(cell(), variants=variants, **kwargs)

    @property
    def component_class(self):
        return self.wrapped.component_class

    @property
    def data_prop(self):
        return self.wrapped.data_prop
    
    def get_props(self, variants: str =None):
        return self.wrapped.get_props(variants=variants)

    @property
    def props(self):
        return self.wrapped.props

    def html(self, cell: DeferredCell):
        return self.wrapped.html(cell())


# class BasicFormatter(Formatter):
#     cell_component = "basic"

#     DTYPES = [
#         "str",
#         "int",
#         "float",
#         "bool",
#     ]

#     def __init__(self, dtype: str = "str"):
#         if dtype not in self.DTYPES:
#             raise ValueError(
#                 f"dtype {dtype} not supported. Must be one of {self.DTYPES}"
#             )
#         self.dtype = dtype

#     def encode(self, cell: Any):
#         # check for native python nan
#         if isinstance(cell, float) and math.isnan(cell):
#             return "NaN"

#         if isinstance(cell, np.generic):
#             if pd.isna(cell):
#                 return "NaN"
#             return cell.item()

#         if hasattr(cell, "as_py"):
#             return cell.as_py()
#         return cell

#     def html(self, cell: Any):
#         cell = self.encode(cell)
#         if isinstance(cell, str):
#             cell = textwrap.shorten(cell, width=100, placeholder="...")
#         return format_array(np.array([cell]), formatter=None)[0]

#     @property
#     def cell_props(self):
#         return {
#             # backwards compatability
#             "dtype": self.dtype
#             if hasattr(self, "dtype")
#             else "str",
#         }


# class NumpyArrayFormatter(BasicFormatter):

#     cell_component = "basic"

#     def encode(self, cell: Any):
#         if isinstance(cell, np.ndarray):
#             return str(f"Tensor of shape {cell.shape}")
#         return super().encode(cell)

#     def html(self, cell: Any):
#         if isinstance(cell, np.ndarray):
#             return str(cell)
#         return format_array(np.array([cell]), formatter=None)[0]


# class IntervalFormatter(NumpyArrayFormatter):
#     cell_component = "interval"

#     def encode(self, cell: Any):
#         if cell is not np.ndarray:
#             return super().encode(cell)

#         if cell.shape[0] != 3:
#             raise ValueError(
#                 "Cell used with `IntervalFormatter` must be np.ndarray length 3 "
#                 "length 3. Got shape {}".format(cell.shape)
#             )

#         return [super().encode(v) for v in cell]

#     def html(self, cell: Any):
#         if isinstance(cell, np.ndarray):
#             return str(cell)
#         return format_array(np.array([cell]), formatter=None)[0]


# class TensorFormatter(BasicFormatter):
#     cell_component = "basic"

#     def encode(self, cell: Any):
#         if isinstance(cell, torch.Tensor):
#             return str(cell)
#         return super().encode(cell)

#     def html(self, cell: Any):
#         if isinstance(cell, torch.Tensor):
#             return str(cell)
#         return format_array(np.array([cell]), formatter=None)[0]


# class WebsiteFormatter(BasicFormatter):
#     cell_component = "website"

#     def encode(self, cell: str):
#         return cell


# class ImageURLFormatter(Formatter):

#     cell_component = "image"

#     def encode(self, cell: Any):
#         if isinstance(cell, str):
#             return cell
#         elif isinstance(cell, FileCell):
#             return cell.url
#         else:
#             raise ValueError("ImageURLFormatter can only be used with str or FileCell")

#     def html(self, cell: Any):
#         return f"""<img src="{self.encode(cell)}" />"""


# class PILImageFormatter(Formatter):

#     cell_component = "image"

#     def encode(
#         self, cell: Union["FileCell", Image, torch.Tensor], thumbnail: bool = False
#     ) -> str:
#         from meerkat.columns.deferred.base import DeferredCell

#         if isinstance(cell, DeferredCell):
#             cell = cell.get()

#         if torch.is_tensor(cell) or isinstance(cell, np.ndarray):
#             from torchvision.transforms.functional import to_pil_image

#             try:
#                 cell = to_pil_image(cell)
#             except ValueError:
#                 if isinstance(cell, np.ndarray):
#                     cell = NumpyArrayFormatter.encode(self, cell)
#                 else:
#                     cell = TensorFormatter.encode(self, cell)
#                 return cell

#         return self._encode(cell, thumbnail=thumbnail)

#     def _encode(self, image: Image, thumbnail: bool = False) -> str:
#         with BytesIO() as buffer:
#             if thumbnail:
#                 image.thumbnail((64, 64))
#             image.save(buffer, "jpeg")
#             return "data:image/jpeg;base64,{im_base_64}".format(
#                 im_base_64=base64.b64encode(buffer.getvalue()).decode()
#             )

#     def html(self, cell: Union["FileCell", Image]) -> str:
#         encoded = self.encode(cell, thumbnail=True)
#         return f'<img src="{encoded}">'


# class CodeFormatter(Formatter):

#     LANGUAGES = [
#         "js",
#         "css",
#         "html",
#         "python",
#     ]

#     def __init__(self, language: str):
#         if language not in self.LANGUAGES:
#             raise ValueError(
#                 f"Language {language} not supported."
#                 f"Supported languages: {self.LANGUAGES}"
#             )
#         self.language = language

#     cell_component = "code"

#     def encode(self, cell: str):
#         return cell

#     def html(self, cell: str):
#         return cell

#     @property
#     def cell_props(self):
#         return {"language": self.language}
