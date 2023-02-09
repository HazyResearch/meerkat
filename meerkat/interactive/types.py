from typing import TYPE_CHECKING, Dict, List, TypeVar, Union

from pydantic import StrictBool, StrictFloat, StrictInt, StrictStr

if TYPE_CHECKING:
    from meerkat.dataframe import DataFrame
    from meerkat.ops.sliceby.sliceby import SliceBy

Primitive = Union[StrictInt, StrictStr, StrictFloat, StrictBool]
Storeable = Union[
    None,
    Primitive,
    List[Primitive],
    Dict[Primitive, Primitive],
    Dict[Primitive, List[Primitive]],
    List[Dict[Primitive, Primitive]],
]
T = TypeVar("T", "DataFrame", "SliceBy")
