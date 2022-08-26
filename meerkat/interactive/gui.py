from ast import Call
from copy import copy
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, List, Union

from IPython.display import IFrame
from pydantic import BaseModel

import meerkat as mk
from meerkat.mixins.collate import identity_collate
from meerkat.mixins.identifiable import IdentifiableMixin
from meerkat.ops.sliceby.sliceby import SliceBy
from meerkat.state import state

from .api.routers.interface import Interface


class GUI:
    @staticmethod
    def launch_interface(interface: Interface, return_url: bool = False):

        if state.network_info is None:
            raise ValueError(
                "Interactive mode not initialized."
                "Run `network, register_api = mk.interactive_mode()` followed by "
                "`register_api()` first."
            )

        url = f"{state.network_info.npm_server_url}/interface?id={interface.id}"
        if return_url:
            return url
        # return HTML(
        #     "<style>iframe{width:100%}</style>"
        #     '<script src="https://cdnjs.cloudflare.com/ajax/libs/iframe-resizer/4.3.1/iframeResizer.min.js"></script>'
        #     f'<iframe id="meerkatIframe" src="{url}"></iframe>'
        #     "<script>iFrameResize({{ log: true }}, '#meerkatIframe')</script>"
        # )
        return IFrame(url, width="100%", height="1000")


class DataPanelGUI(GUI):
    def __init__(self, dp: mk.DataPanel):
        self.dp = dp

    def table(self, return_url: bool = False) -> IFrame:

        interface = Interface(
            component="table",
            props=dict(
                type="table",
                nrows=len(self.dp),
                dp=self.dp.id,
            ),
        )
        return self.launch_interface(interface, return_url=return_url)

    def gallery(self):
        pass


class SliceByGUI(GUI):
    def __init__(self, sb: SliceBy):
        self.sb = sb

    def cards(
        self,
        main_column: str,
        tag_columns: List[str] = None,
        aggregations: Dict[
            str, Callable[[mk.DataPanel], Union[int, float, str]]
        ] = None,
    ) -> IFrame:
        """_summary_

        Args:
            main_column (str): This column will be shown.
            tag_columns (List[str], optional): _description_. Defaults to None.
            aggregations (Dict[
                str, Callable[[mk.DataPanel], Union[int, float, str]]
            ], optional): A dictionary mapping from aggregation names to functions
                that aggregate a DataPanel. Defaults to None.

        Returns:
            IFrame: _description_
        """
        if aggregations is None:
            aggregations = {}

        aggregations = {k: Aggregation(v) for k, v in aggregations.items()}

        interface = Interface(
            component="sliceby-cards",
            props=dict(
                type="sliceby-cards",
                sliceby_id=self.sb.id,
                datapanel_id=self.sb.data.id,
                main_column=main_column,
                tag_columns=tag_columns if tag_columns is not None else [],
                aggregations={
                    k: v.id for k, v in aggregations.items()
                },  # we pass the id of the aggregations to the frontend
            ),
            # and keep a handle on them in the interfaces store so they aren't gc'd
            store={"aggregations": aggregations},
        )
        return self.launch_interface(interface)


class Aggregation(IdentifiableMixin):

    identifiable_group: str = "aggregations"

    def __init__(self, func: Callable[[mk.DataPanel], Union[int, float, str]]):
        self.func = func
        super().__init__()

    def __call__(self, dp: mk.DataPanel) -> Union[int, float, str]:
        return self.func(dp)


class Component:
    pass

    def prepare_config(self):
        pass


class Box(IdentifiableMixin):
    identifiable_group: str = "boxes"

    def __init__(self, obj):
        super().__init__()
        self.obj = obj

        self.children: List[Operation] = []

    @property
    def config(self):
        return BoxConfig(box_id=self.id, type="DataPanel")


class Modification(BaseModel):
    box_id: str
    scope: List[str]

    @property
    def box(self):
        return state.identifiables.get(group="boxes", id=self.box_id)


class Store(IdentifiableMixin):
    identifiable_group: str = "stores"

    def __init__(self, value: Any):
        super().__init__()
        self.value = value

    @property
    def config(self):
        return StoreConfig(
            store_id=self.id,
            value=self.value,
        )


class Pivot(Box):
    pass


class Derived(Box):
    pass


# @dataclass
# class Operation:
#     fn: Callable
#     kwargs: Dict

#     def __call__(self):
#         # deref the kwargs (check which things are Box)
#         #_apply fn
#         pass


def trigger(modifications: List[Modification]) -> List[Modification]:
    modifications = copy(modifications)
    all_modifications = []
    while modifications:
        modification = modifications.pop(0)
        all_modifications.append(modification)
        box = modification.box
        for op in box.children:
            derived = op()
            modifications.append(Modification(box_id=derived.id, scope=[]))

    # ops_to_exec, modifications = topological(pivots)

    # for op in ops_to_exec:
    #     op()

    return all_modifications


class StoreConfig(BaseModel):
    store_id: str
    value: Any


class ComponentConfig(BaseModel):
    component_id: str
    component: str
    props: Dict


class BoxConfig(BaseModel):
    box_id: str
    type: str = "DataPanel"


class PivotConfig(BoxConfig):
    pass


class DerivedConfig(BoxConfig):
    pass


class InterfaceConfig(BaseModel):

    pivots: List[PivotConfig]
    stores: List[StoreConfig]
    components: List[ComponentConfig]


class Component(IdentifiableMixin):

    identifiable_group: str = "components"

    name: str

    @property
    def config(self):
        return ComponentConfig(
            component_id=self.id, component=self.name, props=self.props
        )

    @property
    def props(self):
        return {}


Storeable = Union[int, str, float]


def make_store(value: Union[str, Storeable]):
    if isinstance(value, Store):
        return value
    return Store(value)


class Match(Component):

    name = "Match"

    def __init__(self, pivot: Pivot, against: Union[Store, str]):
        super().__init__()
        self.pivot = pivot
        self.against: Store = make_store(against)
        self.col = Store("")
        self.text = Store("")

    @property
    def props(self):
        return {
            "against": self.against.config,
            "dp": self.pivot.config,
            "col": self.col.config,
            "text": self.text.config,
        }


class Gallery(Component):

    name = "Gallery"

    def __init__(
        self,
        dp: Box,
    ) -> None:
        super().__init__()
        self.dp = dp

    @property
    def props(self):
        return {
            "dp": self.dp.config,
        }


class Plot(Component):
    name: str = "Plot"

    def __init__(
        self,
        dp: Pivot,
        selection: Pivot,
        x: Union[str, Store],
        y: Union[str, Store],
        x_label: Union[str, Store],
        y_label: Union[str, Store],
        type: str = "scatter",
    ) -> None:
        super().__init__()
        self.dp = dp
        self.selection = selection
        self.x = make_store(x)
        self.y = make_store(y)
        self.x_label = make_store(x_label)
        self.y_label = make_store(y_label)
        self.type = type

    @property
    def props(self):
        return {
            "dp": self.dp.config,
            "selection": self.selection.config,
            "x": self.x.config,
            "y": self.y.config,
            "x_label": self.x_label.config,
            "y_label": self.y_label.config,
            "type": self.type,
        }


@dataclass
class Operation:
    fn: Callable
    args: List[Any]
    kwargs: Dict[str, Any]
    result: Derived

    def __call__(self):
        unpacked_args, unpacked_kwargs, _ = _unpack_boxes(*self.args, **self.kwargs)
        result = self.fn(*unpacked_args, **unpacked_kwargs)
        # TODO: result may be multiple
        self.result.obj = result
        return self.result


def _unpack_boxes(*args, **kwargs):
    # TODO(Sabri): this should be nested
    boxes = []
    unpacked_args = []
    for arg in args:
        if isinstance(arg, Box):
            boxes.append(arg)
            unpacked_args.append(arg.obj)
        else:
            unpacked_args.append(arg)

    unpacked_kwargs = {}
    for k, v in kwargs.items():
        if isinstance(v, Box):
            boxes.append(v)
            unpacked_kwargs[k] = v.obj
        else:
            unpacked_kwargs[k] = v

    return unpacked_args, unpacked_kwargs, boxes


def interface_op(fn: Callable):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        # TODO(Sabri): this should be nested
        unpacked_args, unpacked_kwargs, boxes = _unpack_boxes(*args, **kwargs)

        result = fn(*unpacked_args, **unpacked_kwargs)

        if len(boxes) > 0:
            derived = Derived(result)
            op = Operation(fn=fn, args=args, kwargs=kwargs, result=derived)
            for input_box in boxes:
                input_box.children.append(op)
            # TODO(Sabri): this should be nested
            return derived

        return result

    return wrapper


@interface_op
def head(dp: mk.DataPanel, n: int = 5):
    new_dp = dp.head(n)
    import numpy as np

    new_dp["head_column"] = np.zeros(len(new_dp))
    return new_dp


class PlotInterface(Interface):
    def __init__(
        self,
        dp: mk.DataPanel,
        id_column: str,
    ):
        super().__init__()
        self.id_column = id_column

        self.pivots = []
        self.stores = []
        self.dp = dp

        self._layout()

    def pivot(self, obj):
        # checks whether the object is valid pivot

        pivot = Pivot(obj)
        self.pivots.append(pivot)

        return pivot

    def store(self, obj):
        # checks whether the object is valid store

        store = Store(obj)
        self.stores.append(store)

        return store

    def _layout(self):

        # Setup pivots
        dp_pivot = self.pivot(self.dp)
        selection_dp = mk.DataPanel({self.id_column: []})
        selection_pivot = self.pivot(selection_dp)

        # Setup stores
        against = self.store("image")

        # Setup computation graph
        # merge_derived: Derived = mk.merge(
        #     left=dp_pivot, right=selection_pivot, on=self.id_column
        # )
        merge_derived = head(dp_pivot, n=5)

        # Setup components
        match_x: Component = Match(dp_pivot, against=against)
        match_y: Component = Match(dp_pivot, against=against)
        plot: Component = Plot(
            dp_pivot,
            selection=selection_pivot,
            x=match_x.col,
            y=match_y.col,
            x_label=match_x.text,
            y_label=match_y.text,
        )
        gallery: Component = Gallery(merge_derived)

        # TODO: make this more magic
        self.components = [match_x, match_y, plot, gallery]

    @property
    def config(self):
        return InterfaceConfig(
            pivots=[pivot.config for pivot in self.pivots],
            stores=[store.config for store in self.stores],
            components=[component.config for component in self.components],
        )

    def launch(self, return_url: bool = False):

        if state.network_info is None:
            raise ValueError(
                "Interactive mode not initialized."
                "Run `network, register_api = mk.interactive_mode()` followed by "
                "`register_api()` first."
            )

        url = f"{state.network_info.npm_server_url}/interface?id={self.id}"
        if return_url:
            return url
        # return HTML(
        #     "<style>iframe{width:100%}</style>"
        #     '<script src="https://cdnjs.cloudflare.com/ajax/libs/iframe-resizer/4.3.1/iframeResizer.min.js"></script>'
        #     f'<iframe id="meerkatIframe" src="{url}"></iframe>'
        #     "<script>iFrameResize({{ log: true }}, '#meerkatIframe')</script>"
        # )
        return IFrame(url, width="100%", height="1000")
