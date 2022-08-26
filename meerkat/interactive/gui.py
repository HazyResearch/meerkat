from functools import wraps
from typing import Callable, Dict, List, Union, Any

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

class Operation:
    pass

class Modification(BaseModel):
    box_id: int
    scope: List[str]


class Store(Box):
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
    
    def __init__(self, obj):
        super().__init__()
        self.obj = obj

    @property
    def config(self):
        return PivotConfig(
            pivot_id=self.id,
            type="DataPanel"
        )

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


def trigger(pivots: Union[Pivot, List[Pivot]]) ->  List[Modification]:
    
    
    ops_to_exec, modifications = topological(pivots)
    
    for op in ops_to_exec:
        op()
    
    return modifications


class StoreConfig(BaseModel):
    store_id: str
    value: Any 

class ComponentConfig(BaseModel):
    component_id: str
    component: str 
    props: Dict



class PivotConfig(BaseModel):
    pivot_id: str
    type: str = "DataPanel"


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
            component_id=self.id,
            component=self.name,
            props=self.props
        )
    
    @property
    def props(self):
        return {}



class Match(Component):

    name = "Match"
    
    def __init__(
        self, 
        pivot: Pivot,
        against: Union[Store, str]
    ):
        super().__init__()
        self.pivot = pivot
        self.against = against
        self._col = Store('')
        self._text = Store('')

    @property
    def stores(self):
        return [self._col, self._text]

    @property
    def props(self):
        return {}

    @property
    def col(self):
        return self._col

    @property
    def text(self):
        return self._text

        
class Gallery(Component):

    name = "Gallery"

    
    def __init__(
        self, 
        dp: Pivot,
    ) -> None:
        super().__init__()
        self.dp = dp
        

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
    ) -> None:
        super().__init__()
        self.dp = dp
        self.selection = selection
        self.x = x
        self.y = y
        self.x_label = x_label
        self.y_label = y_label

        

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
        

    def layout(self):

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
        merge_derived = None
        
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
            components=[
                component.config for component in self.components
            ]
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