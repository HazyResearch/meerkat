from functools import wraps
from typing import Callable, Dict, List, Union

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


class Box:
    pass 

class Pivot(Box):
    pass 

    def prepare_config(self):
        pass


class Derived(Box):
    pass 
class Store:
    pass

    def prepare_config(self):
        pass

class Component: 
    pass 

    def prepare_config(self):
        pass

class StoreConfig(BaseModel):
    pass

class ComponentConfig(BaseModel):
    pass

class PivotConfig(BaseModel):
    pass




class InterfaceConfig(BaseModel):

    pivots: List[PivotConfig]
    stores: List[StoreConfig]
    Components: List[ComponentConfig]

class PlotInterface:
    
    def __init__(
        self,
        dp: mk.DataPanel,
        id_column: str,
    ):
        self.id_column = id_column

        self.pivots = []
        
        self.layout()

    
    def prepare_config(self):
        return InterfaceConfig(
            pivots=[pivot.prepare_config() for pivot in self.pivots]
            stores=[store.prepare_config() for store in self.stores]
            components=[
                component.prepare_config() for component in self.components
            ]
        )

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
        merge_derived: Derived = mk.merge(
            left=dp_pivot, right=selection_pivot, on=self.id_column
        )
        

        # Setup components
        match_x: Component = mk.gui.Match(dp_pivot, against=against)
        match_y: Component = mk.gui.Match(dp_pivot, against=against)
        plot: Component = mk.gui.Plot(
            dp_pivot, 
            selection=selection_pivot,
            x=match_x.col,
            y=match_y.col,
            x_label=match_x.text,
            y_label=match_y.text,
        )
        gallery: Component = mk.gui.Gallery(merge_derived)

        # TODO: make this more magic
        self.components = [match_x, match_y, plot, gallery]