from copy import copy
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, List, Union

from IPython.display import IFrame
from pydantic import BaseModel

import meerkat as mk
from meerkat.mixins.identifiable import IdentifiableMixin
from meerkat.ops.sliceby.sliceby import SliceBy
from meerkat.state import state

from .api.routers.interface import Interface
from .app.src.lib.interfaces.match_table import MatchTableInterface
from .app.src.lib.interfaces.sliceby import SliceByInterface


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

    def table(self, *args, **kwargs) -> IFrame:
        interface = MatchTableInterface(dp=self.dp, *args, **kwargs)
        return interface.launch()

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
        return SliceByInterface(
            sliceby=self.sb,
            main_column=main_column,
            tag_columns=tag_columns,
            aggregations=aggregations,
        ).launch()


class Aggregation(IdentifiableMixin):

    identifiable_group: str = "aggregations"

    def __init__(self, func: Callable[[mk.DataPanel], Union[int, float, str]]):
        self.func = func
        super().__init__()

    def __call__(self, dp: mk.DataPanel) -> Union[int, float, str]:
        return self.func(dp)
