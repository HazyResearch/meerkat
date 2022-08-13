from typing import List

from IPython.display import HTML, IFrame

import meerkat as mk
from meerkat.ops.sliceby.sliceby import SliceBy
from meerkat.state import state

from .api.routers.interface import Interface


class GUI:
    @staticmethod
    def launch_interface(interface: Interface):

        if state.network_info is None:
            raise ValueError(
                "Interactive mode not initialized."
                "Run `network, register_api = mk.interactive_mode()` followed by "
                "`register_api()` first."
            )

        url = f"{state.network_info.npm_server_url}/interface?id={interface.id}"
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

    def table(self, nrows=10) -> IFrame:

        interface = Interface(
            component="table",
            props=dict(
                type="table",
                nrows=nrows,
                dp=self.dp.id,
            ),
        )
        return self.launch_interface(interface)

    def gallery(self):
        pass


class SliceByGUI(GUI):
    def __init__(self, sb: SliceBy):
        self.sb = sb

    def cards(
        self,
        main_column: str,
        tag_columns: List[str] = None,
    ) -> IFrame:
        interface = Interface(
            component="sliceby-cards",
            props=dict(
                type="sliceby-cards",
                sliceby_id=self.sb.id,
                datapanel_id=self.sb.data.id,
                main_column=main_column,
                tag_columns=tag_columns if tag_columns is not None else [],
            ),
        )
        return self.launch_interface(interface)
