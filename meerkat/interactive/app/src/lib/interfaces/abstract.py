from typing import List

from IPython.display import IFrame
from pydantic import BaseModel

from meerkat.interactive import PivotConfig, StoreConfig
from meerkat.interactive.app.src.lib.component.abstract import ComponentConfig
from meerkat.mixins.identifiable import IdentifiableMixin
from meerkat.state import state


class InterfaceConfig(BaseModel):

    pivots: List[PivotConfig]
    components: List[ComponentConfig]


class Interface(IdentifiableMixin):
    # TODO (all): I think this should probably be a subclassable thing that people
    # implement. e.g. TableInterface

    identifiable_group: str = "interfaces"

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
