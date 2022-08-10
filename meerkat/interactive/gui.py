from IPython.display import HTML, IFrame

import meerkat as mk
from meerkat.state import state

from .api.routers.interface import Interface


class GUI:
    def __init__(self, dp: mk.DataPanel):
        self.dp = dp

    def table(self, nrows=10, return_iframe: bool = True) -> IFrame:

        interface = Interface(
            config=dict(
                type="table",
                nrows=nrows,
                dp=self.dp.id,
            ),
        )
        url = f"{state.network_info.npm_server_url}/interface?id={interface.id}"
        print(url)
        if return_iframe:
            # return HTML(
            #     "<style>iframe{width:100%}</style>"
            #     '<script src="https://cdnjs.cloudflare.com/ajax/libs/iframe-resizer/4.3.1/iframeResizer.min.js"></script>'
            #     f'<iframe id="meerkatIframe" src="{url}"></iframe>'
            #     "<script>iFrameResize({{ log: true }}, '#meerkatIframe')</script>"
            # )
            return IFrame(url, width="100%", height="1000")
        else:
            return url

    def gallery(self):
        pass
