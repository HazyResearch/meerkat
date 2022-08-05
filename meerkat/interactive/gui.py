from IPython.display import IFrame

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
            return IFrame(
                url,
                width=800,
                height=800,
            )
        else:
            return url
