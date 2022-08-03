from IPython.display import IFrame

import meerkat as mk

from .state import add_interface, state


class GUI:
    def __init__(self, dp: mk.DataPanel):
        self.dp = dp

    def table(self, nrows=10) -> IFrame:

        interface_id = add_interface(
            self.dp,
            config=dict(
                type="table",
                params=dict(
                    nrows=nrows,
                ),
            ),
        )

        return IFrame(
            f"{state.network_info.npm_server_url}/interface?id={interface_id}",
            width=800,
            height=800,
        )
