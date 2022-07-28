import meerkat as mk
from IPython.display import IFrame

from .state import add_interface


class GUI:
    def __init__(self, dp: mk.DataPanel):
        self.dp = dp

    def table(self) -> IFrame:

        interface_id = add_interface(self.dp, {"test": 123})

        return IFrame(
            f"http://localhost:5173/interface/{interface_id}",
            width=800,
            height=800,
        )
