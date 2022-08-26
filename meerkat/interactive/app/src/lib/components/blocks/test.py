import numpy as np

import meerkat as mk

mk.DataPanel(
    {
        "a": np.arange(10),
        "color": np.arange(10, 20),
        "time": np.arange(100, 110),
        "label": (np.arange(-5, 5) > 0).astype(int),
    }
)
