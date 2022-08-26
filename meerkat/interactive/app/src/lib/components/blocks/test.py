import meerkat as mk
import numpy as np
from meerkat.interactive import 

mk.DataPanel({
    'a': np.arange(10),
    'color': np.arange(10, 20),
    'time': np.arange(100, 110),
    'label': (np.arange(-5, 5) > 0).astype(int),
})

