from meerkat.env import package_available

if not package_available("meddlr"):
    raise ImportError(
        "`meddlr` package not found. Please install it with `pip install meddlr`."
    )

from .perturbation import MRIPerturbationInference
from .utils import build_slice_df

__all__ = ["MRIPerturbationInference", "build_slice_df"]
