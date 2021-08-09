"""Import nn module classes."""
# flake8: noqa
import warnings

from meerkat.errors import ExperimentalWarning

warnings.warn(
    ExperimentalWarning(
        "The `meerkat.nn` module is experimental and has limited test coverage. "
        "Proceed with caution."
    )
)

from meerkat.nn.embedding_column import EmbeddingColumn
from meerkat.nn.huggingfacemodel import HuggingfaceModel
from meerkat.nn.instances_column import InstancesColumn
from meerkat.nn.prediction_column import ClassificationOutputColumn
from meerkat.nn.segmentation_column import SegmentationOutputColumn
from meerkat.nn.tensormodel import TensorModel
