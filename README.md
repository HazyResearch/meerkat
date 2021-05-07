
<div align="center">
    <img src="docs/mosaic.png" height=100 alt="Mosaic logo"/>
</div>

-----

![GitHub Workflow Status](https://img.shields.io/github/workflow/status/robustness-gym/mosaic/CI)
![GitHub](https://img.shields.io/github/license/robustness-gym/mosaic)
[![codecov](https://codecov.io/gh/robustness-gym/mosaic/branch/main/graph/badge.svg?token=MOLQYUSYQU)](https://codecov.io/gh/robustness-gym/mosaic)
[![Documentation Status](https://readthedocs.org/projects/mosaic/badge/?version=latest)](https://mosaic.readthedocs.io/en/latest/?badge=latest)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

The Mosaic project's goal is to create clean abstractions for ML practitioners to load, manipulate, 
train, evaluate, inspect, visualize and interact with complex high-dimensional, multi-modal data. 

Mosaic's core contribution is the `DataPanel`, a simple columnar data abstraction that 
can house arbitrary columns of complex data side-by-side. 
The `DataPanel` allows users to work with text, image, medical imaging, time-series, 
video, and other complex Python objects with clean, high-level interfaces.

## Supported Columns

| Column             | Supported | Description                                                  |
|--------------------|-----------|--------------------------------------------------------------|
| `ListColumn`       | Yes       | Flexible and can hold any type of data.                      |
| `NumpyArrayColumn` | Yes       | np.ndarray behavior for vectorized operations.               |
| `CellColumn`       | Yes       | Like `ListColumn`, but optimized for `AbstractCell` objects. |
| `SpacyColumn`      | Yes       | Optimized to hold spaCy Doc objects.                         |
| `EmbeddingColumn`  | Planned   | Optimized for embeddings and operations on embeddings.       |
| `PredictionColumn` | Planned   | Optimized for predictions.                                   |


## `DataPanel`

A `DataPanel` is a Python object that contains a collection of columns. 


#### Create a `DataPanel`
```python
from mosaic import DataPanel

# Create a simple DataPanel
dp = DataPanel({
    'text': ['The quick brown fox.', 'Jumped over.', 'The lazy dog.'],
    'imagepath': ['fox.png', 'jump.png', 'dog.png'],
})
```


## `Column`

Mosaic supports a variety of columns.

```python
from mosaic import SomeColumn

sc = SomeColumn(data=some_data)
len(sc) == len(some_data)

```

## About
Mosaic is being developed at Stanford's Hazy Research Lab. Please reach out to `kgoel [at] cs [dot] stanford [dot] edu` if you would like to use or contribute to Mosaic.
