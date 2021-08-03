
<div align="center">
    <img src="docs/meerkat_banner.png" height=100 alt="Meerkat logo"/>
</div>

-----

![GitHub Workflow Status](https://img.shields.io/github/workflow/status/robustness-gym/meerkat/CI)
![GitHub](https://img.shields.io/github/license/robustness-gym/meerkat)
[![Documentation Status](https://readthedocs.org/projects/meerkat/badge/?version=latest)](https://meerkat.readthedocs.io/en/latest/?badge=latest)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
<!---[![codecov](https://codecov.io/gh/robustness-gym/meerkat/branch/main/graph/badge.svg?token=MOLQYUSYQU)](https://codecov.io/gh/robustness-gym/meerkat)--->

Meerkat provides fast and flexible data structures for working with complex machine learning datasets. 

[**Getting Started**](#getting-started)
| [**What is Meerkat?**](#what-is-meerkat)
| [**Supported Columns**](#supported-columns)
| [**Docs**](https://meerkat.readthedocs.io/en/latest/index.html)
| [**Contributing**](CONTRIBUTING.md)
| [**About**](#about)


## Getting started
```bash
pip install meerkat-ml
``` 
> Note: some parts of Meerkat rely on optional dependencies. If you know which optional dependencies you'd like to install, you can do so using something like `pip install meerkat-ml[dev,text]` instead. See `setup.py` for a full list of optional dependencies.   
 
Load your dataset into a `DataPanel` and get going!
```python
import meerkat as mk
dp = mk.DataPanel.from_csv("...")
```


## What is Meerkat?
Meerkat makes it easier for ML practitioners to interact with high-dimensional, multi-modal data. It provides simple abstractions for data inspection, model evaluation and model training supported by efficient and robust IO under the hood.  

Meerkat's core contribution is the `DataPanel`, a simple columnar data abstraction. The Meerkat `DataPanel` can house columns of arbitrary type – from integers and strings to complex, high-dimensional objects like videos, images, medical volumes and graphs. 

**`DataPanel` loads high-dimensional data lazily.**     A full high-dimensional dataset won't typically fit in memory. Behind the scenes, `DataPanel` handles this by only materializing these objects when they are needed. 
```python
import meerkat as mk

# Images are NOT read from disk at DataPanel creation...
dp = mk.DataPanel({
    'text': ['The quick brown fox.', 'Jumped over.', 'The lazy dog.'],
    'image': mk.ImageColumn.from_filepaths(['fox.png', 'jump.png', 'dog.png']),
    'label': [0, 1, 0]
}) 

# ...only at this point is "fox.png" read from disk
dp["image"][0]
```

**`DataPanel` supports advanced indexing.**  Using indexing patterns similar to those of Pandas and NumPy, we can access a subset of a `DataPanel`'s rows and columns. 
```python
import meerkat as mk
dp = ... # create DataPanel

# Pull a column out of the DataPanel
new_col: mk.ImageColumn = dp["image"]

# Create a new DataPanel from a subset of the columns in an existing one
new_dp: mk.DataPanel = dp[["image", "label"]] 

# Create a new DataPanel from a subset of the rows in an existing one
new_dp: mk.DataPanel = dp[10:20] 
new_dp: mk.DataPanel = dp[np.array([0,2,4,8])]

# Pull a column out of the DataPanel and get a subset of its rows 
new_col: mk.ImageColumn = dp["image"][10:20]
```

**`DataPanel` supports `map`, `update` and `filter` operations.**  When training and evaluating our models, we often perform operations on each example in our dataset (*e.g.* compute a model's prediction on each example, tokenize each sentence, compute a model's embedding for each example) and store them . The `DataPanel` makes it easy to perform these operations and produce new columns (via `DataPanel.map`), store the columns alongside the original data (via `DataPanel.update`), and extract an important subset of the datset (via `DataPanel.filter`). Under the hood, dataloading is multiprocessed so that costly I/O doesn't bottleneck our computation. Consider the example below where we use update a `DataPanel` with two new columns holding model predictions and probabilities.  
```python
# A simple evaluation loop using Meerkat 
dp: DataPanel = ... # get DataPanel
model: nn.Module = ... # get the model
model.to(0).eval() # prepare the model for evaluation

@torch.no_grad()
def predict(batch: dict):
    probs = torch.softmax(model(batch["input"].to(0)), dim=-1)
    return {"probs": probs.cpu(), "pred": probs.cpu().argmax(dim=-1)}

# updated_dp has two new `TensorColumn`s: 1 for probabilities and one
# for predictions
updated_dp: mk.DataPanel = dp.update(function=predict, batch_size=128, is_batched_fn=True)
```

**`DataPanel` is extendable.** Meerkat makes it easy for you to make custom column types for our data. The easiest way to do this is by subclassing `AbstractCell`. Subclasses of `AbstractCell` are meant to represent one element in one column of a `DataPanel`. For example, say we want our `DataPanel` to include a column of videos we have stored on disk. We want these videos to be lazily loaded using [scikit-video](http://www.scikit-video.org/stable/index.html), so we implement a `VideoCell` class as follows: 
```python
import meerkat as mk
import skvideo.io

class VideoCell(mk.AbstractCell):
    
    # What information will we eventually  need to materialize the cell? 
    def __init__(filepath: str):
        super().__init__()
        self.filepath = filepath
    
    # How do we actually materialize the cell?
    def get(self):
        return skvideo.io.vread(self.filepath)
    
    # What attributes should be written to disk on `VideoCell.write`?
    @classmethod
    def _state_keys(cls) -> Collection:
        return {"filepath"}

# We don't need to define a `VideoColumn` class and can instead just
# create a CellColumn fro a list of `VideoCell`
vid_column = mk.CellColumn(map(VideoCell, ["vid1.mp4", "vid2.mp4", "vid3.mp4"]))
```
## Supported Columns
Meerkat ships with a number of core column types and the list is growing.
#### Core Columns
| Column             |Description                                                  |
|--------------------|--------------------------------------------------------------|
| `ListColumn`       | Flexible and can hold any type of data.                      |
| `NumpyArrayColumn` | [`np.ndarray`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html) behavior for vectorized operations.               |
| `TensorColumn`     | [`torch.tensor`](https://pytorch.org/docs/stable/tensors.html) behavior for vectorized operations on the GPU.    |
| `ImageColumn`      | Holds images stored on disk (*e.g.* as PNG or JPEG)                              |
| `VideoColumn`      | Holds videos stored on disk (*e.g.* as MP4)                              |
| `MedicalVolumeColumn` |Optimized for medical images stored DICOM or NIFTI format.|
| `SpacyColumn`      | Holds processed text in spaCy Doc objects.                         |
| `EmbeddingColumn`  | Holds embeddings and provides utility methods like `umap` and `build_faiss_index`.|
| `ClassificationOutputColumn` | Holds classifier predictions.|
| `CellColumn`       | Like `ListColumn`, but optimized for `AbstractCell` objects. |


#### Contributed Columns
| Column             | Supported | Description                                                  |
|--------------------|-----------|--------------------------------------------------------------|
| `WILDSInputColumn`       | Yes       | Build `DataPanel`s for the [WILDS benchmark](https://wilds.stanford.edu/).|


## About
Meerkat is being developed at Stanford's Hazy Research Lab. Please reach out to `kgoel [at] cs [dot] stanford [dot] edu` if you would like to use or contribute to Meerkat.
