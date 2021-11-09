
<div align="center">
    <img src="docs/assets/meerkat_banner.png" height=100 alt="Meerkat logo"/>
</div>

-----

![GitHub Workflow Status](https://img.shields.io/github/workflow/status/robustness-gym/meerkat/CI)
![GitHub](https://img.shields.io/github/license/robustness-gym/meerkat)
[![Documentation Status](https://readthedocs.org/projects/meerkat/badge/?version=latest)](https://meerkat.readthedocs.io/en/latest/?badge=latest)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![codecov](https://codecov.io/gh/robustness-gym/meerkat/branch/main/graph/badge.svg?token=MOLQYUSYQU)](https://codecov.io/gh/robustness-gym/meerkat)

Meerkat provides fast and flexible data structures for working with complex machine learning datasets. 

[**Getting Started**](⚡️-Quickstart)
| [**What is Meerkat?**](💡-what-is-Meerkat)
| [**Docs**](https://meerkat.readthedocs.io/en/latest/index.html)
| [**Contributing**](CONTRIBUTING.md)
| [**Blogpost**](https://www.notion.so/sabrieyuboglu/Meerkat-DataPanels-for-Machine-Learning-64891aca2c584f1889eb0129bb747863)
| [**About**](✉️-About)


## ⚡️ Quickstart
```bash
pip install meerkat-ml
``` 
> _Optional_: some parts of Meerkat rely on optional dependencies. If you know which optional dependencies you'd like to install, you can do so using something like `pip install meerkat-ml[dev,text]` instead. See `setup.py` for a full list of optional dependencies.   

> _Installing from dev_: `pip install "meerkat-ml[text] @ git+https://github.com/robustness-gym/meerkat@dev"`
 
Load a dataset into a `DataPanel` and get going!
```python
import meerkat as mk
from meerkat.contrib.imagenette import download_imagenette

download_imagenette(".")
dp = mk.DataPanel.from_csv("imagenette2-160/imagenette.csv")
dp["img"] = mk.ImageColumn.from_filepaths(dp["img_path"])

dp[["label", "split", "img"]].lz[:3]
```
<img width="500" alt="readme_figure" src="https://user-images.githubusercontent.com/32822771/132963373-b4ae2f22-ee89-483c-b131-12e2fa3c9284.png">

To learn more, continue following along in our tutorial:  
[![Open intro](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15kPD6Kym0MOpICafHgO1pCt8T2N_xevM#scrollTo=03nX_l19B5Zt&uniqifier=1) 

## 💡 What is Meerkat?
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

## ✉️ About
Meerkat is being developed at Stanford's Hazy Research Lab. Please reach out to `kgoel [at] cs [dot] stanford [dot] edu` if you would like to use or contribute to Meerkat.
