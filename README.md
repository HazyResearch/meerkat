
<div align="center">
    <img src="docs/assets/meerkat_banner.png" height=100 alt="Meerkat logo"/>
</div>

-----

![GitHub Workflow Status](https://img.shields.io/github/workflow/status/robustness-gym/meerkat/CI)
![GitHub](https://img.shields.io/github/license/robustness-gym/meerkat)
[![Documentation Status](https://readthedocs.org/projects/meerkat/badge/?version=latest)](https://meerkat.readthedocs.io/en/latest/?badge=latest)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![codecov](https://codecov.io/gh/robustness-gym/meerkat/branch/main/graph/badge.svg?token=MOLQYUSYQU)](https://codecov.io/gh/robustness-gym/meerkat)

An interactive and intelligent DataFrame for wrangling complex data types.

[**Getting Started**](⚡️-Quickstart)
| [**What is Meerkat?**](💡-what-is-Meerkat)
| [**Docs**](https://meerkat.readthedocs.io/en/dev/index.html)
| [**Contributing**](CONTRIBUTING.md)
| [**Blogpost**](https://www.notion.so/sabrieyuboglu/Meerkat-DataPanels-for-Machine-Learning-64891aca2c584f1889eb0129bb747863)
| [**About**](✉️-About)

> **_Note_**: Meerkat is a research project in pre-alpha, so users should expect rapid improvements as well as breaking changes to the API. If you are interested in using Meerkat, please reach out to us: eyuboglu [at] stanford [dot] edu. 

## ⚡️ Quickstart
```bash
pip install "meerkat-ml @ git+https://github.com/robustness-gym/meerkat@clever-dev"
``` 
> **_Optional_**: some parts of Meerkat rely on optional dependencies. If you know which optional dependencies you'd like to install, you can do so using something like `pip install meerkat-ml[dev,text]` instead. See `setup.py` for a full list of optional dependencies.   

 
Load a dataset into a `DataFrame` and get going!
```python
import meerkat as mk

df = mk.get("imagenette", version="160px")
df[["label", "split", "img"]].head()
```
<img width="500" alt="readme_figure" src="https://user-images.githubusercontent.com/32822771/132963373-b4ae2f22-ee89-483c-b131-12e2fa3c9284.png">

To learn more, continue following along in our tutorial:  
[![Open intro](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15kPD6Kym0MOpICafHgO1pCt8T2N_xevM#scrollTo=03nX_l19B5Zt&uniqifier=1) 

## 💡 What is Meerkat?
Meerkat is a data-wrangling library that helps data scientists and machine learning practitioners answer statistical questions about complex data types. Questions like:
<!-- TODO: Add links to notebooks demonstrating these techniques. -->
- *Does my model make systematic errors on an unlabeled subset of my data?*
- *Does this large language model generate text that reinforce harmful stereotypes?*
- *How has the distribution of abstract art in the National Gallery of Art shifted over time?*

Meerkat's core contribution is the `DataFrame`, a columnar data abstraction. The Meerkat DataFrame can house columns of arbitrary type – from integers and strings to complex, high-dimensional objects like images, audio, tensors, and graphs. It extends the traditional DataFrame API with (1) intelligent operations backed by machine learning and (2) interactive components that help users control and validate those operations.

Please see our [documentation](https://meerkat.readthedocs.io/en/dev/guide/guide.html) for more information. As we work to make the documentation more comprehensive, please feel free to open an issue or reach out if you have any questions. 

## ✉️ About
Meerkat is being developed at Stanford's Hazy Research Lab. Please reach out to `kgoel [at] cs [dot] stanford [dot] edu, eyuboglu [at] stanford [dot] edu, and arjundd [at] stanford [dot] edu` if you would like to use or contribute to Meerkat.
