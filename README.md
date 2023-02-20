
<div align="center">
    <img src="docs/assets/meerkat_banner.png" height=100 alt="Meerkat logo"/>
</div>

-----

![GitHub Workflow Status](https://img.shields.io/github/workflow/status/robustness-gym/meerkat/CI)
![GitHub](https://img.shields.io/github/license/robustness-gym/meerkat)
[![Documentation Status](https://readthedocs.org/projects/meerkat/badge/?version=latest)](https://meerkat.readthedocs.io/en/latest/?badge=latest)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![codecov](https://codecov.io/gh/robustness-gym/meerkat/branch/main/graph/badge.svg?token=MOLQYUSYQU)](https://codecov.io/gh/robustness-gym/meerkat)


Meerkat is a Python library aimed at technical teams that want to interactively wrangle their unstructured data with foundation models.

[**Quickstart**](⚡️-Quickstart)
| [**Docs**](https://meerkat.readthedocs.io/en/dev/index.html)
| [**Contributing**](CONTRIBUTING.md)
| [**Blogpost**](https://www.notion.so/sabrieyuboglu/Meerkat-DataPanels-for-Machine-Learning-64891aca2c584f1889eb0129bb747863)
| [**About**](✉️-About)


## ⚡️ Quickstart
We recommend installing Meerkat in a fresh virtual environment,
```bash
conda create -n meerkat python=3.10 # or use your favorite virtual environment
conda activate meerkat
pip install meerkat-ml
mk install
```
> **_GPU Install_**: If you want to use Meerkat with a GPU, you will need to install PyTorch with GPU support. See [here](https://pytorch.org/get-started/locally/) for more details.
<!-- ```bash
pip install "meerkat-ml @ git+https://github.com/robustness-gym/meerkat@clever-dev"
```  -->
> **_Optional Dependencies_**: some parts of Meerkat rely on optional dependencies e.g. audio processing may rely on utilities from `torchaudio`. We leave it up to you to install necessary dependencies when required. As a convenience, we provide bundles of optional dependencies that you can install e.g. `pip install meerkat-ml[text]` for text dependencies. See `setup.py` for a full list of optional dependencies.   

Then try one of our demos,
```bash
mk demo match 
# mk demo --help to see a full list of available demos
```

(If this didn't work for you, we'd appreciate if you could open an issue and let us know.)

**Next Steps**.
Check out our [Getting Started page](https://meerkat.readthedocs.io/en/dev/guide/guide.html) and our [documentation](https://meerkat.readthedocs.io/en/dev/guide/guide.html) to start building with Meerkat. As we work to make the documentation more comprehensive, please feel free to open an issue or reach out if you have any questions.

## ✉️ About
Meerkat is being developed at Stanford's Hazy Research Lab. Please reach out to `kgoel [at] cs [dot] stanford [dot] edu, eyuboglu [at] stanford [dot] edu, and arjundd [at] stanford [dot] edu` if you would like to use Meerkat for a project, at your company or if you have any questions.
