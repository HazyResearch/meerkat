Datasets
========================
Meerkat provides a dataset registry that makes it easy to download datasets and load them into Meerkat data structures.
For example, using :func:`~meerkat.datasets.get` we can download and prepare the `Imagenette <https://github.com/fastai/imagenette>`_ dataset: 

.. ipython:: python
   
   import meerkat as mk
   @verbatim
   dp = mk.datasets.get("imagenette")

Some datasets have multiple versions, for example Imagenette provides a full-size version as well as 320 pixel and 160 pixel versions. You can list a dataset's available versions with :func:`~meerkat.datasets.versions`:

.. ipython:: python

   mk.datasets.versions("imagenette")
   
   @verbatim
   mk.datasets.get("imagenette", version="160px")

By default datasets are downloaded to ``~/.meerkat/datasets/{name}/{version}``. However, if you already have the dataset downloaded elsewhere or you want to download to a different location, you can specify the ``dataset_dir`` argument. 

.. code-block:: python
   
      dp = mk.datasets.get("imagenette", dataset_dir="/local/download/of/imagenette/full")

You can also configure Meerkat to use a different default root directory. By setting the ``mk.config.datasets.root_dir = "/local/download/of"``, the default location for datasets will be ``/local/download/of/datasets/{name}/{version}``.

*How does Meerkat's dataset registry fit in with other dataset hubs?*    The purpose of the Meerkat dataset registry is to provide *code* for downloading datasets and loading them into :class:`~meerkat.DataPanel` objects. The Meerkat registry, like `Torchvision Datasets <https://pytorch.org/vision/stable/datasets.html>`_, doesn't actually host any data. 
In contrast, dataset hubs like `HuggingFace Datasets <https://huggingface.co/docs/datasets/index>`_ and `Activeloop Hub <https://www.activeloop.ai/>`_ are great community efforts that *do* host data. So, the Meerkat registry is complementary to these hubs: in fact, we can currently load any dataset in the HuggingFace hubs directly through our registry. For example, we can load the `IMBD dataset <https://huggingface.co/datasets/imdb>`_ hosted on HuggingFace with ``mk.datasets.get("imdb")``. 


.. admonition:: Contributing Datasets

   We encourage users to contribute datasets to the Meerkat registry. If you're already using Meerkat with your dataset, contributing it to the registry is straightforward: you just share the code that you're already using to load the dataset into Meerkat. Please follow the instructions in :doc:`contributing_datasets`. 


    
The table below lists all of the datasets currently in the meerkat registry. 
You can also list these datasets programmatically with ``mk.datasets.catalog``. 


.. raw:: html
   :file: datasets_table.html
   
.. toctree::
   :hidden:
   :maxdepth: 2

   contributing_datasets
   