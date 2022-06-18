Datasets
========================
Meerkat provides a dataset registry that makes it easy to download popular datasets and load them into a :class:`~meerkat.DataPanel`. 
For example, we can load. 

.. code-block:: python
   
   import meerkat as mk
   dp = mk.get("imagenette")

.. admonition:: Contributing Datasets

   We encourage users to share their datasets by contributing to the meerkat registry. To do so, please follow the instructions in 
   :doc:`contributing_datasets`. 

*How does Meerkat's dataset registry compare to other dataset hubs?*

The purpose of the Meerkat dataset registry is to provide **code** for downloading datasets and loading them into :class:`~meerkat.DataPanels`. We do not host any data ourselves. 
Dataset hubs like `HuggingFace Datasets <https://huggingface.co/docs/datasets/index>`_ and `Activeloop Hub <https://www.activeloop.ai/>`_ are great community efforts that do host data. So, you shouldn't think of the Meerkat registry as a replacement for these hubs. In fact, we can currently load any dataset in HugginFace through our registry. 


    
The table below lists all of the datasets currently in the meerkat registry. 
You can also list these datasets programmatically `mk.datasets.catalog`. 


.. raw:: html
   :file: datasets_table.html
   
.. toctree::
   :hidden:
   :maxdepth: 2

   contributing_datasets
   