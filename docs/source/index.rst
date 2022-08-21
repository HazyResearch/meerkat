.. Meerkat documentation master file, created by
   sphinx-quickstart on Fri Jan  1 16:41:09 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Meerkat
==========================================
Meerkat provides fast and flexible data structures for working with complex machine 
learning datasets. It is designed to house your data throughout the machine learning 
lifecycle – along the way enabling interactive data exploration, cross-modal training, and 
fine-grained error analysis. 


Installation
~~~~~~~~~~~~

.. tabbed:: Main

    Meerkat is available on `PyPI <https://pypi.org/project/meerkat-ml/>`_ and can be 
    installed with pip.

    .. code-block:: 

        pip install meerkat-ml
    
    .. admonition:: Optional dependencies
    
        Some parts of Meerkat rely on optional dependencies. To install all optional
        dependencies use: 

        .. code-block:: bash
        
            pip install meerkat-ml[all] 
            
        You can also install specific
        groups optional of dependencies using something like: 

        .. code-block:: bash
        
            pip install meerkat-ml[vision,text]
        
        See `setup.py` for a full list of 
        optional dependencies.   

.. tabbed:: Latest
    
    To install the latest development version of Meerkat use:

    .. code-block:: bash

        pip install "meerkat-ml @ git+https://github.com/robustness-gym/meerkat@dev"

    .. admonition:: Optional Dependencies
    
        Some parts of Meerkat rely on optional dependencies. To install all optional
        dependencies use: 

        .. code-block:: bash

            pip install "meerkat-ml[all] @ git+https://github.com/robustness-gym/meerkat@dev"
        
        You can also install specific groups optional of dependencies using something like: 

        .. code-block:: bash

            ``pip install ``pip install "meerkat-ml[vision,text] @ git+https://github.com/robustness-gym/meerkat@dev"``
            
        See `setup.py` for a full list of optional dependencies.   

.. tabbed:: Editabled

    To install from editable source, clone the meerkat repository and pip install in
    editable mode. 

    .. code-block:: bash

        git clone https://github.com/robustness-gym/meerkat.git
        cd meerkat
        pip install -e .

    .. admonition:: Optional Dependencies
    
        Some parts of Meerkat rely on optional dependencies. To install all optional
        dependencies use: 

        .. code-block:: bash

            pip install -e .[dev]
        
        You can also install specific groups optional of dependencies using something like: 

        .. code-block:: bash

            pip install -e .[vision,text]
            
        See `setup.py` for a full list of optional dependencies.   



Next Steps
~~~~~~~~~~~~

.. panels::

    Get started with Meerkat by following along on Google Colab. 

    .. link-button:: https://drive.google.com/file/d/15kPD6Kym0MOpICafHgO1pCt8T2N_xevM/view?usp=sharing 
        :classes: btn-primary btn-block stretched-link
        :text: Walkthrough Notebook
    ---

    Learn more about the motivation behind Meerkat and what it enables. 

    .. link-button:: https://www.notion.so/sabrieyuboglu/Meerkat-DataPanels-for-Machine-Learning-64891aca2c584f1889eb0129bb747863
        :classes: btn-primary btn-block stretched-link
        :text: Introductory Blog Post 


.. _Issues: https://github.com/robustness-gym/meerkat/issues/


.. toctree::
   :hidden:
   :maxdepth: 2

   guide/guide
   datasets/datasets


.. toctree::
    :hidden:
    :maxdepth: 2

    apidocs/index


..
    Indices and tables
    ==================

    * :ref:`genindex`
    * :ref:`modindex`
    * :ref:`search`

