.. Meerkat documentation master file, created by
   sphinx-quickstart on Fri Jan  1 16:41:09 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Meerkat
==========================================
Meerkat provides fast and flexible data structures for working with complex machine learning datasets. 

..
    Read more about [Meerkat]().

Installation
^^^^^^^^^^^^^

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




Meerkat is *under active development* so expect rough edges.
Feedback and contributions are welcomed and appreciated.
You can submit bugs and feature suggestions on Github Issues_
and submit contributions using a pull request.

You can get started by going to the installation_ page.

.. _Issues: https://github.com/robustness-gym/meerkat/issues/
.. _installation: getting-started/install.md

.. toctree::
   :hidden:
   :maxdepth: 2

   overview.md
   guide/guide.md
   datasets/datasets.md


.. toctree::
    :hidden:
    :maxdepth: 2

    apidocs/meerkat


..
    Indices and tables
    ==================

    * :ref:`genindex`
    * :ref:`modindex`
    * :ref:`search`

