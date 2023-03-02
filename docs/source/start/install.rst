.. _install:

Installation
~~~~~~~~~~~~

.. note::

    If you run into any installation troubles, open an 
    issue on `GitHub <https://github.com/hazyresearch/meerkat/issues>`_ or 
    post a message on our `Discord server <https://discord.gg/jwQMr6em>`_.
    We're happy to help!

.. tab-set::
    .. tab-item:: Standard
        
        Meerkat is available on `PyPI <https://pypi.org/project/meerkat-ml/>`_ and can be 
        installed with pip.
    

        .. code-block:: 

            pip install meerkat-ml
            
        
        .. dropdown:: Optional Dependencies
            
            The core Meerkat package only has a handful of dependencies. Depending on what you 
            use Meerkat for, you may need to install many optional dependencies, since some parts 
            of Meerkat rely on them. 
            
            To install all optional dependencies use: 
            
            .. code-block:: bash
            
                pip install meerkat-ml[all] 
                
            You can also install specific groups of optional of dependencies using something like: 

            .. code-block:: bash
            
                pip install meerkat-ml[vision,text]
            
            See `setup.py` for a full list of 
            optional dependencies.   

    .. tab-item:: Latest on Main
        
        To install the latest development version of Meerkat use:

        .. code-block:: bash

            pip install "meerkat-ml @ git+https://github.com/HazyResearch/meerkat@main"

        .. dropdown:: Optional Dependencies
        
            Some parts of Meerkat rely on optional dependencies. To install all optional
            dependencies use: 

            .. code-block:: bash

                pip install "meerkat-ml[all] @ git+https://github.com/HazyResearch/meerkat@main"
            
            You can also install specific groups of optional of dependencies using something like: 

            .. code-block:: bash

                pip install "meerkat-ml[vision,text] @ git+https://github.com/HazyResearch/meerkat@main"
                
            See `setup.py` for a full list of optional dependencies.   

    .. tab-item:: Editable for Development

        To install from editable source, clone the meerkat repository and pip install in
        editable mode. 

        .. code-block:: bash

            git clone https://github.com/HazyResearch/meerkat.git
            cd meerkat
            pip install -e .

        .. dropdown:: Optional Dependencies
        
            Some parts of Meerkat rely on optional dependencies. To install all optional
            dependencies use: 

            .. code-block:: bash

                pip install -e .[dev]
            
            You can also install specific groups of optional of dependencies using something like: 

            .. code-block:: bash

                pip install -e .[vision,text]
                
            See `setup.py` for a full list of optional dependencies.   

.. tip::

    Installing Meerkat will automatically install the Meerkat CLI, which is used to
    create and run interactive applications.

    See :ref:`meerkat-cli` for more information about how to use the Meerkat CLI. 
    You can also type 

    .. code-block:: bash

        mk --help

    in your terminal to see a list of available commands.


.. admonition:: Additional installation required for custom components in interactive applications

    You will additionally need the following programs if you would like to build custom interactive components. 

    - Node.js (``node``) version ``>=18.0.0``
    - Node Package Manager (``npm``) version ``>=8.0.0``

    Once you have installed Node.js and npm, you can use the Meerkat CLI to
    install dependencies for creating Meerkat components.

    .. code-block:: bash

        mk install

