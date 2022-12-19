
Installation
~~~~~~~~~~~~

.. tab-set::
    .. tab-item:: Main

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

    .. tab-item:: Latest
        
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

    .. tab-item:: Editabled

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

