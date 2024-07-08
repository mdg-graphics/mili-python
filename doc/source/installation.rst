Installation
====================

There are currently two methods of installing the mili-python module. The recommeded method is installing from the WCI Nexus Mirror (LLNL specific). However, if this does not work, users may also install from source using the Git or GitHub repositories.

---------------------------
From the WCI Nexus Mirror
---------------------------

.. code-block:: bash

    # Make sure you are using python > 3.7
    module load python/3.10.8
    # Create and activate a python virtual environment
    python -m venv <venv_name>
    source <venv_name>/bin/activate
    # Upgrade pip (numpy > 1.20.0 will fail to build with the base RZ pip):
    pip install --upgrade pip
    # Install mili
    pip install --upgrade --no-cache mili

**Note:** Using `--upgrade` will upgrade any already-installed copies of the mili module in the venv.

If you want to install the packages into your ~/.local/ python cache so the module is usable with the system python install, try instead not creating and activating a virtual environment and instead (untested and may not work):

.. code-block:: bash

    module load python/3.10.8
    python -m pip install --upgrade pip --user
    python -m pip install --upgrade --user --no-cache mili

---------------------------
From Gitlab Repository
---------------------------

.. code-block:: bash

    git clone ssh://git@rzgitlab.llnl.gov:7999/mdg/mili/mili-python.git
    cd mili-python
    python3 -m venv venv-mili-python
    source venv-mili-python/bin/activate
    pip3 install --upgrade pip
    pip3 install -e .

---------------------------
From GitHub Repository
---------------------------

.. code-block:: bash

    git clone https://github.com/mdg-graphics/mili-python.git
    cd mili-python
    python3 -m venv venv-mili-python
    source venv-mili-python/bin/activate
    pip3 install --upgrade pip
    pip3 install -e .