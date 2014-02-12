.. index:: Python, installing Python, Python modules
.. _preliminaries:

================
 Preliminaries
================

These tutorials make use of a number of Python packages. This section describes
how to install these packages. (If you do not already have Python 3 installed on
your system you may wish to skip to the section :ref:`installing-python`.)

Required Python packages
========================
The tutorials collected here assume that Python version 3.3 or higher is
installed along with the following packages:

- `NumPy <http://numpy.org>`_ version 1.7 or higher
- `SciPy <http://scipy.org>`_ version 0.12 or higher
- `matplotlib <http://matplotlib.org>`_ version 1.2 or higher
- `Pandas <http://pandas.pydata.org/>`_ version 0.11 or higher.
- `NLTK <http://nltk.org>`_ version 3.0 or higher
- `scikit-learn <http://scikit-learn.org>`_ version 0.13.1 or higher

.. note::

    As of this writing, `NLTK 3.0 "alpha" <http://nltk.org/nltk3-alpha>`_ is the
    only version released that works with Python 3. It must be :ref:`installed
    from source <installing-from-source>`. If you have ``pip`` just use:
    ``pip install http://nltk.org/nltk3-alpha/nltk-3.0a3.tar.gz``

If these packages are installed, the following lines of code should run in
a Python interpreter without error.

.. ipython:: python

    import numpy
    numpy.__version__
    import scipy
    scipy.__version__
    import matplotlib
    matplotlib.__version__
    import pandas
    pandas.__version__
    import nltk
    nltk.__version__
    import sklearn
    sklearn.__version__

The following packages are also worth checking out:

- `IPython <http://www.ipython.org>`_ version 1.0 or higher.
- `statsmodels <http://statsmodels.sourceforge.net/>`_ version 0.5.0 or higher.

.. ipython:: python

    import IPython
    IPython.__version__
    import statsmodels
    statsmodels.__version__

.. note:: Why Python 3? Python 3 
   :ref:`stores strings as Unicode <python:textseq>`. This makes working with
   languages other than English immeasurably easier. Python 3.3 in particular
   contains a number of :ref:`significant improvements <python:whatsnew-index>`
   that speed up Python's handling of Unicode text.

.. _installing-python:

Installing Python
=================

Installing Python on Linux
--------------------------
On Debian-based systems such as Ubuntu, Python 3 may be installed with::

    apt-get install python3

On Fedora the command is::

    yum install python3

Depending on how current the operating system is, these commands may or may not
install Python version 3.3 or higher. Find the version of python available by
running ``python3 --version`` in a terminal.

Installing packages on Mac OS X
-------------------------------

Installing Python 3 via `homebrew <http://brew.sh/>`_ is the preferred method
for those comfortable with the OS X command line interface.

`Mac OS X installers <http://www.python.org/download/>`_ for Python may be found
on the official `Python website <http://python.org>`_.

Finally, Python 3.3 may also be installed via `MacPorts <http://macports.org`.

Installing Python on Windows
----------------------------

There are also a number of distributions of Python for Windows that come bundled
with Python packages relevant to scientific computing including as NumPy, SciPy,
and scikit-learn.  One such distribution with excellent support for Python
3 is `Anaconda Python <https://store.continuum.io/cshop/anaconda>`_.

.. _installing-packages:

Installing Python packages
==========================

Installing packages on Linux
-----------------------------
.. note::

    Advanced users may want to consider isolating these packages in
    a `virtual environment <http://docs.python.org/3/library/venv.html>`_.

Using the package manager
~~~~~~~~~~~~~~~~~~~~~~~~~
On recent versions of Debian and Ubuntu as well as Fedora Linux there are
recompiled packages available that cover almost all of the requirements. With
``apt-get`` most of the requirements are installed with the following command::

    sudo apt-get install python3-numpy python3-scipy python3-pandas python3-matplotlib python3-ipython

Using pip
~~~~~~~~~
Installing the required packages is straightforward if the `pip
<http://www.pip-installer.org/>`_ installer is available. For example,
NLTK may be installed with the following command::

    pip install http://nltk.org/nltk3-alpha/nltk-3.0a1.tar.gz

``scikit-learn`` may also be installed with ``pip``::

    pip install scikit-learn

.. _installing-from-source:

Installing from source
~~~~~~~~~~~~~~~~~~~~~~
If ``pip`` is not available, the packages may be installed from source. Source
"tarballs" for NumPy and matplotlib can be obtained and installed with the
following sequence of commands. To install NumPy from source use the following
commands::

    curl -O https://pypi.python.org/packages/source/n/numpy/numpy-1.7.1.tar.gz
    tar zxvf numpy-1.7.1.tar.gz
    cd numpy-1.7.1
    python setup.py install

To install matplotlib from source, enter the following commands::

    curl -O -L https://downloads.sourceforge.net/project/matplotlib/matplotlib/matplotlib-1.2.1/matplotlib-1.2.1.tar.gz
    tar zxvf matplotlib-1.2.1.tar.gz
    cd matplotlib-1.2.1
    python setup.py install

To install NTLK::

    curl -O http://nltk.org/nltk3-alpha/nltk-3.0a3.tar.gz
    tar zxvf nltk-3.0a3.tar.gz
    cd nltk-3.0a3
    python setup.py install

Installing packages on Mac OS X
-------------------------------

Installation of Python 3 and the required packages may be accomplished using
`MacPorts <http://macports.org>`_ or `homebrew <http://brew.sh/>`_. For example,
the following command installs ``matplotlib`` for Python version 3.3 under
MacPorts:

    sudo port install py33-matplotlib

Homebrew has a wiki page `Homebrew and Python
<https://github.com/mxcl/homebrew/wiki/Homebrew-and-Python>`_ that describes how
Python is handled in homebrew.

Installing packages on Windows
------------------------------

There are a number of distributions of Python for Windows that come pre-packaged
with packages relevant to scientific computing such as NumPy and SciPy. They
include, for example, `Anaconda Python
<https://store.continuum.io/cshop/anaconda>`_. Anaconda includes almost all the
packages used here. Also available are `instructions on how to use Python 3 with
Anaconda <http://continuum.io/blog/anaconda-python-3>`.
