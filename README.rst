TAToM: Text Analysis with Topic Models for the Humanities and Social Sciences
=============================================================================

*TAToM: Text Analysis with Topic Models for the Humanities and Social Sciences*
consists of a series of tutorials that introduce basic procedures in
quantitative text analysis with a particular focus on the preparation of a text
corpus for analysis and on exploratory analysis using topic models and machine
learning.

Building
========

The project relies heavily on the ``ipython`` Sphinx directive, which is
somewhat fragile.

Building the documentation may be accomplished by the following sequence of
commands. The core requirements are: Python 3.3 or higher and Sphinx 1.2 or
higher.

#. Install required packages. The packages and version numbers are listed in
   ``requirements.txt``. In theory the command below should download and install
   all the packages but in practice there are dependencies that prevent this
   from working. Some packages may need to be installed one by one.
   
   ``pip install -r requirements.txt``

#. The chapter illustrating the use of MALLET needs a java compiler, on
   Debian-based Linux such as Ubuntu you may install java with: ``sudo apt-get install 

   ``sudo apt-get install openjdk-7-jre-headless``

#. Build the documentation:

   ``make html``
