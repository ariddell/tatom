.. index:: Python, NumPy, matplotlib
.. _getting-started:

=================
 Getting started
=================

For those new to Python
=======================

Those with prior programming experience may benefit from the following
introduction to Python that has been written for an audience in the humanities.

- `Python Programming for the Humanities
  <http://fbkarsdorp.github.io/python-course/>`_ by Folgert Karsdorp and Maarten
  van Gompel 

For those new to programming
----------------------------

Those new to programming may wish to consider the following introductory courses
before diving into the following tutorials. Both these courses are taught using
Python and assume no prior exposure to computer programming. The MIT course is
excellent.

- `Introduction to Computer Science and Programming
  <https://www.edx.org/course/mit/6-00x/introduction-computer-science/586>`_
  (MIT 6.00x) taught by Eric Grimson and John Guttag (among others).

- `Learn to Programm: The Fundamentals
  <https://www.coursera.org/course/programming1>`_ (University of Toronto)
  taught by Jennifer Campbell and Paul Gries.

.. _getting-started-numpy: 

For those new to NumPy
======================

`NumPy <http://www.numpy.org>`_ is a Python package which provides an ``array``
data structure that is widely used in the Python community to represent
a two-dimensional table of data. Such a structure is useful in representing
a text corpus as a table of document-term frequencies.

Those with considerable experience in Python but little exposure to NumPy will
find the following series of `lectures on scientific computing with Python
<http://scipy-lectures.github.io/index.html>`_ helpful, in particular the
chapter `NumPy: creating and manipulating numerical data
<http://scipy-lectures.github.io/intro/numpy/index.html>`_.

While NumPy functions will be introduced as they are used, readers familiar with
R or Octave/Matlab will likely find themselves searching for the Python
equivalent of some function in R or Octave.  For example, the NumPy equivalent
of R's ``apply(X, 1, sum)`` (or ``rowSums(X)``) is ``np.sum(X, axis=1)``.  There
are a number of websites that list such equivalences.

For anyone
----------

- `NumPy: creating and manipulating numerical data
  <http://scipy-lectures.github.io/intro/numpy/index.html>`_ (part of the
  excellent `Python Scientific Lecture Notes
  <http://scipy-lectures.github.io/index.html>`_)
- `NumPy indexing <http://wiki.scipy.org/Cookbook/Indexing>`_

For R an Octave/Matlab users
----------------------------

- `NumPy for R users <http://mathesaurus.sourceforge.net/r-numpy.html>`_
- `NumPy for Matlab Users <http://wiki.scipy.org/NumPy_for_Matlab_Users>`_

.. _getting-started-matplotlib:

For those new to Matplotlib
===========================

Matplotlib is the most popular graphing and visualization package for Python and
this tutorial uses it. An comprehensive introduction to Matplotlib is included
in the `Python Scientific Lecture Notes
<http://scipy-lectures.github.io/index.html>`_ (`Matplotlib: plotting
<http://scipy-lectures.github.io/intro/matplotlib/matplotlib.html>`_).
