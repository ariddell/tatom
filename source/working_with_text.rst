.. index:: document-term matrix, tokenizing, CountVectorizer, n-gram, word frequency
.. _working-with-text:

===================
 Working with text
===================

.. note:: This tutorial is available for interactive use
   with `IPython Notebook <http://ipython.org/notebook.html>`_: :download:`Working with text.ipynb <Working with text.ipynb>`.

Creating a document-term matrix
===============================

Word (or n-gram) frequencies are typical units of analysis when working with
text collections.  It may come as a surprise that reducing a book to a list of
word frequencies retains useful information, but practice has shown this to
be the case. Treating texts as a list of word frequencies (a vector) also makes
available a vast range of mathematical tools developed for `studying and
manipulating vectors <http://en.wikipedia.org/wiki/Euclidean_vector#History>`_.

.. note:: Turning texts into unordered lists (or "bags") of words is easy in
    Python.  `Python Programming for the Humanities
    <http://fbkarsdorp.github.io/python-course/>`_ includes a chapter entitled
    `Text Processing
    <http://nbviewer.ipython.org/urls/raw.github.com/fbkarsdorp/python-course/master/Chapter%203%20-%20Text%20Preprocessing.ipynb>`_
    that describes the steps in detail.

This tutorial assumes some prior exposure to text analysis so we will gather
word frequencies (or term frequencies) associated with texts into
a document-term matrix using the `CountVectorizer
<http://scikit-learn.sourceforge.net/dev/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html>`_
class from the `scikit-learn <http://scikit-learn.sourceforge.net/>`_ package.
(For those familiar with R and the `tm
<http://cran.r-project.org/web/packages/tm/>`_ package, this function performs
the same operation as ``DocumentTermMatrix`` and takes recognizably similar
arguments.)

First we need to import the functions and classes we intend to use, along with
the customary abbreviation for functions in the ``numpy`` package.

.. ipython:: python

    import numpy as np  # a conventional alias
    from sklearn.feature_extraction.text import CountVectorizer

Now we use the `CountVectorizer
<http://scikit-learn.sourceforge.net/dev/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html>`_
class to create a document-term matrix. ``CountVectorizer`` is customizable. For
example, a list of "stop words" can be specified with the ``stop_words``
parameter. Other important parameters include:

- ``lowercase`` (default ``True``) convert all text to lowercase before
  tokenizing
- ``min_df`` (default ``1``) remove terms from the vocabulary that occur in
  fewer than ``min_df`` documents (in a large corpus this may be set to
  ``15`` or higher to eliminate very rare words)
- ``vocabulary`` ignore words that do not appear in the provided list of words 
- ``strip_accents`` remove accents
- ``token_pattern`` (default ``u'(?u)\b\w\w+\b'``) regular expression
  identifying tokens–by default words that consist of a single character 
  (e.g., 'a', '2') are ignored, setting ``token_pattern`` to ``'(?u)\b\w+\b'``
  will include these tokens
- ``tokenizer`` (default unused) use a custom function for tokenizing

For this example we will use texts by Jane Austen and Charlotte Brontë. These
texts are available in :ref:`datasets`.

.. ipython:: python

    filenames = ['data/austen-brontë/Austen_Emma.txt',
                 'data/austen-brontë/Austen_Pride.txt',
                 'data/austen-brontë/Austen_Sense.txt',
                 'data/austen-brontë/CBronte_Jane.txt',
                 'data/austen-brontë/CBronte_Professor.txt',
                 'data/austen-brontë/CBronte_Villette.txt']

    vectorizer = CountVectorizer(input='filename')
    dtm = vectorizer.fit_transform(filenames)  # a sparse matrix
    vocab = vectorizer.get_feature_names()  # a list

Now we have a document-term matrix and a vocabulary list. Before we can query
the matrix and find out, for example, how many times the word 'house' occurs in
*Emma* (the first text in ``filenames``), we need to convert this matrix from
its current format, a `sparse matrix
<http://docs.scipy.org/doc/scipy/reference/sparse.html>`_, into a normal NumPy
array. We will also convert the Python list storing our vocabulary, ``vocab``,
into a NumPy array, as an array supports a greater variety of operations than
a list.

.. ipython:: python
    
    # for reference, note the current class of `dtm`
    type(dtm)
    dtm = dtm.toarray()  # convert to a regular array
    vocab = np.array(vocab)

.. note:: A sparse matrix only records non-zero entries and is used to store
    matrices that contain a significant number of entries that are zero. To
    understand why this matters enough that ``CountVectorizer`` returns a sparse
    matrix by default, consider a 4000 by 50000 matrix of word frequencies that
    is 60% zeros. In Python an integer takes up four bytes, so using a sparse
    matrix saves almost 500M of memory, which is a considerable amount of
    computer memory. (Recall that Python objects such as arrays are stored in
    memory, not on disk).

With this preparatory work behind us, querying the document-term matrix is
simple. For example, the following demonstrate two ways finding how many times
the word 'house' occurs in the first text, *Emma*:

.. ipython:: python

    # the first file, indexed by 0 in Python, is *Emma*
    filenames[0] == 'data/austen-brontë/Austen_Emma.txt'

    # use the standard Python list method index(...)
    # list(vocab) or vocab.tolist() will take vocab (an array) and return a list
    house_idx = list(vocab).index('house')
    dtm[0, house_idx]

    # using NumPy indexing will be more natural for many
    # in R this would be essentially the same, dtm[1, vocab == 'house']
    dtm[0, vocab == 'house']

Although `dtm` is technically a NumPy array, I will keep referring to `dtm` as
a matrix. Note that NumPy arrays do support matrix operations such as dot
product. (If ``X`` and ``Y`` have compatible dimensions, ``X.dot(Y)`` is the
matrix product :math:`XY`.)

.. note:: NumPy does make available a `matrix
    <http://docs.scipy.org/doc/numpy/reference/generated/numpy.matrix.html>`_
    data structure which can be useful if you are doing lots of matrix
    operations such as matrix product, inverse, and so forth. In general,
    however, it is best to stick to NumPy arrays.

Comparing texts
===============

Arranging our texts in a document-term matrix make available a range of
exploratory procedures. For example, calculating a measure of similarity between
texts becomes simple. Since each row of the document-term matrix is a sequence
of a novel's word frequencies, it is possible to put mathematical notions of
similarity (or distance) between sequences of numbers in service of calculating
the similarity (or distnace) between any two novels. One frequently used measure
of distance between vectors (a measure easily converted into a measure of similarity) is `Euclidean
distance <https://en.wikipedia.org/wiki/Euclidean_distance>`_. The Euclidean
distance between two vectors in the plane should be familiar from geometry, as
it is the length of the hypotenuse that joins the two vectors. For instance,
consider the Euclidean distance between the vectors :math:`\vec{x} = (1, 3)` and
:math:`\vec{y} = (4, 2)`. The distance between the two vectors is
:math:`\sqrt{(1-4)^2 + (3-2)^2} = \sqrt{10}`.

.. note::

    Measures of distance can be converted into measures of similarity. If your
    measures of distance are all between zero and one, then a measure of
    similarity could be one minus the distance. (The inverse of the distance
    would also serve as a measure of similarity.)


.. tikz:: Distance between two vectors
   :libs: arrows

    \useasboundingbox (0,0) rectangle (5,5);
    \draw [<->,thick] (0,5) node (yaxis) [above] {} |- (5,0) node (xaxis) [right] {};
    \draw[step=1cm,gray,very thin] (0,0) grid (5,5);

    \draw [->, thick] (0,0) -- (1,3);
    \draw (1,3) node [above] {$(1,3) = \vec{x}$};

    \draw [->, thick] (0,0) -- (4,2);
    \draw (4,1.7) node [below] {$(4,2) =\vec{y}$};

    \draw [-, orange] (1,3) -- (4,2);
    \draw (3.3,2.5) node [above, orange] {$||\vec{x} - \vec{y}|| = \sqrt{10}$};

.. note:: More generally, given two vectors :math:`\vec{x}` and :math:`\vec{y}`
    in :math:`p`-dimensional space,  the Euclidean distance between the two
    vectors is given by

    :math:`||\vec{x} - \vec{y}|| = \sqrt{\sum_{i=1}^p (x_i - y_i)^2}`

This concept of distance is not restricted to two dimensions. For example, it is
not difficult to imagine the figure above translated into three dimensions. We can also persuade ourselves that the measure of distance extends to an arbitrary number of dimensions; for any two matched components in a pair of vectors (such as :math:`x_2` and :math:`y_2`), differences increase the distance.

Since two novels in our corpus now have an expression as vectors, we can
calculate the Euclidean distance between them. We can do this by hand or we can
avail ourselves of the ``scikit-learn`` function ``euclidean_distances``.

.. ipython:: python

    # "by hand"
    n, _ = dtm.shape
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            x, y = dtm[i, :], dtm[j, :]
            dist[i, j] = np.sqrt(np.sum((x - y)**2))
    
    from sklearn.metrics.pairwise import euclidean_distances
    dist = euclidean_distances(dtm)

    np.round(dist, 1)
    # *Pride and Prejudice* is index 1 and *Jane Eyre* is index 3
    filenames[1] == 'data/austen-brontë/Austen_Pride.txt'
    filenames[3] == 'data/austen-brontë/CBronte_Jane.txt'

    # the distance between *Pride and Prejudice* and *Jane Eyre*
    dist[1, 3]

    # which is greater than the distance between *Jane Eyre* and *Villette* (index 5)
    dist[1, 3] > dist[3, 5]

    @suppress
    assert dist[1, 3] > dist[3, 5]


And if we want to use a measure of distance that takes into consideration the
length of the novels (an excellent idea), we can calculate the `cosine
similarity
<http://www.gettingcirrius.com/2010/12/calculating-similarity-part-1-cosine.html>`_
by importing ``sklearn.metrics.pairwise.cosine_similarity`` and use it in place
of `euclidean_distances`.

Keep in mind that cosine similarity is a measure of similarity (rather than
distance) that ranges between 0 and 1 (as it is the cosine of the angle between
the two vectors).  In order to get a measure of distance (or dissimilarity), we
need to "flip" the measure so that a larger angle receives a larger value. The
distance measure derived from cosine similarity is therefore one minus the
cosine similarity between two vectors.

.. ipython:: python

    from sklearn.metrics.pairwise import cosine_similarity
    dist = 1 - cosine_similarity(dtm)
    np.round(dist, 2)

    # the distance between *Pride and Prejudice* (index 1)
    # and *Jane Eyre* (index 3) is
    dist[1, 3]

    # which is greater than the distance between *Jane Eyre* and
    # *Villette* (index 5)
    dist[1, 3] > dist[3, 5]

Those interested in doing the calculation for themselves can use the following
steps:

.. ipython:: python

    norms = np.sqrt(np.sum(dtm * dtm, axis=1, keepdims=True))  # multiplication between arrays is element-wise
    dtm_normed = dtm / norms
    similarities = np.dot(dtm_normed, dtm_normed.T)
    np.round(similarities, 2)
    # similarities between *Pride and Prejudice* and *Jane Eyre* is
    similarities[1, 3]

.. ipython:: python
    :suppress:

    import os
    import pandas as pd
    OUTPUT_HTML_PATH = os.path.join('source', 'generated')
    OUTPUT_FILENAME = 'getting_started_cosine.txt'
    names = [os.path.basename(fn).replace('.txt', '') for fn in filenames]
    ARR, ROWNAMES, COLNAMES = dist, names, names

    html = pd.DataFrame(np.round(ARR, 2), index=ROWNAMES, columns=COLNAMES).to_html()
    with open(os.path.join(OUTPUT_HTML_PATH, OUTPUT_FILENAME), 'w') as f:
        f.write(html)

.. raw:: html
    :file: generated/getting_started_cosine.txt

Visualizing distances
=====================

It is often desirable to visualize the pairwise distances between our texts.
A general approach to visualizing distances is to assign a point in a plane to
each text, making sure that the distance between points is proportional to the
pairwise distances we calculated. This kind of visualization is common enough
that it has a name, "`multidimensional scaling
<https://en.wikipedia.org/wiki/Multidimensional_scaling>`_" (MDS) and family of
functions in ``scikit-learn`` (and R too, see ``mdscale``).

.. ipython:: python

    import os  # for os.path.basename
    import matplotlib.pyplot as plt
    from sklearn.manifold import MDS

    # two components as we're plotting points in a two-dimensional plane
    # "precomputed" because we provide a distance matrix
    # we will also specify `random_state` so the plot is reproducible.
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
    pos = mds.fit_transform(dist)  # shape (n_components, n_samples)

.. ipython:: python

    xs, ys = pos[:, 0], pos[:, 1]
    # short versions of filenames:
    # convert 'data/austen-brontë/Austen_Emma.txt' to 'Austen_Emma'
    names = [os.path.basename(fn).replace('.txt', '') for fn in filenames]
    # color-blind-friendly palette
    for x, y, name in zip(xs, ys, names):
        color = 'orange' if "Austen" in name else 'skyblue'
        plt.scatter(x, y, c=color)
        plt.text(x, y, name)

    @suppress
    plt.tight_layout()

    @savefig plot_getting_started_cosine_mds.png width=8in
    plt.show()

We can also do MDS in three dimensions:

.. ipython:: python

    # après Jeremy M. Stober, Tim Vieira
    # https://github.com/timvieira/viz/blob/master/mds.py

    mds = MDS(n_components=3, dissimilarity="precomputed", random_state=1)
    pos = mds.fit_transform(dist)

.. ipython:: python

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2])
    for x, y, z, s in zip(pos[:, 0], pos[:, 1], pos[:, 2], names):
        ax.text(x, y, z, s)

    @savefig plot_getting_started_cosine_mds_3d.png width=7in
    plt.show()


Clustering texts based on distance
==================================

Clustering texts into discrete groups of similar texts is often a useful
exploratory step. For example, a researcher may be wondering if certain textual
features partition a collection of texts by author or by genre. Pairwise
distances alone do not produce any kind of classification. To put a set of
distance measurements to work in classification requires additional assumptions,
such as a definition of a group or cluster.

The ideas underlying the transition from distances to clusters are, for the most
part, common sense. Any clustering of texts should result in texts that are
closer to each other (in the distance matrix) residing in the same cluster.
There are many ways of satisfying this requirement; there no unique clustering
based on distances that is the "best". One strategy for clustering in
circulation is called `Ward's method
<https://en.wikipedia.org/wiki/Ward%27s_method>`_. Rather than producing
a single clustering, Ward's method produces a hierarchy of clusterings, as we
will see in a moment. All that Ward's method requires is a set of pairwise
distance measurements–such as those we calculated a moment ago.  Ward's method
produces a hierarchical clustering of texts via the following procedure:

#. Start with each text in its own cluster

#. Until only a single cluster remains,
   
   - Find the closest clusters and merge them. The distance between two clusters
     is the change in the sum of squared distances when they are merged.

#. Return a tree containing a record of cluster-merges.

The function `scipy.cluster.hierarchy.ward
<http://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html>`_ performs
this algorithm and returns a tree of cluster-merges. The hierarchy of clusters
can be visualized using ``scipy.cluster.hierarchy.dendrogram``.

.. ipython:: python
    
    from scipy.cluster.hierarchy import ward, dendrogram

    linkage_matrix = ward(dist)
    
    # match dendrogram to that returned by R's hclust()
    dendrogram(linkage_matrix, orientation="right", labels=names);

    @savefig plot_getting_started_ward_dendrogram.png width=7in
    plt.tight_layout()  # fixes margins

For those familiar with R, the procedure is performed as follows:

.. code-block:: r

    labels = c('Austen_Emma', 'Austen_Pride', 'Austen_Sense', 'CBronte_Jane',
               'CBronte_Professor', 'CBronte_Villette')
    dtm_normed = dtm / rowSums(dtm)
    dist_matrix = dist(dtm_normed)
    tree = hclust(dist_matrix, method="ward")
    plot(tree, labels=labels)

Exercises
=========

1. Find two different ways of determining the number of times the word
   'situation' appears in *Emma*. (Make sure the methods produce the same result.)

2. Working with the strings below as documents and using ``CountVectorizer``
   with the ``input='content'`` parameter, create a document-term matrix.
   Apart from the ``input`` parameter, use the default settings.

.. ipython:: python

    text1 = "Indeed, she had a rather kindly disposition."
    text2 = "The real evils, indeed, of Emma's situation were the power of having rather too much her own way, and a disposition to think a little too well of herself;"
    text3 = "The Jaccard distance is a way of measuring the distance from one set to another set."
   
3. Using the document-term matrix just created, calculate the Euclidean
   distance, `Jaccard distance <http://en.wikipedia.org/wiki/Jaccard_index>`_,
   and cosine distance between each pair of documents. Make sure to calculate
   distance (rather than similarity). Are our intuitions about which texts are
   most similar reflected in the measurements of distance?

.. ipython:: python
    :suppress:

    # SOLUTIONS

    vectorizer = CountVectorizer(input='content')
    dtm = vectorizer.fit_transform([text1, text2, text3])  # a sparse matrix
    dtm = dtm.toarray()
    dtm

    from sklearn.metrics.pairwise import euclidean_distances
    dist = euclidean_distances(dtm)
    np.round(dist,3)
    dist[0,1] < dist[0,2]

    from sklearn.metrics.pairwise import cosine_similarity
    dist = 1 - cosine_similarity(dtm)
    np.round(dist,3)
    dist[0,1] < dist[0,2]

    from sklearn.metrics.pairwise import pairwise_distances
    dist = pairwise_distances(dtm, metric='jaccard')
    np.round(dist,3)
    dist[0,1] < dist[0,2]

*For solutions, view the source for this document.*
