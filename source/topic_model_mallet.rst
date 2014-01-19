.. _topic-model-mallet:

============================
 Topic modeling with MALLET
============================

.. ipython:: python
    :suppress:

    import numpy as np; np.set_printoptions(precision=2)

This section illustrates how to use `MALLET <http://mallet.cs.umass.edu/>`_ to
model a corpus of texts using a topic model and how to analyze the results using
Python.

A topic model is a probabilistic model of the words appearing in a corpus of
documents.  (There are a number of general introductions to topic models
available, such as :cite:`blei_introduction_2012`.) The particular topic model
used in this section is Latent Dirichlet Allocation (LDA), a model introduced in
the context of text analysis in 2003 :cite:`blei_latent_2003`. LDA is an
instance of a more general class of models called mixed-membership models. While
LDA involves a greater number of distributions and parameters than the Bayesian
model introduced in the section on :ref:`group comparison
<bayesian-group-comparison>`, both are instances of a Bayesian probabilistic
model. In fact, posterior inference for both models is typically performed in
precisely the same manner, using Gibbs sampling with conjugate priors.

This section assumes prior exposure to topic modeling and proceeds as follows:

1. MALLET is downloaded and used to fit a topic model of six novels, three by
   Brontë and three by Austen. Because these are lengthy texts, the novels are split
   up into smaller sections---a preprocessing step which improves results considerably.
2. The output of MALLET is loaded into Python as a document-topic matrix (a
   2-dimensional array) of topic shares.
3. Topics, discrete distributions over the vocabulary, are analyzed.

Note that :ref:`an entire section <topic-model-visualization>` is devoted to
visualizing topic models. This section focuses on using MALLET and processing
the results.

This section uses six novels by Brontë and Austen. These novels are divided into
parts as follows:

.. ipython:: python

    import os
    CORPUS_PATH = os.path.join('data', 'austen-brontë-split')
    filenames = sorted([os.path.join(CORPUS_PATH, fn) for fn in os.listdir(CORPUS_PATH)])

.. ipython:: python

    # files are located in data/austen-brontë-split
    len(filenames)
    filenames[:5]

Running MALLET
==============

On Linux and BSD-based systems (such as OS X), the following commands should
download and extract MALLET:

.. code-block:: bash

    # alternatively: wget http://mallet.cs.umass.edu/dist/mallet-2.0.7.tar.gz
    curl --remote-name http://mallet.cs.umass.edu/dist/mallet-2.0.7.tar.gz
    tar zxf mallet-2.0.7.tar.gz

We will run MALLET using the default parameters. Using the option
``--random-seed 1`` should guarantee that the results produced match those
appearing below.

.. code-block:: bash

    mallet-2.0.7/bin/mallet import-dir --input data/austen-brontë-split/ --output /tmp/topic-input-austen-brontë.mallet --keep-sequence --remove-stopwords
    mallet-2.0.7/bin/mallet train-topics --input /tmp/topic-input-austen-brontë.mallet --num-topics 20 --output-doc-topics /tmp/doc-topics-austen-brontë.txt --output-topic-keys /tmp/topic-keys-austen-brontë.txt --random-seed 1

Under Windows the commands are similar. For detailed instructions see the
article `"Getting Started with Topic Modeling and MALLET"
<http://programminghistorian.org/lessons/topic-modeling-and-mallet>`_.  The
MALLET homepage also has `instructions on how to install and run the software
under Windows <http://mallet.cs.umass.edu/download.php>`_.

Processing MALLET output
========================

We have already seen that :ref:`a document-term matrix is a convenient way to
represent the word frequencies <working-with-text>` associated with each
document. Similarly, as each document is associated with a set of topic shares,
it will be useful to gather these features into a document-topic
matrix.

.. note:: Topic shares are also referred to as topic *weights*,
   *mixture weights*, or *component weights*. Different communities favor
   different terms.

Manipulating the output of MALLET into a document-topic matrix is not
entirely intuitive. Fortunately the tools required for the job are available in
Python and the procedure is similar to that reviewed in the previous section on
:ref:`grouping texts <grouping-texts>`.

MALLET delivers the topic shares for each document into a file specified by the
``--output-doc-topics`` option. In this case we have provided the output
filename ``/tmp/doc-topics-austen-brontë.txt``. The first lines of this file
should look something like this:

::

   #doc name topic proportion ...
   0	file:/.../austen-brontë-split/Austen_Pride0103.txt	3	0.2110215053763441	14	0.13306451612903225
   1	file:/.../austen-brontë-split/Austen_Pride0068.txt	17	0.19915254237288135	3	0.14548022598870056
   ...

The first two columns of ``doc-topics.txt`` record the document number
(0-based indexing) and the full path to the filename. The rest of the columns are best
considered as (topic-number, topic-share) pairs. There are as many of these
pairs as there are topics. All columns are separated by tabs (there's even
a trailing tab at the end of the line). With the exception of the header (the
first line), this file records data using `tab-separated values
<https://en.wikipedia.org/wiki/Tab-separated_values>`_. There are two challenges
in parsing this file into a document-topic matrix. The first is sorting.
The texts do not appear in a consistent order in ``doc-topics.txt`` and the
topic number and share pairs appear in different columns depending on the
document. We will need to reorder these pairs before assembling them into
a matrix.[#fnmapreduce]_ The second challenge is that the number of columns will
vary with the number of topics specified (``--num-topics``). Fortunately, the
documentation in the Python library `itertools
<http://docs.python.org/dev/library/itertools.html>`_ describes a function
called ``grouper`` using ``itertools.izip_longest`` that solves our problem.

.. [#fnmapreduce] Those familiar with
    `MapReduce <https://en.wikipedia.org/wiki/MapReduce>`_ may recognize the pattern of
    splitting a dataset into smaller pieces and then (re)ordering them.


.. ipython:: python
    :suppress:

    import os
    import shutil
    import subprocess

    N_TOPICS = 20
    MALLET_INPUT = 'source/cache/topic-input-austen-brontë-split.mallet'
    MALLET_TOPICS = 'source/cache/doc-topic-austen-brontë-{}topics.txt'.format(N_TOPICS)
    MALLET_KEYS = 'source/cache/doc-topic-austen-brontë-{}topics-keys.txt'.format(N_TOPICS)
    if not os.path.exists(MALLET_INPUT):
        subprocess.check_call('mallet-2.0.7/bin/mallet import-dir --input data/austen-brontë-split/ --output {} --keep-sequence --remove-stopwords'.format(MALLET_INPUT), shell=True)

.. ipython:: python
    :suppress:

    # again, splitting up to help IPython parse

    shutil.copy(MALLET_INPUT,'/tmp/topic-input-austen-brontë.mallet')

    if not os.path.exists(MALLET_TOPICS):
        subprocess.check_call('mallet-2.0.7/bin/mallet train-topics --input /tmp/topic-input-austen-brontë.mallet --num-topics {} --output-doc-topics {} --output-topic-keys {} --random-seed 1'.format(N_TOPICS, MALLET_TOPICS, MALLET_KEYS), shell=True)
    shutil.copy(MALLET_TOPICS,'/tmp/doc-topics-austen-brontë.txt')
    shutil.copy(MALLET_KEYS,'/tmp/topic-keys-austen-brontë.txt')


.. ipython:: python

    import numpy as np
    import itertools
    import operator
    import os

    def grouper(n, iterable, fillvalue=None):
        "Collect data into fixed-length chunks or blocks"
        # grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
        args = [iter(iterable)] * n
        return itertools.zip_longest(*args, fillvalue=fillvalue)

    doctopic_triples = []
    mallet_docnames = []

    with open("/tmp/doc-topics-austen-brontë.txt") as f:
        f.readline()  # read one line in order to skip the header
        for line in f:
            # ``docnum, docname, *values`` performs "tuple unpacking", useful Python feature
            # ``.rstrip()`` removes the superfluous trailing tab
            docnum, docname, *values = line.rstrip().split('\t')
            mallet_docnames.append(docname)
            for topic, share in grouper(2, values):
                triple = (docname, int(topic), float(share))
                doctopic_triples.append(triple)

    # sort the triples
    # triple is (docname, topicnum, share) so sort(key=operator.itemgetter(0,1))
    # sorts on (docname, topicnum) which is what we want
    doctopic_triples = sorted(doctopic_triples, key=operator.itemgetter(0,1))

    # sort the document names rather than relying on MALLET's ordering
    mallet_docnames = sorted(mallet_docnames)

    # collect into a document-term matrix
    num_docs = len(mallet_docnames)
    num_topics = len(doctopic_triples) // len(mallet_docnames)

    # the following works because we know that the triples are in sequential order
    doctopic = np.zeros((num_docs, num_topics))
    for triple in doctopic_triples:
        docname, topic, share = triple
        row_num = mallet_docnames.index(docname)
        doctopic[row_num, topic] = share

    @suppress
    doctopic_orig = doctopic.copy()

.. ipython:: python
    :suppress:

    assert len(doctopic_triples) % num_docs == 0
    assert np.all(doctopic > 0)
    assert len(doctopic) == len(filenames)
    assert np.allclose(np.sum(doctopic, axis=1), 1)

.. ipython:: python

    # The following method is considerably faster. It uses the itertools library which is part of the Python standard library.
    import itertools
    import operator
    doctopic = np.zeros((num_docs, num_topics))
    for i, (doc_name, triples) in enumerate(itertools.groupby(doctopic_triples, key=operator.itemgetter(0))):
        doctopic[i, :] = np.array([share for _, _, share in triples])

.. ipython:: python
    :suppress:

    assert np.all(doctopic > 0)
    assert np.allclose(np.sum(doctopic, axis=1), 1)
    assert len(doctopic) == len(filenames)
    assert all(doctopic_orig == doctopic)

Now we will calculate the average of the topic shares associated with each
novel. Recall that we have been working with small sections of novels. The
following step combines the topic shares for sections associated with the same
novel.

.. ipython:: python

    novel_names = []
    for fn in filenames:
        basename = os.path.basename(fn)
        # splitext splits the extension off, 'novel.txt' -> ('novel', '.txt')
        name, ext = os.path.splitext(basename)
        # remove trailing numbers identifying chunk
        name = name.rstrip('0123456789')
        novel_names.append(name)
    # turn this into an array so we can use NumPy functions
    novel_names = np.asarray(novel_names)

    @suppress
    assert len(set(novel_names)) == 6

    # use method described in preprocessing section
    num_groups = len(set(novel_names))
    doctopic_grouped = np.zeros((num_groups, num_topics))
    for i, name in enumerate(sorted(set(novel_names))):
        doctopic_grouped[i, :] = np.mean(doctopic[novel_names == name, :], axis=0)

    doctopic = doctopic_grouped

    @suppress
    docnames = sorted(set(novel_names))


.. ipython:: python
    :suppress:

    import pandas as pd
    OUTPUT_HTML_PATH = os.path.join('source', 'generated')
    rownames = sorted(set(novel_names))
    colnames = ["Topic " + str(i + 1) for i in range(doctopic.shape[1])]
    html = pd.DataFrame(np.round(doctopic, 2), index=rownames, columns=colnames).to_html()
    with open(os.path.join(OUTPUT_HTML_PATH, 'topic_model_doctopic.txt'), 'w') as f:
        f.write(html)

.. raw:: html
    :file: generated/topic_model_doctopic.txt


Inspecting the topic model
==========================

The first thing we should appreciate about our topic model is that the twenty
shares do a remarkably good job of summarizing our corpus. For example, they
preserve the distances between novels (see figures below). By this measure, LDA
is good at dimensionality reduction: we have taken a matrix of dimensions 813 by
14862 (occupying almost three megabytes of memory if stored in a spare matrix)
and fashioned a representation that preserves important features in a matrix
that is 813 by 20 (5% the size of the original).

.. ipython:: python

    from sklearn.feature_extraction.text import CountVectorizer

    CORPUS_PATH_UNSPLIT = os.path.join('data', 'austen-brontë-split')
    filenames = [os.path.join(CORPUS_PATH_UNSPLIT, fn) for fn in sorted(os.listdir(CORPUS_PATH_UNSPLIT))]
    vectorizer = CountVectorizer(input='filename')
    dtm = vectorizer.fit_transform(filenames)  # a sparse matrix
    dtm.shape
    dtm.data.nbytes  # number of bytes dtm takes up
    dtm.toarray().data.nbytes  # number of bytes dtm as array takes up

    doctopic_orig.shape
    doctopic_orig.data.nbytes  # number of bytes document-topic shares take up


.. ipython:: python
    :suppress:

    # COSINE SIMILARITY
    import os  # for os.path.basename
    import matplotlib.pyplot as plt
    from sklearn.manifold import MDS
    from sklearn.metrics.pairwise import cosine_similarity

    dist = 1 - cosine_similarity(dtm)
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
    pos = mds.fit_transform(dist)  # shape (n_components, n_samples)

.. ipython:: python
    :suppress:

    assert dtm.shape[0] == doctopic.shape[0]
    # NOTE: the IPython directive seems less prone to errors when these blocks
    # are split up.
    xs, ys = pos[:, 0], pos[:, 1]
    names = sorted(set(novel_names))
    for x, y, name in zip(xs, ys, names):
        color = 'orange' if "Austen" in name else 'skyblue'
        plt.scatter(x, y, c=color)
        plt.text(x, y, name)

    plt.title("Distances calculated using word frequencies")
    @savefig plot_topic_model_cosine_mds.png width=7in
    plt.show()

.. ipython:: python
    :suppress:

    # TOPIC-MODEL
    import os  # for os.path.basename
    import matplotlib.pyplot as plt
    from sklearn.manifold import MDS
    from sklearn.metrics.pairwise import euclidean_distances

    dist = euclidean_distances(doctopic)
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
    pos = mds.fit_transform(dist)  # shape (n_components, n_samples)

.. ipython:: python
    :suppress:

    # NOTE: the IPython directive seems less prone to errors when these blocks
    # are split up.
    xs, ys = pos[:, 0], pos[:, 1]
    names = sorted(set(novel_names))
    for x, y, name in zip(xs, ys, names):
        color = 'orange' if "Austen" in name else 'skyblue'
        plt.scatter(x, y, c=color)
        plt.text(x, y, name)

    plt.title("Distances calculated using topic shares")
    @savefig plot_topic_model_doctopic_euclidean_mds.png width=7in
    plt.show()

Even though a topic model "discards" the "fine-grained" information recorded in
the matrix of word frequencies, it preserves salient details of the underlying
matrix. That is, the topic shares associated with a document have an
interpretation in terms of word frequencies. This is best illustrated by
examining the present topic model.

First let us identify the most significant topics for each text in the corpus.
This procedure does not differ in essence from the procedure for identifying the
most frequent words in each text.

.. ipython:: python

    novels = sorted(set(novel_names))
    print("Top topics in...")
    for i in range(len(doctopic)):
        top_topics = np.argsort(doctopic[i,:])[::-1][0:3]
        top_topics_str = ' '.join(str(t) for t in top_topics)
        print("{}: {}".format(novels[i], top_topics_str))

.. note:: Recall that, like everything else in Python (and C, Java, and many
    other languages), the topics use 0-based indexing; the first topic is topic 0.

Each topic in the topic model can be inspected. Each topic is a distribution
which captures in probabilistic terms, the words associated with the topic and
the strength of the association (the posterior probability of finding a word
associated with a topic). Sometimes this distribution is called a topic-word
distribution (in contrast to the document-topic distribution). Again, this is
best illustrated by inspecting the topic-word distributions provided by MALLET
for our Austen-Brontë corpus.  MALLET places (a subset of) the topic-word
distribution for each topic in a file specified by the command-line option
``--output-topic-keys``. For the run of ``mallet`` used in this section, this
file is ``/tmp/topic-keys-austen-brontë.txt``. The first line of this file
should resemble the following:

::

   0	2.5	long room looked day eyes make voice head till girl morning feel called table turn continued times appeared breakfast

We need to parse this file into something we can work with. Fortunately this
task is not difficult.

.. ipython:: python

    with open('/tmp/topic-keys-austen-brontë.txt') as input:
        topic_keys_lines = input.readlines()
    topic_words = []
    for line in topic_keys_lines:
        _, _, words = line.split('\t')  # tab-separated
        words = words.rstrip().split(' ')  # remove the trailing '\n'
        topic_words.append(words)

    # now we can get a list of the top words for topic 0 with topic_words[0]
    topic_words[0]

Now we have everything we need to list the words associated with each topic.

.. ipython:: python

    for t in range(len(topic_words)):
        print("Topic {}: {}".format(t, ' '.join(topic_words[t][:N_WORDS_DISPLAY])))


There are many ways to inspect and to visualize topic models. Some of the more
common methods are covered in :ref:`next section <topic-model-visualization>`.

Distinctive topics
------------------

Finding distinctive topics is analogous to the task of :ref:`finding distinctive
words <feature-selection>`. The topic model does an excellent job of focusing
attention on recurrent patterns (of co-occurrence) in the word frequencies
appearing in a corpus. To the extent that we are interested in these kinds of
patterns (rather than the rare or isolated feature of texts), working with
topics tends to be easier than working with word frequencies.

Consider the task of finding the distinctive topics in Austen's novels. Here the
simple difference-in-averages provides an easy way of finding topics that tend
to be associated more strongly with Austen's novels than with Brontë's.

.. ipython:: python

    austen_indices, cbronte_indices = [], []
    for index, fn in enumerate(sorted(set(novel_names))):
        if "Austen" in fn:
            austen_indices.append(index)
        elif "CBronte" in fn:
            cbronte_indices.append(index)

    austen_avg = np.mean(doctopic[austen_indices, :], axis=0)
    cbronte_avg = np.mean(doctopic[cbronte_indices, :], axis=0)
    keyness = np.abs(austen_avg - cbronte_avg)
    ranking = np.argsort(keyness)[::-1]  # from highest to lowest; [::-1] reverses order in Python sequences

    # distinctive topics:
    ranking[:10]

.. ipython:: python
    :suppress:

    N_WORDS_DISPLAY = 10
    N_TOPICS_DISPLAY = 10
    topics_display = sorted(ranking[0:N_TOPICS_DISPLAY])
    arr = doctopic[:, topics_display]
    colnames = ["Topic {}".format(t) for t in topics_display]
    rownames = sorted(set(novel_names))
    html = pd.DataFrame(np.round(arr,2), index=rownames, columns=colnames).to_html()
    arr = np.row_stack([topic_words[t][:N_WORDS_DISPLAY] for t in topics_display])
    rownames = ["Topic {}".format(t) for t in topics_display]
    colnames = ['']*N_WORDS_DISPLAY
    html += pd.DataFrame(arr, index=rownames, columns=colnames).to_html()
    with open(os.path.join(OUTPUT_HTML_PATH, 'topic_model_distinctive_avg_diff.txt'), 'w') as f:
        f.write(html)

.. raw:: html
    :file: generated/topic_model_distinctive_avg_diff.txt
