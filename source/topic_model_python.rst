.. index:: topic model, non-negative matrix factorization, NMF
.. _topic-model-python:

==========================
 Topic modeling in Python
==========================

.. ipython:: python
    :suppress:

    import numpy as np; np.set_printoptions(precision=2)

This section illustrates how to do approximate topic modeling in Python. We will
use a technique called `non-negative matrix factorization (NMF)
<https://en.wikipedia.org/wiki/Non-negative_matrix_factorization>`_ that
strongly resembles Latent Dirichlet Allocation (LDA) which we covered in the
previous section, :ref:`topic-model-mallet`. [#fn_nmf]_ Whereas LDA is
a probabilistic model capable of expressing uncertainty about the placement of
topics across texts and the assignment of words to topics, NMF is
a deterministic algorithm which arrives at a single representation of the
corpus. For this reason, NMF is often characterized as a machine learning
algorithm. Like LDA, NMF arrives at its representation of a corpus in terms of
something resembling "latent topics".

.. note:: The name "Non-negative matrix factorization" has the virtue of being
   transparent. A "non-negative matrix" is a matrix containing non-negative
   values (here zero or positive word frequencies). And
   factorization refers to the familiar kind of mathematical factorization.
   Just as a polynomial :math:`x^2 + 3x + 2` may be factored into a simple
   product :math:`(x+2)(x+1)`, so too may a matrix
   :math:`\bigl(\begin{smallmatrix} 6&2&4\\ 9&3&6 \end{smallmatrix} \bigr)` be
   factored into the product of two smaller matrices
   :math:`\bigl(\begin{smallmatrix} 2\\ 3 \end{smallmatrix} \bigr)
   \bigl(\begin{smallmatrix} 3&2&1 \end{smallmatrix} \bigr)`.

This section follows the procedures described in :ref:`topic-model-mallet`,
making the substitution of NMF for LDA where appropriate.

This section uses the novels by Brontë and Austen. These novels are divided into
parts as follows:

.. ipython:: python

    import os
    CORPUS_PATH = os.path.join('data', 'austen-brontë-split')
    filenames = sorted([os.path.join(CORPUS_PATH, fn) for fn in os.listdir(CORPUS_PATH)])

.. ipython:: python

    # files are located in data/austen-brontë-split
    len(filenames)
    filenames[:5]

Using Non-negative matrix factorization
=======================================

As always we need to give Python access to our corpus. In this case we will work
with our familiar document-term matrix.

.. ipython:: python

    import numpy as np  # a conventional alias
    import sklearn.feature_extraction.text as text

    vectorizer = text.CountVectorizer(input='filename', stop_words='english', min_df=20)
    dtm = vectorizer.fit_transform(filenames).toarray()
    vocab = np.array(vectorizer.get_feature_names())

    dtm.shape
    len(vocab)

By analogy with LDA, we will use NMF to get a document-topic matrix (topics here
will also be referred to as "components") and a list of top words for each
topic. We will make analogy clear by using the same variable names:
``doctopic`` and ``topic_words``

.. ipython:: python

    from sklearn import decomposition

    num_topics = 20
    num_top_words = 20

    clf = decomposition.NMF(n_components=num_topics, random_state=1)

    # this next step may take some time

.. ipython:: python
    :suppress:

    # suppress this

    import os
    import pickle

    NMF_TOPICS = 'source/cache/nmf-austen-brontë-doc-topic.pkl'
    NMF_CLF = 'source/cache/nmf-austen-brontë-clf.pkl'

    # the ipython directive seems to have trouble with multi-line indented blocks
    if not os.path.exists(NMF_CLF):
        doctopic = clf.fit_transform(dtm)
        pickle.dump(doctopic, open(NMF_TOPICS, 'wb'))
        pickle.dump(clf, open(NMF_CLF, 'wb'))


    clf = pickle.load(open(NMF_CLF, 'rb'))
    doctopic = pickle.load(open(NMF_TOPICS, 'rb'))

.. code-block:: python

   doctopic = clf.fit_transform(dtm)

.. ipython:: python

    # print words associated with topics
    topic_words = []
    for topic in clf.components_:
        word_idx = np.argsort(topic)[::-1][0:num_top_words]
        topic_words.append([vocab[i] for i in word_idx])

To make the analysis and visualization of NMF components similar to that of
LDA's topic proportions, we will scale the document-component matrix such that
the component values associated with each document sum to one.

.. ipython:: python

    doctopic = doctopic / np.sum(doctopic, axis=1, keepdims=True)

Now we will average those topic shares associated with the same novel together
--- just as we did with the topic shares from MALLET.

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
    colnames = ["NMF Topic " + str(i + 1) for i in range(doctopic.shape[1])]
    html = pd.DataFrame(np.round(doctopic, 2), index=rownames, columns=colnames).to_html()
    with open(os.path.join(OUTPUT_HTML_PATH, 'NMF_doctopic.txt'), 'w') as f:
        f.write(html)

.. raw:: html
    :file: generated/NMF_doctopic.txt

Inspecting the NMF fit
======================

The topics (or components) of the NMF fit preserve the distances between novels (see the figures below).

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
    @savefig plot_nmf_section_austen_brontë_cosine_mds.png width=7in
    plt.show()

.. ipython:: python
    :suppress:

    # NMF
    import os  # for os.path.basename
    import matplotlib.pyplot as plt
    from sklearn.manifold import MDS
    from sklearn.metrics.pairwise import euclidean_distances

    dist = euclidean_distances(doctopic)
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
    pos = mds.fit_transform(dist)  # shape (n_components, n_samples)

.. ipython:: python
    :suppress:

    # NOTE: the IPython directive seems less prone to errors when these blocks are split up
    xs, ys = pos[:, 0], pos[:, 1]
    names = sorted(set(novel_names))
    for x, y, name in zip(xs, ys, names):
        color = 'orange' if "Austen" in name else 'skyblue'
        plt.scatter(x, y, c=color)
        plt.text(x, y, name)

    plt.title("Distances calculated using NMF components")
    @savefig plot_NMF_euclidean_mds.png width=7in
    plt.show()

Even though the NMF fit "discards" the fine-grained detail recorded in the
matrix of word frequencies, the matrix factorization performed allows us to
reconstruct the salient details of the underlying matrix.

As we did in the previous section, let us identify the most significant topics
for each text in the corpus.  This procedure does not differ in essence from the
procedure for identifying the most frequent words in each text.

.. ipython:: python

    novels = sorted(set(novel_names))
    print("Top NMF topics in...")
    for i in range(len(doctopic)):
        top_topics = np.argsort(doctopic[i,:])[::-1][0:3]
        top_topics_str = ' '.join(str(t) for t in top_topics)
        print("{}: {}".format(novels[i], top_topics_str))

And we already have lists of words (``topic_words``) most strongly associated
with the components. For reference, we will display them again:

.. ipython:: python

    # show the top 15 words
    for t in range(len(topic_words)):
        print("Topic {}: {}".format(t, ' '.join(topic_words[t][:15])))


There are many ways to inspect and to visualize topic models. Some of the most
common methods are covered in :ref:`topic-model-visualization`.

Distinctive topics
------------------

Consider the task of finding the topics that are distinctive of Austen using the
NMF "topics". Using the simple difference-in-averages we can find topics that to
be associated with Austen's novels rather than Brontë's.

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

.. FOOTNOTES

.. [#fn_nmf] While there are significant differences between NMF and LDA, there
   are also similarities. Indeed, if the texts in a corpus have certain
   properties, NMF and LDA will arrive at the same representation of a corpus
   :cite:`arora_practical_2013`.

