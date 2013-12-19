.. _feature-selection:

==============================================
 Feature selection: finding distinctive words
==============================================

.. ipython:: python
    :suppress:

    import numpy as np; np.set_printoptions(precision=3)

We often want to know what words distinguish one group of texts from another
group of texts. For instance, we might be working with an archive of two city
newspapers, say, the *Frankfurter Allgemeine Zeitung* and the *Frankfurter
Rundschau* and want to know which words tend to appear in one newspaper rather
than the other. Or we might be interested in comparing word usage in US
Presidents' `State of the Union addresses
<http://en.wikipedia.org/wiki/State_of_the_Union_address>`_ given during
recessions with addresses given during periods of economic growth. Or we might
be comparing the style of several novelists and want to know if one author tends
to use words not found in the works of others.

This section illustrates how distinctive words can be identified using a corpus
of novels containing works by two authors: Jane Austen and Charlotte Brontë.

- Austen, *Emma*
- Austen, *Pride and Prejudice*
- Austen, *Sense and Sensibility*
- \C. Brontë, *Jane Eyre*
- \C. Brontë, *The Professor*
- \C. Brontë, *Villette*

This :ref:`corpus of six novels <datasets>` consists of the following text
files:

.. ipython:: python

    filenames

.. raw:: html
    :file: generated/feature_selection_bayesian.txt

We will find that among the words that reliably distinguish Austen from Brontë
are  "such", "could", and "any". This tutorial demonstrates how we arrived at
these words.

.. ipython:: python
    :suppress:

    N_WORDS_DISPLAY = 23

    import os
    import nltk
    import numpy as np
    import pandas as pd
    from sklearn.feature_extraction.text import CountVectorizer

    CORPUS_PATH = os.path.join('data', 'british-fiction-corpus')
    OUTPUT_HTML_PATH = os.path.join('source', 'generated')
    AUSTEN_FILENAMES = ['Austen_Emma.txt', 'Austen_Pride.txt', 'Austen_Sense.txt']
    CBRONTE_FILENAMES = ['CBronte_Jane.txt', 'CBronte_Professor.txt', 'CBronte_Villette.txt']
    filenames = AUSTEN_FILENAMES + CBRONTE_FILENAMES

.. note:: The following features an introduction to the concepts underlying
    feature selection. Those who are working with a very large corpus and are
    familiar with statistics may wish to skip ahead to the section on
    :ref:`group comparison <bayesian-group-comparison>` or the section
    :ref:`chi2`.


Since we are concerned with words, we begin by extracting word frequencies from
each of the texts in our corpus and :ref:`construct a document-term matrix
<working-with-text>` that records the rate per 1,000 words for each word
appearing in the corpus.  Using rates rather than counts will allow us to
account for differences in the length of the novels. Accounting for differences
in document lengths when dealing with word counts is essential. For example,
a text using "whence" ten times in a 1,000 word article uses the word at a rate
of 10 per 1,000 words, while a 100,000 word novel that uses "whence" 20 times
uses it at a rate of 0.2 per 1,000 words. While the word occurs more in absolute
terms in the second text, the rate is higher in the first text. While there are
other ways to account for document length---a procedure called
"normalization"---considering the rate per 1,000 words will serve us well. An
appealing feature of word rates per 1,000 words is that readers are familiar
with documents of this length (e.g., a newspaper article).

.. ipython:: python

    import os
    import nltk
    import numpy as np
    from sklearn.feature_extraction.text import CountVectorizer

    filenames_with_path = [os.path.join(CORPUS_PATH, fn) for fn in filenames]
    # these texts have underscores ('_') that indicate italics; remove them.
    raw_texts = []
    for fn in filenames_with_path:
        with open(fn) as f:
            text = f.read()
            text = text.replace('_', '')  # remove underscores (italics)
            raw_texts.append(text)

    vectorizer = CountVectorizer(input='content')
    dtm = vectorizer.fit_transform(raw_texts)
    vocab = np.array(vectorizer.get_feature_names())
    # fit_transform returns a sparse matrix (which uses less memory)
    # but we want to work with a normal numpy array.
    dtm = dtm.toarray()

    # normalize counts to rates per 1000 words
    rates = 1000 * dtm / np.sum(dtm, axis=1, keepdims=True)

.. ipython:: python
    :suppress:

    assert rates.shape == dtm.shape

    filenames_short = [fn.rstrip('.txt') for fn in filenames]

    html = pd.DataFrame(np.round(rates[:, 100:110], 3), index=filenames_short, columns=vocab[100:110]).to_html()
    with open(os.path.join(OUTPUT_HTML_PATH, 'feature_selection_rates.txt'), 'w') as f:
        f.write(html)

.. ipython:: python

    # just examine a sample, those at offsets 100 to 110
    rates[:, 100:110]
    vocab[100:110]

.. raw:: html
    :file: generated/feature_selection_rates.txt

Measuring "distinctiveness"
===========================

Finding distinctive words requires a decision about what "distinctive" means.
As we will see, there are a variety of definitions that we might use.  It seems
reasonable to expect that all definitions of distinctive would identify as
distinctive words found exclusively in texts associated with a single author (or
a single group). For example, if Brontë uses the word "access" and Austen never
does, we should count "access" as distinctive. A more challenging question is
how to treat words that occur in both groups of texts but do so with different
rates.

Finding words that are unique to a group is a simple exercise. Indeed, it is
worth treating these words a special case so they will not clutter our work
later on. We will quickly identify these words and remove them. (They tend not
to be terribly interesting words.)

A simple way of identifying words unique to one author would be to calculate the
average rate of word use across all texts for each author and then to look for
cases where the average rate is zero for one author.

.. ipython:: python

    # indices so we can refer to the rows for the relevant author
    austen_indices, cbronte_indices = [], []
    for index, fn in enumerate(filenames):
        if "Austen" in fn:
            austen_indices.append(index)
        elif "CBronte" in fn:
            cbronte_indices.append(index)

    # this kind of slicing should be familiar if you've used R or Octave/Matlab
    austen_rates = rates[austen_indices, :]
    cbronte_rates = rates[cbronte_indices, :]

    # np.mean(..., axis=0) calculates the column-wise mean
    austen_rates_avg = np.mean(austen_rates, axis=0)
    cbronte_rates_avg = np.mean(cbronte_rates, axis=0)

    # since zero times any number is zero, this will identify documents where
    # any author's average rate is zero 
    distinctive_indices = (austen_rates_avg * cbronte_rates_avg) == 0

    # examine words that are unique, ranking by rates
    np.count_nonzero(distinctive_indices)
    ranking = np.argsort(austen_rates_avg[distinctive_indices] + cbronte_rates_avg[distinctive_indices])[::-1]  # from highest to lowest; [::-1] reverses order.
    vocab[distinctive_indices][ranking]

.. ipython:: python
    :suppress:

    arr = np.vstack([austen_rates_avg[distinctive_indices][ranking][0:N_WORDS_DISPLAY],
                     cbronte_rates_avg[distinctive_indices][ranking][0:N_WORDS_DISPLAY]])
    colnames = vocab[distinctive_indices][ranking][0:N_WORDS_DISPLAY]
    rownames = ['Austen', 'Brontë']
    html = pd.DataFrame(np.round(arr,3), index=rownames, columns=colnames).to_html()
    with open(os.path.join(OUTPUT_HTML_PATH, 'feature_selection_distinctive.txt'), 'w') as f:
        f.write(html)

.. raw:: html
    :file: generated/feature_selection_distinctive.txt

Now that we have identified these words, we will remove them from our corpus in
order to focus on identifying distinctive words that appear in texts associated
with every author.

.. ipython:: python

    dtm = dtm[:, np.invert(distinctive_indices)]
    rates = rates[:, np.invert(distinctive_indices)]
    vocab = vocab[np.invert(distinctive_indices)]

    # recalculate variables that depend on rates
    austen_rates = rates[austen_indices, :]
    cbronte_rates = rates[cbronte_indices, :]
    austen_rates_avg = np.mean(austen_rates, axis=0)
    cbronte_rates_avg = np.mean(cbronte_rates, axis=0)


Differences in averages
-----------------------

How can we identify a distinctive word? One approach would compare the average
rate at which authors use a word. A simple quantitative comparison would
calculate the difference between the rates. If one author uses a word often
across his or her oeuvre and another barely uses the word at all, then we
suspect the difference in rates will be large.  This will be the first
definition of distinctiveness (sometimes called "keyness") we will consider.
Using this measure we can calculate the top ten distinctive words in the
Austen-Brontë comparison as follows:

.. ipython:: python

    import numpy as np

    # calculate absolute value because we only care about the magnitude of the difference
    keyness = np.abs(austen_rates_avg - cbronte_rates_avg)
    ranking = np.argsort(keyness)[::-1]  # from highest to lowest; [::-1] reverses order in Python sequences

    # print the top 10 words along with their rates and the difference
    vocab[ranking][0:10]

.. ipython:: python
    :suppress:

    arr = np.vstack([keyness[ranking][0:N_WORDS_DISPLAY],
                     austen_rates[:, ranking][:, 0:N_WORDS_DISPLAY],
                     cbronte_rates[:, ranking][:, 0:N_WORDS_DISPLAY]])
    colnames = vocab[ranking][0:N_WORDS_DISPLAY]
    rownames = ["--keyness--"] + filenames_short
    html = pd.DataFrame(np.round(arr,3), index=rownames, columns=colnames).to_html()
    with open(os.path.join(OUTPUT_HTML_PATH, 'feature_selection_distinctive_avg_diff.txt'), 'w') as f:
        f.write(html)

.. raw:: html
    :file: generated/feature_selection_distinctive_avg_diff.txt

This is a start. The problem with this measure is that it tends to highlight
differences in very frequent words. For example, this method
gives greater attention to a word that occurs
30 times per 1,000 words in Austen and 25 times per 1,000 in Brontë
than it does to a word that occurs 5 times per 1,000 words in
Austen and 0.1 times per 1,000 words in Brontë. This does not seem
right. It seems important to recognize cases when one author uses a word
frequently and another author barely uses it.

As this initial attempt suggests, identifying distinctive words will be
a balancing act. When comparing two groups of texts differences in the rates of
frequent words will tend to be large relative to differences in the rates of
rarer words. Human language is variable; some words occur more frequently than
others regardless of who is writing.  We need to find a way of adjusting our
definition of distinctive in light of this.

One adjustment that is easy to make is to divide the difference in authors'
average rates by the average rate across all authors. Since dividing a quantity
by a large number will make that quantity smaller, our new distinctiveness score
will tend to be lower for words that occur frequently. While this is merely
a heuristic, it does move us in the right direction.

.. ipython:: python

    # we have already calculated the following quantities
    # austen_rates_avg
    # cbronte_rates_avg

    rates_avg = np.mean(rates, axis=0)

    keyness = np.abs(austen_rates_avg - cbronte_rates_avg) / rates_avg
    ranking = np.argsort(keyness)[::-1]  # from highest to lowest; [::-1] reverses order.

    # print the top 10 words along with their rates and the difference
    vocab[ranking][0:10]

.. ipython:: python
    :suppress:

    arr = np.vstack([keyness[ranking][0:N_WORDS_DISPLAY],
                     austen_rates[:, ranking][:, 0:N_WORDS_DISPLAY],
                     cbronte_rates[:, ranking][:, 0:N_WORDS_DISPLAY]])
    colnames = vocab[ranking][0:N_WORDS_DISPLAY]
    rownames = ["--keyness--"] + filenames_short
    html = pd.DataFrame(np.round(arr,3), index=rownames, columns=colnames).to_html()
    with open(os.path.join(OUTPUT_HTML_PATH, 'feature_selection_distinctive_avg_diff_divided_by_avg.txt'), 'w') as f:
        f.write(html)

.. raw:: html
    :file: generated/feature_selection_distinctive_avg_diff_divided_by_avg.txt

This method improves on our initial attempt. It has
the virtue of being simple and easy to implement. Yet it has its flaws. For
example, the method tends to overemphasize very rare words.

Just as there are many definitions of "similarity" or "distance" available to
compare two texts (see :ref:`working-with-text`), there are many definitions of
distinctive that can be used to identify words that characterize a group of
texts.

.. note:: While we used the absolute value of the difference in average rates,
    :math:`|x-y|` we might have easily used the squared difference,
    :math:`(x-y)^2` as it has similar properties (always positive, increasing as
    difference increases).

.. _bayesian-group-comparison:

Bayesian group comparison
=========================

.. note::

   The following sections assume some familiarity with statistics and
   probability. Introductory texts include :cite:`casella_statistical_2001`,
   :cite:`hoff_first_2009`, and :cite:`lee_bayesian_2004`.

.. note::

   The following excursion into the world of Bayesian inference and Gibbs
   sampling is closely related to topic modeling and Latent Dirichlet Allocation
   (LDA). The inference for the model discussed below proceeds using a Gibbs
   sampler from the full condition distribution of each variable of
   interest---precisely the same procedure is used in LDA.

A more nuanced comparison of word use in two groups takes account of the
variability in word use. Consider for instance the word "green"
in Austen and Brontë.  In Austen the word occurs with the following rates: 0.01,
0.03, and 0.06 (0.03 on average).  In Brontë the word is consistently more
frequent: 0.16, 0.36, and 0.22 (0.24 on average). These two groups of rates
look different. But consider how our judgment might change if the rates observed
in Brontë's novels were much more variable, say, 0.03, 0.04, and 0.66 (0.24 on
average).  Although the averages remain the same, the difference does not seem
so pronounced; with only one observation (0.66) noticeably greater than we find in Austen, we
might reasonably doubt that there is evidence of a systematic difference between
the authors. [#fnlyon]_

.. [#fnlyon] Unexpected spikes in word use happen all the time. Word usage in a large corpus
    is notoriously "bursty" (a technical term!) :cite:`church_poisson_1995`.
    Consider, for example, ten French novels, one of which is set in Lyon.
    While "Lyon" might appear in all novels, it would appear much (much) more
    frequently in the novel set in the city.]

One way of formalizing a comparison of two groups that takes account of the
variability of word usage comes from Bayesian statistics. To describe our
beliefs about the word frequencies we observe, we use a probability
distribution, which we will call our a sampling model. Under the model we will
use, the rates are assumed to come from two different normal distributions. The
question we will be asking is how confident we are that the means of the two
normal distributions are different. How confident we are (expressed as
a probability) that the means are indeed different will stand in as our measure
of distinctiveness.

We will use the parameterization below for our two normal sampling
distributions. Group 1 corresponds to Austen and group 2 corresponds to Brontë:

.. math::

    Y_{i,1} = \mu + \delta + \epsilon_{i,1}

    Y_{i,2} = \mu - \delta + \epsilon_{i,2}

    \{\epsilon_{i,j}\} \sim \textrm{i.i.d.} \; \textrm{Normal}(0, \sigma^2)

    n = 1, 2, 3

(i.i.d. stands for `independently and identically distributed
<http://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables>`_)

It is easy to relate this parameterization back to two normal distributions.
Austen's texts come from a normal distribution with mean parameter
:math:`\theta_1 = \mu + \delta` and variance :math:`\sigma^2`, whereas Brontë's
novels come from a distribution with the same variance and with mean parameter
:math:`\theta_2 = \mu - \delta`. :math:`\delta` corresponds to half the
difference between the two means and it is through this parameter that we will
judge how confident we are of a difference between the two distributions.

As we consider the question of what prior distributions to assign to
:math:`\mu`, :math:`\delta`, and :math:`\sigma^2` we need to keep in mind that
the word rates must be positive even though we are using normal distributions
(which will always assign some, potentially quite small, probability to negative
values).  A compromise that will allow us to make use of
computationally-convenient conjugate prior distributions will be to use normal
prior distributions that favor positive values in most cases. As we will be
modeling more than ten thousand of vocabulary items, computational speed will be
important. These are the prior distributions that we will use:

.. math::

    \mu \sim \textrm{Normal}(\mu_0, \tau_0^2)

    \delta \sim \textrm{Normal}(0, \gamma_0^2)

    \sigma^2 \sim \textrm{Inverse-Gamma}(\nu_0/2, \nu_0\sigma_0^2/2)

We need to determine suitable values for the priors' parameters
(called hyperparameters): :math:`\mu_0,
\tau_0^2, \gamma_0^2, \nu_0, \text{and} \sigma_0^2`. Let us consider
:math:`\mu_0` and :math:`\sigma_0^2` first. While words like "the" and "she"
occur quite frequently, the almost all words occur less than four times per
1,000 words:

.. ipython:: python

    np.mean(rates < 4)

    np.mean(rates > 1)

    from scipy.stats.mstats import mquantiles  # analgous to R's quantiles
    mquantiles(rates, prob=[0.01, 0.5, 0.99])

In keeping with this observation we will set :math:`\mu_0` to be 3 and
:math:`\gamma_0^2` to be :math:`1.5^2`, with the reasoning that when drawing
from a normal distribution, the great majority (.95) of observations will fall
between two standard deviations of the mean. There isn't tremendous variability
in rates across the works of a single author, so we will set :math:`\sigma_0^2`
to be 1 and :math:`\nu_0` to be 1. (If we were to use non-conjugate priors we
could more realistically model our prior beliefs about rates.) We know there is
considerable variability in the rates *between* authors, so we will choose
:math:`\tau_0^2` to be :math:`1.5^2`, as :math:`\delta` represents half the
difference between the means and its value is unlikely to be greater than 3 in
absolute value.

With these conjugate priors it is possible to use a Gibbs sampler to sample
efficiently from the posterior distribution, using the full conditional
distributions for the parameters of interest :cite:`hoff_first_2009`:

.. math::

    \{\mu|\mathbf{y_1}, \mathbf{y_2}, \delta, \sigma^2\} &\sim \textrm{Normal}(\mu_n, \gamma_n^2)\\
        \mu_n &= \gamma_n^2 \times [\mu_0/\gamma_0^2 + \sum_{i=1}^{n_1} (y_{i,1} - \delta)/\sigma^2 +
            \sum_{i=1}^{n_2} (y_{i,2} - \delta)/\sigma^2 ] \\
        \gamma_n^2 &= [1/\gamma_0^2 + (n_1+n_2)/\sigma^2]^{-1} \\

    \{\delta|\mathbf{y_1}, \mathbf{y_2}, \mu, \sigma^2\} &\sim \textrm{Normal}(\delta_n, \tau_n^2)\\
        \delta_n &= \tau_n^2 \times [ \delta_0/\tau_0^2 +
            \sum_{i=1}^{n_1} (y_{i,1} - \mu)/\sigma^2 - \sum_{i=1}^{n_2} (y_{i,2} - \mu)/\sigma^2 ]\\
        \tau_n^2 &= [1/\tau_0^2 + (n_1+n_2)/\sigma^2]^{-1} \\

    \{\sigma^2|\mathbf{y_1}, \mathbf{y_2}, \delta, \mu\} &\sim \textrm{Inverse-Gamma}(\nu_n/2, \nu_n\sigma_n^2/2)\\
        \nu_n &= \nu_0 + n_1 + n_2 \\
        \nu_n\sigma_n^2 &= \nu_0\sigma_0^2 +
            \sum_{i=1}^{n_1} (y_{i,1} - (\mu+\delta)) + \sum_{i=1}^{n_2} (y_{i,2} - (\mu - \delta)) \\

In Python, we can wrap the Gibbs sampler in single function and use it to get
a distribution of posterior values for :math:`\delta`, which is the variable we
care about in this context as it characterizes our belief about the difference
in authors' word usage.

.. ipython:: python

    def sample_posterior(y1, y2, mu0, sigma20, nu0, delta0, gamma20, tau20, S):
        """Draw samples from posterior distribution using Gibbs sampling
        Parameters
        ----------
        `S` is the number of samples
        Returns
        -------
        chains : dict of array
            Dictionary has keys: 'mu', 'delta', and 'sigma2'.
        """
        n1, n2 = len(y1), len(y2)
        # initial values
        mu = (np.mean(y1) + np.mean(y2))/2
        delta = (np.mean(y1) - np.mean(y2))/2
        vars = ['mu', 'delta', 'sigma2']
        chains = {key: np.empty(S) for key in vars}
        for s in range(S):
            # update sigma2
            a = (nu0+n1+n2)/2
            b = (nu0*sigma20 + np.sum((y1-mu-delta)**2) + np.sum((y2-mu+delta)**2))/2
            sigma2 = 1 / np.random.gamma(a, 1/b)
            # update mu
            mu_var = 1/(1/gamma20 + (n1+n2)/sigma2)
            mu_mean = mu_var * (mu0/gamma20 + np.sum(y1-delta)/sigma2 +
                                np.sum(y2+delta)/sigma2)
            mu = np.random.normal(mu_mean, np.sqrt(mu_var))
            # update delta
            delta_var = 1/(1/tau20 + (n1+n2)/sigma2)
            delta_mean = delta_var * (delta0/tau20 + np.sum(y1-mu)/sigma2 -
                                    np.sum(y2-mu)/sigma2)
            delta = np.random.normal(delta_mean, np.sqrt(delta_var))
            # save values
            chains['mu'][s] = mu
            chains['delta'][s] = delta
            chains['sigma2'][s] = sigma2
        return chains

.. ipython:: python

    # data
    word = "green"
    y1, y2 = austen_rates[:, vocab == word], cbronte_rates[:, vocab == word]

    # prior parameters
    mu0 = 3
    tau20 = 1.5**2

    nu0 = 1
    sigma20 = 1

    delta0 = 0
    gamma20 = 1.5**2

    # number of samples
    S = 2000

    chains = sample_posterior(y1, y2, mu0, sigma20, nu0, delta0, gamma20, tau20, S)

    delta = chains['delta']


These samples reflect what our belief about :math:`\delta` ought to be given our
prior specification. Our interest is in :math:`\delta`, which represents the
half the difference between the population means for the distributions
characterizing word rates in Austen and Brontë. We aren't concerned with whether
or not it is negative or positive, but we do care whether or not it is likely to
be zero. In fact, we need to have a measure of how confident we are that
:math:`\deta` is something other than zero (implying no difference in means).
If, for instance, the moment that samples of :math:`\delta` tend to be negative;
we need to know the posterior probability of its being definitively less than
zero, :math:`\textrm{p}(\delta < 0)`. This probability can be estimated from the
output of the Gibbs sampler. The following demonstrates the calculation of this
probability for two different words, 'green' and 'dark', both words more
characteristic of the Brontë novels than the Austen novels.

.. ipython:: python

    y1 = austen_rates[:, vocab == 'green']
    y2 = cbronte_rates[:, vocab == 'green']
    chains = sample_posterior(y1, y2, mu0, sigma20, nu0, delta0, gamma20, tau20, S)
    delta_green = chains['delta']

    y1 = austen_rates[:, vocab == 'dark']
    y2 = cbronte_rates[:, vocab == 'dark']
    chains = sample_posterior(y1, y2, mu0, sigma20, nu0, delta0, gamma20, tau20, S)
    delta_dark = chains['delta']

    # estimate of p(delta < 0)
    np.mean(delta_dark < 0)


.. ipython:: python

    words = ['dark', 'green']
    ix = np.in1d(vocab, words)

    @suppress
    assert all(vocab[ix] == words)  # order matters for subsequent display

    keyness = np.asarray([np.mean(delta_dark < 0), np.mean(delta_green < 0)])

.. ipython:: python
    :suppress:

    arr = [keyness, austen_rates_avg[ix], cbronte_rates_avg[ix]]
    colnames = vocab[ix]
    rownames = ['p(delta<0)', 'Austen average', 'Bronte average']
    html = pd.DataFrame(np.round(arr,3), index=rownames, columns=colnames).to_html()
    with open(os.path.join(OUTPUT_HTML_PATH, 'feature_selection_bayesian_dark_green.txt'), 'w') as f:
        f.write(html)

.. raw:: html
    :file: generated/feature_selection_bayesian_dark_green.txt

As 'dark' is more distinctive of Brontë than 'green' is, the probabilities
(our measure of distinctiveness or keyness) reflect this.

If we want to apply this "feature selection" method *en masse* to every word
occurring in the corpus, we need only write one short loop and make an
adjustment for the fact that we don't care whether or not :math:`\delta` is
positive or negative:

.. ipython:: python

    # fewer samples to speed things up, this may take several minutes to run
    S = 200

    def delta_confidence(rates_one_word):
        austen_rates = rates_one_word[0:3]
        bronte_rates = rates_one_word[3:6]
        chains = sample_posterior(austen_rates, bronte_rates, mu0, sigma20, nu0,
                                  delta0, gamma20, tau20, S)
        delta = chains['delta']
        return np.max([np.mean(delta < 0), np.mean(delta > 0)])

.. ipython:: python
    :suppress:

    # because this computation takes so long, we will try to cache it
    CACHE_PATH = os.path.join('source', 'cache')
    KEYNESS_FILENAME = os.path.join(CACHE_PATH, 'feature_selection_keyness.npy')
    os.path.exists(KEYNESS_FILENAME)
    keyness = np.load(KEYNESS_FILENAME) if os.path.exists(KEYNESS_FILENAME) else np.apply_along_axis(delta_confidence, axis=0, arr=rates)
    np.save(KEYNESS_FILENAME, keyness)
    os.path.exists(KEYNESS_FILENAME)

.. code-block:: python

    # apply the function over all columns
    In [117]: keyness = np.apply_along_axis(delta_confidence, axis=0, arr=rates)

.. ipython:: python

    ranking = np.argsort(keyness)[::-1]  # from highest to lowest; [::-1] reverses order.

    # print the top 10 words along with their rates and the difference
    vocab[ranking][0:10]

.. ipython:: python
    ::suppress::

    arr = np.vstack([keyness[ranking][0:N_WORDS_DISPLAY],
                     austen_rates[:, ranking][:, 0:N_WORDS_DISPLAY],
                     cbronte_rates[:, ranking][:, 0:N_WORDS_DISPLAY]])
    colnames = vocab[ranking][0:N_WORDS_DISPLAY]
    rownames = ["--keyness--"] + filenames_short
    html = pd.DataFrame(np.round(arr,3), index=rownames, columns=colnames).to_html()
    with open(os.path.join(OUTPUT_HTML_PATH, 'feature_selection_bayesian.txt'), 'w') as f:
        f.write(html)

.. raw:: html
    :file: generated/feature_selection_bayesian.txt

This produces a useful ordering of characteristic words. Unlikely `frequentist
<https://en.wikipedia.org/wiki/Frequentist_inference>`_ methods discussed below
(chi-squared and log likelihood) this approach considers the variability of
observations within groups. This method will also work for small corpora
provided useful prior information is available. To the extent that we are
interested in a close reading of differences of vocabulary use, the Bayesian
method should be preferred. [#fnunderwood]_

.. _chi2:

Log likelihood ratio and :math:`\chi^2` feature selection
=========================================================

We can recast our discussions about measuring distinctiveness in terms of
hypothesis testing. This turns out to be a satisfying way of thinking about the
problem and it also allows us to introduce a range of feature selection methods,
including the log likelihood test and the :math:`\chi^2` test.

One hypothesis that we might test comes as no surprise: rather than two groups
of texts characterized by different word rates, this hypothesis claims that
there is, in fact, a single group. Words are examined one at a time; those words
for which this hypothesis seems most wrong will be counted as distinctive
(classical statistics is always a workout in counterfactual language).

Consider again the word "green". Taking all the Austen texts together, the word
"green" occurs 11 times out of ~370,000 words (0.03 per 1,000 words). In the
novels by Brontë, "green" occurs 96 times out of ~400,000 (0.24 per 1,000
words). We do not really need statistics to tell us that this is a large
difference: picking a word from each author-specific corpus at random, one is ten
times more likely to find "green" in the Brontë corpus. To summarize the
appearance of the word "green" we may assemble a table with the following code:

.. ipython:: python

    green_austen = np.sum(dtm[austen_indices, vocab == "green"])
    nongreen_austen = np.sum(dtm[austen_indices, :]) - green_austen
    green_cbronte = np.sum(dtm[cbronte_indices, vocab == "green"])
    nongreen_cbronte = np.sum(dtm[cbronte_indices, :]) - green_cbronte

    green_table = np.array([[green_austen, nongreen_austen],
                            [green_cbronte, nongreen_cbronte]])
    green_table

.. ipython:: python
    ::suppress::

    arr = green_table
    colnames = ['"green"', 'not "green"']
    rownames = ['Austen', 'C. Brontë']
    html = pd.DataFrame(arr, index=rownames, columns=colnames).to_html()
    with open(os.path.join(OUTPUT_HTML_PATH, 'feature_selection_green_table.txt'), 'w') as f:
        f.write(html)

.. raw:: html
    :file: generated/feature_selection_green_table.txt

The hypothesis being tested is that the grouping of the counts by author is
unnecessary, that :math:`P(word = "green" | author = "Austen") = P(word
= "green" | author != "Austen")`. If this were the case, then the rate of
"green" in the corpus is the same, namely 0.14 per 1,000 words, and we would
anticipate seeing the following frequencies given the total number of words
for each group of texts:

.. ipython:: python

    prob_green = np.sum(dtm[:, vocab == "green"]) / np.sum(dtm)
    prob_notgreen = 1 - prob_green
    labels = []
    for fn in filenames:
        label = "Austen" if "Austen" in fn else "CBrontë"
        labels.append(label)
    n_austen = np.sum(dtm[labels == "Austen", :])
    n_cbronte = np.sum(dtm[labels != "Austen", :])

    expected_table = np.array([[prob_green * n_austen, prob_notgreen * n_austen],
                               [prob_green * n_cbronte, prob_notgreen * nongreen_cbronte]])
    expected_table

    # same result, but more concise and more general
    from sklearn.preprocessing import LabelBinarizer
    X = dtm[:, vocab == "green"]
    X = np.append(X, np.sum(dtm[:, vocab != "green"], axis=1, keepdims=True), axis=1)
    y = LabelBinarizer().fit_transform(labels)
    y = np.append(1 - y, y, axis=1)
    green_table = np.dot(y.T, X)
    green_table

    feature_count = np.sum(X, axis=0, keepdims=True)
    class_prob = np.mean(y, axis=0, keepdims=True)
    expected_table = np.dot(class_prob.T, feature_count)

In classical statistics, hypothesis tests typically have a quantity called
a test statistic associated with them. If the test statistic is greater than
a critical value the hypothesis is rejected. In this case, the test statistic is
identical with our measure of distinctiveness. The test commonly used to analyze
the present hypothesis (that two distinct groups are unnecessary) is the log
likelihood ratio test, and its statistic is called the log likelihood ratio
(alternatively a `G-test <http://en.wikipedia.org/wiki/G-test>`_ statistic or
`Dunning log likelihood <http://acl.ldc.upenn.edu/J/J93/J93-1003.pdf>`_
:cite:`dunning_accurate_1993`).  Various symbols are associated with this
statistic, including :math:`G`, :math:`G^2`, :math:`l`,  and :math:`\lambda`.
(The theoretical underpinnings of the log likelihood ratio test and its
application to corpus analysis are covered in chapter 8 of Casella and Berger
(2001) and Dunning (1993) :cite:`casella_statistical_2001`
:cite:`dunning_accurate_1993`.)

The log likelihood ratio is calculated as follows:

.. math::

    \sum_i O_i \times \ln \frac{O_i/E_i}

where :math:`i` indexes the cells. (Note the similarity of this formula to the
calculation of :ref:`mutual information <mutual_information>`.) In Python:

.. ipython:: python

    G = np.sum(green_table * np.log(green_table / expected_table))

The higher the value of the test statistic, the more pronounced the deviation is
from the hypothesis---and, for our purposes, the more "distinctive" the word is.

Pearson's :math:`\chi^2` test statistic approximates the log likelihood ratio
test (:math:`\chi^2` is read chi-squared). It is computationally easier to
calculate. The Python library ``scikit-learn`` provides a function
``sklearn.feature_selection.chi2`` that allows us to use this test statistic as
a feature selection method:

.. ipython:: python

    from sklearn.feature_selection import chi2
    labels = []
    for fn in filenames:
        label = "Austen" if "Austen" in fn else "CBrontë"
        labels.append(label)

    # chi2 returns two arrays, the chi2 test statistic and an
    # array of "p-values", which we'll ignore
    keyness, _ = chi2(dtm, labels)
    ranking = np.argsort(keyness)[::-1]
    vocab[ranking][0:10]

.. ipython:: python
    :suppress:

    arr = np.vstack([keyness[ranking][0:N_WORDS_DISPLAY],
                     austen_rates[:, ranking][:, 0:N_WORDS_DISPLAY],
                     cbronte_rates[:, ranking][:, 0:N_WORDS_DISPLAY]])
    colnames = vocab[ranking][0:N_WORDS_DISPLAY]
    rownames = ["--keyness--"] + filenames_short
    html = pd.DataFrame(np.round(arr,2), index=rownames, columns=colnames).to_html()
    with open(os.path.join(OUTPUT_HTML_PATH, 'feature_selection_distinctive_chi2.txt'), 'w') as f:
        f.write(html)

.. raw:: html
    :file: generated/feature_selection_distinctive_chi2.txt


.. note::

    Logarithms are expensive. Calculating the log likelihood ratio over
    a vocabulary of 10,000 words will involve taking 40,000 logarithms. The
    :math:`\chi^2` test statistic, by contrast, involves taking the square of
    a quantity the same number of times. On my computer, calculating the
    logarithm takes about twenty times longer than taking the square (simple
    multiplication):

    .. ipython:: python

        import timeit
        time_log = timeit.timeit("import numpy as np; np.log(np.arange(40000))", number=100)
        time_square = timeit.timeit("import numpy as np; np.square(np.arange(40000))", number=100)
        time_log / time_square

.. _mutual_information:

Mutual information feature selection
====================================

Feature selection based on mutual information also delivers good results.
Good introductions to the method can be found in `Cosma Shalizi's Data Mining
course <http://www.stat.cmu.edu/~cshalizi/350/>`_ (`Finding Informative Features
<http://www.stat.cmu.edu/~cshalizi/350/lectures/05/lecture-05.pdf>`_) and in
`section 13.5
<http://www-nlp.stanford.edu/IR-book/html/htmledition/feature-selection-1.html>`_
in :cite:`manning_introduction_2008`.

Feature selection as exploratory data analysis
==============================================

If nothing else, studying methods of feature selection forces us to think
critically about what we mean when we say some characteristic is "distinctive".

In practice, these methods let us quickly identify features (when they exist)
that appear more or less often in one group of texts.  As such, these methods
are useful for dimensionality reduction and exploratory data analysis.  For
example, if we suspect that there is a meaningful partition of a collection of
texts, we can use one of the methods described above to pull out features that
characterize the proposed groups of texts and explore whether those features
make sense given other information. Or we may be confronted with a massive
dataset---perhaps all 1-, 2-, and 3-grams in the corpus---and need to reduce the
space of features so that our analyses can run on a computer with limited
memory.

Feature selection needs to be used with care when working with a small number of
observations and a relatively large number of features---e.g., a corpus with of
a small number of documents and a very large vocabulary. Feature selection is
perfectly capable of pulling out features that are characteristic of any
division of texts.

.. note:: The shorthand :math:`n >> p` is used to describe situations where
    the number of variables greatly outnumbers the number observations.
    :math:`n` is the customary label for the number of observations and
    :math:`p` refers to the number of covariates.

A brief demonstration that feature selection "works" as expected can be seen by
plotting the cosine distance among texts in the corpus before and after feature
selection is applied. ``chi2`` is the feature selection used in the bottom
figure and the top 50 words are used.

.. ipython:: python
    ::suppress::

    import matplotlib.pyplot as plt
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.manifold import MDS
    dist = 1 - cosine_similarity(dtm)
    mds = MDS(n_components=2, dissimilarity="precomputed")
    pos = mds.fit_transform(dist)  # shape (n_components, n_samples)

.. ipython:: python
    ::suppress::

    xs, ys = pos[:, 0], pos[:, 1]
    names = [os.path.basename(fn).replace('.txt', '') for fn in filenames]
    for x, y, name in zip(xs, ys, names):
        color = 'orange' if "Austen" in name else 'skyblue'
        plt.scatter(x, y, c=color)
        plt.text(x, y, name)

    @savefig plot_feature_selection_mds_before.png width=7in
    plt.title("Before feature selection")


.. ipython:: python
    ::suppress::

    keyness, _ = chi2(dtm, names)
    selected = np.argsort(keyness)[::-1][0:50]
    dtm_chi2 = dtm[:, selected]
    dist = 1 - cosine_similarity(dtm_chi2)
    mds = MDS(n_components=2, dissimilarity="precomputed")
    pos = mds.fit_transform(dist)  # shape (n_components, n_samples)


.. ipython:: python
    ::suppress::

    xs, ys = pos[:, 0], pos[:, 1]
    for x, y, name in zip(xs, ys, names):
        color = 'orange' if "Austen" in name else 'skyblue'
        plt.scatter(x, y, c=color)
        plt.text(x, y, name)

    @savefig plot_feature_selection_mds_after.png width=7in
    plt.title("After feature selection")

Exercises
=========

1. Using the two groups of texts (Austen and C. Brontë), find the top 40
   characteristic words by the :math:`\chi^2` statistic. Feel free to use
   scikit-learn's ``chi2``.

2. The following is a random partition of the texts. Find the top 40
   characteristic words by the :math:`\chi^2` statistic. How do these
   compare with those you found in exercise 1?

.. ipython:: python
    ::suppress::

    import random
    random.seed(1)
    shuffled = filenames.copy()
    random.shuffle(shuffled)
    group_a = shuffled[:len(filenames)//2]
    group_b = shuffled[len(filenames)//2:]

.. ipython:: python
    ::suppress::

    group_a
    group_b

3. Reconstruct the corpus using only these 40 words. Find the cosine distances
   between pairs of texts and visualize these using multi-dimensional scaling
   (see :ref:`working-with-text` for a refresher). Compare this plot to the MDS
   plot of the distances between texts using the full vocabulary.


.. FOOTNOTES

.. [#fnunderwood] Ted Underwood has written a `blog post discussing some of the
   drawbacks of using the log likelihood and chi-squared test statistic in the
   context of literary studies <http://tedunderwood.com/2011/11/09/identifying-the-terms-that-characterize-an-author-or-genre-why-dunnings-may-not-be-the-best-method/>`_.]


