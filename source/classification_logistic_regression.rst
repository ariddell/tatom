.. _classification-machine-learning:

===========================================================
 Classification, Machine Learning, and Logistic Regression
===========================================================

Previous tutorials have illustrated how probabilistic topic models can be used
to navigate a large corpus. As general purpose tools for identifying recurrent
themes in a corpus, topic models and non-negative matrix factorization are
useful. They perform better than methods previously used for similar
purposes, such as principle component analysis (PCA) and latent semantic
analysis (LSA). For tasks such as classifying texts into a known set of categories, however,
there exist methods that are better suited to the problem. One family of such methods
goes under the heading of neural networks (or, more recently, "deep learning").
An essential conceptual and practical building block for these methods is
logistic regression, which we will review briefly in this tutorial.

.. note:: Discussion of the role of logistic regression in neural networks may
    be found in section 5.1 of Bishop (2007) :cite:`bishop_pattern_2007`.

Predicting genre classifications
================================

The bag-of-words model is a horrible model of a text. Its failure to distinguish
word order ('the cat ate the fish' from 'the fish ate the cat') is the least of
its failings. In most cases, knowing the frequency with which a word occurs in
a text tells us very little. Without additional context it is
difficult to know how to interpret a word's frequency. For example, the word
'heart' might occur in a discussion of courtly love, of physical exercise, or in
a cookbook (e.g., "heart of palm"). And even when a word seems to have a single
interpretation, its meaning may depend on words occurring around it.

Nevertheless, sometimes the frequency of words appears to be correlated with
useful information, such as pre-existing classifications (or classifications in
which we happen to believe). Consider the word "ennemis" ("enemies") in the
context of a corpus of :ref:`French classical theatre <datasets>`. This corpus
includes only plays classified as tragedy or comedy. The word "ennemis" is not,
at first glance, a word particularly troubled by problems of polysemy.
Considered as an indicator of whether or not a play is a tragedy or a comedy,
the frequency of "ennemis" seems to be a reliable guide; the word tends to occur
more often in tragedies.

The first way we can verify this is simply to calculate the percentage of plays
classified as tragedy in which the word 'ennemis' occurs and compare that
percentage with the corresponding percentage for comedies. As usual, in order to
have a better sense of the variability of language in French classical theatre,
we have split the plays into approximately 1,000-word sections.

.. ipython:: python

    import os

    import numpy as np
    from sklearn.feature_extraction.text import CountVectorizer

    data_dir = 'data/french-tragedies-and-comedies-split/'

    filenames = np.array(os.listdir(data_dir))
    filenames_with_path = [os.path.join(data_dir, fn) for fn in filenames]

    # tragedies and comedies are coded with 'TR' or 'CO',
    # e.g., PCorneille_TR-V-1647-Heraclius0001.txt
    genre = []
    for fn in filenames:
        genre.append('tragedy' if '_TR-' in fn else 'comedy')
    genre = np.array(genre)

    # .strip() removes the trailing newline '\n' from each line in the file
    french_stopwords = [l.strip() for l in open('data/stopwords/french.txt')]
    vectorizer = CountVectorizer(input='filename', min_df=15, max_df=.95, stop_words=french_stopwords, max_features=3000)
    dtm = vectorizer.fit_transform(filenames_with_path)
    dtm = dtm.toarray()
    vocab = np.array(vectorizer.get_feature_names())

    # texts are split into documents of approximately equal length, so we will
    # skip the normalization step and deal directly with counts

Having assembled the corpus, it is easy to calculate the number of play sections
in which 'ennemis' occurs.

.. ipython:: python

    word = "ennemis"
    tragedy_counts = dtm[genre == 'tragedy', vocab == word]
    comedy_counts = dtm[genre == 'comedy', vocab == word]

    # tragedy percentage
    np.count_nonzero(tragedy_counts) / len(tragedy_counts)
    # comedy percentage
    np.count_nonzero(comedy_counts) / len(comedy_counts)

    # overall percentage
    np.count_nonzero(dtm[:, vocab == word]) / len(dtm)

    # text in which "ennemis" appears the most
    filenames[np.argmax(dtm[:, vocab == word])], np.max(dtm[:, vocab == word])

In our sample, if a play section is a tragedy it features the word 'ennemis' about a third
of time. Among comedy sections, the word appears in only five percent. (Recall, however,
that in the majority of play sections the word *does not appear* at all.) While this
gives us a rough sense of the relationship between the word 'ennemis' and genre,
we may want to describe the relationship more precisely.  First, we would like to
consider the relationship between the word's frequency (rather than just its
presence or absence) and a text's classification. Second, we want to
predict the classification of a section of a play for which we do not have
a classification ready at hand. Logistic regression accomplishes both of these
tasks.

Like linear regression, logistic regression will happily make predictions based
on aleatory patterns in our data. It is therefore important to make sure we have
some additional basis for believing there might be a correlation between the
frequency of the word 'ennemis' and a genre classification. Our intuition tells
us that the word (particularly in its plural form) does not belong in a comedy
(or at least not in any great frequency), whereas we can imagine a variety of
sentences using the word appearing in a tragedy.  Consider, for example, the
section of Racine's *Thebaide* which features the six occurrences of the word
(and plenty of 'ennemi' as well):

::

   Plus qu'à mes ennemis la guerre m'est mortelle,
   Et le courroux du ciel me la rend trop cruelle ;
   Il s'arme contre moi de mon propre dessein,
   Il se sert de mon bras pour me percer le sein.
   La guerre s'allumait, lorsque pour mon supplice,
   Hémon m'abandonna pour servir Polynice ;
   Les deux frères par moi devinrent ennemis,
   Et je devins, Attale, ennemi de mon fils.
   ...

In quantitative text analysis, a common way to represent a classification is as
a binary outcome, e.g., 0 for comedy or 1 for tragedy. Whereas linear regression
relates some quantity ``x`` to another quantity ``y``, logistic regression
relates a quantity ``x`` to the *probability* of something being a member of one
of two groups, that is, the probability of ``y`` having a value of 1.

For reasons covered in greater detail at the :ref:`end of this section
<logistic-regression>`, the probability of classification is expressed not in
terms of probability (from 0 to 1) but in log `odds
<https://en.wikipedia.org/wiki/Odds>`_. This is not a mysterious transformation.
Indeed, in certain countries (and among individuals involved in
gambling) expressing the likelihood of an event in terms of odds is common.
Moving between probability, odds, and log odds is somewhat tedious but not
difficult---e.g., an event occurring with probability 0.75, it occurs with odds
3 (often expressed 3:1) and with log odds 1.1. Logistic regression delivers, for
any value of ``x``, here the frequency of the word 'ennemis', the log odds of
a play section being from a tragedy.  Typically we immediately convert the log
odds into probability as the latter is more familiar.

.. note:: For very rare or very probable events using odds (and even log
    odds) can be preferable to using probabilities. Consider the
    `Intergovernmental Panel on Climate Change's <https://en.wikipedia.org/wiki/Intergovernmental_Panel_on_Climate_Change>`_
    `guidance on addressing uncertainties <https://www.ipcc.ch/pdf/supporting-material/uncertainty-guidance-note_ar4.pdf>`_.

        ======================   ======================  ============  ============
        Terminology              Likelihood              Odds          Log odds
        ======================   ======================  ============  ============
        Virtually certain        99% probability         99 (or 99:1)  > 4.6
        Very likely              > 90% probability       > 9           > 2.2
        Likely                   > 66% probability       > 2           > 0.7
        About as likely as not   33 to 66% probability   0.5 to 2      -0.7 to 0.7
        Unlikely                 < 33% probability       < 0.5         < -0.7
        Very unlikely            < 10% probability       < .1          < -2.2
        Exceptionally unlikely   < 1% probability        < 0.01        < -4.6
        ======================   ======================  ============  ============

    Note that whereas moving from a likelihood of 33% to 66% corresponds to
    moving from 0.5 to 2 on the odds scale, moving from 90% to 99% entails
    moving from 9 to 99 on the odds scale. The odds scale expresses better 
    the difference between an event that happens 9 out of 10 times versus an
    event that happens 99 times out of 100.

First we will fit the logistic regression model using the ``statsmodels``
package and then, converting from log odds to the more familiar scale of
probability, we will plot this estimated relationship.

.. ipython:: python

    import statsmodels.api as sm

    wordfreq = dtm[:, vocab == "ennemis"]

    # we need to add an intercept (whose coefficient is related to the
    # probability that a novel will be classified a tragedy when the
    # predictor is zero.
    # This is done automatically in R and by sklearn's LogisticRegression
    X = sm.add_constant(wordfreq)
    model = sm.GLM(genre == 'tragedy', X, family=sm.families.Binomial())
    fit = model.fit()
    fit.params

For those accustomed to fitting regression models in R, the following code
produces precisely the same results:

.. code-block:: r

    data = data.frame(wordfreq = wordfreq, genre = genre == 'tragedy')
    fit = glm(genre ~ wordfreq, data = data, family = binomial(link="logit"))
    coef(fit)

    # note that R is implicitly adding a constant term. We can make this
    # term explicit in our model if we choose (the results should be the same)
    fit = glm(genre ~ 1 + wordfreq, data = data, family = binomial(link="logit"))
    coef(fit)

Using the fitted parameters of the model we can make a prediction for any given
word frequency. For example, the probability of a section in which 'ennemis'
occurs twice given by

.. ipython:: python

    def invlogit(x):
        """Convert from log odds to probability"""
        return 1/(1+np.exp(-x))

    x = 2
    invlogit(fit.params[0] + fit.params[1] * x)


The following code plots the relationship between a section's word frequency and
the model's estimate of the probability of a section being from a tragedy.  The
points on the figure mark the observations in the corpus. (The points have been
jittered to improve readability.)

.. ipython:: python

    xs = np.arange(min(wordfreq), max(wordfreq) + 1, 0.1)
    ys = np.array([invlogit(x) for x in xs])
    plt.plot(xs, ys, linewidth=2)
    # jitter the outcomes (0 or 1) a bit
    jitter = np.random.random(len(genre)) / 5
    ys_outcomes = np.abs((genre == 'tragedy') - 0.01 - jitter)
    alpha = 0.7
    # use different colors for the different classes
    plt.plot(wordfreq[genre == 'tragedy'], ys_outcomes[genre == 'tragedy'], 'b.', alpha=alpha)
    plt.plot(wordfreq[genre != 'tragedy'], ys_outcomes[genre != 'tragedy'], 'y.', alpha=alpha)
    plt.xlabel("Word frequency")
    plt.ylabel("Predicted probability of play section being a tragedy")

    @suppress
    assert np.max(wordfreq) == 6

    plt.title("Predicting genre by the frequency of 'ennemis'")
    # make some final aesthetic adjustments of the plot boundary
    @savefig plot_logistic_ennemis.png width=7in
    plt.xlim(-0.1, max(wordfreq) + 0.2); plt.tight_layout()

The figure illustrates what the model infers: if 'ennemis' appears more than
three times in a section it will tend to be a tragedy with high probability.

As an experiment and an illustration of `cross validation
<https://en.wikipedia.org/wiki/Cross-validation_%28statistics%29#K-fold_cross-validation>`_
(also called out-of-sample validation), consider the task of predicting the
classification of a section of text based on the frequency of 'ennemis' alone.
From the 3,429 play sections in our corpus we will take one third of them at
random and ask the model to predict their classification with the model
fitted on the remaining sections. We will do this three times (once for each
held-out third). The scikit-learn package makes this procedure embarrassingly
easy, provided we use its version of logistic regression, which is designed for
large datasets and differs slightly from the version provided by R and
statsmodels. [#fn_sklearn_logisticregression]_

.. ipython:: python

    from sklearn import cross_validation
    from sklearn import linear_model

    clf = linear_model.LogisticRegression()
    cross_validation.cross_val_score(clf, wordfreq, genre == 'tragedy')

Since 'ennemis' only appears in 20% of the sections and appears more than once
in only 5% of the sections, the model will only have useful information to work
with in a fraction of the cases presented to it. Nevertheless, it does
considerably better than a baseline of simply picking 'tragedy' every time, which
would be expected to achieve 52% accuracy, as sections from tragedies make up 52% of the sections.

Of course, if we give the model access to all the word frequencies in the corpus
(not just 'ennemis') and ask it to make predictions it does much better:

.. ipython:: python

    clf = linear_model.LogisticRegression()
    cross_validation.cross_val_score(clf, dtm, genre == 'tragedy')

.. note:: Those interested in using a large number of predictors---such as
    a matrix with 3,000 features---should use the implementation of logistic
    regression found in scikit-learn. Unlike the default version provided by
    R or statsmodels, scikit-learn's version includes a `penalty or
    regularization term
    <https://en.wikipedia.org/wiki/Regularization_%28mathematics%29>`_, which
    tends to help prevent `overfitting
    <https://en.wikipedia.org/wiki/Overfitting>`_ that can occur in models using
    a large number of predictors.

.. _logistic-regression:

Logistic regression
===================

.. note:: Resources for those interested in learning about logistic (and linear)
    regression include Gelman and Hill (2006) :cite:`gelman_data_2006` and
    Bishop (2007) :cite:`bishop_pattern_2007`. Stanford's OpenClassroom also has
    a `series of lectures devoted to logistic regression
    <http://openclassroom.stanford.edu/MainFolder/CoursePage.php?course=DeepLearning>`_.

Linear regression is one way of thinking about the relationship between two
variables. Logistic regression is a linear model as well; it assumes a linear,
additive relationship between the predictors and the *log odds* of a classification.
With a single predictor and an intercept term, the relationship between
a classification and a predictor has the following symbolic expression:

.. math::

   P(y_i = \mathrm{tragedy}) &= \mathrm{logit}^{-1}(\beta_0 + \beta_1 x_i)\\
              &= \frac{e^{\beta_0 + \beta x_i}}{1+e^{\beta_0 + \beta_1 x_i}}\\
              &= \frac{1}{1+e^{-(\beta_0 + \beta_1 x_i)}}\\
              &= \sigma(\beta_0 + \beta_1 x_i)\\

Typically we have more than one observation. Letting :math:`\sigma(x_i\beta)`
stand in for :math:`\frac{1}{1+e^{-(\beta_0 + \beta_1 x_i)}}` the `maximum
likelihood estimate
<https://en.wikipedia.org/wiki/Maximum_likelihood_estimate>`_ for :math:`\beta`
is the value of :math:`\beta` which maximizes the log
likelihood of the observations:

.. math::

   \log \prod_{i=1}^n P(y_i = \mathrm{tragedy}) &= \sum \left( y_i \log \sigma(x_i \beta) + (1 - y_i) \log (1 - \sigma(x_i \beta)) \right)\\
   
While for linear regression there is frequently a closed-form solution for the
maximum, logistic regression lacks a tidy solution. The solution (there is
indeed a unique maximum) is typically found using `iteratively reweighted least
squares <https://en.wikipedia.org/wiki/Iteratively_reweighted_least_squares>`_.

The solution may be found in Python using ``statsmodels.api.GLM`` or in R using
the built-in ``glm`` function. The two functions should yield identical results.

.. FOOTNOTES

.. [#fn_sklearn_logisticregression] Scikit-learn's ``LogisticRegression``
  includes a penalty term which prevents overfitting, something that is
  a major concern when the number of predictors exceeds the number of
  observations.  Those wishing for a logistic regression model that mirrors
  R's ``glm()`` should use ``statsmodels``'s ``GLM``.
