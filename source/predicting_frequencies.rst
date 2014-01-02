.. _predicting:

=============================
 Predicting Word Frequencies
=============================

.. ipython:: python
    :suppress:

    import numpy as np; np.set_printoptions(precision=3)

The following is an exercise intended to illustrate Bayesian inference. This
section shows how to combine a sampling model and prior beliefs in order to make
predictions about an unobserved event.  In this case, the unobserved quantity
will be the word frequencies in the chapters of Jane Austen's novel, *Pride and
Prejudice*.

**Problem**: predict the number of times "and" occurs in chapter six of *Pride
and Prejudice*. You have the full text of Jane Austen's *Mansfield Park* and the
number of times "and" occurs in chapters one through five in *Pride and
Prejudice*.

Predicting something that we know or could easily look up, such as the rate of
"and" in chapter six of *Pride and Prejudice*, may seem like a vacuous exercise.
In fact, it is a useful and intuitively appealing way to assess a model. Just as
asking a model of the behavior of the stock market or the weather in Berlin to
predict future events is a good way to test whether a model is credible,
predicting unseen (or "held-out") quantities is a convenient way to check
a probabilistic description of something.

Rates of frequent words such as "the", "and", "to", and "by" turn out to be
remarkably reliable indicators of an author's style. For example, the frequency
of the word "and" (and other conjunctions) tells us something about a writer's
tendency to join sentences together. In order to get a sense of how rates of
a single frequent word such as "and" can be indicative of an author's style,
consider the rates of the word in `|MP|
<http://en.wikipedia.org/wiki/Mansfield_Park>`_ and three other texts: Edward
Said's `|CI| <http://en.wikipedia.org/wiki/Culture_and_Imperialism>`_, Getrude
Stein's `|TL| <http://www.gutenberg.org/ebooks/15408>`_, , Ernest Hemingway's
`|TSAR| <http://en.wikipedia.org/wiki/The_Sun_Also_Rises>`_. As these texts lack
the frequent chapter divisions present in *Mansfield Park* and *Pride and
Prejudice*, the rate of "and" will be calculated for segments (or, less
elegantly, "chunks") of roughly 1,000 words.


.. Hacks to get italicized hyperlinks

.. |MP| replace:: *Mansfield Park*
.. |PP| replace:: *Pride and Prejudice*
.. |CI| replace:: *Culture and Imperialism*
.. |TSAR| replace:: *The Sun Also Rises*
.. |TL| replace:: *Three Lives*


.. ipython:: python

    import os
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.stats
    from distantreader import get_num_sentences, get_dtm_and_vocab, get_word_rates
    
    get_fns_in_path = lambda path: [os.path.join(path, fn)
                                    for fn in sorted(os.listdir(path))]
    mansfield_fns = get_fns_in_path('data/mansfield-park-chunks/')
    said_fns = get_fns_in_path('data/culture-and-imperialism-chunks/')
    stein_fns = get_fns_in_path('data/three-lives-chunks/')
    hemingway_fns = get_fns_in_path('data/sun-also-rises-chunks/')
    
    mansfield_and = get_word_rates("and", mansfield_fns)
    said_and = get_word_rates("and", said_fns)
    stein_and = get_word_rates("and", stein_fns)
    hemingway_and = get_word_rates("and", hemingway_fns)
    
    # remove the last chunks as they are much less than 1,000 words
    del mansfield_and[-1]
    del said_and[-1]
    del stein_and[-1]
    del hemingway_and[-1]


.. ipython:: python

   data = [mansfield_and, said_and, stein_and, hemingway_and]
   titles = ["Austen", "Said", "Stein", "Hemingway"]
   
   fig = plt.figure(figsize=(13,8), dpi=100)
   for i, rates, title in zip(range(4), data, titles):
       ax = fig.add_subplot(2, 2, i + 1)
       ax.hist(rates, bins=range(10,90,3), normed=True)
       ax.set_xlim((10.0, 90.0))
       ax.set_title(title)



These are distinctive distributions. Said and Austen are somewhat difficult to
tell apart but they are easy to distinguish from the two modernists. The
distribution of rates in *The Sun Also Rises* is the most unusual, with one
section having a rate above 85 per 1,000 words.[#fnsun]_

.. [#] Here are several sentences from the section of *The Sun Also Rises* that
    clocks in at a rate of 85 "and"s per 1,000 words: There was a little stream
    and a bridge, and Spanish carabineers, with patent-leather Bonaparte hats,
    and short guns on their backs, on one side, and on the other fat Frenchmen
    in kepis and mustaches. They only opened one bag and took the passports in
    and looked at them. ... The road went along the summit of the Col and then
    dropped down, and the driver had to honk, and slow up, and turn out to avoid
    running into two donkeys that were sleeping in the road. We came down out of
    the mountains and through an oak forest, and there were white cattle grazing
    in the forest. Down below there were grassy plains and clear streams, and
    then we crossed a stream and went through a gloomy little village, and
    started to climb again."

For comparison, consider the rates of "und" for the following four texts written
in German: Rosa Luxemborg's *Massenstreik, Partei, und Gewerkshaften*, Teresa
Huber's *Die Familie Seldorf*, Goethe's *Die Leiden des jungen Werters*, and
Robert Musil's *Die Verwirrungen des Zöglings Törleß*.  The distributions of
rates of "und" for these writers are not as distinctive as those above. Musil,
however, does fall out; he tends to use "und" less frequently that the other
writers.


.. ipython:: python

    massenstreik_fns = get_fns_in_path('data/massenstreik-partei-gewerkschaften-chunks/')
    familie_fns = get_fns_in_path('data/familie-seldorf-chunks/')
    leiden_fns = get_fns_in_path('data/leiden-des-jungen-werthers-chunks/')
    verwirrung_fns = get_fns_in_path('data/verwirrung-chunks/')
    
    massenstreik_und = get_word_rates("und", massenstreik_fns)
    familie_und = get_word_rates("und", familie_fns)
    leiden_und = get_word_rates("und", leiden_fns)
    verwirrung_und = get_word_rates("und", verwirrung_fns)

    # remove the last chunks as they are much less than 1,000 words
    del massenstreik_und[-1]
    del familie_und[-1]
    leiden_und = np.concatenate([leiden_und[:18], leiden_und[19:-1]])
    del verwirrung_und[-1]

    data = [massenstreik_und, familie_und, leiden_und, verwirrung_und]
    titles = ["Luxemburg", "Huber", "Goethe", "Musil"]
    
    fig = plt.figure(figsize=(13, 8), dpi=100)
    for i, rates, title in zip(range(4), data, titles):
        ax = fig.add_subplot(2, 2, i + 1)
        ax.hist(rates, bins=range(10,90,3), normed=True)
        ax.set_xlim((10.0, 90.0))
        ax.set_title(title)

To connect these figures to something more tangible in our experience as
readers, consider the association between the frequency of "and" and the number
of sentences in a section of text. The length of sentences is something,
I think, that is comparatively easy for readers to notice. It seems a reasonable
bet that the more writers use "and" the longer their sentences will tend to be.
The following figure is a scatterplot of the number of sentences and the number
of times "and" occurs per 1,000 words in *Mansfield Park*.


.. ipython:: python

    mansfield_and = get_word_rates("and", mansfield_fns)
    mansfield_dtm, _ = get_dtm_and_vocab(mansfield_fns)  # dtm == document-term-matrix
    mansfield_chunk_word_counts = np.sum(mansfield_dtm, axis=1).ravel()
    
    # calculate number of sentences per 1,000 words
    window = 1000
    mansfield_num_sent = window * \
    np.array(get_num_sentences(mansfield_fns)) / mansfield_chunk_word_counts
    
    plt.figure(figsize=(13, 8), dpi=100)
    plt.scatter(mansfield_num_sent, mansfield_and)
    plt.xlabel("number of sentences per 1,000 words")
    plt.ylabel("'and' per 1,000 words");


.. ipython:: python

    # calculate correlation
    np.corrcoef(mansfield_num_sent, mansfield_and)[0, 1]


This is what we anticipated. The chapters in which "and" occurs more often have
fewer sentences. Note that there is barely any correlation between the rate of
"she" (another frequent word) and the number of sentences.


.. ipython:: python

    mansfield_she = get_word_rates("she", mansfield_fns)
    plt.figure(figsize=(13, 8), dpi=100)
    plt.scatter(mansfield_num_sent, mansfield_she)
    plt.xlabel("number of sentences per 1,000 words")
    plt.ylabel("'she' per 1,000 words");


.. ipython:: python

    # calculate correlation
    np.corrcoef(mansfield_num_sent, mansfield_and)[0, 1]

(A priori, there probably should be some relationship between the rate of many
high frequency words and the number of sentences, as many of the high frequency
words do not occur alone---e.g., "of", "by", "to"---and weakly indicate the
verbosity of the chapter, and therefore negative correlated with the number of
sentences.)

So it turns out that the rate of "and" in a text, something that I suspect
initially strikes many as inconsequential, turns out to be informative.


Sampling Model and Prior
------------------------

.. note:: Kadane :cite:`kadane_principles_2011` covers the normal model in `chapter
   8 of his textbook <http://uncertainty.stat.cmu.edu/>`_. MacKay
   :cite:`mackay_information_2003` discusses the model in `chapter 24
   <http://www.inference.phy.cam.ac.uk/mackay/itprnn/ps/319.323.pdf>`_. Both
   textbooks are accessible online. Hoff :cite:`hoff_first_2009` and Lee
   :cite:`lee_bayesian_2004` are excellent but unfortunately not freely available.

Bayesian inference asks us to specify two things: our prior beliefs (or just
"prior") about the quantity in question and a "sampling model" (or likelihood)
for data we observe.  It is the latter that allows us to update our prior
beliefs and characterize a "posterior" set of beliefs which can be used for
prediction.

Let's start with specifying our prior beliefs. Even before we see the frequency
of the word "and" in chapters one through five in  *Pride and Prejudice*, we can
make an educated guess. First, based on our knowledge of the English language we
can say that it would be shocking to see a rate of one "and" per ten words (100
per 1000 words). Having the information in *Mansfield Park* allows an even
better guess. Based on the rates of "and" found there, we would be wise to
suspect that the rate of "and" per 1,000 words in *Pride and Prejudice* is
likely to be more than 20 and less than 50. We know this based on the observed
frequencies in *Mansfield Park*:

.. ipython:: python

    r, l = (10, 70)
    plt.figure(figsize=(13, 8), dpi=100)
    plt.hist(mansfield_and, 10, normed=True)
    plt.xlim(r, l);

Frequent words such as "and" often turn out to be well approximated by a normal
distribution. A normal distribution is, however, *not appropriate for the vast
majority of words* that you will encounter (see Church and Gale
:cite:`church_poisson_1995`). As we will see, it will be easier to work with
this distribution as a description of our knowledge about the rates of "and" in
*Mansfield Park*. The unwiedly alternative is to keep track of all the
observations.


.. ipython:: python

    x = mansfield_and
    mu, sigma = np.mean(x), np.std(x)
    
    # calculate 99% prior credible interval
    scipy.stats.norm(mu, sigma).ppf([0.005, 0.995])
    
    # overlay normal distribution
    xs = np.arange(r, l, 0.1)
    plt.figure(figsize=(13, 8), dpi=100)
    plt.hist(x, 10, normed=True)
    plt.plot(xs, scipy.stats.norm(mu, sigma).pdf(xs), linewidth=2)
    plt.xlim(r, l);


The normal distribution shown above may be expressed by just two numbers. That is, it is uniquely characterized by its mean and
variance (its first two central moments). For reasons that will become clear in
a moment, let's denote these as $\mu_0$ and $\sigma^2$ respectively.
The distribution can be written as

$$\label{eqn:normal} f(z_i) = (2 \pi \sigma^2)^{-1/2} \exp(-\frac{1}{2\sigma^2}(z_i - \mu_0)^2)$$

where $z_i$ is the rate of "and" we anticipate observing were a chapter of
*Mansfield Park* selected at random and the rate of "and" reported.

The normal distribution is also usefully parameterized by its precision,
$\phi = 1/\sigma^2$. Using the precision parameterization will make
subsequent calculations much easier. The distribution using the precision
parameterization reads

$$f(z_i) = (2\pi)^{-1/2} \phi^{1/2} \exp(-\frac{\phi}{2}(z_i - \mu_0)^2)$$

Since $f(z_i)$ is a probability distribution, it must satisfy $\int
f(z_i) dz_i = 1$. That is, it must assign probability 1 to observing some value of
$z_i$. This means that we can ignore terms in the equation above that do not
depend on $z_i$ and be secure in the knowledge that we can recover them by
integrating the unnormalized expression. This means that the equation above
may also be expressed as

$$f(z_i) \propto \exp(-\frac{\phi}{2}(z_i - \mu_0)^2)$$

Now we can use this normal distribution based on *Mansfield Park* to inform our
prior beliefs about the likely rate of "and" in *Pride and Prejudice*. While it
would be unreasonable to believe that the rate of "and" will be precisely the
same, it does seem like a good bet that it will resemble the rates
observed in *Mansfield Park*. That is, it would be shocking to observe that
Austen used "and" once or twice per 1000 words or that she used it more than
seventy times per 1,000 words. To express this uncertainty, we can specify that
the rate of "and" in *Pride and Prejudice* will follow a normal distribution but
we will express only a vague sense about the center (or mean) of that
distribution, using the prior information gathered from *Mansfield Park*. We
will use :math:`\mu` to denote the mean of the distribution of the "and" rates
in Austen's most famous novel and we will characterize our beliefs about
:math:`\mu` with a normal distribution. In symbols this reads

$$x_i \sim N(\mu, \sigma^2)$$

$$\mu \sim N(\mu_0, \tau_0^2)$$

where $x_i$ denotes the rate of "and" observed in a chapter of *Pride and
Prejudice*. $x_i \sim N(\mu, \sigma^2)$ is shorthand for the normal
distribution shown in equation \ref{eqn:normal}. We will make things considerably
simpler by assuming that we know $\sigma$. We will take its value directly
from the observed standard deviation of the rates of "and" in *Mansfield Park*.
We will also use those rates for $\mu_0$ and, indirectly, for
$\tau_0$. Since we do not know that much about the relationship between
rates across novels, we will pick a value of $\tau_0$ that is quite a bit
larger than $\sigma$ in order to express this uncertainty. The rationale behind
this prior specification is that the spread of rates in the *chapters* of
*Mansfield Park* seems a reasonable rough guide for the spread of average rates
in other *novels* by the same author.

To visualize our prior uncertainty about the rate of "and" in *Pride and
Prejudice* it will be helpful to simulate from or otherwise examine the *prior
predictive distribution* of rates by first simulating a value for $\mu$
from the prior distribution and then plotting the implied distribution for the
rate of "and". The following figure shows a random sample of these
distributions.


.. ipython:: python

    mu0 = np.mean(mansfield_and)
    tau0 = np.sqrt(2) * np.std(mansfield_and)
    r, l = (0, 90)
    xs = np.arange(r, l, 0.01)
    plt.figure(figsize=(13, 8), dpi=100)
    plt.plot(xs, scipy.stats.norm(mu0, np.sqrt(sigma**2 + tau0**2)).pdf(xs), linewidth=2, alpha=0.5)
    plt.xlim(r, l);



The distribution roughly expresses the belief that the rate of "and"
lies with high probability somewhere between 0 and 80. (Note that this
distribution assigns very low---but positive---probability to negative
rates. This is regrettable and sloppy but it will not pose a problem for
this pedagogical example.)

Now we need to update our beliefs based on seeing the first five
1000-word sections of *Pride and Prejudice*. Designate these first five
rates as $\mathbf{x} = (x_1, \ldots, x_5)$. If we knew $\mu$ and
$\sigma^2$ we could again use a normal distribution to describe the
probability of these observations and write

$$p(\mathbf{x}) = (2\pi\sigma^2)^{-n/2} \exp(-\frac{1}{2\sigma^2}(\sum_{i=1}^n (x_i - \mu)^2)$$

where $n$ equals 5.

Bayes' Rule tells us how to update our beliefs about $\mu$ after
observing $\mathbf{x}$:

$$p(\mu|\mathbf{x}) = \frac{p(\mathbf{x}|\mu) p(\mu)}{p(\mathbf{x})}$$

Since this is a probability distribution, we can focus only on terms
that involve $\mu$ and write

$$p(\mu|\mathbf{x}) \propto p(\mathbf{x}|\mu) p(\mu)$$

Plugging in our distributions with precision parameterizations ($\phi
= 1/\sigma^2$ and $\psi_0 = 1/\tau_0^2$) we arrive at the following
unnormalized posterior distribution for $\mu$

$$p(\mu|\mathbf{x}) \propto \exp(-\frac{\phi}{2}(\sum_{i=1}^n (x_i - \mu)^2)) \exp(-\frac{\psi_0}{2} (\mu - \mu_0)^2)$$

If we focus only on the exponent and ignore the $-\frac{1}{2}$ factor we
may expand the quadratic expressions and [complete the
square](http://en.wikipedia.org/wiki/Completing_the_square).

$$\phi(\sum_{i=1}^n x_i^2 - 2\mu\sum_{i=1}^n x_i + \mu^2) + \psi_0(\mu^2 - 2\mu\mu_0 + \mu^2)\\
= a\mu^2 - 2\mu b + c$$

where

$$a = n\phi + \psi_0$$
$$b = \phi \sum_{i=1}^n x_i + \psi_0 \mu_0$$
$$c = c(\mathbf{x}, \mu_0, \phi, \psi_0)$$

$$p(\mu|\mathbf{x}) \propto \exp(-\frac{1}{2}(a\mu^2 - 2\mu b)$$
$$\propto \exp(-\frac{a}{2}(\mu^2 - 2\mu \frac{b}{a})$$
$$\propto \exp(-\frac{a}{2}(\mu^2 - 2\mu \frac{b}{a} + \frac{b^2}{a^2})$$
$$\propto \exp(-\frac{a}{2}(\mu - \frac{b}{a})^2$$

We should recognize the last line as a normal distribution. (The
penultimate step, adding a constant that doesn't depend on $\mu$, is
justified by the fact that adding a constant value in the exponent is
equivalent to multiplying the expression by a constant.) This normal
distribution has the following mean and precision parameters

$$\mu_n = \frac{b}{a} = \frac{n \phi \bar{x} + \psi_0 \mu_0}{n\phi + \psi_0}$$
$$\psi_n = n\phi + \psi_0$$

where $\bar{x} = \frac{1}{n}\sum_{i=1}^n x_i$. We need only specify
$\mu_0$ and $\tau_0$ in order to use this distribution to predict the
rate of "and" in the sixth chapter as we already have the rates in
chapters one through five ($\mathbf{x}$). (The value for $\phi$ we
assume is fixed; all novels by Jane Austen under this assumption have
the same variability in rates.) And we already have appropriate values
for $\mu_0$ and $\tau_0$ that we arrived at by considering what our
beliefs about the rates in other novels by Jane Austen should be given
what we saw in *Mansfield Park*. Using those values, we arrive at the
following *posterior predictive distribution* for the rate of "and" in
chapter six.

.. ipython:: python

    # load the data from pride and prejudice
    pp_fns = get_fns_in_path('data/pride-and-prejudice-chapters/')
    pp_and = get_word_rates("and", pp_fns, window=1000)
    pp_dtm, _ = get_dtm_and_vocab(pp_fns)
    pp_chp_word_counts = np.sum(pp_dtm, axis=1).ravel()
    pp_num_sent = 1000 * np.array(get_num_sentences(pp_fns)) / pp_chp_word_counts
    
    # get rates for first five chapters
    x = pp_and[0:5] x
    np.mean(x)
    n = len(x)
    
    # calculate posterior parameters
    phi = 1/sigma**2 ; psi0 = 1/tau0**2
    psin = n * phi + psi0
    mun = (phi * np.sum(x) + psi0 * mu0)/psin
    
    # plot the distribution
    plt.figure(figsize=(13, 8), dpi=100)
    plt.plot(xs, scipy.stats.norm(mun, np.sqrt(1/phi + 1/psin)).pdf(xs), linewidth=2);

Now we can see how well our prediction did by peeking at the actual rate in
chapter six. The rate of "and" in chapter six is the red vertical line. As can
be seen, the observed value lands nicely within the range of predicted values.

.. ipython:: python

    plt.figure(figsize=(13, 8), dpi=100)
    plt.plot(xs, scipy.stats.norm(mun, np.sqrt(1/phi + 1/psin)).pdf(xs), color='b', linewidth=2)
    plt.axvline(x=x_chp6, linewidth=2, color='r');



.. rubric:: Footnotes

[#fnsun]: Here are several sentences from the section of *The Sun Also Rises* that
    clocks in at a rate of 85 "and"s per 1,000 words: There was a little stream
    and a bridge, and Spanish carabineers, with patent-leather Bonaparte hats,
    and short guns on their backs, on one side, and on the other fat Frenchmen
    in kepis and mustaches. They only opened one bag and took the passports in
    and looked at them. ... The road went along the summit of the Col and then
    dropped down, and the driver had to honk, and slow up, and turn out to avoid
    running into two donkeys that were sleeping in the road. We came down out of
    the mountains and through an oak forest, and there were white cattle grazing
    in the forest. Down below there were grassy plains and clear streams, and
    then we crossed a stream and went through a gloomy little village, and
    started to climb again."
