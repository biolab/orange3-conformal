Orange3 Conformal Prediction
============================

Conformal Prediction is an add-on for
`Orange3 <http://orange.biolab.si>`__ data mining software package. It
provides an extensive toolset for conformal prediction.

Installation
------------

To install the add-on, run

::

    python setup.py install

To register this add-on with Orange, but keep the code in the
development directory (do not copy it to Python's site-packages
directory), run

::

    python setup.py develop

Usage
-----

The library in the add-on can be used in Python scripts. The add-on does
not provide any GUI widgets.

The example below evaluates an inductive conformal predictor at 0.1
significance level on the Iris dataset (spliting it into a training and
testing set in ratio 2:1). The nonconformity scores used by the
conformal predictor are based on the probabilities returned by a Naive
Bayes classifier.

::

    import Orange
    import orangecontrib.conformal as cp

    tab = Orange.data.Table('iris')
    nc = cp.nonconformity.InverseProbability(Orange.classification.NaiveBayesLearner())
    ic = cp.classification.InductiveClassifier(nc)
    r = cp.evaluation.run(ic, 0.1, cp.evaluation.RandomSampler(tab, 2, 1))
    print(r.accuracy())

Documentation
-------------

Please see doc/Orange-ConformalPrediction.pdf. Documentation in other formats can also be built using Sphinx from the
doc directory.

Online documentation is available at `<https://orange3-conformal.readthedocs.io>`_.
