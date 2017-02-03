Evaluation
==========

The evaluation module provides many useful classes and functions for evaluating the performance and validity of conformal predictions.
The main two classes, which represent the results of a conformal classifier and regressor, are :py:class:`conformal.evaluation.ResultsClass` and :py:class:`conformal.evaluation.ResultsRegr`.

For ease of use, the evaluation results can be obtained using utility functions that evaluate the selected conformal predictor on data defined by the provided sampler (:py:func:`conformal.evaluation.run`) or explicitly provided by the user (:py:func:`conformal.evaluation.run_train_test`).

As an example, let's take a look at how to quickly evaluate a conformal classifier on a test data set and compute some of the performance metrics:

    >>> import Orange
    >>> import orangecontrib.conformal as cp
    >>> iris = Orange.data.Table('iris')
    >>> train, test = iris[::2], iris[1::2]
    >>> lr = Orange.classification.LogisticRegressionLearner()
    >>> ip = cp.nonconformity.InverseProbability(lr)
    >>> ccp = cp.classification.CrossClassifier(ip, 5)
    >>> res = cp.evaluation.run_train_test(ccp, 0.1, train, test)

The results are an instance of :py:class:`conformal.evaluation.ResultsClass` mentioned above, and can be used to compute the accuracy of predictions (fraction of predictions including the actual class). For a *valid* predictor it needs to hold that the error (1 - accuracy) is lower or equal to the specified significance level.
In addition to *validity*, we are often interested in the *efficiency* of a predictor. For classification, this is often measured with the fraction of cases with a single predicted class (:py:func:`conformal.evaluation.ResultsClass.singleton_criterion`). For regression, one might measure the widths of predicted intervals and e.g. report the average value (:py:func:`conformal.evaluation.ResultsRegr.mean_range`).

    >>> print('Accuracy:', res.accuracy())
    Accuracy: 0.946666666667
    >>> print('Singletons:', res.singleton_criterion())
    Singletons: 0.96

Another very useful visual validation approach is to plot the dependency of the actual measured error rate at different levels of the specified significance so the user can quickly see that the error is indeed controlled by the parameter.
There is a function in the evaluation module that prepares a calibration plot for the specified predictor and data:

    >>> cp.evaluation.calibration_plot(ccp, iris, fname='calibration.png')
