Regression
==========

For regression inductive and cross conformal prediction are implemented along
with several nonconformity measures.

Similarly to the classification example, let's combine some standard components
to show how to train and use a conformal prediction model for regression.

Let's load the housing data set and try to make a prediction for the last
instance using the rest for learning.

    >>> import Orange
    >>> import orangecontrib.conformal as cp
    >>> housing = Orange.data.Table('housing')
    >>> train = housing[:-1]
    >>> test_instance = housing[-1]

We will use a LinearRegressionLearner from Orange and the absolute error
nonconformity score in a 5-fold cross conformal regressor.

    >>> lr = Orange.regression.LinearRegressionLearner()
    >>> abs_err = cp.nonconformity.AbsError(lr)
    >>> ccr = cp.regression.CrossRegressor(abs_err, 5, train)

Predicting the 90% and 99% prediction regions gives the following results.

    >>> print('Actual target value:', test_instance.get_class())
    Actual target value: 11.900
    >>> print(ccr(test_instance.x, 0.1))
    (13.708550425853684, 31.417230194137165)
    >>> print(ccr(test_instance.x, 0.01))
    (-0.98542733224618217, 46.111207952237031)

We can see that in the first case the predicted interval was smaller, but did
not contain the correct value (this should not happend more than 10% of the
time). In the second case, with a much lower tolerance for errors, the model
predicted a larger interval, which did contain the correct value.

