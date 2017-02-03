Classification
==============

All 3 types of conformal prediction are implemented for classification
(transductive, inductive and cross), with several different nonconformity
measures to choose from.

We will show how to train and use a conformal predictive model in the following
simple, but fully functional example.

Let's load the iris data set and try to make a prediction for the last
instance using the rest for learning.

    >>> import Orange
    >>> import orangecontrib.conformal as cp
    >>> iris = Orange.data.Table('iris')
    >>> train = iris[:-1]
    >>> test_instance = iris[-1]

We will use a LogisticRegressionLearner from Orange and the inverse probability
nonconformity score in a 5-fold cross conformal prediction classifier.

    >>> lr = Orange.classification.LogisticRegressionLearner()
    >>> ip = cp.nonconformity.InverseProbability(lr)
    >>> ccp = cp.classification.CrossClassifier(ip, 5, train)

Predicting the 90% and 99% prediction regions gives the following results.

    >>> print('Actual class:', test_instance.get_class())
    Actual class: Iris-virginica
    >>> print(ccp(test_instance.x, 0.1))
    ['Iris-virginica']
    >>> print(ccp(test_instance.x, 0.01))
    ['Iris-versicolor', 'Iris-virginica']

We can see that in the first case only the correct class of 'Iris-virginica'
was predicted.  In the second case, with a much lower tolerance for errors, the
model claims only that the instance belongs to one of two possible classes
'Iris-versicolor' or 'Iris-virginica', but not the third 'Iris-setosa'.

