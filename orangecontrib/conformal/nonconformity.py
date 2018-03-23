"""Nonconformity module contains nonconformity scores for classification and regression.

Structure:

- ClassNC (classification scores)
    - ClassModelNC (model based)
        :py:class:`InverseProbability`, :py:class:`ProbabilityMargin`, :py:class:`SVMDistance`,
        :py:class:`LOOClassNC`
    - ClassNearestNeighboursNC (nearest neighbours based)
        :py:class:`KNNDistance`, :py:class:`KNNFraction`
- RegrNC (regression scores)
    - RegrModelNC (model based)
        :py:class:`AbsError`, :py:class:`AbsErrorRF` :py:class:`AbsErrorNormalized`, :py:class:`LOORegrNC`,
        :py:class:`ErrorModelNC`
    - RegrNearestNeighboursNC (nearest neighbours based)
        :py:class:`AbsErrorKNN`, :py:class:`AvgErrorKNN`
"""

import math
from copy import deepcopy

import numpy as np

from Orange.base import Model
from Orange.data import Table, Instance
from Orange.distance import DistanceModel, Euclidean

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC, LinearSVC, NuSVC


# ----- CLASSIFICATION ----- #

class ClassNC:
    """Base class for classification nonconformity scores.

    Extending classes should implement :py:func:`fit` and :py:func:`nonconformity` methods.
    """

    def __str__(self):
        return self.__class__.__name__

    def fit(self, data):
        """Process the data used for later calculation of nonconformities.

        Args:
            data (Table): Data set.
        """
        raise NotImplementedError

    def nonconformity(self, instance):
        """Compute the nonconformity score of the given `instance`."""
        raise NotImplementedError


# --- model-based --- #

class ClassModelNC(ClassNC):
    """Base class for classification nonconformity scores that are based on an underlying classifier.

    Extending classes should implement :py:func:`ClassNC.nonconformity` method.

    Attributes:
        learner: Untrained underlying classifier.
        model: Trained underlying classifier.
    """

    def __init__(self, classifier):
        """Store the provided classifier as :py:attr:`learner`."""
        self.learner = classifier
        self.model = None

    def __str__(self):
        return str(self.learner)

    def fit(self, data):
        """Train the underlying classifier on provided data and store the trained model."""
        self.model = self.learner(data)


class InverseProbability(ClassModelNC):
    """Inverse probability nonconformity score returns :math:`1 - p`, where :math:`p` is the probability
    assigned to the actual class by the underlying classification model (:py:attr:`ClassModelNC.model`).

    Examples:
        >>> train, test = next(LOOSampler(Table('iris')))
        >>> tp = TransductiveClassifier(InverseProbability(NaiveBayesLearner()), train)
        >>> print(tp(test[0].x, 0.1))
    """

    def __str__(self):
        return "InverseProbability ({})".format(super().__str__())

    def nonconformity(self, instance):
        predictions = self.model(instance, ret=Model.Probs)[0]
        return 1 - predictions[int(instance.get_class())]


class ProbabilityMargin(ClassModelNC):
    """Probability margin nonconformity score measures the difference :math:`d_p` between the predicted probability
    of the actual class and the largest probability corresponding to some other class. To put the values on scale
    from 0 to 1, the nonconformity function returns :math:`(1 - d_p) / 2`.

    Examples:
        >>> train, test = next(LOOSampler(Table('iris')))
        >>> tp = TransductiveClassifier(ProbabilityMargin(LogisticRegressionLearner()), train)
        >>> print(tp(test[0].x, 0.1))
    """

    def __str__(self):
        return "ProbabilityMargin ({})".format(super().__str__())

    def nonconformity(self, instance):
        predictions = self.model(instance, ret=Model.Probs)[0]
        y = int(instance.get_class())
        py = predictions[y]
        pz = max(p for z, p in enumerate(predictions) if z != y)
        return (1.0 - (py - pz)) / 2


class SVMDistance(ClassNC):
    """SVMDistance nonconformity score measures the distance from the SVM's decision boundary. The score depends
    on the distance and the side of the decision boundary that the example lies on.
    Examples that lie on the correct side of the decision boundary and would therefore result in a
    correct prediction using the SVM classifier have a nonconformity score less than 1, while the incorrectly
    predicted examples have a score more than 1.

    .. math::
        \\mathit{nc} =
        \\begin{cases}
        \\frac{1}{1+d} & \\text{correct}\\\\
        1+d &\\text{incorrect}
        \\end{cases}

    The provided SVM classifier must be a sklearn's SVM classifier (SVC, LinearSVC, NuSVC)
    providing the decision_function() which computes the distance to decision boundary. This nonconformity works
    only for binary classification problems.

    Examples:
        >>> from sklearn.svm import SVC
        >>> train, test = next(LOOSampler(Table('titanic')))
        >>> train, calibrate = next(RandomSampler(train, 2, 1))
        >>> icp = InductiveClassifier(SVMDistance(SVC()), train, calibrate)
        >>> print(icp(test[0].x, 0.1))
    """

    def __init__(self, classifier):
        assert isinstance(classifier, (SVC, LinearSVC, NuSVC)), \
            "Classifier must be a sklearn's SVM classifier (SVC, LinearSVC, NuSVC)."
        self.clf = classifier
        self.model = None

    def __str__(self):
        return "SVMDistance ({})".format(self.clf.__class__.__name__)

    def fit(self, data):
        assert len(data.domain.class_var.values) == 2, \
            "SVMDistance supports only binary classification problems."
        self.clf.fit(data.X, data.Y)

    def nonconformity(self, instance):
        y = self.clf.predict(np.atleast_2d(instance.x))[0]
        d = self.clf.decision_function(np.atleast_2d(instance.x))[0]
        if instance.y == y:
            return 1/(1+abs(d))
        else:
            return 1+abs(d)


# --- knn-based --- #

class NearestNeighbours:
    """Base class for nonconformity measures based on nearest neighbours.

    Attributes:
        distance: Distance measure.
        k (int):  Number of nearest neighbours.
    """

    def __init__(self, distance=Euclidean(), k=1):
        """Store the distance measure and the number of neighbours."""
        self.distance = distance
        self.k = k

    def fit(self, data):
        """Store the data for finding nearest neighbours."""
        self.data = data
        # fit the distance measure if uninitialized
        if not isinstance(self.distance, DistanceModel):
            self.distance = self.distance.fit(data)

    def neighbours(self, instance):
        """Compute distances to all other data instances using the distance measure (:py:attr:`distance`).

        Excludes data instances that are equal to the provided `instance`.

        Returns:
            List of pairs (distance, instance) in increasing order of distances.
        """
        other = self.data[np.array([not np.array_equal(row.x, instance.x) for row in self.data])]
        dist = self.distance(instance, other)[0]
        return sorted([(d, row) for d, row in zip(dist, other)], key=lambda x: x[0])


class ClassNearestNeighboursNC(NearestNeighbours, ClassNC):
    """Base class for nearest neighbrours based classification nonconformity scores."""
    pass


class KNNDistance(ClassNearestNeighboursNC):
    """Computes the sum of distances to k nearest neighbours of the same class as the given instance and
    the sum of distances to k nearest neighbours of other classes. Returns their ratio.

    Examples:
        >>> from Orange.distance import Euclidean
        >>> train, test = next(LOOSampler(Table('iris')))
        >>> cp = CrossClassifier(KNNDistance(Euclidean(), 10), 2, train)
        >>> print(cp(test[0].x, 0.1))
    """

    def __str__(self):
        return "KNNDistance (k={})".format(self.k)

    def nonconformity(self, instance):
        dist = self.neighbours(instance)
        same = [d for d, row in dist if row.get_class() == instance.get_class()]
        diff = [d for d, row in dist if row.get_class() != instance.get_class()]
        same, diff = same[:self.k], diff[:self.k]
        return sum(same) / sum(diff)


class KNNFraction(ClassNearestNeighboursNC):
    """Computes the k nearest neighbours of the given instance. Returns the fraction of instances of the
    same class as the given instance within its k nearest neighbours.

    Weighted version uses weights :math:`1/d_i` based on distances instead of simply counting the instances.
    Non-weighted version is equivalent to using a value 1 for all weights.

    Examples:
        >>> train, test = next(LOOSampler(Table('iris')))
        >>> cp = CrossClassifier(KNNFraction(Euclidean(), 10, weighted=True), 2, train)
        >>> print(cp(test[0].x, 0.1))
    """

    def __init__(self, distance=Euclidean(), k=1, weighted=False):
        super().__init__(distance, k)
        self.weighted = weighted

    def __str__(self):
        return "KNNFraction (k={})".format(self.k)

    def nonconformity(self, instance):
        dist = self.neighbours(instance)[:self.k]
        if self.weighted:
            diff = [1/d for d, row in dist if row.get_class() != instance.get_class()]
            return sum(diff) / sum(1/d for d, _ in dist)
        else:
            diff = [1 for d, row in dist if row.get_class() != instance.get_class()]
            return sum(diff) / len(dist)


# --- model-knn --- #

class LOOClassNC(NearestNeighbours, ClassNC):
    """
    .. math::
        \\mathit{nc} = \\mathit{error} + (1 - p)
        \\quad \\text{or} \\quad
        \\mathit{nc} = \\frac{1 - p}{\\mathit{error}}

    :math:`p` ... probability of actual class predicted from :math:`N_k(z^*)` - k nearest neighbours
    of the instance :math:`z^*`

    The first nonconformity score is used when the parameter :py:attr:`relative` is set to *False*
    and the second one when it is set to *True*.

    .. math::
        \\mathit{error} = \\frac {\\sum_{z_i \\in N_k(z^*)} w_i (1 - p_i)} {\\sum_{z_i \\in N_k(z^*)} w_i},
        \\quad w_i = \\frac{1}{d(x^*, x_i)}

    :math:`p_i` ... probability of actual class predicted from :math:`N_k(z') \\setminus z_i` or
    :math:`N_k(z') \\setminus z_i \\cup z^*` if the parameter :py:attr:`include` is set to *True*. :math:`z'` is
    :math:`z^*` if the :py:attr:`neighbourhood` parameter is '*fixed*' and :math:`z_i` if it's '*variable*'.
    """

    def __init__(self, classifier, distance=Euclidean(), k=10, relative=True, include=False, neighbourhood='fixed'):
        """Initialize the parameters."""
        super().__init__(distance, k)
        self.classifier = classifier
        self.relative = relative
        self.include = include
        assert neighbourhood in ['fixed', 'variable']
        self.neighbourhood = neighbourhood

    def fit(self, data):
        """Store the data for finding nearest neighbours and initialize cache."""
        super().fit(data)
        self.cache = {}

    def get_neighbourhood(self, inst):
        """Construct an Orange data Table consisting of instance's k nearest neighbours.
        Cache the results for later calls with the same instance."""
        t = tuple(inst.x)
        if t not in self.cache:
            neigh = self.neighbours(inst)[:self.k]
            neigh_d, neigh_list = zip(*neigh)
            neigh_tab = Table(neigh_list[0].domain, list(neigh_list))
            self.cache[t] = neigh_tab
        return self.cache[t]

    def error(self, inst, neighbours):
        """Compute the average weighted probability prediction error for predicting the actual class
        of each neighbour from the other ones. Include the new example among the neighbours
        if the parameter :py:attr:`include` is True."""
        sc = 0
        ws = []
        for i in range(len(neighbours)):
            neigh = Instance(neighbours.domain, neighbours[i])
            w = 1/self.distance(inst, neigh)
            ws.append(w)

            if self.neighbourhood == 'fixed':
                neighbours_i = neighbours[np.arange(len(neighbours)) != i]
            else:
                neighbours_i = self.get_neighbourhood(neigh)
            if self.include:
                neighbours_i = neighbours_i.copy()
                neighbours_i.append(inst)

            model = self.classifier(neighbours_i)
            sc += w * (1-model(neigh, ret=Model.Probs)[0][int(neigh.get_class())])
        return float(sc / sum(ws))

    def nonconformity(self, inst):
        neighbours = self.get_neighbourhood(inst)
        model = self.classifier(neighbours)
        error = self.error(inst, neighbours)
        yp = model(inst, ret=Model.Probs)[0][int(inst.get_class())]
        if self.relative:
            return (1-yp) / (1e-6 + error)
        else:
            return error + (1-yp)


# ----- REGRESSION ----- #

class RegrNC:
    """Base class for regression nonconformity scores.

    Extending classes should implement :py:func:`fit`, :py:func:`nonconformity` and :py:func:`predict` methods.
    """

    def __str__(self):
        return self.__class__.__name__

    def fit(self, data):
        """Process the data used for later calculation of nonconformities.

        Args:
            data (Table): Data set.
        """
        raise NotImplementedError

    def nonconformity(self, instance):
        """Compute the nonconformity score of the given `instance`."""
        raise NotImplementedError

    def predict(self, inst, nc):
        """Compute the inverse of the nonconformity score. Determine a range of values for which the
        nonconformity of the given `instance` does not exceed `nc`."""
        raise NotImplementedError


# --- model-based --- #

class RegrModelNC(RegrNC):
    """Base class for regression nonconformity scores that are based on an underlying classifier.

    Extending classes should implement :py:func:`RegrNC.nonconformity` and :py:func:`RegrNC.predict` methods.

    Attributes:
        learner: Untrained underlying classifier.
        model: Trained underlying classifier.
    """

    def __init__(self, classifier):
        """Store the provided classifier as :py:attr:`learner`."""
        self.learner = classifier
        self.model = None

    def __str__(self):
        return str(self.learner)

    def fit(self, data):
        """Train the underlying classifier on provided data and store the trained model."""
        self.model = self.learner(data)


class AbsError(RegrModelNC):
    """Absolute error nonconformity score returns the absolute difference between
    the predicted value (:math:`\\hat{y}`) by the underlying :py:attr:`RegrModelNC.model`
    and the actual value (:math:`y^{*}`).

    .. math::
        \\mathit{nc} = |\\hat{y}-y^{*}|

    Examples:
        >>> train, test = next(LOOSampler(Table('housing')))
        >>> cr = CrossRegressor(AbsError(LinearRegressionLearner()), 2, train)
        >>> print(cr(test[0].x, 0.1))
    """

    def __str__(self):
        return "AbsError ({})".format(super().__str__())

    def nonconformity(self, instance):
        return abs(instance.get_class() - float(self.model(instance)))

    def predict(self, inst, nc):
        y = float(self.model(inst))
        return y-nc, y+nc


class AbsErrorRF(RegrModelNC):
    """AbsErrorRF is based on an underlying regressor and a random forest. The prediction errors of regressor
    are used as nonconformity scores and are normalized by the standard deviation of predictions coming from
    individual trees in the forest.

    .. math::
        \\mathit{nc} = \\frac{|\\hat{y}-y^{*}|}{\sigma_\\mathit{RF} + \\beta}

    Examples:
        >>> from sklearn.ensemble import RandomForestRegressor
        >>> icr = InductiveRegressor(AbsErrorRF(RandomForestRegressionLearner(), RandomForestRegressor()))
        >>> r = run(icr, 0.1, CrossSampler(Table('housing'), 10))
        >>> print(r.accuracy(), r.median_range(), r.interdecile_mean())
    """

    def __init__(self, classifier, rf, beta=0.5):
        """Store the classifier and beta parameter."""
        assert isinstance(rf, RandomForestRegressor), \
            "Second parameter must be an instance of sklearn's RandomForestRegressor."
        self.learner, self.model = classifier, None
        self.rf = rf
        self.beta = beta

    def __str__(self):
        return "AbsErrorRF ({})".format(super().__str__())

    def fit(self, data):
        """Train the underlying classifier on provided data and store the trained model."""
        self.model = self.learner(data)
        self.rf = self.rf.fit(data.X, data.Y)

    def norm(self, inst):
        """Normalization factor is equal to the standard deviation of predictions from trees in a random forest
        plus a constant term :py:attr:`beta`."""
        ys = [estimator.predict(np.atleast_2d(inst.x))[0] for estimator in self.rf.estimators_]
        return np.std(ys) + self.beta

    def nonconformity(self, inst):
        y = float(self.model(inst))
        nc = abs(inst.get_class() - y)
        return nc / self.norm(inst)

    def predict(self, inst, nc):
        y = float(self.model(inst))
        norm = self.norm(inst)
        return y - nc*norm, y + nc*norm


class ErrorModelNC(RegrModelNC):
    """ErrorModelNC is based on two underlying regressors. The first one is trained to predict the value while the
    second one is used for predicting logarithms of the errors made by the first one.

    H. Papadopoulos and H. Haralambous. *Reliable prediction intervals with regression neural networks*.
    Neural Networks (2011).

    .. math::
        \\mathit{nc} = \\frac{|\\hat{y}-y^{*}|}{\\exp(\\mu)-1 + \\beta}

    :math:`\\mu` ... prediction for the value of :math:`\\log(|\\hat{y}-y^{*}|+1)` returned by the second regressor

    Parameter :py:attr:`loo` determines whether to use a leave-one-out schema for building the training set
    of errors for the second regressor or not.

    Examples:
        >>> nc = ErrorModelNC(SVRLearner(), LinearRegressionLearner())
        >>> icr = InductiveRegressor(nc)
        >>> r = run(icr, 0.1, CrossSampler(Table('housing'), 10))
        >>> print(r.accuracy(), r.median_range(), r.interdecile_mean())
    """

    def __init__(self, classifier, error_classifier, beta=0.5, loo=False):
        super().__init__(classifier)
        self.error_learner = error_classifier
        self.error_model = None
        self.beta = beta
        self.loo = loo

    def __str__(self):
        return "ErrorModelNC ({}, {})".format(self.learner, self.error_learner)

    def fit(self, data):
        if self.loo:
            ys = []
            for i in range(len(data)):
                train = data[np.arange(len(data)) != i]
                inst = data[i]
                self.model = self.learner(train)
                ys.append(math.log(abs(inst.get_class() - self.model(inst))+1))
            self.model = self.learner(data)
        else:
            self.model = self.learner(data)
            ys = [math.log(abs(row.get_class() - self.model(row))+1) for row in data]
            error_data = Table.from_numpy(data.domain, data.X, ys)
        error_data = Table.from_numpy(data.domain, data.X, ys)
        self.error_model = self.error_learner(error_data)

    def nonconformity(self, inst):
        nc = abs(inst.get_class() - float(self.model(inst)))
        norm = math.exp(float(self.error_model(inst)))-1 + self.beta
        return nc / norm

    def predict(self, inst, nc):
        y = float(self.model(inst))
        norm = math.exp(float(self.error_model(inst)))-1 + self.beta
        return y - nc*norm, y + nc*norm


class ExperimentalNC(RegrModelNC):
    def __init__(self, rf):
        self.rf = rf

    def fit(self, data):
        self.rf = self.rf.fit(data.X, data.Y)
        ys = sorted(data.Y)
        d = len(ys)//10
        self.ir = ys[-(d+1)] - ys[d]

    def norm(self, inst):
        ys = [estimator.predict(np.atleast_2d(inst.x))[0] for estimator in self.rf.estimators_]
        return np.std(ys) / self.ir

    def nonconformity(self, inst):
        y = float(self.rf.predict(np.atleast_2d(inst.x))[0])
        return abs(inst.get_class() - y / self.norm(inst))

    def predict(self, inst, nc):
        y = float(self.rf.predict(np.atleast_2d(inst.x))[0])
        norm = self.norm(inst)
        a, b = (y-nc)*norm, (y+nc)*norm
        return min(a,b), max(a,b)


# --- model-knn --- #

class AbsErrorNormalized(RegrModelNC, NearestNeighbours):
    """Normalized absolute error prediction uses an underlying regression model to predict the value,
    which is then normalized by the distance and variance of the nearest neighbours.

    H. Papadopoulos, V. Vovk and A. Gammerman. *Regression Conformal Prediction with Nearest Neighbours*.
    Journal of Artificial Intelligence Research (2011).

    .. math::
        \\mathit{nc} = \\frac{|\\hat{y}-y^{*}|}{\\exp(\\gamma \\lambda^*) + \exp(\\rho \\xi^*)}
        \\quad \\text{or} \\quad
        \\mathit{nc} = \\frac{|\\hat{y}-y^{*}|}{\\gamma + \\lambda^* + \\xi^*}

    The first nonconformity score is used when the parameter :py:attr:`exp` is set to *True*
    and the second one when it is set to *False*.

    .. math::
        \\lambda^* = \\frac{d_k(z^*)}{\\mathit{median}(\\{d_k(z), z \\in T\\})}, \\quad
        d_k(z) = \\sum_{z_i \\in N_k(z)} distance(x, x_i)

    .. math::
        \\xi^* = \\frac{\\sigma_k(z^*)}{\\mathit{median}(\\{\\sigma_k(z), z \\in T\\})}, \\quad
        \\sigma_k(z) = \\sqrt{\\frac{1}{k} \\sum_{z_i \\in N_k(z)}(y_i-\\bar{y})}

    Parameter :py:attr:`rf` enables the use of a random forest for computing the standard deviation
    of predictions instead of the nearest neighbours.
    """

    def __init__(self, classifier, distance=Euclidean(), k=10, gamma=0.5, rho=0.5, exp=True, rf=None):
        """Initialize the parameters."""
        RegrModelNC.__init__(self, classifier)
        NearestNeighbours.__init__(self, distance, k)
        self._gamma = gamma  # distance sensitivity
        self._rho = rho  # variance sensitivity
        self.exp = exp  # type of normalization
        self.rf = rf  # random forest for normalization
        if self.rf:
            assert isinstance(rf, RandomForestRegressor), \
                "Rf must be an instance of sklearn's RandomForestRegressor."

    def __str__(self):
        return "AbsErrorNormalized ({}, k={})".format(self.learner, self.k)

    def fit(self, data):
        """Train the underlying model and precompute medians for nonconformity scores."""
        RegrModelNC.fit(self, data)
        NearestNeighbours.fit(self, data)
        if self.rf:
            self.rf = self.rf.fit(data.X, data.Y)
        # compute medians in the training set used for normalization
        self.median_d = np.median([self._d(inst) for inst in self.data])
        self.median_s = np.median([self._sigma(inst) for inst in self.data])

    def _d(self, inst):
        """Sum of distances to nearest neighbours."""
        neigh = self.neighbours(inst)[:self.k]
        return sum(dist for dist, row in neigh)

    def _lambda(self, inst):
        """Normalized distance measure."""
        return self._d(inst) / self.median_d

    def _sigma(self, inst):
        """Standard deviation of y values. This comes either from the nearest neighbours or from the
        predictions of individual trees in a random forest if the :py:attr:`rf` is provided."""
        if not self.rf:
            neigh = self.neighbours(inst)[:self.k]
            return np.std([row.get_class() for dist, row in neigh])
        else:
            preds = [estimator.predict(np.atleast_2d(inst.x))[0] for estimator in self.rf.estimators_]
            return np.std(preds)

    def _xi(self, inst):
        """Normalized variance measure."""
        return self._sigma(inst) / self.median_s

    def norm(self, inst):
        """Compute the normalization factor."""
        if self.exp:
            return math.exp(self._gamma*self._lambda(inst)) + math.exp(self._rho*self._xi(inst))
        else:
            return self._gamma + self._lambda(inst) + self._xi(inst)

    def nonconformity(self, inst):
        yp = float(self.model(inst))
        return abs(inst.get_class() - yp) / self.norm(inst)

    def predict(self, inst, nc):
        y = float(self.model(inst))
        norm = self.norm(inst)
        return y-nc*norm, y+nc*norm


class LOORegrNC(NearestNeighbours, RegrNC):
    """
    .. math::
        \\mathit{nc} = \\mathit{error} + |\\hat{y}-y^{*}|
        \\quad \\text{or} \\quad
        \\mathit{nc} = \\frac{|\\hat{y}-y^{*}|}{\\mathit{error}}

    :math:`\\hat{y}` ... value predicted from :math:`N_k(z^*)`

    The first nonconformity score is used when the parameter :py:attr:`relative` is set to *False*
    and the second one when it is set to *True*.

    .. math::
        \\mathit{error} = \\frac {\\sum_{z_i \\in N_k(z^*)} w_i |\\hat{y_i}-y_i|} {\\sum_{z_i \\in N_k(z^*)} w_i},
        \\quad w_i = \\frac{1}{d(x^*, x_i)}

    :math:`\\hat{y_i}` ... value predicted from :math:`N_k(z') \\setminus z_i` or
    :math:`N_k(z') \\setminus z_i \\cup z^*` if the parameter :py:attr:`include` is set to *True*. :math:`z'` is
    :math:`z^*` if the :py:attr:`neighbourhood` parameter is '*fixed*' and :math:`z_i` if it's '*variable*'.
    """

    def __init__(self, classifier, distance=Euclidean(), k=10, relative=True, include=False, neighbourhood='fixed'):
        """Initialize the parameters."""
        super().__init__(distance, k)
        self.classifier = classifier
        self.relative = relative
        self.include = include
        assert neighbourhood in ['fixed', 'variable']
        self.neighbourhood = neighbourhood

    def fit(self, data):
        """Store the data for finding nearest neighbours and initialize cache."""
        super().fit(data)
        self.cache = {}

    def get_neighbourhood(self, inst):
        """Construct an Orange data Table consisting of instance's k nearest neighbours.
        Cache the results for later calls with the same instance."""
        t = tuple(inst.x)
        if t not in self.cache:
            neigh = self.neighbours(inst)[:self.k]
            neigh_d, neigh_list = zip(*neigh)
            neigh_tab = Table(neigh_list[0].domain, list(neigh_list))
            self.cache[t] = neigh_tab
        return self.cache[t]

    def error(self, inst, neighbours):
        """Compute the average weighted error for predicting the value of each neighbour from the other ones.
        Include the new example among the neighbours if the parameter :py:attr:`include` is True."""
        sc = 0
        ws = []
        for i in range(len(neighbours)):
            neigh = Instance(neighbours.domain, neighbours[i])
            w = 1/self.distance(inst, neigh)
            ws.append(w)

            if self.neighbourhood == 'fixed':
                neighbours_i = neighbours[np.arange(len(neighbours)) != i]
            else:
                neighbours_i = self.get_neighbourhood(neigh)
            if self.include:
                neighbours_i = neighbours_i.copy()
                neighbours_i.append(inst)

            model = self.classifier(neighbours_i)
            sc += w * abs(model(neigh) - neigh.get_class())
        return float(sc / sum(ws))

    def nonconformity(self, inst):
        neighbours = self.get_neighbourhood(inst)
        model = self.classifier(neighbours)
        error = self.error(inst, neighbours)
        yp = float(model(inst))
        if self.relative:
            return abs(yp - inst.get_class()) / error
        else:
            return error + abs(yp - inst.get_class())

    def predict(self, inst, nc):
        neighbours = self.get_neighbourhood(inst)
        model = self.classifier(neighbours)
        yp = float(model(inst))
        inst.set_class(yp)
        error = self.error(inst, neighbours)
        if self.relative:
            return yp - (nc*error), yp + (nc*error)
        else:
            return yp - (nc-error), yp + (nc-error)


# --- knn-based --- #

class RegrNearestNeighboursNC(NearestNeighbours, RegrNC):
    """Base class for nearest neighbours based regression nonconformity scores."""
    pass


class AbsErrorKNN(RegrNearestNeighboursNC):
    """Absolute error of k nearest neighbours computes the average value of the k nearest neighbours and
    returns an absolute difference between this average (:math:`y_k`) and the actual value (:math:`y^{*}`).

    .. math::
        \\bar{y} &= 1/k \sum_{N_k(x^{*})} y_i \\\\
        \\mathit{nc} &= |\\bar{y} - y^{*}|

    Weighted version can normalize by average and/or variance.

    .. math::
        \\mathit{nc} = \\frac{ |\\bar{y}-y^{*}| } { \\bar{y} \cdot y_{\\sigma} }

    Attributes:
        average (bool): Normalize by average.
        variance (bool): Normalize by variance.

    Examples:
        >>> train, test = next(LOOSampler(Table('housing')))
        >>> cr = CrossRegressor(AbsErrorKNN(Euclidean(), 10, average=True), 2, train)
        >>> print(cr(test[0].x, 0.1))
    """

    def __init__(self, distance=Euclidean(), k=10, average=False, variance=False):
        """Initialize the distance measure, number of nearest neighbours to consider and
        whether to normalize by average and by variance."""
        super().__init__(distance, k)
        self.average = average
        self.variance = variance

    def __str__(self):
        return "AbsErrorKNN (k={})".format(self.k)

    def stats(self, instance):
        """Computes mean and standard deviation of values within the k nearest neighbours."""
        dist = self.neighbours(instance)
        neigh = [row.get_class() for d, row in dist[:self.k]]
        return np.mean(neigh), np.std(neigh)

    def norm(self, avg, std):
        """Compute the normalization factor according to the chosen properties."""
        norm = 1
        if self.average: norm /= avg
        if self.variance: norm /= std
        return norm

    def nonconformity(self, instance):
        avg, std = self.stats(instance)
        nc = abs(instance.get_class() - avg)
        return nc * self.norm(avg, std)

    def predict(self, inst, nc):
        avg, std = self.stats(inst)
        norm = self.norm(avg, std)
        return avg - nc/norm, avg + nc/norm


class AvgErrorKNN(RegrNearestNeighboursNC):
    """Average error of k nearest neighbours computes the average absolute error of the actual value (:math:`y^{*}`)
    compared to the k nearest neighbours (:math:`y_i`).

    .. math::
        \\mathit{nc} = 1/k \sum_{N_k(x^{*})} |y^{*} - y_i|

    Note:
        There might be no suitable `y` values for the required significance level at the time of prediction.
        In such cases, the predicted range is [nan, nan].

    Examples:
        >>> train, test = next(LOOSampler(Table('housing')))
        >>> cr = CrossRegressor(AvgErrorKNN(Euclidean(), 10), 2, train)
        >>> print(cr(test[0].x, 0.1))
    """

    def __str__(self):
        return "AvgErrorKNN (k={})".format(self.k)

    def avg_abs(self, y, ys):
        return np.mean([abs(y-yi) for yi in ys])

    def avg_abs_inv(self, nc, ys):
        ys = sorted(ys)
        i, j = (len(ys)-1)//2, len(ys)//2
        if self.avg_abs(ys[i], ys) > nc:
            return np.nan, np.nan
        # lower bound
        while i-1 >= 0 and self.avg_abs(ys[i-1], ys) <= nc:
            i -= 1
        i1, i2 = i, len(ys)-i
        lo = ys[i] + len(ys)*(nc-self.avg_abs(ys[i], ys))/(i1-i2)
        # upper bound
        while j+1 < len(ys) and self.avg_abs(ys[j+1], ys) <= nc:
            j += 1
        j1, j2 = j+1, len(ys)-(j+1)
        hi = ys[j] + len(ys)*(nc-self.avg_abs(ys[j], ys))/(j1-j2)
        return lo, hi

    def nonconformity(self, instance):
        dist = self.neighbours(instance)[:self.k]
        ys = [row.get_class() for d, row in dist]
        return self.avg_abs(instance.get_class(), ys)

    def predict(self, inst, nc):
        dist = self.neighbours(inst)[:self.k]
        ys = [row.get_class() for d, row in dist]
        return self.avg_abs_inv(nc, ys)
