"""Unittests for conformal prediction."""

from unittest import TestCase
import os

import numpy as np
import sklearn.svm as skl_svm
from sklearn.ensemble import RandomForestRegressor

from Orange.classification import NaiveBayesLearner, LogisticRegressionLearner, SVMLearner, KNNLearner
from Orange.data import Table, RowInstance
from Orange.distance import Euclidean, Cosine
from Orange.preprocess import Normalize
from Orange.regression import LinearRegressionLearner, KNNRegressionLearner, SVRLearner, RandomForestRegressionLearner

from orangecontrib.conformal.classification import TransductiveClassifier, InductiveClassifier, CrossClassifier, LOOClassifier
from orangecontrib.conformal.evaluation import LOOSampler, CrossSampler, RandomSampler, run, calibration_plot, run_train_test, ResultsClass, \
    ResultsRegr
from orangecontrib.conformal.nonconformity import InverseProbability, AbsError, KNNDistance, KNNFraction, AbsErrorKNN, ProbabilityMargin, \
    AvgErrorKNN, AbsErrorNormalized, LOORegrNC, LOOClassNC, SVMDistance, AbsErrorRF, ExperimentalNC, ErrorModelNC
from orangecontrib.conformal.regression import InductiveRegressor, CrossRegressor, LOORegressor
from orangecontrib.conformal.utils import get_instance, split_data, shuffle_data


class TestTransductive(TestCase):
    def test_inverse_probability(self):
        tab = Table('iris')
        train, test = get_instance(tab, 0)
        tcp = TransductiveClassifier(InverseProbability(NaiveBayesLearner()), train)
        pred = tcp(test.x, 0.1)
        self.assertEqual(pred, ['Iris-setosa'])

        train, test = get_instance(tab, 0)
        tcp = TransductiveClassifier(InverseProbability(NaiveBayesLearner()))
        tcp.fit(train)
        pred = tcp(test.x, 0.1)
        self.assertEqual(pred, ['Iris-setosa'])

    def test_nearest_neighbours(self):
        tab = Table('iris')
        train, test = get_instance(tab, 0)
        tcp = TransductiveClassifier(KNNDistance(Euclidean), train)
        pred = tcp(test.x, 0.1)
        self.assertEqual(pred, ['Iris-setosa'])

    def test_validate_transductive(self):
        tab = Table('iris')
        eps = 0.1
        correct, num, all = 0, 0, len(tab)
        for i in range(all):
            train, test = get_instance(tab, i)
            tcp = TransductiveClassifier(InverseProbability(LogisticRegressionLearner()), train)
            pred = tcp(test.x, eps)
            if test.get_class() in pred: correct += 1
            num += len(pred)
        self.assertAlmostEqual(correct/all, 1.0-eps, delta=0.01)


class TestInductive(TestCase):
    def setUp(self):
        self.tab = Table('iris')
        train, self.test = get_instance(self.tab, 0)
        self.train, self.calibrate = split_data(shuffle_data(train), 2, 1)

    def test_inverse_probability(self):
        icp = InductiveClassifier(InverseProbability(NaiveBayesLearner()), self.train, self.calibrate)
        pred = icp(self.test.x, 0.01)
        self.assertEqual(pred, ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])

        icp = InductiveClassifier(InverseProbability(NaiveBayesLearner()))
        icp.fit(self.train, self.calibrate)
        pred = icp(self.test.x, 0.1)
        self.assertEqual(pred, ['Iris-setosa'])

    def test_nearest_neighbours(self):
        icp = InductiveClassifier(KNNDistance(Euclidean), self.train, self.calibrate)
        pred = icp(self.test.x, 0.1)
        self.assertEqual(pred, ['Iris-setosa'])

    def test_validate_inductive(self):
        eps = 0.1
        correct, num, all = 0, 0, len(self.tab)
        for i in range(all):
            train, test = get_instance(self.tab, i)
            train, calibrate = split_data(shuffle_data(train), 2, 1)
            icp = InductiveClassifier(InverseProbability(LogisticRegressionLearner()), train, calibrate)
            pred = icp(test.x, eps)
            if test.get_class() in pred: correct += 1
            num += len(pred)
        self.assertAlmostEqual(correct/all, 1.0-eps, delta=0.01)

    def test_validate_regression(self):
        tab = Table('housing')
        eps = 0.1
        correct, num, all = 0, 0, len(tab)
        for i in range(all):
            train, test = get_instance(tab, i)
            train, calibrate = split_data(shuffle_data(train), 2, 1)
            icr = InductiveRegressor(AbsError(LinearRegressionLearner()), train, calibrate)
            y_min, y_max = icr(test.x, eps)
            if y_min <= test.y <= y_max: correct += 1
            num += y_max - y_min
        self.assertAlmostEqual(correct/all, 1.0-eps, delta=0.02)


class TestCross(TestCase):
    def test_cross_classification(self):
        tab = Table('iris')
        train, test = get_instance(tab, 0)
        train = shuffle_data(train)
        ccp = CrossClassifier(InverseProbability(LogisticRegressionLearner()), 3, train)
        pred = ccp(test.x, 0.1)
        self.assertEqual(pred, ['Iris-setosa'])

        ccp = CrossClassifier(InverseProbability(LogisticRegressionLearner()), 3)
        ccp.fit(train)
        pred = ccp(test.x, 0.1)
        self.assertEqual(pred, ['Iris-setosa'])

    def test_loo(self):
        train, test = get_instance(Table('iris'), 0)
        loocp = LOOClassifier(InverseProbability(LogisticRegressionLearner()), train)
        pred = loocp(test.x, 0.1)
        self.assertEqual(pred, ['Iris-setosa'])

        train, test = get_instance(Table('housing'), 0)
        loocr = LOORegressor(AbsError(LinearRegressionLearner()), train)
        lo, hi = loocr(test.x, 0.1)
        self.assertLess(hi-lo, 20)

    def test_validate_cross_classification(self):
        tab = shuffle_data(Table('iris'))
        eps = 0.1
        correct, num, all = 0, 0, len(tab)
        for i in range(all):
            train, test = get_instance(tab, i)
            ccp = CrossClassifier(InverseProbability(NaiveBayesLearner()), 5, train)
            pred = ccp(test.x, eps)
            if test.get_class() in pred: correct += 1
            num += len(pred)
        self.assertAlmostEqual(correct/all, 1.0-eps, delta=0.02)

    def test_validate_cross_regression(self):
        tab = shuffle_data(Table('housing'))
        eps = 0.1
        correct, num, all = 0, 0, len(tab)
        for i in range(all):
            train, test = get_instance(tab, i)
            ccr = CrossRegressor(AbsError(LinearRegressionLearner()), 5, shuffle_data(train))
            y_min, y_max = ccr(test.x, eps)
            if y_min <= test.y <= y_max: correct += 1
            num += y_max - y_min
        self.assertAlmostEqual(correct/all, 1.0-eps, delta=0.02)


class TestMondrian(TestCase):
    def test_transductive(self):
        cpm = TransductiveClassifier(InverseProbability(LogisticRegressionLearner()), mondrian=True)
        rm = run(cpm, 0.1, LOOSampler(Table('iris')))
        self.assertGreater(rm.accuracy(), 0.85)
        self.assertGreater(rm.singleton_criterion(), 0.85)

    def test_inductive(self):
        cpm = InductiveClassifier(InverseProbability(LogisticRegressionLearner()), mondrian=True)
        rm = run(cpm, 0.1, LOOSampler(Table('iris')))
        self.assertGreater(rm.accuracy(), 0.85)
        self.assertGreater(rm.singleton_criterion(), 0.85)

    def test_cross(self):
        cpm = CrossClassifier(InverseProbability(LogisticRegressionLearner()), 4, mondrian=True)
        rm = run(cpm, 0.1, LOOSampler(Table('iris')))
        self.assertGreater(rm.accuracy(), 0.85)
        self.assertGreater(rm.singleton_criterion(), 0.85)

    def test_accuracy(self):
        tab = Table('iris')[:120]
        results = []
        for m in [False, True]:
            cp = CrossClassifier(InverseProbability(LogisticRegressionLearner()), 2, mondrian=m)
            r = run(cp, 0.2, LOOSampler(tab), rep=10)
            res = [r.accuracy(y) for y in range(3)]
            results.append(res)
            print(r.accuracy(), res)
        span = max(results[0])-min(results[0])
        span_mondrian = max(results[1])-min(results[1])
        self.assertLess(span_mondrian, span)


class TestUSPS(TestCase):
    def test_nonexchangeability(self):
        tab = Table(os.path.join(os.path.dirname(__file__), '../data/usps.tab'))
        train, test = split_data(tab, 7291, 2007)
        test = test[:200]
        train, calibrate = split_data(train, 3, 1)
        icp = InductiveClassifier(InverseProbability(LogisticRegressionLearner()), train, calibrate)
        err = [inst.get_class() not in icp(inst.x, 0.1) for inst in test]
        self.assertGreater(sum(err)/len(test), 0.13)


class TestClassificationNC(TestCase):
    def setUp(self):
        self.train, self.test = get_instance(Table('iris'), 0)
        self.train, self.calibrate = split_data(shuffle_data(self.train), 2, 1)

    def test_model_based(self):
        icp = InductiveClassifier(InverseProbability(SVMLearner(probability=True)), self.train, self.calibrate)
        pred = icp(self.test.x, 0.1)
        self.assertEqual(pred, ['Iris-setosa'])

        icp = InductiveClassifier(ProbabilityMargin(SVMLearner(probability=True)), self.train, self.calibrate)
        pred = icp(self.test.x, 0.1)
        self.assertEqual(pred, ['Iris-setosa'])

    def test_knn_speed(self):
        tab = Table(os.path.join(os.path.dirname(__file__), '../datasets-class/spambase.tab'))
        train, calibrate = next(RandomSampler(tab, 2, 1))
        icp = InductiveClassifier(KNNDistance(Euclidean), train, calibrate)

    def test_knn_distance(self):
        icp = InductiveClassifier(KNNDistance(Euclidean), self.train, self.calibrate)
        pred = icp(self.test.x, 0.1)
        self.assertEqual(pred, ['Iris-setosa'])

        icp = InductiveClassifier(KNNDistance(Cosine, 5), self.train, self.calibrate)
        pred = icp(self.test.x, 0.1)
        self.assertEqual(pred, ['Iris-setosa'])

        from Orange.distance import Mahalanobis
        icp = InductiveClassifier(KNNDistance(Mahalanobis, 5), self.train, self.calibrate)
        pred = icp(self.test.x, 0.1)
        self.assertEqual(pred, ['Iris-setosa'])

        from Orange.distance import MahalanobisDistance
        Mah = MahalanobisDistance(self.train)
        icp = InductiveClassifier(KNNDistance(Mah, 5), self.train, self.calibrate)
        pred = icp(self.test.x, 0.1)
        self.assertEqual(pred, ['Iris-setosa'])

    def test_knn_fraction(self):
        icp = InductiveClassifier(KNNFraction(Euclidean, 10), self.train, self.calibrate)
        pred = icp(self.test.x, 0.1)
        self.assertEqual(pred, ['Iris-setosa'])

        icp = InductiveClassifier(KNNFraction(Euclidean, 10, weighted=True), self.train, self.calibrate)
        pred = icp(self.test.x, 0.1)
        self.assertEqual(pred, ['Iris-setosa'])

    def test_LOOClassNC(self):
        for incl in [False, True]:
            for rel in [False, True]:
                for neigh in ['fixed', 'variable']:
                    nc = LOOClassNC(NaiveBayesLearner(), Euclidean, 20,
                                    relative=rel, include=incl, neighbourhood=neigh)
                    icp = InductiveClassifier(nc, self.train, self.calibrate)
                    pred = icp(self.test.x, 0.1)
                    print(pred)
                    self.assertEqual(pred, ['Iris-setosa'])

        icp = InductiveClassifier(LOOClassNC(NaiveBayesLearner(), Euclidean, 20))
        r = run(icp, 0.1, CrossSampler(Table('iris'), 4))
        self.assertGreater(r.accuracy(), 0.85)
        self.assertGreater(r.singleton_criterion(), 0.8)

    def test_SVM(self):
        iris = Table('iris')
        tab = Table(iris.X[50:], iris.Y[50:]-1)  # versicolor, virginica
        # clear cases
        train, test = get_instance(tab, 30)
        train, calibrate = next(RandomSampler(train, 2, 1))
        icp = InductiveClassifier(SVMDistance(skl_svm.SVC()), train, calibrate)
        pred = icp(test.x, 0.1)
        self.assertEqual(pred, ['v1'])
        train, test = get_instance(tab, 85)
        train, calibrate = next(RandomSampler(train, 2, 1))
        icp = InductiveClassifier(SVMDistance(skl_svm.SVC()), train, calibrate)
        pred = icp(test.x, 0.1)
        self.assertEqual(pred, ['v2'])
        # border case
        train, test = get_instance(tab, 27)
        train, calibrate = next(RandomSampler(train, 2, 1))
        icp = InductiveClassifier(SVMDistance(skl_svm.SVC()), train, calibrate)
        pred = icp(test.x, 0.2)
        self.assertEqual(pred, [])
        pred = icp(test.x, 0.01)
        self.assertEqual(pred, ['v1', 'v2'])

    def test_nc_type(self):
        nc_class = InverseProbability(LogisticRegressionLearner())
        nc_regr = AbsError(LinearRegressionLearner())
        TransductiveClassifier(nc_class)
        self.assertRaises(AssertionError, TransductiveClassifier, nc_regr)
        InductiveClassifier(nc_class)
        self.assertRaises(AssertionError, InductiveClassifier, nc_regr)
        CrossClassifier(nc_class, 5)
        self.assertRaises(AssertionError, CrossClassifier, nc_regr, 5)


class TestRegressionNC(TestCase):
    def setUp(self):
        self.train, self.test = get_instance(Table('housing'), 0)
        self.train, self.calibrate = split_data(shuffle_data(self.train), 2, 1)

    def test_abs_error_normalized(self):
        tab = Table('housing')
        normalizer = Normalize(zero_based=True, norm_type=Normalize.NormalizeBySpan)
        tab = normalizer(tab)

        icr = InductiveRegressor(AbsError(LinearRegressionLearner()))
        icr_knn = InductiveRegressor(AbsError(KNNRegressionLearner(4)))
        icr_norm = InductiveRegressor(AbsErrorNormalized(KNNRegressionLearner(4), Euclidean, 4, exp=False))
        icr_norm_exp = InductiveRegressor(AbsErrorNormalized(KNNRegressionLearner(4), Euclidean, 4, exp=True))
        icr_norm_rf = InductiveRegressor(AbsErrorNormalized(KNNRegressionLearner(4), Euclidean, 4,
                                                            rf=RandomForestRegressor()))

        r, r_knn, r_norm, r_norm_exp, r_norm_rf = ResultsRegr(), ResultsRegr(), ResultsRegr(), ResultsRegr(), ResultsRegr()
        eps = 0.05
        for rep in range(10):
            for train, test in CrossSampler(tab, 10):
                train, calibrate = next(RandomSampler(train, len(train)-100, 100))
                r.concatenate(run_train_test(icr, eps, train, test, calibrate))
                r_knn.concatenate(run_train_test(icr_knn, eps, train, test, calibrate))
                r_norm.concatenate(run_train_test(icr_norm, eps, train, test, calibrate))
                r_norm_exp.concatenate(run_train_test(icr_norm_exp, eps, train, test, calibrate))
                r_norm_rf.concatenate(run_train_test(icr_norm_rf, eps, train, test, calibrate))

        print(r.median_range(), r.interdecile_mean(), 1-r.accuracy())
        print(r_knn.median_range(), r_knn.interdecile_mean(), 1-r_knn.accuracy())
        print(r_norm.median_range(), r_norm.interdecile_mean(), 1-r_norm.accuracy())
        print(r_norm_exp.median_range(), r_norm_exp.interdecile_mean(), 1-r_norm_exp.accuracy())
        print(r_norm_rf.median_range(), r_norm_rf.interdecile_mean(), 1-r_norm_rf.accuracy())
        self.assertGreater(r.accuracy(), 1-eps-0.03)
        self.assertGreater(r_knn.accuracy(), 1-eps-0.03)
        self.assertGreater(r_norm.accuracy(), 1-eps-0.03)
        self.assertGreater(r_norm_exp.accuracy(), 1-eps-0.03)
        self.assertGreater(r_norm_rf.accuracy(), 1-eps-0.03)
        """
        19.739259734 20.4378007266 0.051185770751
        22.225 22.2995182806 0.0474308300395
        15.0819327682 17.4239613065 0.048418972332
        14.3463382738 18.2976916462 0.0462450592885
        13.6700968865 18.0934053343 0.051185770751
        """

    def test_LOORegrNC(self):
        for incl in [False, True]:
            for rel in [False, True]:
                for neigh in ['fixed', 'variable']:
                    nc = LOORegrNC(LinearRegressionLearner, Euclidean, 150,
                                   relative=rel, include=incl, neighbourhood=neigh)
                    icr = InductiveRegressor(nc, self.train, self.calibrate)
                    lo, hi = icr(self.test.x, 0.1)
                    print(lo, hi)
                    self.assertLess(hi-lo, 20.0)

        tab = Table('housing')
        icr = InductiveRegressor(LOORegrNC(LinearRegressionLearner, Euclidean, 150))
        r = run(icr, 0.1, CrossSampler(tab, 4))
        self.assertGreater(r.accuracy(), 0.85)
        self.assertLess(r.mean_range(), 15.0)

    def test_error_model(self):
        for loo in [False, True]:
            icr = InductiveRegressor(ErrorModelNC(LinearRegressionLearner(), LinearRegressionLearner(), loo=loo),
                                     self.train, self.calibrate)
            lo, hi = icr(self.test.x, 0.1)
            self.assertLess(hi-lo, 30.0)

        icr = InductiveRegressor(AbsError(RandomForestRegressionLearner()))
        r = run(icr, 0.1, CrossSampler(Table('housing'), 20))
        self.assertGreater(r.accuracy(), 0.85)
        print(r.accuracy(), r.median_range(), r.interdecile_mean())

        icr = InductiveRegressor(ErrorModelNC(RandomForestRegressionLearner(), LinearRegressionLearner()))
        r = run(icr, 0.1, CrossSampler(Table('housing'), 20))
        self.assertGreater(r.accuracy(), 0.85)
        print(r.accuracy(), r.median_range(), r.interdecile_mean())

        icr = InductiveRegressor(ErrorModelNC(RandomForestRegressionLearner(), LinearRegressionLearner(), loo=True))
        r = run(icr, 0.1, CrossSampler(Table('housing'), 20))
        self.assertGreater(r.accuracy(), 0.85)
        print(r.accuracy(), r.median_range(), r.interdecile_mean())

    def test_abs_error_rf(self):
        icr = InductiveRegressor(AbsErrorRF(RandomForestRegressionLearner(), RandomForestRegressor()),
                                 self.train, self.calibrate)
        lo, hi = icr(self.test.x, 0.1)
        self.assertLess(hi-lo, 30.0)

        icr = InductiveRegressor(AbsErrorRF(LinearRegressionLearner(), RandomForestRegressor()),
                                 self.train, self.calibrate)
        lo, hi = icr(self.test.x, 0.1)
        self.assertLess(hi-lo, 30.0)

        icr = InductiveRegressor(AbsErrorRF(RandomForestRegressionLearner(), RandomForestRegressor()))
        r = run(icr, 0.1, CrossSampler(Table('housing'), 10))
        self.assertGreater(r.accuracy(), 0.85)
        print(r.median_range(), r.interdecile_mean())

    def test_abs_error_knn(self):
        icr = InductiveRegressor(AbsErrorKNN(Euclidean, 5), self.train, self.calibrate)
        lo, hi = icr(self.test.x, 0.1)
        self.assertLess(hi-lo, 30.0)

        icr = InductiveRegressor(AbsErrorKNN(Euclidean, 5, average=True), self.train, self.calibrate)
        lo, hi = icr(self.test.x, 0.1)
        self.assertLess(hi-lo, 30.0)

        icr = InductiveRegressor(AbsErrorKNN(Euclidean, 5, variance=True), self.train, self.calibrate)
        lo, hi = icr(self.test.x, 0.1)
        self.assertLess(hi-lo, 30.0)

    def test_avg_error_knn(self):
        ncm = AvgErrorKNN(Euclidean)
        self.assertEqual(ncm.avg_abs_inv(6/5, [1, 2, 3, 4, 5]), (3, 3))
        for odd in [0, 1]:
            ys = np.random.uniform(0, 1, 10+odd)
            nc = 0.4
            lo, hi = ncm.avg_abs_inv(nc, ys)
            self.assertGreater(ncm.avg_abs(lo-0.001, ys), nc)
            self.assertLess(ncm.avg_abs(lo+0.001, ys), nc)
            self.assertLess(ncm.avg_abs(hi-0.001, ys), nc)
            self.assertGreater(ncm.avg_abs(hi+0.001, ys), nc)

        icr = InductiveRegressor(AvgErrorKNN(Euclidean, 10), self.train, self.calibrate)
        lo, hi = icr(self.test.x, 0.1)
        self.assertLess(hi-lo, 30.0)

        r = run(InductiveRegressor(AvgErrorKNN(Euclidean, 10)), 0.1, RandomSampler(Table("housing"), 2, 1), rep=10)
        self.assertFalse(any([np.isnan(w) for w in r.widths()]))

    def test_validate_AvgErrorKNN(self):
        eps = 0.1
        correct, num, all = 0, 0, 0
        for it in range(10):
            train, test = split_data(shuffle_data(Table('housing')), 4, 1)
            train, calibrate = split_data(shuffle_data(train), 3, 1)
            icr = InductiveRegressor(AvgErrorKNN(Euclidean, 10), train, calibrate)
            for i, inst in enumerate(test):
                y_min, y_max = icr(inst.x, eps)
                if not np.isnan(y_min):
                    if y_min <= inst.y <= y_max: correct += 1
                    num += y_max - y_min
                all += 1
            print(correct/all, num/all)
        self.assertAlmostEqual(correct/all, 1.0-eps, delta=0.03)

    def test_validate_AbsErrorKNN(self):
        eps = 0.1
        correct, num, all = 0, 0, 0
        for it in range(10):
            train, test = split_data(shuffle_data(Table('housing')), 4, 1)
            train, calibrate = split_data(shuffle_data(train), 3, 1)
            icr = InductiveRegressor(AbsErrorKNN(Euclidean, 10, average=True, variance=True), train, calibrate)
            for i, inst in enumerate(test):
                y_min, y_max = icr(inst.x, eps)
                if y_min <= inst.y <= y_max: correct += 1
                num += y_max - y_min
                all += 1
            print(correct/all, num/all)
        self.assertAlmostEqual(correct/all, 1.0-eps, delta=0.03)

    def test_nc_type(self):
        nc_regr = AbsError(LinearRegressionLearner())
        nc_class = InverseProbability(LogisticRegressionLearner())
        InductiveRegressor(nc_regr)
        self.assertRaises(AssertionError, InductiveRegressor, nc_class)
        CrossRegressor(nc_regr, 5)
        self.assertRaises(AssertionError, CrossRegressor, nc_class, 5)

    def test_experimental(self):
        icr = InductiveRegressor(ExperimentalNC(RandomForestRegressor(n_estimators=20)), self.train, self.calibrate)
        r = run(icr, 0.1, CrossSampler(Table('housing'), 10))
        print(r.accuracy(), r.median_range())


class TestSamplers(TestCase):
    def setUp(self):
        self.data = Table('iris')

    def test_random(self):
        a, b = 3, 2
        s = RandomSampler(self.data, a, b)
        train, test = next(s)
        self.assertTrue(isinstance(train[0], RowInstance))
        self.assertAlmostEqual(len(train)/len(test), a/b)

    def test_cross(self):
        folds = 7
        s = CrossSampler(self.data, k=folds)
        l = [(len(train), len(test)) for train, test in s]
        self.assertEqual(len(l), folds)
        self.assertTrue(all(a+b == len(self.data) for a, b in l))
        t = [b for a, b in l]
        self.assertLessEqual(max(t)-min(t), 1)

        s = CrossSampler(self.data, k=folds)
        ids = frozenset(frozenset(inst.id for inst in test) for train, test in s)
        self.assertEqual(len(ids), folds)

    def test_loo(self):
        s = LOOSampler(self.data)
        x = [len(test) == 1 for train, test in s]
        self.assertTrue(all(x))
        self.assertEqual(len(x), len(self.data))

    def test_repeat(self):
        rep = 5
        s = RandomSampler(self.data, 3, 2)
        ids = frozenset(frozenset(inst.id for inst in test) for train, test in s.repeat(5))
        self.assertEqual(len(ids), 5)

        s = CrossSampler(self.data, 3)
        ids = frozenset(frozenset(inst.id for inst in test) for train, test in s.repeat(5))
        self.assertEqual(len(ids), 15)


class TestEvaluation(TestCase):
    def test_run(self):
        tab = Table('iris')
        cp = CrossClassifier(InverseProbability(LogisticRegressionLearner()), 5)
        r = run(cp, 0.1, RandomSampler(tab, 4, 1), rep=3)
        self.assertEqual(len(r.preds), 3*1/5*len(tab))

        tab = Table('housing')
        cr = InductiveRegressor(AbsError(LinearRegressionLearner()))
        r = run(cr, 0.1, CrossSampler(tab, 4), rep=3)
        self.assertEqual(len(r.preds), 3*len(tab))

    def test_run_train_test(self):
        tab = shuffle_data(Table('iris'))
        cp = CrossClassifier(InverseProbability(LogisticRegressionLearner()), 4)
        r = run_train_test(cp, 0.1, tab[:100], tab[100:])
        cp = InductiveClassifier(InverseProbability(LogisticRegressionLearner()))
        r = run_train_test(cp, 0.1, tab[:50], tab[100:], tab[50:100])

    def test_accuracy(self):
        tab = Table('iris')
        cp = CrossClassifier(InverseProbability(LogisticRegressionLearner()), 5)
        eps = 0.1
        r = run(cp, eps, LOOSampler(tab))
        acc = r.accuracy()
        self.assertAlmostEqual(acc, 1-eps, delta=0.03)

    def test_time(self):
        tab = Table('iris')
        cp = CrossClassifier(InverseProbability(LogisticRegressionLearner()), 5)
        r = run(cp, 0.1, LOOSampler(tab))
        r.time()

    def test_results_class(self):
        tab = Table('iris')
        cp = CrossClassifier(InverseProbability(LogisticRegressionLearner()), 5)
        empty = run(cp, 0.5, RandomSampler(tab, 2, 1)).empty_criterion()
        self.assertGreater(empty, 0.0)
        single = run(cp, 0.1, RandomSampler(tab, 2, 1)).singleton_criterion()
        self.assertGreater(single, 0.8)
        multiple = run(cp, 0.01, RandomSampler(tab, 2, 1)).multiple_criterion()
        self.assertGreater(multiple, 0.1)

        results = run(cp, 0.1, RandomSampler(tab, 2, 1))
        self.assertGreater(results.singleton_correct(), 0.8)
        self.assertGreater(results.confidence(), 0.9)
        self.assertGreater(results.credibility(), 0.4)

    def test_results_regr(self):
        tab = Table('housing')
        cr = CrossRegressor(AbsErrorKNN(Euclidean, 10, average=True), 5)
        r1 = run(cr, 0.1, RandomSampler(tab, 2, 1))
        r5 = run(cr, 0.5, RandomSampler(tab, 2, 1))
        self.assertGreater(r1.median_range(), r5.median_range())
        self.assertGreater(r1.mean_range(), r5.mean_range())
        self.assertGreater(r1.interdecile_range(), r5.interdecile_range())
        self.assertGreater(r1.interdecile_mean(), r5.interdecile_mean())
        self.assertGreater(r1.std_dev(), r5.std_dev())

    def test_pickle(self):
        import pickle
        train, test = next(RandomSampler(Table('iris'), 2, 1))
        train, cal = next(RandomSampler(train, 2, 1))
        ic = InductiveClassifier(InverseProbability(NaiveBayesLearner()))
        ic.fit(train, cal)
        print(ic(test[0].x, 0.1))
        with open('temp.cp','wb') as f:
            pickle.dump(ic, f)
        with open('temp.cp','rb') as f:
            ic2 = pickle.load(f)
        print(ic2(test[0].x, 0.1))

    """
    def test_calibration_plot(self):
        data = Table("iris")
        nc = InverseProbability(LogisticRegressionLearner())
        cp = InductiveClassifier(nc)
        calibration_plot(cp, data, rep=20)
    """


class TestEfficiency(TestCase):
    def test_individual_classification(self):
        train, test = get_instance(Table('iris'), 123)  # borderline case
        train = shuffle_data(train)
        ccp = CrossClassifier(InverseProbability(LogisticRegressionLearner()), 3, train)
        pred = ccp.predict(test.x)
        cred, conf = pred.credibility(), pred.confidence()
        self.assertLess(cred, 0.5)
        d = 1e-6
        self.assertEqual(len(ccp(test.x, 1-(conf-d))), 1)
        self.assertGreater(len(ccp(test.x, 1-(conf+d))), 1)
        self.assertEqual(len(ccp(test.x, (cred+d))), 0)
        self.assertGreater(len(ccp(test.x, (cred-d))), 0)
