from unittest import TestCase

import Orange

import orangecontrib.conformal as cp


class TestSamplers(TestCase):
    def setUp(self):
        self.data = Orange.data.Table('iris')

    def test_random(self):
        a, b = 3, 2
        s = cp.evaluation.RandomSampler(self.data, a, b)
        train, test = next(s)
        self.assertTrue(isinstance(train[0], Orange.data.RowInstance))
        self.assertAlmostEqual(len(train)/len(test), a/b)

    def test_cross(self):
        folds = 7
        s = cp.evaluation.CrossSampler(self.data, k=folds)
        l = [(len(train), len(test)) for train, test in s]
        self.assertEqual(len(l), folds)
        self.assertTrue(all(a+b == len(self.data) for a, b in l))
        t = [b for a, b in l]
        self.assertLessEqual(max(t)-min(t), 1)

        s = cp.evaluation.CrossSampler(self.data, k=folds)
        ids = frozenset(frozenset(inst.id for inst in test) for train, test in s)
        self.assertEqual(len(ids), folds)

    def test_loo(self):
        s = cp.evaluation.LOOSampler(self.data)
        x = [len(test) == 1 for train, test in s]
        self.assertTrue(all(x))
        self.assertEqual(len(x), len(self.data))

    def test_repeat(self):
        rep = 5
        s = cp.evaluation.RandomSampler(self.data, 3, 2)
        ids = frozenset(frozenset(inst.id for inst in test) for train, test in s.repeat(5))
        self.assertEqual(len(ids), 5)

        s = cp.evaluation.CrossSampler(self.data, 3)
        ids = frozenset(frozenset(inst.id for inst in test) for train, test in s.repeat(5))
        self.assertEqual(len(ids), 15)


class TestEvaluation(TestCase):
    def test_run(self):
        tab = Orange.data.Table('iris')[::2]
        nc = cp.nonconformity.InverseProbability(Orange.classification.LogisticRegressionLearner())
        pred = cp.classification.InductiveClassifier(nc)
        r = cp.evaluation.run(pred, 0.1, cp.evaluation.CrossSampler(tab, 3), rep=2)
        self.assertEqual(len(r.preds), 2*len(tab))

        tab = Orange.data.Table('housing')[:100]
        nc = cp.nonconformity.AbsError(Orange.regression.LinearRegressionLearner())
        pred = cp.regression.InductiveRegressor(nc)
        r = cp.evaluation.run(pred, 0.1, cp.evaluation.RandomSampler(tab, 3, 1), rep=2)
        self.assertEqual(len(r.preds), 2*(1/4)*len(tab))

    def test_run_train_test(self):
        tab = Orange.data.Table('iris')
        nc = cp.nonconformity.InverseProbability(Orange.classification.LogisticRegressionLearner())
        pred = cp.classification.InductiveClassifier(nc)
        train, test, calibrate = tab[0::3], tab[1::3], tab[2::3]
        r = cp.evaluation.run_train_test(pred, 0.1, train, test, calibrate)
        self.assertEqual(len(r.preds), len(test))

    def test_accuracy(self):
        tab = Orange.data.Table('iris')
        nc = cp.nonconformity.InverseProbability(Orange.classification.LogisticRegressionLearner())
        pred = cp.classification.CrossClassifier(nc, 5)
        eps = 0.1
        res = cp.evaluation.run(pred, eps, cp.evaluation.CrossSampler(tab, 3))
        acc = res.accuracy()
        self.assertAlmostEqual(acc, 1-eps, delta=0.05)

    def test_time(self):
        tab = Orange.data.Table('iris')
        nc = cp.nonconformity.InverseProbability(Orange.classification.LogisticRegressionLearner())
        pred = cp.classification.InductiveClassifier(nc)
        res = cp.evaluation.run(pred, 0.1, cp.evaluation.CrossSampler(tab, 3))
        self.assertLess(res.time(), 10.0)

    def test_results_class(self):
        tab = Orange.data.Table('iris')
        nc = cp.nonconformity.InverseProbability(Orange.classification.LogisticRegressionLearner())
        pred = cp.classification.CrossClassifier(nc, 3)

        r = cp.evaluation.run(pred, 0.5, cp.evaluation.RandomSampler(tab, 2, 1))
        self.assertGreater(r.empty_criterion(), 0.0)
        r = cp.evaluation.run(pred, 0.1, cp.evaluation.RandomSampler(tab, 2, 1))
        self.assertGreater(r.singleton_criterion(), 0.8)
        r = cp.evaluation.run(pred, 0.01, cp.evaluation.RandomSampler(tab, 2, 1))
        self.assertGreater(r.multiple_criterion(), 0.1)

        r = cp.evaluation.run(pred, 0.1, cp.evaluation.RandomSampler(tab, 2, 1))
        self.assertGreater(r.singleton_correct(), 0.8)
        self.assertGreater(r.confidence(), 0.9)
        self.assertGreater(r.credibility(), 0.4)

        self.assertAlmostEqual(sum(r.accuracy(class_value=c) for c in tab.domain.class_var.values)/3,
                               r.accuracy(),
                               delta=1e-2)

        self.assertGreater(r.accuracy(eps=0.01), r.accuracy())
        self.assertGreater(r.accuracy(), r.accuracy(eps=0.2))

    def test_results_regr(self):
        tab = Orange.data.Table('housing')[:300]
        nc = cp.nonconformity.AbsErrorKNN(Orange.distance.Euclidean(), 10, average=True)
        pred = cp.regression.InductiveRegressor(nc)
        train, test = next(cp.evaluation.RandomSampler(tab, 4, 1))
        r1 = cp.evaluation.run_train_test(pred, 0.1, train, test)
        r5 = cp.evaluation.run_train_test(pred, 0.5, train, test)
        self.assertGreater(r1.median_range(), r5.median_range())
        self.assertGreater(r1.mean_range(), r5.mean_range())
        self.assertGreater(r1.interdecile_range(), r5.interdecile_range())
        self.assertGreater(r1.interdecile_mean(), r5.interdecile_mean())
        self.assertGreater(r1.std_dev(), r5.std_dev())

    def test_pickle(self):
        import pickle, os
        train, test = next(cp.evaluation.RandomSampler(Orange.data.Table('iris'), 2, 1))
        train, cal = next(cp.evaluation.RandomSampler(train, 2, 1))
        nc = cp.nonconformity.InverseProbability(Orange.classification.NaiveBayesLearner())
        ic = cp.classification.InductiveClassifier(nc)
        ic.fit(train, cal)
        pred1 = ic(test[0], 0.1)
        with open('temp.cp','wb') as f:
            pickle.dump(ic, f)
        with open('temp.cp','rb') as f:
            ic2 = pickle.load(f)
        os.remove('temp.cp')
        pred2 = ic2(test[0], 0.1)
        self.assertListEqual(pred1, pred2)
