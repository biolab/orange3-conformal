Introduction
============

The Conformal Predictions add-on expands the Orange library with
implementations of algorithms from the theoretical framework of conformal
predictions (CP) to obtain error calibration under classification and
regression settings.

In contrast with standard supervised machine learning, which for a given new
data instance typically produces :math:`\hat{y}`, called a *point prediction*, here we are
interested in making a *region prediction*. For example, with conformal
prediction we could produce a 95% prediction region --- a set :math:`\Gamma^{0.05}`
that contains the true label :math:`y` with probability at least 95%.  In the case of
regression, where :math:`y` is a number, :math:`\Gamma^{0.05}` is typically an interval around :math:`\hat{y}`.
In the case of classification, where :math:`y` has a limited number of possible values,
:math:`\Gamma^{0.05}` may consist of a few of these values or, in the ideal case, just one.
For a more detailed explanation of the conformal predictions theory refer to the paper [Vovk08]_
or the book [Shafer05]_.

In this library the final method for conformal predictions is obtained by
selecting a combination of pre-prepared components.  Starting with the learning
method (either classification or regression) used to fit predictive models, we
need to link it with a suitable nonconformity measure and use them together in
a selected conformal predictions procedure: transductive, inductive or cross.
These CP procedures differ in the way data is split and used for training the
predictive model and calibration, which computes the distribution of
nonconformity scores used to evaluate possible new predicitions. Inductive CP
requires two disjoint data sets to be provided - one for training, the other
for calibration. Cross CP uses a single training data set and automatically
prepares k different splits into training and calibration sets in the same
manner as k-fold crossvalidation. Transductive CP on the other hand does not
need a separate calibration set at all, but retrains the model with a new test
instance included for each of its possible labels and compares the
nonconformity to those of the labelled instances. This allows it to use the
complete training set, but makes it computationally more expensive.

Sections below will explain how to use the implemented methods from this
library through practical examples and use-cases. For a detailed documentation
of implemented methods and classes along with their parameters consult the
:ref:`library_reference`.  For more code examples, take a look at the tests
module.


References
----------

.. [Vovk08] Glenn Shafer, Vladimir Vovk. Tutorial on Conformal Predictions. *Journal of Machine Learning Research* 9 (2008) 371-421
.. [Shafer05] Vladimir Vovk, Alex Gammerman, and Glenn Shafer. *Algorithmic Learning in a Random World*. Springer, New York, 2005.

