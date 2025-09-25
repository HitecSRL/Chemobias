"""
Functions to calculate useful PLS metrics.

While these values can be calculated from the data and the PLS regression,
functions in this module use precalculated metrics to avoid multiple recalculation.
"""

from __future__ import annotations

from typing import Any, Generator

import numpy as np
from numpy.random import RandomState
from scipy import stats
from sklearn import metrics, model_selection
from sklearn import utils as skutils
from sklearn.base import check_is_fitted
from sklearn.cross_decomposition import PLSRegression

from ._typing import Array, Scalar, Vector
from .util import copy_regression_object


def hotelling_t2(
    scores: Array, n_components: int, n_features: int, confidence_level: float = 0.95
) -> tuple[Vector, Scalar]:
    """Calculate the Hotelling's t-squared statistic and
    the critical value for a given confidence level.

    Samples with large Hotelling’s T-squared values display deviations
    within the model.

    Parameters
    ----------
    scores: numpy array
        Scores of the regression.
        Typically the x_scores of a regression.
    n_components: int
        Number of components in the PLS regression.
    n_features: int
        Number of features in the PLS regression.
        In spectroscopy, corresponds to the number of wavelengths.
    confidence_level: float, optional (default=0.95)
        lower tail probability.

    Returns
    -------
    t2: float
        Hotelling's t-squared.
    t2_cut: float
        Critical value for a given confidence level.

    See Also
    --------
    reduced_hotelling_t2 for a normalized version.

    References
    ----------
    [Bu2013] Ec. 2 and 3
    """

    nc = n_components
    nf = n_features

    value = (
        nc
        * (nf - 1)
        / (nf - nc)
        * stats.f.ppf(q=confidence_level, dfn=nc, dfd=(nf - nc))
    )

    return np.sum((scores / np.std(scores, axis=0)) ** 2, axis=1), value


def reduced_hotelling_t2(
    scores: Array, n_components: int, n_features: int, confidence_level: float = 0.95
) -> Vector:
    """Calculate the Hotelling's t-squared statistic
    normalized by the critical value for a given confidence level.

    Samples with large Hotelling’s T-squared values display deviations
    within the model.

    Parameters
    ----------
    scores: numpy array
        Scores of the regression.
        Typically the x_scores of a regression.
    n_components: int
        Number of components in the PLS regression.
    n_features: int
        Number of features in the PLS regression.
        In spectroscopy, corresponds to the number of wavelengths.
    confidence_level: float, optional (default=0.95)
        lower tail probability.

    Returns
    -------
    reduced_t2: float
        Reduced hotelling's t-squared.

    See Also
    --------
    hotelling_t2 for a unnormalized version.

    References
    ----------
    [Bu2013] Ec. 2 and 3
    """

    t2, t2_cut = hotelling_t2(scores, n_components, n_features, confidence_level)

    return t2 / t2_cut


def q_residual(
    X: Array, scores: Array, loadings: Array, confidence_level: float = 0.05
) -> tuple[Vector, float]:
    """Calculate the Q-residuals for each sample and
    the critical value for a given confidence level.

    Those with large Q-residuals are the ones that are not well explained
    by the model.

    Parameters
    ----------
    X: numpy array
        Input data.
    scores: numpy array
        Scores of the regression.
        Typically the x_scores of a regression.
    loadings: int
        x_loadings of a regression.
    confidence_level: float, optional (default=0.95)
        lower tail probability.

    Returns
    -------
    qr: float
        Q-residuals.
    qcut: float

    See Also
    --------
    reduced_q_residual for a normalized version.

    References
    ----------
    [Bu2013] Ec. 5
    """

    T = scores
    P = loadings

    err = X - np.dot(T, P.T)

    Q = np.sum(err**2, axis=1)

    i = np.max(Q) + 1
    while 1 - np.sum(Q > i) / np.sum(Q > 0) > confidence_level:
        i -= 1

    return Q, i


def reduced_q_residual(
    X: Array, scores: Array, loadings: Array, confidence_level: float = 0.95
) -> Vector:
    """Calculate the Q-residuals for each sample and value
    normalized by the critical value for a given confidence level.

    Those with large Q-residuals are the ones that are not well explained
    by the model.

    Parameters
    ----------
    X: numpy array
        Input data.
    scores: numpy array
        Scores of the regression.
        Typically the x_scores of a regression.
    loadings: int
        Loadings of the regression.
        Typically the x_loadings of a regression.
    confidence_level: float, optional (default=0.95)
        lower tail probability.

    Returns
    -------
    reduced_qr: float
        Reduced Q-residuals.

    See Also
    --------
    q_residual for a unnormalized version.

    References
    ----------
    [Bu2013] Ec. 5
    """

    q, qcut = q_residual(X, scores, loadings, confidence_level)

    return q / qcut


def leverage(score_cal: Array, score_pred: Array | None = None) -> Vector | Scalar:
    """Calcualte the leverage statistics.

    Leverage is measure of how far away the independent variable values
    of an observation are from those of the other observations.

    High-leverage points, if any, are outliers with respect
    to the independent variables

    Parameters
    ----------
    score_cal: numpy array
        Scores of the calibration regression.
        Prediction, Typically the x_scores of a regression.
    score_pred: numpy array, optional (default=None)
        Scores of the prediction regression.
        If `None`, the calibration will be used.
    confidence_level: float, optional (default=0.95)
        lower tail probability.

    Returns
    -------
    leverage: numpy array
        leverage for each independent value.

    References
    ----------
    [Bu2013] Ec. 4
    """
    if score_pred is None:
        score_pred = score_cal

    return np.diag(score_pred @ np.linalg.pinv(score_cal))


def studentized_y_residual(
    y: Vector, y_pred: Vector, lev: Vector, n_components: int
) -> Vector:
    """Calculate the studentized y residuals

    This metric is useful to assess the influence and fit of individual
    data points. These residuals are scaled by their estimated standard
    deviation, which accounts for potential differences in variability
    across observations.

    This makes them particularly useful for identifying outliers and
    assessing the assumption of homoscedasticity in regression.

    Parameters
    ----------
    y: numpy array
        Target calibration vector.
    y_pred: numpy array
        Prediction vector.
    lev: numpy array
        leverage vector
    n_components: int
        Number of components in the PLS regression.

    Returns
    -------
    stdt_y_res: numpy array
        Studentized y residuals.

    References
    ----------
    https://wiki.eigenvector.com/index.php?title=Pls
    """

    n_samples = len(y)
    residuals = y - y_pred

    rss = np.sum(residuals**2)
    degrees_of_freedom = n_samples - n_components
    residual_standard_error = np.sqrt(rss / degrees_of_freedom)

    return residuals / (residual_standard_error * np.sqrt(1 - lev))


def cross_validation(
    model: PLSRegression, X: Array, y: Vector, cv: Any
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Cross validate fit, returing average and standard deviation of
    train and test set.

    The score used is the root mean squared error (RMSE).

    Parameters
    ----------
    model: PLSRegression
        PLS regression object.
    X: numpy array
        Training vectors.
    y: numpy array
        Target vectors
    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.

        See `cv` of `sklearn.model_selection.cross_validate` for more.

    Returns
    -------
    train_score: (float, float)
        mean and standard deviation of the train RMSE.
    test_score: (float, float)
        mean and standard deviation of the test RMSE.
    """

    score_dct = model_selection.cross_validate(
        model, X, y, cv=cv, scoring="neg_mean_squared_error", return_train_score=True
    )
    train_score = np.sqrt(-score_dct["train_score"])
    test_score = np.sqrt(-score_dct["test_score"])

    return (
        (train_score.mean(), train_score.std()),
        (test_score.mean(), test_score.std()),
    )


def components_cross_validation(
    model: PLSRegression,
    X: Array,
    y: Vector,
    max_components: int,
    test_size: float | int = 0.2,
) -> tuple[Vector, Array, Array, Vector]:
    """Cross validate fit as a function of the number of components.

    The score used is the root mean squared error (RMSE).

    High q-latent can be used to decide which component could be discarded.

    Parameters
    ----------
    model: PLSRegression
        Object to be used as a template to build the PLSRegression object during the cross validation.
    X: numpy array
        Training vectors.
    y: numpy array
        Target vectors
    max_components: int
        Maximum number of components of the PLSRegression that will be cross validated.
    test_size:
        if float, should be between 0.0 and 1.0 and represent
        the proportion of the dataset to include in the train split.
        If int, represents the absolute number of train samples.

    Returns
    -------
    rmsec: numpy array (shape=(N, ))
        using the full calibration data
    rmsecv: numpy array (shape=(N, 2))
        mean and standard deviation of the train scores.
    rmsecv: numpy array (shape=(N, 2))
        mean and standard deviation of the test scores.
    qlatent: numpy array (shape=(N, ))
        q-latent metric
    """
    splitv = model_selection.ShuffleSplit(n_splits=10, test_size=test_size)

    rmsec = np.full(max_components - 1, np.nan)

    train_rmsecv = np.full((max_components - 1, 2), np.nan)
    test_rmsecv = np.full((max_components - 1, 2), np.nan)

    for ndx, n_components in enumerate(np.arange(1, max_components)):
        pls = copy_regression_object(model, PLSRegression, n_components=n_components)
        a, b = cross_validation(pls, X, y, splitv)

        train_rmsecv[ndx, :] = a
        test_rmsecv[ndx, :] = b

        pls.fit(X, y)
        rmsec[ndx] = np.sqrt(metrics.mean_squared_error(y, pls.predict(X)))

    ##########
    # This is useful to calculate Q

    # [Abdi2010], Ec. 11
    K = X.shape[1]  # dependent variables
    I = X.shape[0]  # observations
    rmsec0 = K * (I - 1)

    rmsec = np.asarray(rmsec, dtype=np.float64)
    train_rmsecv = np.asarray(train_rmsecv, dtype=np.float64)
    test_rmsecv = np.asarray(test_rmsecv, dtype=np.float64)

    # TODO: not clear if one should divide by rmsec or train_rmsecv
    Q = 1 - test_rmsecv[1:, 0] / train_rmsecv[:-1, 0]
    # Q = 1 - test_rmsecv[1:, 0] / rmsec[:-1]
    Q = np.insert(Q, 0, 1 - test_rmsecv[0, 0] / rmsec0, axis=0)

    return rmsec, train_rmsecv, test_rmsecv, Q


def vip(fitted_model: PLSRegression) -> Vector:
    """Calculate Variable Importance in Projection (VIP)

    It estimates the importance of each variable in the projection
    used in a PLS model and is often used for variable selection.
    A variable with a VIP Score close to or greater
    than 1 (one) can be considered important in given model.

    Variables with VIP scores significantly less than 1 (one) are less important
    and might be good candidates for exclusion from the model.

    Parameters
    ----------
    fitted_model: PLSRegression
        A fitted regression object.

    Returns
    -------
    vip: numpy array
        The VIP metric for each feature (wavelength in spectroscopy)

    References
    ----------
    [Chong2005] Ec. 2
    """

    check_is_fitted(fitted_model)

    t = fitted_model.x_scores_
    w = fitted_model.x_weights_
    q = fitted_model.y_loadings_
    features_, _ = w.shape
    vip = np.zeros(shape=(features_,))
    inner_sum = np.diag(t.T @ t @ q.T @ q)
    SS_total = np.sum(inner_sum)
    vip = np.sqrt(features_ * (w**2 @ inner_sum) / SS_total)

    return vip


def yield_bootstrap_residuals_model(
    fitted_model: PLSRegression,
    X: Array,
    y: Vector,
    repeats: int = 1000,
    seed: int | RandomState | None = None,
) -> Generator[PLSRegression, Any, None]:
    """Generate multiple predictions for a new observation
    by boostrapping the residuals of a calibration dataset.

    Parameters
    ----------
    fitted_model: PLSRegression
        A fitted regression object.
    X: numpy array
        Training vectors.
    y: numpy array
        Target vectors
    repeats : int, optional (default=1000)
        Number of samples to generate.
    seed: int, RandomState instance or None, default=None
        Determines random number generation for shuffling the data.
        Pass an int for reproducible results across multiple function calls.

    Yields
    ------
    bootstrapped_model: PLSRegression
        A model trained with data generated
        by boostrapping the residuals of a calibration dataset.

    See Fernández2003, Section 2.3
    """

    check_is_fitted(fitted_model)

    y_pred = fitted_model.predict(X)
    residual = y - y_pred

    new_model = copy_regression_object(fitted_model, PLSRegression)

    for r in skutils.resample(residual, n_samples=repeats, random_state=seed):
        y_boot = y_pred + r
        new_model.fit(X, y_boot)
        yield new_model


def yield_noise_addition_model(
    fitted_model: PLSRegression,
    X: Array,
    y: Vector,
    repeats: int = 1000,
    rng: np.random.Generator | None = None,
) -> Generator[PLSRegression, Any, None]:
    """Generate multiple predictions for a new observation
    by adding noise with the standard deviation of the
    calibration residuals.

    Parameters
    ----------
    fitted_model: PLSRegression
        A fitted regression object.
    X: numpy array
        Training vectors.
    y: numpy array
        Target vectors
    repeats : int, optional (default=1000)
        Number of samples to generate.
    rng: np.random.Generator None, default=None
        Determines random number generation for shuffling the data.
        Pass an int for reproducible results across multiple function calls.

    Yields
    ------
    noise_addition_model: PLSRegression
        A model trained with data generated
        by adding noise.

    See Fernández2003, Section 2.4
    """

    check_is_fitted(fitted_model)

    if rng is None:
        rng = np.random.default_rng()

    y_pred = fitted_model.predict(X)
    residual = y - y_pred

    stdev = np.std(residual)

    new_model = copy_regression_object(fitted_model, PLSRegression)

    for _ in range(repeats):
        y_rand = rng.normal(y, stdev)
        new_model.fit(X, y_rand)
        yield new_model
