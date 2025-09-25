"""
Highlevel objects to organize and handle calibration dataset and regression.
"""

from __future__ import annotations

from collections import abc
from dataclasses import KW_ONLY, dataclass, field
from types import EllipsisType
from typing import Any, Iterator, Literal, NamedTuple, Self

import numpy as np
import pandas as pd
from sklearn import metrics as skmetrics
from sklearn import model_selection
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression

from . import metrics
from ._typing import Array, Vector
from .util import copy_regression_object, parse_wavelength


@dataclass(frozen=True)
class CalibrationDataset:
    """Calibration data containing spectra and reference values for multiple samples.

    Parameters
    ----------
    spectra: numpy array (shape=(S, W))
        Spectral measurement with W wavelengths for S samples.
    reference: numpy array (shape=(S, R))
        Reference values.
    wavelengths: numpy array (shape=(W, )), optional
        Wavelengths corresponding to the spectra
        If not given, an [1, W] range will be assigned.
    reference_names: tuple of str, (len=R), optional
        Name of each reference value.
        If not given, an [1, R] range will be assigned.
    sample_names: tuple of str, (len=S), optional
        Name of each sample.
        If not given, an [1, S] range will be assigned.
    metadatada: dict
        Extra information about the dataset.
    """

    spectra: Array
    reference: Array
    _: KW_ONLY

    wavelengths: Vector = field(default_factory=np.ndarray)
    reference_names: tuple[str, ...] = ()
    sample_names: tuple[str, ...] = ()

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        S = self.spectra.shape[0]
        W = self.spectra.shape[1]
        R = self.reference.shape[1]

        if S != self.reference.shape[0]:
            raise ValueError(
                f"The first dimension of spectra and reference, which correspond to the number of samples, do not match ({S} vs {self.reference.shape[0]})"
            )

        if len(self.sample_names) == 0:
            object.__setattr__(self, "sample_names", tuple(range(1, S + 1)))
        elif S != len(self.sample_names):
            raise ValueError(
                f"The first dimension of spectra, which correspond to the number of samples, and length of `sample_names` do not match ({S} vs {len(self.sample_names)})"
            )

        if len(self.wavelengths) == 0:
            object.__setattr__(self, "wavelengths", tuple(range(1, W + 1)))
        elif W != len(self.wavelengths):
            raise ValueError(
                f"The second dimension of spectra, which correspond to the number of wavelengths, and length of `wavelength` do not match ({W} vs {len(self.wavelengths)})"
            )

        if len(self.reference) == 0:
            object.__setattr__(self, "reference_names", tuple(range(1, R + 1)))
        elif R != len(self.reference_names):
            raise ValueError(
                f"The second dimension of spectra, which correspond to the number of reference, and length of `wavelength` do not match ({R} vs {len(self.reference_names)})"
            )

    @classmethod
    def from_dataframe(cls, dataframe: pd.DataFrame) -> "CalibrationDataset":
        """Create calibration dataset from pandas dataframe.

        The dataframe structure is the following
        1. Samples are given by rows.
        2. Wavelengths of the spectra and reference measurements are given by column.
        3. If a column named `sample` is found, it will be used to label each sample.
        4. Remaining columns will be parsed in the following manner
            a. If a number or a number followed by `nm`, it will interpreted as wavelength.
            b. If not a number, it will interpreted as reference.
        4. It CAN contain a dict in `attrs` from which the metada
        """
        wl_col = []
        rn_col = []

        wavelengths = []
        sample_names = []
        reference_names = []

        for col in dataframe.columns:
            if isinstance(col, (float, int)):
                wavelengths.append(col)
                wl_col.append(col)
            elif isinstance(col, str):
                if col.strip() == "sample":
                    sample_names = dataframe[col].to_list()
                    continue
                try:
                    wavelengths.append(parse_wavelength(col))
                    wl_col.append(col)
                except:
                    reference_names.append(col.replace(r"\n", "").strip())
                    rn_col.append(col)
            else:
                raise ValueError(f"Cannot interpret column name of type {type(col)}")

        if not wl_col:
            raise ValueError(
                f"No suitable column for wavelength was found within {dataframe.columns}"
            )
        if not rn_col:
            raise ValueError(
                f"No suitable column for reference was found within {dataframe.columns}"
            )

        return CalibrationDataset(
            dataframe[wl_col].to_numpy(dtype=np.float64),
            dataframe[rn_col].to_numpy(dtype=np.float64),
            wavelengths=np.asarray(wavelengths, dtype=np.float64),
            reference_names=tuple(reference_names),
            sample_names=tuple(sample_names),
            metadata=dict(dataframe.attrs),
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Convert calibration dataset to a pandas dataframe."""

        df = pd.DataFrame(
            np.hstack(
                (
                    np.atleast_2d(np.asarray(self.sample_names)).T,
                    self.spectra,
                    self.reference,
                )
            ),
            columns=("sample",)
            + tuple(f"{wl} nm" for wl in self.wavelengths)
            + self.reference_names,
        )
        for k, v in self.metadata.items():
            df.attrs[k] = v

        return df

    def __str__(self):
        return (
            f"# Samples: {len(self.sample_names)}\n"
            f"# Wavelengths: {len(self.wavelengths)}\n"
            f"# References: {len(self.reference_names)}\n"
            f"Reference names: {self.reference_names}"
        )

    __repr__ = __str__

    @property
    def X(self) -> Array:
        """Predictor values."""
        return self.spectra

    @property
    def Y(self) -> Array:
        """Target values."""
        return self.reference

    def get_Xy(self, reference_name: str = "") -> tuple[Array, Vector]:
        """Get predictor, and target value for a single reference."""
        if reference_name not in self.reference_names:
            raise ValueError(f"{reference_name} must be either {self.reference_names}")
        return self.spectra, self.reference[
            :, self.reference_names.index(reference_name)
        ]

    def __getitem__(
        self, include_rows: abc.Iterable[bool] | EllipsisType = Ellipsis, /
    ) -> Self:
        """A dataset object with a subset of rows."""
        return self.__class__(
            self.spectra[include_rows, :],
            self.reference[include_rows, :],
            wavelengths=self.wavelengths,
            reference_names=self.reference_names,
            sample_names=list(np.asarray(self.sample_names)[include_rows]),
            metadata=dict(self.metadata),
        )

    def select(
        self,
        reference_name: str = "",
        include_rows: abc.Iterable[bool] | EllipsisType = Ellipsis,
    ) -> CalibrationDatasetST:
        """Create CalibrationDatasetST object, with the same predictors
        and a single target.

        Parameters
        ----------
        reference_name: str, optional

            If not given, the first of reference names will be used.
        include_rows: iterable of bool or ellipsis, optional
            If not given, all rows will be used.
        """
        names = self.reference_names

        if not reference_name:
            if len(names) != 1:
                raise ValueError(
                    f"Provide a the name of the reference to use. Valid options: {names}"
                )
            else:
                reference_name = names[0]
        elif reference_name not in names:
            raise ValueError(
                f"{reference_name} is not a valid reference name. Valid options: {names}"
            )

        k = self.reference_names.index(reference_name)

        return CalibrationDatasetST(
            self.spectra[include_rows, :],
            self.reference[include_rows, k],
            wavelengths=self.wavelengths,
            reference_name=reference_name,
            sample_names=list(np.asarray(self.sample_names)[include_rows]),
            metadata=dict(self.metadata),
        )


@dataclass(frozen=True)
class CalibrationDatasetST:
    """Single target calibration data containing spectra and reference values for multiple samples.

    Parameters
    ----------
    spectra: numpy array (shape=(S, W))
        Spectral measurement with W wavelengths for S samples.
    reference: numpy array (shape=(S, R))
        Reference values.
    wavelengths: numpy array (shape=(W, )), optional
        Wavelengths corresponding to the spectra
        If not given, an [1, W] range will be assigned.
    reference_names: tuple of str, (len=R), optional
        Name of each reference value.
        If not given, an [1, R] range will be assigned.
    sample_names: tuple of str, (len=S), optional
        Name of each sample.
        If not given, an [1, S] range will be assigned.
    metadatada: dict
        Extra information about the dataset.
    """

    spectra: Array
    reference: Vector
    _: KW_ONLY

    wavelengths: Vector = field(default_factory=np.ndarray)
    reference_name: str = ""
    sample_names: tuple[str, ...] = ()

    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self):
        return (
            f"# Samples: {len(self.sample_names)}\n"
            f"# Wavelengths: {len(self.wavelengths)}\n"
            f"Reference name: {self.reference_name}"
        )

    __repr__ = __str__

    @property
    def X(self) -> Array:
        """Predictor values."""
        return self.spectra

    @property
    def y(self) -> Vector:
        """Target values."""
        return self.reference

    def __getitem__(self, k: abc.Iterable[bool] | EllipsisType = Ellipsis, /) -> Self:
        """A single target dataset object with a subset of rows."""
        return self.__class__(
            self.spectra[k, :],
            self.reference[k],
            wavelengths=self.wavelengths,
            reference_name=self.reference_name,
            sample_names=list(np.asarray(self.sample_names)[k]),
            metadata=dict(self.metadata),
        )


class PLSCalibration(PLSRegression):
    """A PLS based transformation for a single reference observable that
    can be fitted from calibration data and provides uncertainty calculations.

    Uncertainty is calculated with several methods described in `predict_mu_std`.
    """

    #: Calibration info, is None when the regression is not fitted.
    #: Per sample values are stored in the rows.
    #: Global values are stored in the attrs dict.
    calibration_info: pd.DataFrame | None = None

    def fit_calibration_data(self, calibration_data: CalibrationDatasetST) -> Self:
        """Fit model to single target calibration dataset.

        Parameters
        ----------
        calibration_data: CalibrationDatasetST
            A single target calibration dataset.

        Returns
        -------
        fitted_model: PLSCalibration
            Fitted model, with calibration info filled.
        """

        X, y = calibration_data.X, calibration_data.y
        model = self.fit(X, y)

        n_samples = len(y)

        y_pred = model.predict(X)

        lev = metrics.leverage(self.x_scores_)

        info = pd.DataFrame(
            dict(
                sample=calibration_data.sample_names,
                y=y,
                y_pred=y_pred,
                reduced_hotelling_t2=metrics.reduced_hotelling_t2(
                    self.x_scores_, self.n_components, self.n_features_in_
                ),
                reduced_q_residual=metrics.reduced_q_residual(
                    X, self.x_scores_, self.x_loadings_
                ),
                leverage=lev,
                studentized_y_residual=metrics.studentized_y_residual(
                    y, y_pred, lev, self.n_components
                ),
            )
        )

        info.attrs["reference_name"] = calibration_data.reference_name
        info.attrs["X"] = X
        info.attrs["features"] = calibration_data.wavelengths

        info.attrs["n_samples"] = n_samples
        info.attrs["rmsec"] = np.sqrt(skmetrics.mean_squared_error(y, y_pred))
        info.attrs["r2"] = skmetrics.r2_score(y, y_pred)

        self.calibration_info = info

        return model

    def empty_copy(self) -> Self:
        """Create an new, unfitted, regression object."""
        obj = copy_regression_object(self)
        return obj

    PRED_METHOD = Literal[
        "Höskuldsson1988", "Faber2002", "BootstrapResiduals", "NoiseAddition"
    ]

    def predict_mu_std(
        self, X_new: Array, method: PRED_METHOD, *, exp_V_DY: float | None = None
    ) -> tuple[float, float]:
        """Predict targets and their uncertainties from the spectra of given samples
        as the mean and standard deviation of the predictions of all models.

        Implemented methods are:
        - Höskuldsson1988: see [Zhang2009] Ec. 13
        - Faber2002: see [Faber2002] Ec. 9
        - BootstrapResiduals: see [Fernández2003], Section 2.3
        - NoiseAddition: see [Fernández2003], Section 2.4

        Parameters
        ----------
        X_new: Array
            Spectrum of 1 or more samples from which the reference
            will be predicted.

        Returns
        -------
        mu: float
            Prediction value.
        unc: float
            Uncertainty of the prediction.
        """

        if self.calibration_info is None:
            raise ValueError("Model is not fitted yet.")

        X_new = np.atleast_2d(X_new)

        scores = self.transform(X_new)
        mu = self.predict(X_new)

        ho = metrics.leverage(self.x_scores_, scores)

        # Names used in several papers.
        # A             number of PLS factors (components)
        # N             number of samples in the calibration set.
        # P             number of features (wavelengths).
        # ho            leverage of the new observation.
        # V_e           variance due to the unmodeled part of yo (model error even when no measurement exists in both X and y).
        # V_DX          X-residual variance.
        # V_Dy          y-residual variance.
        # V_Xo          unexplained variance in the X.

        # V_dy_val      y-residual variance validation set.
        # V_X_tol_val   X-residual average  in the validation set.

        A = self.n_components
        N = self.calibration_info.attrs["n_samples"]
        P = self.n_features_in_

        cal_res = self.calibration_info["y"] - self.calibration_info["y_pred"]
        V_Dy = np.var(cal_res)

        X = self.calibration_info.attrs["X"]
        y = self.calibration_info["y"]

        REPEATS = 100

        match method:
            case "Höskuldsson1988":
                # [Zhang2009] Ec. 13 / Höskuldsson 1988
                var = V_Dy * (1 + ho + 1 / N)
            case "Faber2002":
                # [Faber2002] Ec. 9 and mean centered version of Ec. 10
                if exp_V_DY is None:
                    exp_V_DY = V_Dy
                MSEC = np.sum(cal_res**2) / (N - A - 1)
                var = (1 + ho) * MSEC - exp_V_DY
            case "BootstrapResiduals":
                values = np.asarray(
                    [
                        model.predict(X_new)
                        for model in metrics.yield_bootstrap_residuals_model(
                            self, X, y, repeats=REPEATS
                        )
                    ]
                )
                var = np.var(values, axis=0)
            case "NoiseAddition":
                values = np.asarray(
                    [
                        model.predict(X_new)
                        for model in metrics.yield_noise_addition_model(
                            self, X, y, repeats=REPEATS
                        )
                    ]
                )
                var = np.var(values, axis=0)
            case _:
                raise ValueError(f"Unknown method {method}")

        return mu, np.sqrt(var)


class PLSHitecCalibration(PLSRegression):
    """A PLS based transformation for a single reference observable that
    can be fitted from calibration data and provides uncertainty calculations.

    Uncertainty is calculated with the method described in `predict_mu_std`.
    """

    #: Calibration info, is None when the regression is not fitted.
    #: Per sample values are stored in the rows.
    #: Global values are stored in the attrs dict.
    calibration_info: pd.DataFrame | None = None

    #: Intermediate models.
    models: list[tuple[PLSRegression, LinearRegression]] | None = None

    def fit_calibration_data(self, calibration_data: CalibrationDatasetST) -> Self:
        """Fit model to single target calibration dataset.

        Parameters
        ----------
        calibration_data: CalibrationDatasetST
            A single target calibration dataset.

        Returns
        -------
        fitted_model: PLSCalibration
            Fitted model, with calibration info filled.
        """

        X, y = calibration_data.X, calibration_data.y
        model = super().fit(X, y)

        self.fit(X, y)

        n_samples = len(y)

        y_pred = model.predict(X)

        lev = metrics.leverage(self.x_scores_)

        info = pd.DataFrame(
            dict(
                sample=calibration_data.samples,
                y=y,
                y_pred=y_pred,
                reduced_hotelling_t2=metrics.reduced_hotelling_t2(
                    self.x_scores_, self.n_components, self.n_features_in_
                ),
                reduced_q_residual=metrics.reduced_q_residual(
                    X, self.x_scores_, self.x_loadings_
                ),
                leverage=lev,
                studentized_y_residual=metrics.studentized_y_residual(
                    y, y_pred, lev, self.n_components
                ),
            )
        )

        info.attrs["reference_name"] = calibration_data.reference_name
        info.attrs["X"] = X
        info.attrs["features"] = calibration_data.features

        info.attrs["n_samples"] = n_samples
        info.attrs["rmsec"] = np.sqrt(skmetrics.mean_squared_error(y, y_pred))
        info.attrs["r2"] = skmetrics.r2_score(y, y_pred)

        self.calibration_info = info

        return model

    def empty_copy(self):
        """Create an new, unfitted, regression object."""
        obj = copy_regression_object(self)
        return obj

    def fit(self, X: Array, y: Array) -> Self:
        splitv = model_selection.ShuffleSplit(n_splits=10, test_size=0.2)

        models: list[tuple[PLSRegression, LinearRegression]] = []

        for _split_number, (train_indx, test_indx) in enumerate(splitv.split(X, y)):
            pls = copy_regression_object(self, PLSRegression)

            # En HITEC
            # MCM = Modelo de calibración por cuadrados mínimos
            # MMV: Modelo de calibración multivariado (usualmente PCR, PLS, Integral)

            # xcal valor obtenido por método de referencia utilizado para el desarrollo del MMV, calibracion
            # ycal valor que se calcula a partir del MMV ya desarrollado, calibracion

            # xval valor obtenido por método de referencia utilizado para el desarrollo del MMV, validacion
            # yval valor que se calcula a partir del MMV ya desarrollado, validacion

            # yunk: valor que se calcula a partir del MMV ya desarrollado, y que está relacionado con el
            # espectro 𝑆𝑢𝑛𝑘 de una muestra desconocida a medir por el método NIR

            # 𝑥𝑝𝑟𝑒𝑑: valor que se calcula a partir del MCM ya desarrollado, ingresando al MCM con el valor 𝑦𝑣𝑎𝑙
            # 𝑦𝑝𝑟𝑒𝑑: valor que se calcula a partir del MCM ya desarrollado, ingresando al MCM con el valor 𝑥𝑣𝑎𝑙

            pls.fit(X[train_indx], y[train_indx].reshape(-1, 1))

            lr = LinearRegression()
            lr.fit(pls.predict(X[test_indx]), y[test_indx].reshape(-1, 1))

            models.append((pls, lr))

        self.models = models

        return self

    def predict(self, X: Array, copy: bool = True) -> Array:
        """Predict targets from the spectra of given samples.

        Parameters
        ----------
        X_new: Array
            Spectrum of 1 or more samples from which the reference
            will be predicted.

        Returns
        -------
        mu: float
            Prediction value.
        """
        return self.predict_mu_std(X)[0]

    def predict_mu_std(self, X_new: Array) -> tuple[float, float]:
        """Predict targets and their uncertainties from the spectra of given samples
        as the mean and standard deviation of the predictions of all models.

        Parameters
        ----------
        X_new: Array
            Spectrum of 1 or more samples from which the reference
            will be predicted.

        Returns
        -------
        mu: float
            Prediction value.
        unc: float
            Uncertainty of the prediction.
        """

        y = []
        for pls, lr in self.models:
            y.append(lr.predict(pls.predict(X_new)).flatten())

        stack = np.stack(y)

        return stack.mean(axis=0), stack.std(axis=0)


class PLSCalibrationEnsemble(NamedTuple):
    """A PLS based transformation for a single reference observable,
    build upon an ensemble of PLS transformation.
    """

    submodels: tuple[PLSRegression, ...]

    @classmethod
    def from_iterator(
        cls, model_template: PLSRegression, xy: Iterator[tuple[Array, Array]]
    ):
        """Construct PLSCalibrationEnsemble from a model template
        and an iterator of predictor and target values and provides
        uncertainty calculations.

        A PLSRegression is build from each element of the iterator.

        Uncertainty is calculated with the method described in `predict_mu_std`.
        """
        model_class = model_template.__class__
        kw = {
            k: getattr(model_template, k)
            for k in model_class._parameter_constraints.keys()
        }

        all_pls = []
        for x, y in xy:
            pls = model_class(**kw)
            all_pls.append(pls.fit(x, y))

        return cls(tuple(all_pls))

    def __call__(self, spectrum: Array) -> Array:
        return self.predict(spectrum)

    def predict(self, X_new: Array) -> Array:
        """Predict targets of given samples as the mean of all models.

        Parameters
        ----------
        X_new: Array
            Spectrum of 1 or more samples from which the reference
            will be predicted.

        Returns
        -------
        mu: float
            Prediction value.
        """
        return self.predict_mu_std(X_new)[0]

    def predict_mu_std(self, X_new: Array) -> tuple[Array, Array]:
        """Predict targets and their uncertainties from the spectra of given samples
        as the mean and standard deviation of the predictions of all models.

        Parameters
        ----------
        X_new: Array
            Spectrum of 1 or more samples from which the reference
            will be predicted.

        Returns
        -------
        mu: float
            Prediction value.
        unc: float
            Uncertainty of the prediction.
        """
        if X_new.ndim == 1:
            X_new.shape = (1, X_new.size)

        output = []
        for t in self.submodels:
            output.append(t.predict(X_new))

        stack = np.stack(output)
        return stack.mean(axis=0), stack.std(axis=0)
