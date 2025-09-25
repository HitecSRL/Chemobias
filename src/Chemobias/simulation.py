"""
Functions to simulate spectra.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable

import numpy as np

from ._typing import Array, Scalar, Vector
from .highlevel import CalibrationDatasetST


def _raised_cosine(z: Any, s: Any) -> Any:
    return (1 + np.cos(z * np.pi)) / 2 / s


def raised_cosine(x: Any, mu: Scalar, s: Scalar) -> Any:
    z = (x - mu) / s
    if isinstance(z, float):
        if abs(z) > 1:
            return 0.0
        else:
            return _raised_cosine(z, s)

    out = np.zeros_like(z)
    valid = abs(z) < 1
    out[valid] = _raised_cosine(z[valid], s)
    return out


def simulated_spectra_bump(
    wl: Vector, c: float, f1: float, baseline: Vector | float = 0.0
) -> Vector:
    bump1 = c * raised_cosine(wl, 1100, 50)

    return baseline + f1 * bump1


def simulated_spectra_shift(
    wl: Vector, c: float, f1: float, baseline: Vector | float = 0.0
) -> Vector:
    bump1 = raised_cosine(wl, 950 + c * 500, 50)

    return baseline + f1 * bump1


def simulated_spectra_exchange(
    wl: Vector, c: float, f1: float, f2: float, baseline: Vector | float = 0.0
) -> Vector:
    bump1 = c * raised_cosine(wl, 1100, 50)
    bump2 = (1 - c) * raised_cosine(wl, 1400, 50)

    return baseline + f1 * bump1 + f2 * bump2


def simulated_multi_spectra_shift(
    wl: Vector, c: float, f1: float, baseline: Vector | float = 0.0
) -> Vector:
    bump1 = raised_cosine(wl, 950 + c * 500, 15 * (-c + 2))
    bump2 = raised_cosine(wl, 1450 - c * 500, 15 * (c + 1))
    bump3 = c * np.cos(2 * np.pi * wl / 100 + c * 2 * np.pi)

    return baseline + f1 * bump1 - f1 / 4 * bump2 + 0.1 * bump3


def simulated_multi_exchange(
    wl: Vector, c: float, f1: float, baseline: Vector | float = 0.0
) -> Vector:
    bump1 = c * raised_cosine(wl, 1050, 50)
    bump2 = (1 - c) * raised_cosine(wl, 1100, 60)

    bump3 = 1.1 * c**2 * raised_cosine(wl, 1200 - 10 * c, 43)
    bump4 = 1.1 * (1 - c) ** 2 * raised_cosine(wl, 1260 + 10 * c, 60)

    bump5 = 0.1 * np.sqrt(c) * raised_cosine(wl, 1350 * c, 25)

    return 0 * baseline + f1 * (bump1 + bump2 + bump3 + bump4 + bump5)


WL = np.arange(950, 1350).astype(np.float64)
BASELINE = np.exp(-(WL - 400) / 500)
MODEL1 = partial(simulated_spectra_bump, wl=WL, f1=10, baseline=BASELINE)
MODEL2 = partial(simulated_spectra_shift, wl=WL, f1=10, baseline=BASELINE)
MODEL3 = partial(simulated_spectra_exchange, wl=WL, f1=10, f2=10, baseline=BASELINE)
MODEL4 = partial(simulated_multi_spectra_shift, wl=WL, f1=10, baseline=BASELINE)
MODEL5 = partial(simulated_multi_exchange, wl=WL, f1=10, baseline=BASELINE)


def simulate_calibration_data(
    model: Callable[
        [
            float,
        ],
        Array,
    ],
    n_samples: int,
    generator: np.random.Generator | None = None,
) -> CalibrationDatasetST:
    if generator is None:
        generator = np.random.default_rng()

    y = generator.uniform(0.4, 0.9, 60)

    X = np.stack([model(c=c) for c in y])

    return CalibrationDatasetST(
        X,
        y,
        wavelengths=WL,
        reference_name="concentration",
        sample_names=tuple([str(ndx) for ndx in range(1, n_samples + 1)]),
    )
