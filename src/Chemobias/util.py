"""
Useful general functions.
"""

from __future__ import annotations

from typing import Any

from sklearn.cross_decomposition import PLSRegression


def copy_regression_object[T: PLSRegression](
    obj: T, cls: type[PLSRegression] | None = None, **extra_kwargs: Any
) -> T:
    """Create an new, unfitted, regression object."""

    if cls is None:
        cls = obj.__class__

    kw = {k: getattr(obj, k) for k in cls._parameter_constraints.keys()}
    kw.update(extra_kwargs)
    return cls(**kw)


def parse_wavelength(value: str) -> float:
    value = value.replace(r"\n", " ").strip()
    if value.endswith("nm"):
        return float(value[:-2])
    elif value.endswith("nm."):
        return float(value[:-3].strip())
    else:
        return float(value)
