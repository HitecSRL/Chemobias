"""
Functions to visualize data and regressions.
"""

from __future__ import annotations

import numpy as np
from matplotlib import colors, figure
from matplotlib import pyplot as plt

from . import metrics
from ._typing import Array, Vector
from .highlevel import CalibrationDataset, CalibrationDatasetST, PLSCalibration

SIZE_SCALING = 2


def plot_calibration_data_spectra(
    ax: figure.Axes, wavelengths: Array, spectra: Array, reference_values: Array
) -> figure.Colorbar:
    """Plot spectra."""

    lims = reference_values.min(), reference_values.max()
    norm = colors.Normalize(vmin=lims[0], vmax=lims[1])
    cbar = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.viridis),
        orientation="vertical",
        ax=ax,
    )

    lines = ax.plot(wavelengths, spectra.T)
    for line, reference_value in zip(lines, reference_values):
        line.set_color(plt.cm.viridis(norm(reference_value)))

    return cbar


def plot_calibration_data(calibration_data: CalibrationDatasetST):
    """Plot calibration data."""

    spectra, reference_values = calibration_data.X, calibration_data.y

    wavelengths = calibration_data.wavelengths
    samples = calibration_data.sample_names

    fig, (ax_sp, ax_ref) = plt.subplots(
        1,
        2,
        sharey=False,
        layout="tight",
        figsize=(11.69 / SIZE_SCALING, 8.27 / SIZE_SCALING),
    )

    ax_ref.scatter(samples, reference_values, c=reference_values)
    ax_ref.set_xlabel("Sample")
    ax_ref.set_yticklabels([])

    cbar = plot_calibration_data_spectra(ax_sp, wavelengths, spectra, reference_values)

    if isinstance(wavelengths, np.ndarray) and np.issubdtype(
        wavelengths.dtype, np.number
    ):
        ax_sp.set_xlabel("Wavelength / nm")
    else:
        ax_sp.set_xlabel("Wavelength")

    cbar.ax.set_ylim(ax_ref.get_ylim())

    fig.suptitle(calibration_data.reference_name)

    return fig


def plot_calibration_dataset(
    calibration_file: CalibrationDataset, reference_name: str | tuple[str,] = ""
):
    """Plot calibration file."""

    if not isinstance(reference_name, str) or reference_name == "":
        for name in calibration_file.reference_names:
            plot_calibration_data(calibration_file.select(name))
    else:
        plot_calibration_data(calibration_file.select(reference_name))


def plot_model(
    model: PLSCalibration, *other_models: PLSCalibration, extra_title: str = ""
):
    """Plot model prediction."""

    if model.calibration_info is None:
        raise ValueError("Model is not fitted yet.")

    for ndx, ot in enumerate(other_models):
        if ot.calibration_info is not None:
            raise ValueError(f"Model is not fitted yet (element {ndx} in other_models)")

    fig, (ax, axr) = plt.subplots(
        2,
        1,
        sharey=False,
        layout="tight",
        figsize=(11.69 / SIZE_SCALING, 8.27 / SIZE_SCALING),
        height_ratios=[2, 1],
        sharex=True,
    )

    for ot in other_models:
        assert ot.calibration_info is not None
        y, y_pred = ot.calibration_info["y"], ot.calibration_info["y_pred"]
        ax.scatter(y, y_pred, c="k", alpha=0.1, zorder=1)
        axr.scatter(y, y_pred - y, c="k", alpha=0.1, zorder=1)

    y, y_pred = model.calibration_info["y"], model.calibration_info["y_pred"]
    ax.scatter(y, y_pred, c="r", zorder=2)
    axr.scatter(y, y_pred - y, c="r", zorder=2)

    ax.plot([0, 1], [0, 1], transform=ax.transAxes, c="k", ls=":")

    reference_name = model.calibration_info.attrs["reference_name"]

    axr.set_xlabel(f"{reference_name}\nfrom reference")
    ax.set_ylabel(f"{reference_name}\nfrom spectra")
    axr.set_ylabel("residuals")

    mx = np.max(np.abs(axr.get_ylim()))
    axr.set_ylim(-mx, mx)
    axr.axhline(y=0, ls=":", c="k")

    ptp = np.ptp(y)
    mn = y.min() - ptp / 20
    mx = y.max() + ptp / 20

    ax.set_xlim(mn, mx)
    ax.set_ylim(mn, mx)

    plt.suptitle(extra_title)

    return fig


def plot_vip(model: PLSCalibration):
    """Plot model prediction."""

    if model.calibration_info is None:
        raise ValueError("Model is not fitted yet.")

    reference_values = model.calibration_info["y"]
    spectra = model.calibration_info.attrs["X"]
    wavelengths = model.calibration_info.attrs["features"]

    fig, (ax_sp, ax_vip) = plt.subplots(
        2, 1, sharex=True, figsize=(11.69 / SIZE_SCALING, 8.27 / SIZE_SCALING)
    )

    cbar = plot_calibration_data_spectra(ax_sp, wavelengths, spectra, reference_values)

    if np.issubdtype(wavelengths.dtype, np.number):
        ax_vip.set_xlabel("wavelength / nm")
    else:
        ax_vip.set_xlabel("wavelength")

    cbar.set_label(model.calibration_info.attrs["reference_name"])

    ax_vip.plot(wavelengths, metrics.vip(model), label="VIP")
    ax_vip.set_xlabel("wavelength / nm")
    ax_vip.axhline(y=1, c="gray", ls=":")
    ax_vip.legend(bbox_to_anchor=(1.03, 0), loc=3)

    pos = ax_sp.get_position()
    pos2 = ax_vip.get_position()
    ax_vip.set_position([pos.x0, pos2.y0, pos.width, pos2.height])

    return fig


def plot_components_cross_validation(
    rmsec: Vector, train_rmsecv: Array, test_rmsecv: Array, qlatent: float = 1.0
):
    """Plot cross validation RMSE vs the number of components and Q value.

    train_rmsecv and test_rmsecv are (N, 2) arrays:
    - column 0: mean value
    - column 1: std
    """

    components = np.arange(1, 20)

    fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True)

    ax0.errorbar(
        components,
        train_rmsecv[:, 0],
        train_rmsecv[:, 1],
        fmt=".-",
        label="RMSECV (train)",
    )
    ax0.errorbar(
        components,
        test_rmsecv[:, 0],
        test_rmsecv[:, 1],
        fmt=".-",
        label="RMSECV (test)",
    )
    ax0.plot(components, rmsec, label="RMSEC")
    ax0.set_xticks(components)
    ax0.set_ylabel("RMSE")
    ax0.legend()

    ax1.semilogy(components, qlatent)
    ax1.axhline(y=1 - 0.1**2, ls=":", c="gray")
    ax1.axhline(y=1 - 0.95**2, ls=":", c="gray")
    ax1.set_xlabel("Number of components")
    ax1.set_ylabel("$Q_L^2$")

    return fig


def plot_outlier_detection(model: PLSCalibration, vs_samples: bool = True):
    """Plot statistics useful for outlier detection."""

    if model.calibration_info is None:
        raise ValueError("Model is not fitted yet.")

    samples = model.calibration_info["sample"]
    y = model.calibration_info["y"]
    qr = model.calibration_info["reduced_q_residual"]
    ht2 = model.calibration_info["reduced_hotelling_t2"]
    lev = model.calibration_info["leverage"]
    ystdtres = model.calibration_info["studentized_y_residual"]

    if vs_samples:
        fig, (ax_sam, ax_qr, ax_ht2, ax_lev, ax_yst) = plt.subplots(
            5,
            1,
            sharex=True,
            figsize=(11.69 / SIZE_SCALING, 5 / 3 * 8.27 / SIZE_SCALING),
        )
        ax_sam.scatter(samples, y, c=y)
        ax_sam.set_ylabel(model.calibration_info.attrs["reference_name"])

        ax_qr.scatter(samples, qr, c=y)
        ax_qr.axhline(y=1, ls=":", c="gray")
        ax_qr.set_ylabel("reduced\nQ-residuals")

        ax_ht2.scatter(samples, ht2, c=y)
        ax_ht2.axhline(y=1, ls=":", c="gray")
        ax_ht2.set_ylabel("reduced\nHotelling $t^2$")

        ax_lev.scatter(samples, lev, c=y)
        # ax_lev.axhline(y=qr_cut, ls=":", c="gray")
        ax_lev.set_ylabel("Leverage")

        ax_yst.scatter(samples, ystdtres, c=y)
        ax_yst.axhline(y=0, ls=":", c="gray")
        ax_yst.axhline(y=+3, ls=":", c="gray")
        ax_yst.axhline(y=-3, ls=":", c="gray")
        ax_yst.set_ylabel("Y stdt residuals")

        ax_yst.set_xlabel("Samples")
    else:
        fig, (ax0, ax1) = plt.subplots(
            1, 2, figsize=(2 * 8.27 / SIZE_SCALING, 8.27 / SIZE_SCALING)
        )
        ax0.scatter(ht2, qr, c=y)
        ax0.axhline(y=1, ls=":", c="gray")
        ax0.axvline(x=1, ls=":", c="gray")
        ax0.set_xlabel("reduced\nHotelling $t^2$")
        ax0.set_ylabel("reduced\nQ-Residuals")

        ax1.scatter(lev, ystdtres, c=y)
        ax1.set_xlabel("Leverage")
        ax1.set_ylabel("Y stdt residual")
        ax1.axhline(y=0, ls=":", c="gray")
        ax1.axhline(y=+3, ls=":", c="gray")
        ax1.axhline(y=-3, ls=":", c="gray")
        mx = max(np.max(np.abs(ax1.get_ylim())), 4)
        ax1.set_ylim([-mx, +mx])

    return fig
