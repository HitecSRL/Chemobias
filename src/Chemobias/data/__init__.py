"""
Test datasets.

A curated set of general purpose and datasets used in tests, examples,
and documentation.

Available setS:
- animal_feed
- cgl
- cocoa_beans
- gasoline
- wheat

"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

from .. import io

if TYPE_CHECKING:
    from ..highlevel import CalibrationDataset

current_path = pathlib.Path(__file__).absolute().parent


def load(name: str) -> CalibrationDataset:
    return io.read_excel((current_path / name).with_suffix(".xlsx"))


def animal_feed() -> CalibrationDataset:
    return load("animal_feed")


def cgl() -> CalibrationDataset:
    return load("cgl")


def cocoa_beans() -> CalibrationDataset:
    return load("cocoa_beans")


def gasoline() -> CalibrationDataset:
    return load("gasoline")


def wheat() -> CalibrationDataset:
    return load("wheat")


__all__ = ["animal_feed", "cgl", "cocoa_beans", "gasoline", "wheat"]
