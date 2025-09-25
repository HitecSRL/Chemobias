"""
Functions to read and write calibration files.
"""

from __future__ import annotations

import pathlib

import pandas as pd

from .highlevel import CalibrationDataset


def read_excel(path: pathlib.Path | str) -> CalibrationDataset:
    """Read calibration dataset from an excel file.

    The excel file contains the following sheets:

    - spectra
        - rows: sample.
        - columns: wavelength.
        - row headers: sample number.
        - column headers: "sample" followed by wavelength in nanometers.
    - reference:
        - rows: sample.
        - columns: quantities measured by the reference method.
        - row headers: sample number.
        - column headers: "sample" followed by the quantities name.
    - metadata: (optional)
        - rows: key, value pair.
        - column header: "key", "value".

    Parameters
    ----------
    path: str, or path object
        Any valid string path is acceptable. The string could be a URL. Valid
        URL schemes include http, ftp, s3, and file. For file URLs, a host is
        expected. A local file could be: ``file://localhost/path/to/table.xlsx``.

    Returns
    -------
    CalibrationDataset
        A CalibrationDataset from the passed in Excel file.

    See Also
    --------
    write_excel: Write a calibration file to an Excel file.

    """
    df_spectra = pd.read_excel(path, sheet_name="spectra", index_col="sample")
    df_reference = pd.read_excel(path, sheet_name="reference", index_col="sample")

    df = df_spectra.join(df_reference, on="sample", how="outer", validate="one_to_one")

    try:
        df_metadata = pd.read_excel(path, sheet_name="metadata")
        df.attrs = dict(zip(df_metadata["key"].values, df_metadata["value"].values))
    except Exception:
        pass

    return CalibrationDataset.from_dataframe(df)


def write_excel(calds: CalibrationDataset, path: pathlib.Path | str):
    """Write calibration data to a standarized xlsx file.

    Parameters
    ----------
    calds: CalibrationDataset

    path: str, or path object
        Any valid string path is acceptable. The string could be a URL. Valid
        URL schemes include http, ftp, s3, and file. For file URLs, a host is
        expected. A local file could be: ``file://localhost/path/to/table.xlsx``

    See Also
    --------
    read_excel: Write a calibration file to an Excel file.

    """
    with pd.ExcelWriter(path) as xlsx:
        calds.spectra.to_excel(xlsx, sheet_name="spectra", index=False)
        calds.reference.to_excel(xlsx, sheet_name="reference", index=False)
        if calds.metadata:
            tmp = pd.DataFrame(
                dict(
                    nombre=list(calds.metadata.keys()),
                    valor=list(calds.metadata.values()),
                )
            )
            tmp.reference.to_excel(xlsx, sheet_name="metadata", index=False)
