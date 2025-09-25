import numpy as np

type Scalar = float | int | np.float64
type Vector = np.ndarray[tuple[int,], np.dtype[np.float64]]
type Array = np.ndarray[tuple[int, int], np.dtype[np.float64]]
