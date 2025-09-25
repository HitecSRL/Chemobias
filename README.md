# Chemobias

A helper package for PLS regressions by Hitec SRL (sensing & control).

## Example

```python
>>> import Chemobias
>>> from Chemobias import data, visualization, metrics
>>> calds = data.gasoline()
>>> print(calds)
>>> caldata = calds.select("octane_number")
>>> print(caldata)
>>> model = Chemobias.PLSCalibration(n_components=3)
>>> model.fit_calibration_data(caldata)
>>> visualization.plot_model(model).show() # this is a figure!
>>> model.predict_mu_std(new_spectra, method="Höskuldsson1988")
```

See `notebooks/example.ipynb` for more usage examples
and docstring for each function and class.


## Install

```sh
pip install Chemobias  # from PyPI, once it is published.
pip install Chemobias.zip  # from a zip file.
```
