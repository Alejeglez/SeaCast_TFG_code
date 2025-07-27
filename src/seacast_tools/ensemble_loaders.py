
from typing import List
from .np_loaders import PredictionsFromNumpy
import gc
import xarray as xr

class EnsembleFromNpLoaders:
    def __init__(self, loaders: List[PredictionsFromNumpy]):
        self.loaders = loaders

    def build_ensemble(self) -> xr.Dataset:
        datasets = [loader.load_chunk() for loader in self.loaders]
        ensemble = xr.concat(datasets, dim='number')

        return ensemble