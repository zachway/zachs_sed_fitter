import numpy as np
import pandas as pd
from tqdm import tqdm
import os


def get_mass_luminosity(logg, teff):
    """
    Docstring for get_mass_luminosity
    
    :param logg: Description
    :param teff: Description
    """
    c_ = np.array([5,4,3,2,1.4,1.2,1])
    c = c_[np.sum((logg>0) + (logg>0.9) + (logg>1.6) + (logg>2) + (logg>3) + (logg>4))]
    mass = c*(teff/5770)**2 # in solar masses
    luminosity = mass**3 # in solar luminosities

    return mass, luminosity


class SynthPhotGrid:
    def __init__(self, synthphot_filename, interpolation="rbf", interpolator_kwargs={}):
        if not os.path.exists("files/" + synthphot_filename):
            raise FileNotFoundError(f"Synthetic photometry file '{synthphot_filename}' not found in 'files/' directory.")
        self.synthphot_filename = synthphot_filename
        self.interpolation = interpolation
        self.interpolator_kwargs = interpolator_kwargs
        #self.load_synthphot_grid()
        #self.create_interpolated_grid()

    def load_synthphot_grid(self):
        self.synthphot_grid_data = pd.read_csv("files/" + self.synthphot_filename, sep=r'\t', engine='python')

        self.label_grid = self.synthphot_grid_data[["Teff", "logg", "feh", "ah"]].values
        self.synthphot_grid = self.synthphot_grid_data.drop(columns=["Teff", "logg", "feh", "ah", "file_type"]).values
        self.normalize_label_grid()

    def normalize_label_grid(self):
        self.label_mins = np.min(self.label_grid, axis=0)
        self.label_maxs = np.max(self.label_grid, axis=0)
        self.norm_label_grid = (self.label_grid - self.label_mins) / (np.max(self.label_grid, axis=0) - np.min(self.label_grid, axis=0))
        self.label_to_norm_label = lambda label: (label - self.label_mins) / (self.label_maxs - self.label_mins)
        self.norm_label_to_label = lambda norm_label: norm_label * (self.label_maxs - self.label_mins) + self.label_mins

    def create_interpolated_grid(self, interpolation="rbf", interpolator_kwargs={'kernel': 'thin_plate_spline', 'smoothing': 0, 'degree': 1}, 
                                 save=False, save_filename="interpolated_grid.npz"):
        print("Creating interpolated grid...")
        print("Interpolation method:", self.interpolation)
        if self.interpolation == "rbf":
            from scipy.interpolate import RBFInterpolator
            self.interpolated_grid = RBFInterpolator(self.norm_label_grid, self.synthphot_grid, **self.interpolator_kwargs)
        if save:
            if save_filename is None:
                raise ValueError("save_filename must be provided if save is True.")
            np.savez("files/" + save_filename, interpolated_grid=self.interpolated_grid, label_mins=self.label_mins, label_maxs=self.label_maxs)


if __name__ == "__main__":
    synthphot_grid = SynthPhotGrid("synthetic_photometry.txt")
    synthphot_grid.load_synthphot_grid()
    synthphot_grid.create_interpolated_grid(save=True, save_filename="interpolated_grid.npz")

