import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from utils import get_filter_wavelength_vals, load_starphot
import astropy.units as u

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
    # def __init__(self)):
    #     self.interpolation = interpolation
    #     self.interpolator_kwargs = interpolator_kwargs
    #     #self.load_synthphot_grid()
    #     #self.create_interpolated_grid()

    def load_synthphot_grid(self, synthphot_filename=None):
        if not os.path.exists("files/" + synthphot_filename):
            raise FileNotFoundError(f"Synthetic photometry file '{synthphot_filename}' not found in 'files/' directory.")
        self.synthphot_filename = synthphot_filename
        self.synthphot_grid_data = pd.read_csv("files/" + self.synthphot_filename, sep=r'\t', engine='python')
        self.label_grid = self.synthphot_grid_data[["Teff", "logg", "feh", "ah"]].values
        self.synthphot_grid = self.synthphot_grid_data.drop(columns=["Teff", "logg", "feh", "ah", "file_type"]).values
        self.synthphot_filters = self.synthphot_grid_data.columns.drop(["Teff", "logg", "feh", "ah", "file_type"])
        self.normalize_label_grid()

    def normalize_label_grid(self):
        self.label_mins = np.min(self.label_grid, axis=0)
        self.label_maxs = np.max(self.label_grid, axis=0)
        self.norm_label_grid = (self.label_grid - self.label_mins) / (np.max(self.label_grid, axis=0) - np.min(self.label_grid, axis=0))
        self.label_to_norm_label = lambda label: (label - self.label_mins) / (self.label_maxs - self.label_mins)
        self.norm_label_to_label = lambda norm_label: norm_label * (self.label_maxs - self.label_mins) + self.label_mins

    def create_interpolated_grid(self, interpolation="rbf", interpolator_kwargs={'kernel': 'thin_plate_spline', 'smoothing': 0, 'degree': 1}, 
                                 save=False, save_filename="interpolated_grid.npz"):
        if self.label_grid is None or self.synthphot_grid is None:
            raise ValueError("Synthphot grid must be loaded before creating interpolated grid. Call load_synthphot_grid() first.")
        self.interpolation = interpolation
        self.interpolator_kwargs = interpolator_kwargs
        print("Creating interpolated grid...")
        print("Interpolation method:", self.interpolation)
        if self.interpolation == "rbf":
            from scipy.interpolate import RBFInterpolator
            self.interpolated_grid = RBFInterpolator(self.norm_label_grid, self.synthphot_grid, **self.interpolator_kwargs)
        if save:
            if save_filename is None:
                raise ValueError("save_filename must be provided if save is True.")
            np.savez("files/" + save_filename, 
                     interpolated_grid=self.interpolated_grid, 
                     label_mins=self.label_mins, 
                     label_maxs=self.label_maxs, 
                     synthphot_filters=self.synthphot_filters,
                     interpolation=self.interpolation,
                     interpolator_kwargs=self.interpolator_kwargs,
                     allow_pickle=True)
    
    def load_interpolated_grid(self, filename="interpolated_grid.npz"):
        print("Loading interpolated grid from file:", filename)
        if not os.path.exists("files/" + filename):
            raise FileNotFoundError(f"Interpolated grid file '{filename}' not found in 'files/' directory.")
        data = np.load("files/" + filename, allow_pickle=True)
        self.interpolated_grid = data['interpolated_grid'].item()
        self.label_mins = data['label_mins']
        self.label_maxs = data['label_maxs']
        self.label_to_norm_label = lambda label: (label - self.label_mins) / (self.label_maxs - self.label_mins)
        self.norm_label_to_label = lambda norm_label: norm_label * (self.label_maxs - self.label_mins) + self.label_mins
        self.synthphot_filters = data['synthphot_filters']
        self.interpolation = data['interpolation'].item()
        self.interpolator_kwargs = data['interpolator_kwargs'].item()

    def return_synthphot(self, theta, ):
        # TODO: figure out how this scales with PHOENIX luminosity
        from dust_extinction.parameter_averages import G23
        Teff, logg, feh, ah, dist, Av = theta
        label = np.array([Teff, logg, feh, ah])
        norm_label = self.label_to_norm_label(label)
        out_phot = self.interpolated_grid(norm_label.reshape(1, -1))[0]
        filter_wvls = np.array([get_filter_wavelength_vals(filter=f)[2] for f in self.synthphot_filters])
        reddening = G23().extinguish(filter_wvls*u.AA, Av=Av)
        out_phot = out_phot * reddening / (4*np.pi*dist**2)
        return out_phot
        

class StarPhot:
    def __init__(self, gaia_source_id, synthphot_grid):
        self.gaia_source_id = gaia_source_id
        self.data = load_starphot(gaia_source_id)
        if not isinstance(synthphot_grid, SynthPhotGrid):
            raise ValueError("synthphot_grid must be an instance of SynthPhotGrid class.")
        self.synthphot_grid = synthphot_grid
        self.photometry_array = self.organize_phot()
    
    def organize_phot(self):
        photometry_array = np.zeros(len(self.synthphot_grid.synthphot_filters))
        for i, filter in enumerate(self.synthphot_grid.synthphot_filters):
            if filter in self.data.columns:
                photometry_array[i] = self.data[filter]
            else:
                photometry_array[i] = np.nan
        return photometry_array
    
    def get_photometry_array(self):
        return self.photometry_array
    

class SEDFitter:
    # TODO: ERRORS!!!!!!!
    def __init__(self, star_phot, synthphot_grid):
        if not isinstance(star_phot, StarPhot):
            raise ValueError("star_phot must be an instance of StarPhot class.")
        if not isinstance(synthphot_grid, SynthPhotGrid):
            raise ValueError("synthphot_grid must be an instance of SynthPhotGrid class.")
        self.star_phot = star_phot
        self.synthphot_grid = synthphot_grid

    def fit_chi2(self):
        photometry_array = self.star_phot.get_photometry_array()
        valid_filters = ~np.isnan(photometry_array)
        if np.sum(valid_filters) == 0:
            raise ValueError("No valid photometry available for fitting.")
        
        from scipy.optimize import minimize
        initial_guess = np.array([5770, 4.44, 0, 0, 100, 0]) # Solar values as initial guess
        def chi2(theta):
            model_photometry = self.synthphot_grid.return_synthphot(theta)
            return np.sum(((photometry_array[valid_filters] - model_photometry[valid_filters])**2) / model_photometry[valid_filters])
        result = minimize(chi2, initial_guess, bounds=[(3000, 10000), (0, 5), (-2.5, 0.5), (-0.5, 0.5), (0, None), (0, 0.1)])
        best_fit_label = result.x
        best_fit_photometry = self.synthphot_grid.interpolated_grid(self.synthphot_grid.label_to_norm_label(best_fit_label[:4]).reshape(1, -1))[0]

        return best_fit_label, best_fit_photometry
    

    
if __name__ == "__main__":
    synthphot_grid = SynthPhotGrid()
    synthphot_grid.load_synthphot_grid(synthphot_filename="synthetic_photometry.txt")
    synthphot_grid.create_interpolated_grid(save=True, 
                                            save_filename="interpolated_grid.npz",
                                            interpolation="rbf",
                                            interpolator_kwargs={'kernel': 'thin_plate_spline', 'smoothing': 0}
                                            )
    
    synthphot_grid = SynthPhotGrid()
    synthphot_grid.load_interpolated_grid(filename="interpolated_grid.npz")

    sp = StarPhot(gaia_source_id=6583864876821240576, synthphot_grid=synthphot_grid)
    print("Photometry array for star with Gaia source ID 6583864876821240576:", sp.photometry_array)
    print("synthphot grid filters:", synthphot_grid.synthphot_filters)
    print("Mass and luminosity for star with logg=4.44 and Teff=5770:", get_mass_luminosity(logg=4.44, teff=5770))

    fitter = SEDFitter(star_phot=sp, synthphot_grid=synthphot_grid)
    best_fit_label, best_fit_photometry = fitter.fit_chi2()
    print("Best fit label:", best_fit_label)
    print("Best fit photometry:", best_fit_photometry)

    filter_effective_wavelengths = np.array([get_filter_wavelength_vals(filter = f)[2] for f in synthphot_grid.synthphot_filters])
    sorted_indices = np.argsort(filter_effective_wavelengths)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.scatter(filter_effective_wavelengths[sorted_indices], sp.photometry_array[sorted_indices], label="Observed Photometry", color='red')
    plt.scatter(filter_effective_wavelengths[sorted_indices], best_fit_photometry[sorted_indices], label="Best Fit Photometry", color='blue')
    plt.xlabel("wavelength (Angstroms)")
    plt.ylabel("Magnitude")
    plt.title(f"Observed vs Best Fit Photometry\n"
              f"Teff={best_fit_label[0]:.1f} K, logg={best_fit_label[1]:.2f}, [Fe/H]={best_fit_label[2]:.2f}, [alpha/Fe]={best_fit_label[3]:.2f}, dist={best_fit_label[4]:.1f} pc, Av={best_fit_label[5]:.2f}"
              )
    plt.legend()
    plt.xticks(rotation=45)
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.show()
