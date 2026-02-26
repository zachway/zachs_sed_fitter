import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from utils import get_filter_wavelength_vals, load_starphot
import astropy.units as u
import astropy.constants as c

def get_mass_luminosity_radius(logg, teff):
    """
    Docstring for get_mass_luminosity_radius
    
    :param logg: Description
    :param teff: Description
    """
    const_ = np.array([5,4,3,2,1.4,1.2,1])
    const = const_[np.sum((logg>0) + (logg>0.9) + (logg>1.6) + (logg>2) + (logg>3) + (logg>4))]
    mass = const*(teff/5770)**2 * u.M_sun # in solar masses
    radius = np.sqrt(((c.G * mass))/((10**logg) * u.cm/u.s**2)).to(u.R_sun) # in solar radii
    luminosity = ((teff/5770)**4 * (radius)**2).value * u.L_sun # in solar luminosities

    return mass.value, luminosity.value, radius.value


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
        if self.interpolation == "nearest":
            from scipy.interpolate import NearestNDInterpolator
            self.interpolated_grid = NearestNDInterpolator(self.norm_label_grid, self.synthphot_grid)
        #if self.interpolation == "regular_grid":
        #    from scipy.interpolate import RegularGridInterpolator
        #    grid_points = [np.unique(self.norm_label_grid[:, i]) for i in range(self.norm_label_grid.shape[1])]
        #    self.interpolated_grid = RegularGridInterpolator(grid_points, self.synthphot_grid.reshape([len(gp) for gp in grid_points] + [self.synthphot_grid.shape[1]]), **self.interpolator_kwargs)
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
        mass, luminosity, radius = get_mass_luminosity_radius(logg=logg, teff=Teff)
        label = np.array([Teff, logg, feh, ah])
        norm_label = self.label_to_norm_label(label)
        out_phot = self.interpolated_grid(norm_label.reshape(1, -1))[0]
        filter_wvls = np.array([get_filter_wavelength_vals(filter=f)[2] for f in self.synthphot_filters])
        reddening = G23().extinguish(filter_wvls*u.AA, Av=Av)
        out_phot = out_phot * reddening * (2.25e-8*radius/dist)**2
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
    
    def get_filter_effective_wavelengths(self):
        return np.array([get_filter_wavelength_vals(filter=f)[2] for f in self.synthphot_grid.synthphot_filters])
    

class SEDFitter:
    # TODO: ERRORS and upper limits on photometry!!!!!!
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
        initial_guess = np.array([5500, 4.50, -0.5, 0, 975, 0]) # Solar values as initial guess
        def chi2(theta):
            model_photometry = self.synthphot_grid.return_synthphot(theta)
            effective_wavelengths = self.star_phot.get_filter_effective_wavelengths()
            return np.sum(((np.log10(photometry_array[valid_filters]*effective_wavelengths[valid_filters]) \
                                 - np.log10(model_photometry[valid_filters]*effective_wavelengths[valid_filters]))**2))
        result = minimize(chi2, initial_guess, 
                          bounds=[(3000, 10000), (0, 5), (-2.5, 0.5), (-0.5, 0.5), (10, 2000), (0, 10.0)],
                          method='L-BFGS-B',
                          )
        best_fit_photometry = self.synthphot_grid.return_synthphot(result.x)

        return result, best_fit_photometry
    

    
if __name__ == "__main__":
    synthphot_grid = SynthPhotGrid()
    synthphot_grid.load_synthphot_grid(synthphot_filename="synthetic_photometry.txt")
    synthphot_grid.create_interpolated_grid(save=True, 
                                            save_filename="interpolated_grid.npz",
                                            interpolation="nearest",
                                            #interpolator_kwargs={'kernel': 'thin_plate_spline', 'smoothing': 0}
                                            )
    
    synthphot_grid = SynthPhotGrid()
    synthphot_grid.load_interpolated_grid(filename="interpolated_grid.npz")

    source_id = 2762664699324800
    sp = StarPhot(gaia_source_id=source_id, synthphot_grid=synthphot_grid)
    print(f"Photometry array for star with Gaia source ID {source_id}:", sp.photometry_array)
    print("synthphot grid filters:", synthphot_grid.synthphot_filters)
    print("Mass/luminosity/radius for star with logg=4.44 and Teff=5770:", get_mass_luminosity_radius(logg=4.44, teff=5770))

    fitter = SEDFitter(star_phot=sp, synthphot_grid=synthphot_grid)
    fitter_result, best_fit_photometry = fitter.fit_chi2()
    print("Best fit label:", fitter_result.x)
    print("Best fit photometry:", best_fit_photometry)
    print()
    #print("Fitter result:", fitter_result)

    filter_effective_wavelengths = np.array([get_filter_wavelength_vals(filter = f)[2] for f in synthphot_grid.synthphot_filters])
    sorted_indices = np.argsort(filter_effective_wavelengths)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.scatter(filter_effective_wavelengths[sorted_indices], filter_effective_wavelengths[sorted_indices]*sp.photometry_array[sorted_indices], label="Observed Photometry", color='red')
    plt.scatter(filter_effective_wavelengths[sorted_indices], filter_effective_wavelengths[sorted_indices]*best_fit_photometry[sorted_indices], label="Best Fit Photometry", color='blue')
    for i, filter in enumerate(synthphot_grid.synthphot_filters):
        plt.annotate(filter, (filter_effective_wavelengths[i], filter_effective_wavelengths[i]*best_fit_photometry[i]), textcoords="offset points", xytext=(0,10), ha='center')
    plt.xlabel("wavelength (Angstroms)")
    plt.ylabel("Magnitude")
    plt.title(f"Observed vs Best Fit Photometry\n"
              f"Teff={fitter_result.x[0]:.1f} K, logg={fitter_result.x[1]:.2f}, [Fe/H]={fitter_result.x[2]:.2f}, [alpha/Fe]={fitter_result.x[3]:.2f}, dist={fitter_result.x[4]:.1f} pc, Av={fitter_result.x[5]:.2f}"
              )
    plt.legend()
    plt.xticks(rotation=45)
    plt.xscale('log')
    #plt.yscale('log')
    plt.tight_layout()
    plt.show()
