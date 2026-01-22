from glob import glob
import numpy as np
import os
from astropy.io import fits

dir_path = os.path.dirname(os.path.realpath(__file__))

filter_files = glob(f"{dir_path}/filters/*")
surveys = [ff.split("/")[-1].split("_")[1].split(".")[0] 
           for ff in filter_files]
filters = [ff.split("/")[-1].split(".")[1] 
           for ff in filter_files]


class phot_filter:
    def __init__(self, survey, filter):
        self.survey = survey
        self.filter = filter

        self.surveys = surveys
        self.filters = filters

        if self.survey not in self.surveys or self.filter not in self.filters:
            raise Exception(
                f"Survey {self.survey} and filter {self.filter} not in the "
                f"valid surveys ({self.surveys}) and filters ({self.filters})"
            )

        survey_dict = {"2MASS": "2MASS_2MASS",
                       "GAIA3": "GAIA_GAIA3",
                       "WISE": "WISE_WISE",
                       }
        filter_filename = f"{survey_dict[self.survey]}.{self.filter}.dat"
        try:
            self.data = np.genfromtxt("filters/" + filter_filename, 
                                      delimiter=" ")
        except Exception as e:
            print(f"Something wrong with filename {filter_filename}")
            raise e

        self.wvl = self.data.T[0]
        self.trans = self.data.T[1]


class model_spectrum:
    def __init__(self, model_dir, teff, logg, feh, ah):
        self.teff = teff
        self.logg = logg
        self.feh = feh
        self.ah = ah
        self.model_dir = model_dir
        feh_str = f"+{self.feh}" if self.feh > 0 else f"{self.feh}"
        ah_str = f"+{self.ah}" if self.ah > 0 else f"{self.ah}"
        teff_str = str(self.teff).zfill(5)[:3]
        file_string = (f"{self.model_dir}/*lte{teff_str}-{self.logg}"
                       f"*{feh_str}*{ah_str}*.fits")
        possible_files = glob(file_string)
        if len(possible_files) == 0:
            raise Exception(f"No model file found for Teff={teff}, "
                            f"logg={logg}, feh={feh}, ah={ah}")
        elif len(possible_files) > 1:
            raise Exception(f"Multiple model files found for Teff={teff},"
                            f" logg={logg}, feh={feh}, ah={ah}")
        self.filename = possible_files[0]

        try:
            self.data = fits.open(self.filename)
        except Exception as e:
            print(f"Something wrong with filename {self.filename}")
            raise e

        self.wvl = self.data[1].data['WAVELENGTH']  # in Angstroms
        self.flux = self.data[1].data['FLUX']  # in erg/s/cm2/Angstrom

    def get_flux_in_filter(self, phot_filter):
        from scipy.interpolate import interp1d
        from scipy.integrate import simps

        # Interpolate the filter transmission to the model wavelength grid
        interp_trans = interp1d(phot_filter.wvl, phot_filter.trans,
                                kind="linear", fill_value=0,
                                bounds_error=False)
        trans_interp = interp_trans(self.wvl)  # interpolated transmission
        # Calculate the flux in the filter using the formula
        numerator = simps(self.flux * trans_interp * self.wvl, self.wvl)
        denominator = simps(trans_interp * self.wvl, self.wvl)
        flux_in_filter = numerator / denominator
        return flux_in_filter


def list_available_models(model_dir):
    model_files = glob(f"{model_dir}/*.fits")
    model_params = []
    for mf in model_files:
        base = os.path.basename(mf)
        teff = int(base[3:6])*100
        logg = float(base[7:10])
        feh = float(base[10:14])
        ah = float(base[15:19])
        model_params.append((teff, logg, feh, ah))
    return model_params


def list_available_filters():
    return [(s, f) for s, f in zip(surveys, filters)]