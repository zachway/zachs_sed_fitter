from glob import glob
import numpy as np
import os

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
            self.data = np.genfromtxt(filter_filename, delimiter=" ")
        except Exception as e:
            print(f"Something wrong with filename {filter_filename}")
            raise e

        self.wvl = self.data.T[0]
        self.trans = self.data.T[1]


class model_spectrum:
    def __init__(self, model_dir, teff, logg, feh):
        self.teff = teff
        self.logg = logg
        self.feh = feh
        self.model_dir = model_dir
        self.filename = (
            f"{self.model_dir}/btsett-cifist+_teff{self.teff}_"
            f"logg{self.logg}_feh{self.feh}_spec.dat"
        )

        try:
            self.data = np.genfromtxt(self.filename, delimiter=" ")
        except Exception as e:
            print(f"Something wrong with filename {self.filename}")
            raise e

        self.wvl = self.data.T[0]*1e4  # convert from microns to Angstroms
        self.flux = self.data.T[1]  # in erg/s/cm2/Angstrom

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
    model_files = glob(f"{model_dir}/*.dat")
    model_params = []
    for mf in model_files:
        base = os.path.basename(mf)
        parts = base.split("_")
        teff = int(parts[1].replace("teff", ""))
        logg = float(parts[2].replace("logg", ""))
        feh = float(parts[3].replace("feh", "").replace("_spec.dat", ""))
        model_params.append((teff, logg, feh))
    return model_params


def list_available_filters():
    return [(s, f) for s, f in zip(surveys, filters)]