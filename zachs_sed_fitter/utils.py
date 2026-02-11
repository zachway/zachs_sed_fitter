from glob import glob
import numpy as np
import os
from astropy.io import fits
#from astroquery.simbad import Simbad
from astroquery.gaia import Gaia

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
    def __init__(self, model_dir, teff, logg, feh, ah, file_type):
        self.teff = teff
        self.logg = logg
        self.feh = feh
        self.ah = ah
        self.file_type = file_type
        self.model_dir = model_dir
        logg_str = f"+{-self.logg}" if self.logg < 0 else f"-{self.logg}"
        feh_str = f"+{self.feh}" if self.feh > 0 else f"{self.feh}"
        ah_str = f"+{self.ah}" if self.ah > 0 else f"{self.ah}"
        teff_str = str(self.teff).zfill(5)[:3]
        file_string = (f"{self.model_dir}/*lte{teff_str}{logg_str}"
                       f"*{feh_str}*{ah_str}*.{self.file_type}")
        possible_files = glob(file_string)
        if len(possible_files) == 0:
            print(file_string)
            raise Exception(f"No model file found for Teff={teff}, "
                            f"logg={logg}, feh={feh}, ah={ah}")
        elif len(possible_files) > 1:
            print(file_string)
            raise Exception(f"Multiple model files found for Teff={teff},"
                            f" logg={logg}, feh={feh}, ah={ah}")
        self.filename = possible_files[0]

        if self.file_type=="fits":
            try:
                self.data = fits.open(self.filename)
            except Exception as e:
                print(f"Something wrong with filename {self.filename}")
                raise e
            self.wvl = self.data[1].data['WAVELENGTH']  # in Angstroms
            self.flux = self.data[1].data['FLUX']  # in erg/s/cm2/Angstrom
        
        # functionality from https://github.com/pkgw/pwkit/blob/388f1b049a4f95a41e46c4a256927d4a96532944/pwkit/phoenix.py
        elif self.file_type=="7":
            try:
                self.wvl, self.lflam = np.loadtxt(self.filename, usecols=(0, 1)).T
            except ValueError:
                with open(self.filename, "rb") as f:
                    def lines():
                        for line in f:
                            yield line.replace(b"D", b"e")
                    self.wvl, self.lflam = np.genfromtxt(lines(), delimiter=(13, 13)).T

            self.flux = 10**(self.lflam - 8.0)  # convert from log to linear flux in erg/s/cm2/Angstrom

            self.flux = self.flux[np.argsort(self.wvl)]
            self.lflam = self.lflam[np.argsort(self.wvl)]
            self.wvl = self.wvl[np.argsort(self.wvl)]


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
    model_files = glob(f"{model_dir}/*.fits") + glob(f"{model_dir}/*.7")
    model_params = []
    for mf in model_files:
        if mf[-5:] == ".fits":
            try:
                base = os.path.basename(mf)
                teff = int(base[3:6])*100
                logg = float(base[7:10])
                feh = float(base[10:14])
                ah = float(base[15:19])
                file_type = "fits"
                model_params.append((teff, logg, feh, ah, file_type))
            except Exception as e:
                print(f"Error parsing filename {mf}: {e}")
                raise e
        elif mf[-2:] == ".7":
            try:
                base = os.path.basename(mf)
                teff = int(float(base.split('.BT-Settl.')[0].split('a')[0].split('lte')[1][:-8])*100)
                logg = -1*float(base.split('.BT-Settl.')[0].split('a')[0][-8:-4])
                feh = float(base.split('.BT-Settl.')[0].split('a')[0][-4:])
                ah = float(base.split('.BT-Settl.')[0][-4:])
                file_type = "7"
                model_params.append((teff, logg, feh, ah, file_type))
            except Exception as e:
                print(f"Error parsing filename {mf}: {e}")
                raise e
    return model_params


def list_available_filters():
    return [(s, f) for s, f in zip(surveys, filters)]


def get_starphot(gaia_source_id, save=False):
    # REMINDER: test when Gaia Archive is back up
    # REMINDER: get fluxes instead of magnitudes, and convert to same units as synthetic photometry
    Gaia.ROW_LIMIT = 2
    job = Gaia.launch_job_async(
        f"WITH cte AS (SELECT TOP 1 source_id FROM gaiadr3.gaia_source WHERE source_id={gaia_source_id}) "
        "SELECT dr3.source_id, dr3.ra, dr3.dec, dr3.pmra, dr3.pmdec, "
        "dr3.parallax, radial_velocity, radial_velocity_error, "
        "phot_g_mean_mag, phot_rp_mean_mag, phot_bp_mean_mag, "
        "phot_bp_rp_excess_factor, ruwe, ipd_frac_multi_peak, "
        "ipd_frac_odd_win, ipd_gof_harmonic_amplitude, "
        "duplicated_source, rv_chisq_pvalue, rv_renormalised_gof, "
        "rv_nb_transits, tmass.j_m, tmass.j_msigcom, tmass.h_m, "
        "tmass.h_msigcom, tmass.ks_m, tmass.ks_msigcom, "
        "allwise.w1mpro, allwise.w1mpro_error, allwise.w2mpro, "
        "allwise.w2mpro_error, allwise.w3mpro, allwise.w3mpro_error, "
        "allwise.w4mpro, allwise.w4mpro_error, * "
        "FROM cte "
        "LEFT JOIN gaiadr3.gaia_source dr3 ON cte.source_id = dr3.source_id "
        "LEFT JOIN gaiadr3.allwise_best_neighbour AS a_xmatch "
        "ON (cte.source_id = a_xmatch.source_id) "
        "LEFT JOIN gaiadr1.allwise_original_valid AS allwise "
        "ON (a_xmatch.allwise_oid = allwise.allwise_oid) "
        "LEFT JOIN gaiadr3.tmass_psc_xsc_best_neighbour AS xmatch "
        "ON (cte.source_id = xmatch.source_id) "
        "LEFT JOIN gaiadr3.tmass_psc_xsc_join AS xjoin "
        "ON (xmatch.clean_tmass_psc_xsc_oid = xjoin.clean_tmass_psc_xsc_oid) "
        "LEFT JOIN gaiadr1.tmass_original_valid AS tmass ON "
        "xjoin.original_psc_source_id = tmass.designation "
        "LEFT JOIN gaiadr3.synthetic_photometry_gspc AS gaia_synth "
        "ON (cte.source_id = gaia_synth.source_id) "
    )
    result = job.get_results()
    if len(result) == 0:
        raise Exception(f"No results found for Gaia source ID {gaia_source_id}")
    elif len(result) > 1:
        raise Exception(f"Multiple results found for Gaia source ID {gaia_source_id}")
    
    #if save:
    #   with open(f"starphot/{name}_photometry.txt", "w") as f:
    #        for band, flux in photometry.items():
    #            f.write(f"{band}\t{flux}\n")

    return result

if __name__ == "__main__":
    # # Example usage
    # model_dir = "models/phoenix"
    # teff = 5000
    # logg = 4.5
    # feh = 0.0
    # ah = 0.0
    # file_type = "fits"
    # model_spec = model_spectrum(model_dir, teff, logg, feh, ah, file_type)

    # survey = "GAIA3"
    # filter_name = "G"
    # phot_filt = phot_filter(survey, filter_name)

    # flux_in_filter = model_spec.get_flux_in_filter(phot_filt)
    # print(f"Flux in {survey} {filter_name} filter: {flux_in_filter}")

    result = get_starphot(gaia_source_id=6583864876821240576)
    print(result)