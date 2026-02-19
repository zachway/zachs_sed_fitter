from utils import (
    model_spectrum,
    phot_filter,
    list_available_models,
    list_available_filters,
    load_starphot,
    get_starphot,
)
from tqdm import tqdm
from joblib import Parallel, delayed


def generate_synthetic_photometry(model_dir, teff, logg,
                                  feh, ah, file_type, survey, filter):
    # Load the model spectrum
    model_spec = model_spectrum(model_dir, teff, logg, feh, ah, file_type)

    # Load the filter curve
    phot_filt = phot_filter(survey, filter)

    return model_spec.get_flux_in_filter(phot_filt)


def create_fluxes_from_models(model_dir, output_filename, ncores=1):
    filter_survey_pairs = list_available_filters()
    model_params = list_available_models(model_dir)

    with open(output_filename, "w") as f:
        f.write(f"Teff\tlogg\tfeh\tah\tfile_type\t"
                f"{'\t'.join(map(str, [fsp[1] for fsp in filter_survey_pairs]))}\n"
                )

    def process_model(teff, logg, feh, ah, file_type):
        with open(output_filename, "a") as f:
            flux_ = []
            for survey, filter in filter_survey_pairs:
                flux_.append(generate_synthetic_photometry(
                    model_dir, teff, logg, feh, ah, file_type, survey, filter
                ))
            output_line = (
                f"{teff}\t{logg}\t{feh}\t{ah}\t{file_type}\t"
                f"{'\t'.join(map(str, flux_))}\n"
            )
            f.write(output_line)

    Parallel(n_jobs=ncores)(
        delayed(process_model)(teff, logg, feh, ah, file_type)
        for teff, logg, feh, ah, file_type in tqdm(model_params)
    )

class starphot:
    def __init__(self, gaia_source_id):
        self.gaia_source_id = gaia_source_id
        self.data = load_starphot(gaia_source_id)
    
    def get_flux_in_filter(self, survey, filter):
        row = self.data[(self.data["survey"] == survey) & (self.data["filter"] == filter)]
        if len(row) == 0:
            raise ValueError(f"No data found for survey {survey} and filter {filter}")
        elif len(row) > 1:
            raise ValueError(f"Multiple rows found for survey {survey} and filter {filter}")
        return row["flux"][0]

if __name__ == "__main__":
    model_dir = (
        "/Volumes/ExternalDrive/Comparing_M_Dwarf_Models/"
        "MODELOS_BTSESTTLCIFIST_TEST"
    )
    output_filename = "files/synthetic_photometry.txt"

    create_fluxes_from_models(model_dir, output_filename, ncores=7)
