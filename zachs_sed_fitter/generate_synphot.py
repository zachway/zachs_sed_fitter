from zachs_sed_fitter.utils import (
    model_spectrum,
    phot_filter,
    list_available_models,
    list_available_filters,
)
from tqdm import tqdm


def generate_synthetic_photometry(model_dir, teff, logg, feh, survey, filter):
    # Load the model spectrum
    model_spec = model_spectrum(model_dir, teff, logg, feh)

    # Load the filter curve
    phot_filt = phot_filter(survey, filter)

    return model_spec.get_flux_in_filter(phot_filt)


def create_fluxes_from_models(model_dir, output_filename):
    filter_survey_pairs = list_available_filters()
    model_params = list_available_models(model_dir)
    for teff, logg, feh in tqdm(model_params):
        with open(output_filename, "a") as f:
            for survey, filter in filter_survey_pairs:
                flux = generate_synthetic_photometry(
                    model_dir, teff, logg, feh, survey, filter
                )
                f.write(f"{teff}\t{logg}\t{feh}\t{survey}\t{filter}\t{flux}\n")


if __name__ == "__main__":
    model_dir = "path_to_model_directory"
    output_filename = "synthetic_photometry.txt"
    create_fluxes_from_models(model_dir, output_filename)
