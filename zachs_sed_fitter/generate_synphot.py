from utils import (
    model_spectrum,
    phot_filter,
    list_available_models,
    list_available_filters,
)
from tqdm import tqdm


def generate_synthetic_photometry(model_dir, teff, logg,
                                  feh, ah, file_type, survey, filter):
    # Load the model spectrum
    model_spec = model_spectrum(model_dir, teff, logg, feh, ah, file_type)

    # Load the filter curve
    phot_filt = phot_filter(survey, filter)

    return model_spec.get_flux_in_filter(phot_filt)


def create_fluxes_from_models(model_dir, output_filename):
    filter_survey_pairs = list_available_filters()
    model_params = list_available_models(model_dir)
    for teff, logg, feh, ah, file_type in tqdm(model_params):
        with open(output_filename, "a") as f:
            for survey, filter in filter_survey_pairs:
                flux = generate_synthetic_photometry(
                    model_dir, teff, logg, feh, ah, file_type, survey, filter
                )
                output_line = (
                    f"{teff}\t{logg}\t{feh}\t{ah}\t{file_type}\t{survey}\t"
                    f"{filter}\t{flux}\n"
                )
                f.write(output_line)


if __name__ == "__main__":
    model_dir = (
        "/Volumes/ExternalDrive/Comparing_M_Dwarf_Models/"
        "SyntheticModelLibraries/btsett-cifist"
    )
    output_filename = "files/synthetic_photometry.txt"
    with open(output_filename, "w") as f:
        f.write("Teff\tlogg\tfeh\tah\tfile_type\tsurvey\tfilter\tflux\n")
    create_fluxes_from_models(model_dir, output_filename)
