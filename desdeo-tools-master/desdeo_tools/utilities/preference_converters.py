"""Provides implementations that convert one type of preference information to another."""

import numpy as np


def classification_to_reference_point(
    classification_preference: dict, ideal: np.ndarray, nadir: np.ndarray
) -> dict:
    """Convert classification type of preference (e.g. NIMBUS) to reference point
    preference.

    Args:
        classification_preference (dict): A dict containing keys 'current solution',
            'levels', and 'classifications'. Read the NIMBUS paper for more details.
        ideal (np.ndarray): The ideal point of the problem.
        nadir (np.ndarray): The nadir point of the problem.

    Returns:
        dict: The preference in the form of a reference point. Contains one key:
            "reference point", which maps to the preference in a numpy array structure.
    """
    z_bar = np.zeros_like(nadir, dtype=float)

    improve_inds = np.where(
        np.array(classification_preference["classifications"]) == "<"
    )[0]
    acceptable_inds = np.where(
        np.array(classification_preference["classifications"]) == "="
    )[0]
    free_inds = np.where(np.array(classification_preference["classifications"]) == "0")[
        0
    ]
    improve_until_inds = np.where(
        np.array(classification_preference["classifications"]) == "<="
    )[0]
    impaire_until_inds = np.where(
        np.array(classification_preference["classifications"]) == ">="
    )[0]

    z_bar[improve_inds] = ideal[improve_inds]
    z_bar[improve_until_inds] = classification_preference["levels"][improve_until_inds]
    z_bar[acceptable_inds] = classification_preference["current solution"][
        acceptable_inds
    ]
    z_bar[impaire_until_inds] = classification_preference["levels"][impaire_until_inds]
    z_bar[free_inds] = nadir[free_inds]

    return {"reference point": z_bar}
