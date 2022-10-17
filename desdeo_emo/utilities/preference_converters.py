# The contents of this file should be transferred to a file in desdeo-tools with the
# same file name.

import numpy as np


def UPEMO(preference: np.ndarray):
    """Converts different kinds of preference information to a reference point.

    Args:
        preference (np.ndarray): It can be a single reference point or solution as a
            1-d or 2-d array. It can be multiple reference points or solutions as a 2-d
            array such that each row is a reference point or a solution. It can also be
            a pair or preferred ranges in a 2d array with two rows for the lower and
            upper ranges.

    Returns:
        _type_: _description_
    """
    preference = np.atleast_2d(preference)
    return preference.max(axis=0)
