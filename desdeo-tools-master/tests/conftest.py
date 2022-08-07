import pytest

import numpy as np


@pytest.fixture
def SimpleVectorValuedFunction(xs: np.ndarray):
    """A simple vector valued function for testing.
    
    Args:
        xs (np.ndarray): A 2D numpy array with argument vectors as its rows.
        Each vector consists of four values.
    
    Returns:
        np.ndarray: A 2D array with function evaluation results for each of
        the argument vectors on its rows. Each row contains three values.
    """
    f1 = xs[:, 0] + xs[:, 1]
    f2 = xs[:, 1] - xs[:, 2]
    f3 = xs[:, 2] * xs[:, 3]

    return np.vstack((f1, f2, f3)).T


if __name__ == "__main__":
    xs = np.array([[1, 2, 3, 4], [9, 8, 7, 6], [1, 5, 7, 3]])

    print(SimpleVectorValuedFunction(xs))
