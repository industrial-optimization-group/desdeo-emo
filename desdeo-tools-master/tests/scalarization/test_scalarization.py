import pytest
import numpy as np

from desdeo_tools.scalarization.Scalarizer import Scalarizer, DiscreteScalarizer
from desdeo_tools.scalarization.ASF import PointMethodASF


def simple_vector_valued_fun(xs: np.ndarray, extra: int = 0):
    """A simple vector valued function for testing.

    Args:
        xs (np.ndarray): A 2D numpy array with argument vectors as its rows.
            Each vector consists of four values.

    Returns:
        np.ndarray: A 2D array with function evaluation results for each of
            the argument vectors on its rows. Each row contains three values.
    """
    f1 = xs[:, 0] + xs[:, 1] + extra
    f2 = xs[:, 1] - xs[:, 2] + extra
    f3 = xs[:, 2] * xs[:, 3] + extra

    return np.vstack((f1, f2, f3)).T


def simple_scalarizer(ys: np.ndarray, extra: int = 0):
    res = np.sum(ys, axis=1)

    if extra > 0:
        return -res

    else:
        return res


def test_scalarizer_simple():
    scalarizer = Scalarizer(simple_vector_valued_fun, simple_scalarizer)
    xs = np.array([[1, 2, 3, 4], [9, 8, 7, 6], [1, 5, 7, 3]])

    res = scalarizer.evaluate(xs)

    assert np.array_equal(res, [14, 60, 25])


def test_scalarizer_simple_with_arg():
    scalarizer = Scalarizer(
        simple_vector_valued_fun,
        simple_scalarizer,
        evaluator_args={"extra": 1},
        scalarizer_args={"extra": 4},
    )
    xs = np.array([[1, 2, 3, 4], [9, 8, 7, 6], [1, 5, 7, 3]])

    res = scalarizer.evaluate(xs)

    assert np.array_equal(res, [-17, -63, -28])


def test_scalarizer_asf():
    asf = PointMethodASF(np.array([10, 10, 10]), np.array([-10, -10, -10]))
    ref = np.atleast_2d([1, 5, 2.5])
    scalarizer = Scalarizer(
        simple_vector_valued_fun, asf, scalarizer_args={"reference_point": ref}
    )

    res = scalarizer.evaluate(np.atleast_2d([2, 1, 1, 1]))

    assert np.allclose(res, 0.1000002)


def test_discrete():
    vectors = np.array([[1, 1, 1], [2, 2, 2], [4, 5, 6.0]])
    dscalarizer = DiscreteScalarizer(lambda x: np.sum(x, axis=1))
    res = dscalarizer(vectors)

    assert np.array_equal(res, [3, 6, 15])


def test_discrete_1d():
    vector = np.array([1, 2, 3.0])
    dscalarizer = DiscreteScalarizer(lambda x: np.sum(x, axis=1))
    res_1d = dscalarizer(vector)

    assert np.array_equal(res_1d, [6.0])


def test_discrete_args():
    vectors = np.array([[1, 1, 1], [2, 2, 2], [4, 5, 6.0]])
    dscalarizer = DiscreteScalarizer(
        lambda x, a=1: a * np.sum(x, axis=1), scalarizer_args={"a": 2}
    )
    res = dscalarizer(vectors)

    assert np.array_equal(res, [6, 12, 30])


if __name__ == "__main__":
    asf = PointMethodASF(np.array([10, 10, 10]), np.array([-10, -10, -10]))
    ref = np.atleast_2d([2.5, 2.5, 2.5])
    scalarizer = Scalarizer(
        simple_vector_valued_fun, asf, scalarizer_args={"reference_point": ref}
    )

    res = scalarizer.evaluate(np.atleast_2d([2, 1, 1, 1]))
    print(res)

    asf.nadir = np.array([9, 9, 9])

    res = scalarizer.evaluate(np.atleast_2d([2, 1, 1, 1]))
    print(res)
