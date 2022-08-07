import numpy as np

from typing import Any, Callable, Dict, Optional


class Scalarizer:
    """Implements a class for scalarizing vector valued functions with a
    given scalarization function.
    """

    def __init__(
        self, evaluator: Callable, scalarizer: Callable, evaluator_args: Dict = None, scalarizer_args: Dict = None
    ):
        """
        Args:
            evaluator (Callable): A Callable object returning a numpy array.
            scalarizer (Callable): A function which should accepts as its
                arguments the output of evaluator and return a single value.
            evaluator_args (Any, optional): Optional arguments to be passed to
                evaluator. Defaults to None.
            scalarizer_args (Any, optional): Optional arguments to be passed to
                scalarizer. Defaults to None.
        """
        self._evaluator = evaluator
        self._scalarizer = scalarizer
        self._evaluator_args = evaluator_args
        self._scalarizer_args = scalarizer_args

    def evaluate(self, xs: np.ndarray) -> np.ndarray:
        """Evaluates the scalarized function with the given arguments and
        returns a scalar value for each vector of variables given in a numpy
        array.

        Args:
            xs (np.ndarray): A 2D numpy array containing vectors of variables
                on each of its rows.

        Returns:
            np.ndarray: A 1D numpy array with the values returned by the
                scalarizer for each row in xs.
        """
        if self._evaluator_args is not None:
            res_eval = self._evaluator(xs, **self._evaluator_args)
        else:
            res_eval = self._evaluator(xs)

        if self._scalarizer_args is not None:
            res_scal = self._scalarizer(res_eval, **self._scalarizer_args)
        else:
            res_scal = self._scalarizer(res_eval)

        return res_scal

    def __call__(self, xs: np.ndarray) -> np.ndarray:
        """Wrapper to the evaluate method.
        """
        return self.evaluate(xs)


class DiscreteScalarizer:
    """Implements a class to scalarize discrete vectors given a scalarizing function.
    """

    def __init__(self, scalarizer: Callable, scalarizer_args: Dict = None):
        self._scalarizer = scalarizer
        self._scalarizer_args = scalarizer_args

    def evaluate(self, vectors: np.ndarray) -> np.ndarray:
        # guarantee two dimensions, makes sure works with single vectors as well.
        vectors = np.atleast_2d(vectors)

        if self._scalarizer_args is not None:
            res_scal = self._scalarizer(vectors, **self._scalarizer_args)
        else:
            res_scal = self._scalarizer(vectors)

        return res_scal

    def __call__(self, vectors: np.ndarray):
        return self.evaluate(vectors)


if __name__ == "__main__":
    vectors = np.array([[1, 1, 1], [2, 2, 2], [4, 5, 6.0]])
    vector = np.array([1, 2, 3])
    dscalarizer = DiscreteScalarizer(lambda x, a=1: a * np.sum(x, axis=1), scalarizer_args={"a": 2})
    res = dscalarizer(vectors)
    res_1d = dscalarizer(vector)
    print(res)
    print(res_1d)
