import abc
import numpy as np

from abc import abstractmethod
from typing import List, Union


class ASFError(Exception):
    """Raised when an error related to the ASF classes is encountered."""


class ASFBase(abc.ABC):
    """A base class for representing achievement scalarizing functions.
    Instances of the implementations of this class should function as
    function.
    """

    @abstractmethod
    def __call__(self, objective_vector: np.ndarray, reference_point: np.ndarray) -> Union[float, np.ndarray]:
        """Evaluate the ASF.

        Args:
            objective_vectors (np.ndarray): The objective vectors to calculate
                the values.
            reference_point (np.ndarray): The reference point to calculate the
                values.

        Returns:
            Union[float, np.ndarray]: Either a single ASF value or a vector of
                values if objective is a 2D array.

        Note:
            The reference point may not always necessarily be feasible, but
            it's dimensions should match that of the objective vector.
        """
        pass


class SimpleASF(ASFBase):
    """Implements a simple order-representing ASF.

    Args:
        weights (np.ndarray): A weight vector that holds weights. It's
            length should match the number of objectives in the underlying
            MOO problem the achievement problem aims to solve.

    Attributes:
        weights (np.ndarray): A weight vector that holds weights. It's
            length should match the number of objectives in the underlying
            MOO problem the achievement problem aims to solve.
    """

    def __init__(self, weights: np.ndarray):
        self.weights = weights

    def __call__(self, objective_vector: np.ndarray, reference_point: np.ndarray) -> Union[float, np.ndarray]:
        """Evaluate the simple order-representing ASF.

        Args:
            objective_vector (np.ndarray): A vector representing a solution in
                the solution space.
            reference_point (np.ndarray): A vector representing a reference
                point in the solution space.

        Note:
            The shaped of objective_vector and reference_point must match.
        """

        return np.max(
            np.where(np.isnan(reference_point), -np.inf, self.weights * (objective_vector - reference_point)), axis=-1
        )


class ReferencePointASF(ASFBase):
    """Uses a reference point q and preferential factors to scalarize a MOO problem.

    Args:
        preferential_factors (np.ndarray): The preferential factors.
        nadir (np.ndarray): The nadir point of the MOO problem to be
            scalarized.
        utopian_point (np.ndarray): The utopian point of the MOO problem to be
            scalarized.
        rho (float): A small number to be used to scale the sm factor in the
            ASF. Defaults to 0.1.

    Attributes:
        preferential_factors (np.ndarray): The preferential factors.
        nadir (np.ndarray): The nadir point of the MOO problem to be
            scalarized.
        utopian_point (np.ndarray): The utopian point of the MOO problem to be
            scalarized.
        rho (float): A small number to be used to scale the sm factor in the
            ASF. Defaults to 0.1.

    References:
        Miettinen, K.; Eskelinen, P.; Ruiz, F. & Luque, M.
        NAUTILUS method: An interactive technique in multiobjective
        optimization based on the nadir point
        Europen Journal of Operational Research, 2010, 206, 426-434
    """

    def __init__(
        self, preferential_factors: np.ndarray, nadir: np.ndarray, utopian_point: np.ndarray, rho: float = 1e-6
    ):
        self.preferential_factors = preferential_factors
        self.nadir = nadir
        self.utopian_point = utopian_point
        self.rho = rho

    def __call__(self, objective_vector: np.ndarray, reference_point: np.ndarray) -> Union[float, np.ndarray]:
        mu = self.preferential_factors
        f = objective_vector
        q = reference_point
        rho = self.rho
        z_nad = self.nadir
        z_uto = self.utopian_point

        max_term = np.max(mu * (f - q), axis=-1)
        sum_term = rho * np.sum((f - q) / (z_nad - z_uto), axis=-1)

        return max_term + sum_term


class MaxOfTwoASF(ASFBase):
    """Implements the ASF used in NIMBUS, which takes the maximum of two terms.

    Args:
        nadir (np.ndarray): The nadir point.
        ideal (np.ndarray): The ideal point.
        lt_inds (List[int]): Indices of the objectives categorized to be
            decreased.
        lte_inds (List[int]): Indices of the objectives categorized to be
            reduced until some value is reached.
        rho (float): A small number to form the utopian point.
        rho_sum (float): A small number to be used as a weight for the sum
            term.

    Attributes:
        nadir (np.ndarray): The nadir point.
        ideal (np.ndarray): The ideal point.
        lt_inds (List[int]): Indices of the objectives categorized to be
            decreased.
        lte_inds (List[int]): Indices of the objectives categorized to be
            reduced until some value is reached.
        rho (float): A small number to form the utopian point.
        rho_sum (float): A small number to be used as a weight for the sum
            term.

    References:
        Miettinen, K. & Mäkelä, Marko M.
        Synchronous approach in interactive multiobjective optimization
        European Journal of Operational Research, 2006, 170, 909-922
    """

    def __init__(
        self,
        nadir: np.ndarray,
        ideal: np.ndarray,
        lt_inds: List[int],
        lte_inds: List[int],
        rho: float = 1e-6,
        rho_sum: float = 1e-6,
    ):
        self.nadir = nadir
        self.ideal = ideal
        self.lt_inds = lt_inds
        self.lte_inds = lte_inds
        self.rho = rho
        self.rho_sum = rho_sum

    def __call__(self, objective_vector: np.ndarray, reference_point: np.ndarray) -> Union[float, np.ndarray]:
        # assure this function works with single objective vectors
        if objective_vector.ndim == 1:
            f = objective_vector.reshape((1, -1))
        else:
            f = objective_vector

        ii = self.lt_inds
        jj = self.lte_inds
        z = reference_point
        nad = self.nadir
        ide = self.ideal
        uto = self.ideal - self.rho

        lt_term = (f[:, ii] - ide[ii]) / (nad[ii] - uto[ii])
        lte_term = (f[:, jj] - z[jj]) / (nad[jj] - uto[jj])
        max_term = np.max(np.hstack((lt_term, lte_term)), axis=1)
        sum_term = self.rho_sum * np.sum(f / (nad - uto), axis=1)

        return max_term + sum_term


class StomASF(ASFBase):
    """Implementation of the satisfying trade-off method (STOM).

    Args:
        ideal (np.ndarray): The ideal point.
        rho (float): A small number to form the utopian point.
        rho_sum (float): A small number to be used as a weight for the sum
            term.

    Attributes:
        ideal (np.ndarray): The ideal point.
        rho (float): A small number to form the utopian point.
        rho_sum (float): A small number to be used as a weight for the sum
            term.

    References:
        Miettinen, K. & Mäkelä, Marko M.
        Synchronous approach in interactive multiobjective optimization
        European Journal of Operational Research, 2006, 170, 909-922
    """

    def __init__(self, ideal: np.ndarray, rho: float = 1e-6, rho_sum: float = 1e-6):
        self.ideal = ideal
        self.rho = rho
        self.rho_sum = rho_sum

    def __call__(self, objective_vectors: np.ndarray, reference_point: np.ndarray) -> Union[float, np.ndarray]:
        # assure this function works with single objective vectors
        if objective_vectors.ndim == 1:
            f = objective_vectors.reshape((1, -1))
        else:
            f = objective_vectors

        z = reference_point
        uto = self.ideal - self.rho

        max_term = np.max((f - uto) / (z - uto), axis=1)
        sum_term = self.rho_sum * np.sum((f) / (z - uto), axis=1)

        return max_term + sum_term


class PointMethodASF(ASFBase):
    """Implementation of the reference point based ASF.

    Args:
        nadir (np.ndarray): The nadir point.
        ideal (np.ndarray): The ideal point.
        rho (float): A small number to form the utopian point.
        rho_sum (float): A small number to be used as a weight for the sum
            term.

    References:
        Miettinen, K. & Mäkelä, Marko M.
        Synchronous approach in interactive multiobjective optimization
        European Journal of Operational Research, 2006, 170, 909-922
    """

    def __init__(self, nadir: np.ndarray, ideal: np.ndarray, rho: float = 1e-6, rho_sum: float = 1e-6):
        self.nadir = nadir
        self.ideal = ideal
        self.rho = rho
        self.rho_sum = rho_sum

    def __call__(self, objective_vectors: np.ndarray, reference_point: np.ndarray):
        # assure this function works with single objective vectors
        if objective_vectors.ndim == 1:
            f = objective_vectors.reshape((1, -1))
        else:
            f = objective_vectors

        z = reference_point
        nad = self.nadir
        uto = self.ideal - self.rho

        max_term = np.max((f - z) / (nad - uto), axis=1)
        sum_term = self.rho_sum * np.sum((f) / (nad - uto), axis=1)

        return max_term + sum_term


class AugmentedGuessASF(ASFBase):
    """Implementation of the augmented GUESS related ASF.

    Args:
        nadir (np.ndarray): The nadir point.
        ideal (np.ndarray): The ideal point.
        index_to_exclude (List[int]): The indices of the objective functions to
            be excluded in calculating the first term of the ASF.
        rho (float): A small number to form the utopian point.
        rho_sum (float): A small number to be used as a weight for the sum
            term.

    References:
        Miettinen, K. & Mäkelä, Marko M.
        Synchronous approach in interactive multiobjective optimization
        European Journal of Operational Research, 2006, 170, 909-922
    """

    def __init__(
        self,
        nadir: np.ndarray,
        ideal: np.ndarray,
        index_to_exclude: List[int],
        rho: float = 1e-6,
        rho_sum: float = 1e-6,
    ):
        self.nadir = nadir
        self.ideal = ideal
        self.index_to_exclude = index_to_exclude
        self.rho = rho
        self.rho_sum = rho_sum

    def __call__(self, objective_vectors: np.ndarray, reference_point: np.ndarray):
        # assure this function works with single objective vectors
        if objective_vectors.ndim == 1:
            f = objective_vectors.reshape((1, -1))
        else:
            f = objective_vectors

        if reference_point.ndim == 1:
            z = reference_point
        elif reference_point.ndim == 2 and reference_point.shape[0] == 1:
            z = reference_point[0]
        else:
            msg = "Error interpreting reference point"
            raise ASFError(msg)

        nad = self.nadir
        uto = self.ideal - self.rho
        ex_mask = np.full((f.shape[1]), True, dtype=bool)
        ex_mask[self.index_to_exclude] = False

        max_term = np.max((f[:, ex_mask] - nad[ex_mask]) / (nad[ex_mask] - z[ex_mask]), axis=1)
        sum_term_1 = self.rho_sum * np.sum((f[:, ex_mask]) / (nad[ex_mask] - z[ex_mask]), axis=1)
        # avoid division by zeros
        sum_term_2 = self.rho_sum * np.sum((f[:, ~ex_mask]) / (nad[~ex_mask] - uto[~ex_mask]), axis=1)

        return max_term + sum_term_1 + sum_term_2


class GuessASF(ASFBase):
    """Implementation of the naive or GUESS ASF.

    Args:
        nadir (np.ndarray): The nadir point of the problem being scalarized.

    References:
        Miettinen, K., Mäkelä, M.
        On scalarizing functions in multiobjective optimization
        OR Spectrum 24, 193–213 (2002)
    """

    def __init__(self, nadir: np.ndarray):
        self.nadir = nadir

    def __call__(self, objective_vectors: np.ndarray, reference_point: np.ndarray):
        nad = self.nadir
        f = np.atleast_2d(objective_vectors)
        z = reference_point

        max_term = np.max((f - nad) / (nad - z), axis=1)

        return max_term
