import numpy as np


class BP_mutation:
    def __init__(
        self,
        lower_limits: np.ndarray,
        upper_limits: np.ndarray,
        ProM: float = None,
        DisM: float = 20,
    ):
        self.lower_limits = lower_limits
        self.upper_limits = upper_limits
        if ProM is None:
            self.ProM = 1/len(lower_limits)
        else:
            self.ProM = ProM
        self.DisM = DisM

    def do(self, offspring: np.ndarray):
        """Conduct bounded polynomial mutation. Return the mutated individuals.

        Parameters
        ----------
        offspring : np.ndarray
            The array of offsprings to be mutated.

        Returns
        -------
        np.ndarray
            The mutated offsprings
        """
        min_val = np.ones_like(offspring) * self.lower_limits
        max_val = np.ones_like(offspring) * self.upper_limits
        k = np.random.random(offspring.shape)
        miu = np.random.random(offspring.shape)
        temp = np.logical_and((k <= self.ProM), (miu < 0.5))
        offspring_scaled = (offspring - min_val) / (max_val - min_val)
        offspring[temp] = offspring[temp] + (
            (max_val[temp] - min_val[temp])
            * (
                (
                    2 * miu[temp]
                    + (1 - 2 * miu[temp])
                    * (1 - offspring_scaled[temp]) ** (self.DisM + 1)
                )
                ** (1 / (self.DisM + 1))
                - 1
            )
        )
        temp = np.logical_and((k <= self.ProM), (miu >= 0.5))
        offspring[temp] = offspring[temp] + (
            (max_val[temp] - min_val[temp])
            * (
                1
                - (
                    2 * (1 - miu[temp])
                    + 2 * (miu[temp] - 0.5) * offspring_scaled[temp] ** (self.DisM + 1)
                )
                ** (1 / (self.DisM + 1))
            )
        )
        offspring[offspring > max_val] = max_val[offspring > max_val]
        offspring[offspring < min_val] = min_val[offspring < min_val]
        return offspring
