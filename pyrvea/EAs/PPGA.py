from pyrvea.EAs.baseEA import BasePPGA
from random import sample
import numpy as np
from scipy.stats import bernoulli as bn

class PPGA(BasePPGA):

    def set_params(
        self,
        population: "Population",
        population_size: int = None,
        target_pop_size: int = 200,
        interact: bool = False,
        a_priori_preference: bool = False,
        generations_per_iteration: int = 10,
        iterations: int = 1,
        plotting: bool = True,
        prob_omit: float = 0.3,
        prob_crossover: float = 0.8,
        prob_mutation: float = 0.3,
        mut_strength: float = 0.7
    ):
        """Set up the parameters.

        Parameters
        ----------
        population : Population
            Population object
        population_size : int
            Population Size
        interact : bool
            bool to enable or disable interaction. Enabled if True
        a_priori_preference : bool
            similar to interact
        generations_per_iteration : int
            Number of generations per iteration.
        iterations : int
            Total Number of iterations.
        plotting : bool
            Useless really.
        prob_crossover : float
            Probability of crossover occurring
        prob_mutation : float
            Probability of mutation occurring
        mut_strength : float
            Strength of the mutation
        prob_omit : float
            Probability of deactivating some connections

        Returns
        -------

        """

        ppgaparams = {
            "population": population,
            "population_size": population_size,
            "target_pop_size": target_pop_size,
            "predator_pop_size": 50,
            "interact": interact,
            "a_priori": a_priori_preference,
            "generations": generations_per_iteration,
            "iterations": iterations,
            "ploton": plotting,
            "current_iteration_gen_count": 0,
            "current_iteration_count": 0,
            "prob_crossover": prob_crossover,
            "prob_mutation": prob_mutation,
            "mut_strength": mut_strength,
            "prob_omit": prob_omit,
            "kill_intrvl": 7,
            "max_rank": 20

        }

        lattice = Lattice(60, 60, ppgaparams)

        return ppgaparams


class Lattice:

    def __init__(self, size_x, size_y, params):

        self.size_x = size_x
        self.size_y = size_y
        self.arr = np.zeros((self.size_y, self.size_x))
        self.place_prey(params)
        self.place_pred(params)

    def place_prey(self, params):
        """Set some of the individual weights to zero, find an empty
        position in the lattice and place the prey"""

        # Eliminate some weights
        flag = bn.rvs(p=1 - params["prob_omit"], size=np.shape(params["population"].individuals))
        random_numbers = np.zeros(np.shape(params["population"].individuals))
        params["population"].individuals[flag == 0] = random_numbers[flag == 0]

        for i in range(np.shape(params["population"].individuals)[0]):

            x = np.random.randint(self.size_x)
            y = np.random.randint(self.size_y)

            while self.arr[x][y] != 0:
                x = np.random.randint(self.size_x)
                y = np.random.randint(self.size_y)

            # +1 to offset zero index individual
            self.arr[y][x] = i+1

    def place_pred(self, params):

        # Initialize Predator population

        predators = np.linspace(0, 1, num=params["predator_pop_size"])

        for i in range(np.shape(predators)[0]):

            x = np.random.randint(self.size_x)
            y = np.random.randint(self.size_y)

            while self.arr[x][y] != 0:
                x = np.random.randint(self.size_x)
                y = np.random.randint(self.size_y)

            # +1 to offset zero index individual, set negative number to
            # identify predators from prey in the lattice

            self.arr[y][x] = -1*(i+1)

    def move_prey(self):
        pass

    def move_pred(self):
        pass