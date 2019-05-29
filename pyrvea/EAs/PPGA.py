from pyrvea.EAs.baseEA import BaseEA
from random import sample
import numpy as np
from random import choice, randint
from scipy.stats import bernoulli as bn
from pyrvea.Population.population_evonn import Population


class PPGA(BaseEA):

    def __init__(self, population: "Population", ea_parameters: dict = None):

        self.params = self.set_params(population, ea_parameters)
        self.lattice = Lattice(60, 60, self.params)
        self._next_iteration(population)

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
            "prey_max_moves": 5,
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

        return ppgaparams

    def _next_iteration(self, population: "Population"):
        """Run one iteration of EA.

        One iteration consists of a constant or variable number of
        generations. This method leaves EA.params unchanged, except the current
        iteration count and gen count.

        Parameters
        ----------
        population : "Population"
            Contains current population
        """
        self.params["current_iteration_gen_count"] = 1
        while self.continue_iteration():
            self._next_gen(population)
            self.params["current_iteration_gen_count"] += 1
        self.params["current_iteration_count"] += 1

    def _next_gen(self, population: "Population"):
        """Run one generation of EA.

        This method leaves method.params unchanged. Intended to be used by
        next_iteration.

        Parameters
        ----------
        population: "Population"
            Population object
        """

        self.lattice.move_prey()

        for ind in range(population.individuals.shape[0]-1):

            mate_idx = self.lattice.choose_mate(ind)
            if not mate_idx:
                continue
            else:
                offspring = population.mate(ind, mate_idx)

        # Try to place the offspring to lattice, add to population if successful
        if self.lattice.place_prey(offspring):
            population.add(offspring)

        # Kill bad individuals every n generations
        if self.params["current_iteration_gen_count"] >= self.params["kill_intrvl"]:
            selected = self.select(population)
            population.delete_or_keep(selected, "delete")

        # Move predators
        self.lattice.move_pred()

    def select(self, population) -> list:
        """Describe a selection mechanism. Return indices of selected
        individuals.

        Parameters
        ----------
        population : Population
            Contains the current population and problem
            information.

        Returns
        -------
        list
            List of indices of individuals to be selected.
        """
        pass

    def continue_iteration(self):
        """Checks whether the current iteration should be continued or not."""
        return self.params["current_iteration_gen_count"] <= self.params["generations"]

    def continue_evolution(self) -> bool:
        """Checks whether the current iteration should be continued or not."""
        pass


class Lattice:

    def __init__(self, size_x, size_y, params):

        self.size_x = size_x
        self.size_y = size_y
        self.arr = np.zeros((self.size_y, self.size_x))
        self.preys = {}
        self.predators = {}
        self.params = params
        self.init_pred()
        self.init_prey()

    def init_pred(self):
        """Initialize the predator population, linearly distributed in [0,1]
        and place them in the lattice randomly"""

        # Initialize the predator population
        predators = np.linspace(0, 1, num=self.params["predator_pop_size"])

        # Take random indices from free (==zero) lattice spaces
        free_space = np.transpose(np.nonzero(self.arr == 0))
        indices = sample(free_space.tolist(), predators.shape[0])

        for i in range(np.shape(predators)[0]):

            # Keep track of predators in a dictionary
            self.predators[i] = [indices[i][0], indices[i][1]]

            # +1 to offset zero index individual, set negative number to
            # identify predators from prey in the lattice
            self.arr[indices[i][0]][indices[i][1]] = -1*(i+1)

    def init_prey(self):
        """Set some of the individual weights to zero, find an empty
        position in the lattice and place the prey"""

        # Eliminate some weights
        flag = bn.rvs(p=1 - self.params["prob_omit"], size=np.shape(self.params["population"].individuals))
        random_numbers = np.zeros(np.shape(self.params["population"].individuals))
        self.params["population"].individuals[flag == 0] = random_numbers[flag == 0]

        # Take random indices from free (==zero) lattice spaces
        free_space = np.transpose(np.nonzero(self.arr == 0))
        indices = sample(free_space.tolist(), self.params["population"].individuals.shape[0])

        for i in range(self.params["population"].individuals.shape[0]):

            # Keep track of preys in a dictionary
            self.preys[i] = [indices[i][0], indices[i][1]]

            # +1 to offset zero index individual
            self.arr[indices[i][0]][indices[i][1]] = i+1

    def move_prey(self):
        """Find an empty position in the moore neighbourhood for the Prey to move in.
        Ensure that the lattice is toroidal."""

        for prey, pos in self.preys.items():
            attempts = 0
            while attempts <= self.params["prey_max_moves"]:

                neighbours = self.neighbours(self.arr, pos[0], pos[1])
                free_space = np.transpose(np.nonzero(neighbours == 0))
                dy, dx = free_space[randint(0, len(free_space)-1)]

                dest_y = dy - 1 + pos[0]
                dest_x = dx - 1 + pos[1]

                # Check boundaries of the lattice
                if dest_y not in range(self.size_y) or dest_x not in range(self.size_x):
                    dest_y, dest_x = self.lattice_wrap_idx((dest_y, dest_x), np.shape(self.arr))

                # Move prey, clear previous location
                self.arr[dest_y][dest_x] = prey+1
                self.arr[pos[0]][pos[1]] = 0

                # Update prey location in the dictionary
                pos[0], pos[1] = dest_y, dest_x

                # Find mates in the moore neighbourhood
                neighbours = self.neighbours(self.arr, pos[0], pos[1])
                mates = neighbours[neighbours > prey+1]
                if len(mates) > 0:
                    break
                else:
                    attempts += 1

    def choose_mate(self, ind):

        neighbours = self.neighbours(self.arr, self.preys[ind][0], self.preys[ind][1])
        mates = neighbours[neighbours > ind + 1]
        if not mates:
            return
        else:
            mate = int(choice(mates))
            return mate

    def move_pred(self):
        pass

    @staticmethod
    def lattice_wrap_idx(index, lattice_shape):
        """Returns periodic lattice index
        for a given iterable index

        Parameters
        ----------
        index : tuple
            one integer for each axis
        lattice_shape : tuple
            the shape of the lattice to index to
        """
        if not hasattr(index, '__iter__'): return index  # handle integer slices
        if len(index) != len(lattice_shape): return index  # must reference a scalar
        if any(type(i) == slice for i in index): return index  # slices not supported
        if len(index) == len(lattice_shape):  # periodic indexing of scalars
            mod_index = tuple(((i % s + s) % s for i, s in zip(index, lattice_shape)))
            return mod_index
        raise ValueError('Unexpected index: {}'.format(index))

    @staticmethod
    def neighbours(arr, x, y, n=3):
        """Given a 2D-array, returns an nxn array whose "center" element is arr[x,y]"""
        arr = np.roll(np.roll(arr, shift=-x + 1, axis=0), shift=-y + 1, axis=1)
        return arr[:n, :n]
