from pyrvea.EAs.baseEA import BaseEA
from pyrvea.Selection.PPGA_select import ppga_select
from random import choice, sample
import numpy as np
from pyrvea.Population.Population import Population


class PPGA(BaseEA):
    """Predatory-Prey algorithm

    If you have any questions about the code, please contact:

    Bhupinder Saini: bhupinder.s.saini@jyu.fi

    Project researcher at University of Jyväskylä.

    Parameters
    ----------
    population : object
        The population object
    ea_parameters : dict
        PPGA specific parameters

    References
    ----------
    """

    def __init__(self, population: "Population", EA_parameters: dict = {}):

        self.params = self.set_params(population, **EA_parameters)
        self.lattice = Lattice(60, 60, self.params)
        #self._next_iteration(population)

    def set_params(
        self,
        population: "Population",
        population_size: int = None,
        target_pop_size: int = 300,
        generations_per_iteration: int = 10,
        iterations: int = 10,
        interact: bool = False,
        plotting: bool = True,
        logging: list = False,
        logfile=None,
        opt: bool = False,
        crossover_type: str = None,
        mutation_type: str = None,
        kill_interval: int = 7,
        prob_crossover: float = 0.8,
        prob_mutation: float = 0.3,
        prob_prey_move: float = 0.3,
        mut_strength: float = 0.7,
    ):
        """Set up the parameters.

        Parameters
        ----------
        population : Population
            Population object
        population_size : int
            Population Size
        target_pop_size : int
            Target population size
        generations_per_iteration : int
            Number of generations per iteration.
        iterations : int
            Total Number of iterations.
        plotting : bool
            Useless really.
        logging : list = [bool, logfile]
            If true, append parameters to a logfile
        prob_crossover : float
            Probability of crossover occurring
        prob_mutation : float
            Probability of mutation occurring
        mut_strength : float
            Strength of the mutation

        Returns
        -------

        """

        ppgaparams = {
            "population": population,
            "population_size": population.pop_size,
            "target_pop_size": target_pop_size,
            "predator_pop_size": 50,
            "prey_max_moves": 10,
            "prob_prey_move": prob_prey_move,
            "offspring_place_attempts": 10,
            "generations": generations_per_iteration,
            "iterations": iterations,
            "total_generations": iterations * generations_per_iteration,
            "interact": interact,
            "plotting": plotting,
            "logging": logging,
            "opt": opt,
            "logfile": logfile,
            "current_iteration_gen_count": 0,
            "current_total_gen_count": 0,
            "current_iteration_count": 0,
            "prob_crossover": prob_crossover,
            "prob_mutation": prob_mutation,
            "mut_strength": mut_strength,
            "kill_interval": kill_interval,
            "max_rank": 20,
        }

        # If logging enabled, write params to file
        if ppgaparams["logging"]:
            for k, v in ppgaparams.items():
                print(k, v, file=ppgaparams["logfile"])

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
            self.params["current_total_gen_count"] += 1
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

        # Predator max moves for gen
        self.params["predator_max_moves"] = int(
            (
                len(self.params["population"].individuals)
                - self.params["target_pop_size"]
            )
            / self.params["predator_pop_size"]
        )

        # Move prey
        self.lattice.move_prey()

        # Choose and create offspring
        offspring = []

        # Calculate standard deviation
        self.params["std_dev"] = (4 / 3) * (
                1
                - self.params["current_total_gen_count"]
                / self.params["total_generations"]
        )

        # If optimizing instead of training, perform crossover over the entire pop at once
        if self.params["opt"]:
            offspring = population.mate(params=self.params)
        else:
            for ind in range(len(population.individuals)):

                mate_idx = self.lattice.choose_mate(ind)
                if not mate_idx:
                    continue
                else:
                    mating_pop = [ind, mate_idx]
                    offsprings = population.mate(mating_pop, self.params)
                    for n in offsprings:
                        offspring.append(n)

        # Try to place the offspring to lattice, add to population if successful
        placed_indices = self.lattice.place_offspring(len(offspring))

        # Remove from offsprings the ones that didn't get placed
        mask = np.ones(len(offspring), dtype=bool)
        mask[placed_indices] = False
        offspring = np.array(offspring)[~mask]

        # Add the successfully placed offspring to the population
        population.add(offspring)

        # Kill bad individuals every n generations
        if (
            self.params["current_iteration_gen_count"] % self.params["kill_interval"]
            == 0
        ):
            selected = self.select(population, self.params["max_rank"])
            self.lattice.update_lattice(selected)
            population.delete_or_keep(selected, "delete")

        # Move predators
        self.lattice.move_predator()

    def _run_interruption(self, population: "Population"):
        """Run the interruption phase of PPGA.

        Use this phase to make changes to PPGA.params or other objects.

        Parameters
        ----------
        population : Population
        """
        if self.params["interact"]:
            pass

        #if population.plotting:
        #    population.plot_objectives()

    def select(self, population, max_rank=20) -> list:
        """Of the population, individuals lower than max_rank are selected.
        Return indices of selected individuals.

        Parameters
        ----------
        population : Population
            Contains the current population and problem
            information.
        max_rank : int
            Select only individuals lower than max_rank

        Returns
        -------
        list
            List of indices of individuals to be selected.
        """
        selection = ppga_select(population.fitness, max_rank)
        return selection

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
        self.arr = np.zeros((self.size_y, self.size_x), int)
        self.predator_pop = np.empty((0, 1))
        self.predators_loc = []
        self.preys_loc = []
        self.params = params
        self.init_predators()
        self.init_prey()

    def init_predators(self):
        """Initialize the predator population, linearly distributed in [0,1]
        and place them in the lattice randomly"""

        # Initialize the predator population
        self.predator_pop = np.linspace(0, 1, num=self.params["predator_pop_size"])

        # Take random indices from free (==zero) lattice spaces
        free_space = np.transpose(np.nonzero(self.arr == 0))
        indices = sample(free_space.tolist(), self.predator_pop.shape[0])

        for i in range(self.predator_pop.shape[0]):

            # Keep track of predator locations in a list
            self.predators_loc.append([indices[i][0], indices[i][1]])

            # +1 to offset zero index individual, set negative number to
            # identify predators from prey in the lattice
            self.arr[indices[i][0]][indices[i][1]] = int(-1 * (i + 1))

    def init_prey(self):
        """Find an empty position in the lattice and place the prey"""

        # Take random indices from free (==zero) lattice spaces
        free_space = np.transpose(np.nonzero(self.arr == 0))
        indices = sample(
            free_space.tolist(), len(self.params["population"].individuals)
        )

        for i in range(len(self.params["population"].individuals)):

            # Keep track of preys in a list
            self.preys_loc.append([indices[i][0], indices[i][1]])

            # +1 to offset zero index individual
            self.arr[indices[i][0]][indices[i][1]] = int(i + 1)

    def move_prey(self):
        """Find an empty position in the moore neighbourhood for the Prey to move in.
        Ensure that the lattice is toroidal."""

        for prey, pos in enumerate(self.preys_loc):

            if np.random.random() < self.params["prob_prey_move"]:

                for i in range(self.params["prey_max_moves"]):

                    neighbours = self.neighbours(self.arr, pos[0], pos[1])

                    dy = np.random.randint(neighbours.shape[0])
                    dx = np.random.randint(neighbours.shape[1])

                    # If neighbouring cell is occupied, skip turn
                    if neighbours[dy][dx] != 0:
                        continue

                    dest_y = dy - 1 + pos[0]
                    dest_x = dx - 1 + pos[1]

                    # Check boundaries of the lattice
                    if dest_y not in range(self.size_y) or dest_x not in range(
                        self.size_x
                    ):
                        dest_y, dest_x = self.lattice_wrap_idx(
                            (dest_y, dest_x), np.shape(self.arr)
                        )

                    # Move prey, clear previous location
                    self.arr[dest_y][dest_x] = int(prey + 1)
                    self.arr[pos[0]][pos[1]] = 0

                    # Update prey location in the list
                    pos[0], pos[1] = dest_y, dest_x

    def choose_mate(self, ind):
        """Choose a mate from moore neighbourhood, return None if no mates
        are found."""

        neighbours = self.neighbours(
            self.arr, self.preys_loc[ind][0], self.preys_loc[ind][1]
        )
        mates = neighbours[(neighbours > 0) & (neighbours != ind + 1)]
        if len(mates) < 1:
            return
        else:
            # -1 for lattice offset
            mate = int(choice(mates)) - 1
            return mate

    def place_offspring(self, offspring):
        """Try to place the offsprings to the lattice.

        Parameters
        ----------
        offspring : int
            number of offsprings

        Returns
        -------
            list of successfully placed offspring indices
        """

        placed_offspring = []

        for i in range(offspring):

            y = np.random.randint(self.size_y)
            x = np.random.randint(self.size_x)

            for j in range(self.params["offspring_place_attempts"]):
                if self.arr[y][x] != 0:
                    continue

                else:

                    # Keep track of offspring in a list
                    if self.arr[y][x] == 0:
                        # Append the offspring to the list of preys.
                        # len(self.preys_loc) is the index of the current last prey in the list
                        self.arr[y][x] = int(len(self.preys_loc) + 1)
                        self.preys_loc.append([y, x])
                        placed_offspring.append(i)

        return placed_offspring

    def move_predator(self):
        """Find an empty position in the moore neighbourhood for the Predators to move in."""

        # Calculate predator moves
        # predator_max_moves = int(
        #     (
        #         len(self.params["population"].individuals)
        #         - self.params["target_pop_size"]
        #     )
        #     / self.params["predator_pop_size"]
        # )

        predator_max_moves = self.params["predator_max_moves"]

        # Track killed preys in list and remove them at the end of the function
        to_be_killed = []

        for predator, pos in enumerate(self.predators_loc):

            for i in range(predator_max_moves):

                neighbours = self.neighbours(self.arr, pos[0], pos[1])
                targets = neighbours[neighbours > 0]

                # If preys found in the neighbourhood, calculate their fitness and kill the weakest
                if len(targets) > 0:
                    fitness = []
                    for target in targets:

                        obj1 = self.params["population"].fitness[target - 1][0]
                        obj2 = self.params["population"].fitness[target - 1][1]

                        fc = (
                            self.predator_pop[predator] * obj1
                            + (1 - self.predator_pop[predator]) * obj2
                        )

                        fitness.append((fc, target))
                        fitness.sort()
                        weakest_prey = fitness[-1][1] - 1

                    # Kill the weakest prey and move the predator to its place
                    self.arr[self.preys_loc[weakest_prey][0]][
                        self.preys_loc[weakest_prey][1]
                    ] = -1 * (predator + 1)

                    # Set the old predator location to zero
                    self.arr[pos[0]][pos[1]] = 0

                    # Update predator location in the list of predators
                    pos[0], pos[1] = (
                        self.preys_loc[weakest_prey][0],
                        self.preys_loc[weakest_prey][1],
                    )

                    # Set the killed prey as dead in the list of prey locations and end predator turn
                    to_be_killed.append(weakest_prey)
                    self.preys_loc[weakest_prey] = None

                else:
                    dy = np.random.randint(neighbours.shape[0])
                    dx = np.random.randint(neighbours.shape[1])

                    # If neighbouring cell is occupied by another predator, skip turn
                    if neighbours[dy][dx] < 0:
                        continue

                    dest_y = dy - 1 + pos[0]
                    dest_x = dx - 1 + pos[1]

                    # Check boundaries of the lattice
                    if dest_y not in range(self.size_y) or dest_x not in range(
                        self.size_x
                    ):
                        dest_y, dest_x = self.lattice_wrap_idx(
                            (dest_y, dest_x), np.shape(self.arr)
                        )

                    # Move predator, clear previous location
                    self.arr[dest_y][dest_x] = -1 * (predator + 1)
                    self.arr[pos[0]][pos[1]] = 0

                    # Update predator location in the list
                    pos[0], pos[1] = dest_y, dest_x

        self.params["population"].delete_or_keep(to_be_killed, "delete")
        self.update_lattice()

    def update_lattice(self, selected=None):

        # Remove selected individuals from the lattice
        if selected is not None:
            for i in selected:
                self.arr[self.preys_loc[i][0]][self.preys_loc[i][1]] = 0
                self.preys_loc[i] = None

        # Update the list of prey locations
        updated_preys = [x for x in self.preys_loc if x is not None]
        self.preys_loc = updated_preys

        # Update lattice
        for prey, pos in enumerate(self.preys_loc):
            self.arr[pos[0]][pos[1]] = prey + 1

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
        if not hasattr(index, "__iter__"):
            return index  # handle integer slices
        if len(index) != len(lattice_shape):
            return index  # must reference a scalar
        if any(type(i) == slice for i in index):
            return index  # slices not supported
        if len(index) == len(lattice_shape):  # periodic indexing of scalars
            mod_index = tuple(((i % s + s) % s for i, s in zip(index, lattice_shape)))
            return mod_index
        raise ValueError("Unexpected index: {}".format(index))

    @staticmethod
    def neighbours(arr, x, y, n=3):
        """Given a 2D-array, returns an nxn array whose "center" element is arr[x,y]

        Parameters
        ----------
        arr : ndarray
            A 2D-array where to get the neighbouring cells
        x : int
            X coordinate for the center element
        y : int
            Y coordinate for the center element
        n : int
            Radius of the neighbourhood

        Returns
        -------
        The neighbouring cells of x, y in radius n. Defaults to Moore neighbourhood (n=3)
        """
        arr = np.roll(np.roll(arr, shift=-x + 1, axis=0), shift=-y + 1, axis=1)
        return arr[:n, :n]
