from random import choice, sample
import numpy as np
from pygmo import fast_non_dominated_sorting as nds
from desdeo_emo.EAs.BaseEA import BaseEA, eaError


class PPGA(BaseEA):
    """Predatory-Prey genetic algorithm.

    A population of prey signify the various models or solutions to the problem at hand.
    Weaker prey, i.e. bad models or solutions, are killed by predators.
    The predators and prey are placed in a lattice, in which they are free to roam.

    In each generation, each predator gets a certain number of turns to move about and
    hunt in its neighbourhood, killing the weaker prey, according to a fitness criteria.
    After this, each prey gets a certain number of moves to pursue a random walk and to
    reproduce with other prey. Each reproduction step generates two new prey from two
    parents, by crossing over their attributes and adding random mutations. After each
    prey has completed its move, the whole process starts again.

    As the weaker individuals get eliminated in each generation, the population as a
    whole becomes more fit, i.e. the individuals get closer to the true pareto-optimal
    solutions.

    If you have any questions about the code, please contact:

    Bhupinder Saini: bhupinder.s.saini@jyu.fi
    Project researcher at University of Jyväskylä.

    Parameters
    ----------
    population : object
        The population object


    Notes
    -----
    The algorithm has been created earlier in MATLAB, and this Python implementation
    has been using that code as a basis. See references [4] for the study during which
    the original MATLAB version was created.
    Python code has been written by Niko Rissanen under the supervision of professor
    Nirupam Chakraborti.

    For the MATLAB implementation, see:
    N. Chakraborti. Data-Driven Bi-Objective Genetic Algorithms EvoNN and BioGP and
    Their Applications in Metallurgical and Materials Domain. In Datta, Shubhabrata,
    Davim, J. Paulo (eds.), Computational Approaches to Materials Design: Theoretical
    and Practical Aspects, pp. 346-369, 2016.

    References
    ----------
    [1] Laumanns, M., Rudolph, G., & Schwefel, H. P. (1998). A spatial predator-prey
    approach to multi-objective
    optimization: A preliminary study.
    In International Conference on Parallel Problem Solving from Nature (pp. 241-249).
    Springer, Berlin, Heidelberg.

    [2] Li, X. (2003). A real-coded predator-prey genetic algorithm for multiobjective
    optimization. In International
    Conference on Evolutionary Multi-Criterion Optimization (pp. 207-221). Springer,
    Berlin, Heidelberg.

    [3] Chakraborti, N. (2014). Strategies for evolutionary data driven modeling in
    chemical and metallurgical Systems.
    In Applications of Metaheuristics in Process Engineering (pp. 89-122). Springer,
    Cham.

    [4] Pettersson, F., Chakraborti, N., & Saxén, H. (2007). A genetic algorithms based
    multi-objective neural net
    applied to noisy blast furnace data. Applied Soft Computing, 7(1), 387-397.

    """

    def __init__(
        self,
        problem,
        population_size: int = 100,
        population_params=None,
        initial_population=None,
        n_iterations: int = 10,
        n_gen_per_iter: int = 10,
        predator_pop_size: int = 50,
        prey_max_moves: int = 10,
        prob_prey_move: float = 0.3,
        offspring_place_attempts: int = 10,
        kill_interval: int = 7,
        max_rank: int = 20,
        neighbourhood_radius: int = 3,
    ):
        super().__init__(n_gen_per_iter=n_gen_per_iter, n_iterations=n_iterations)
        if initial_population is None:
            msg = "Provide initial population"
            raise eaError(msg)
        self.population = initial_population
        self.target_pop_size = population_size
        self.predator_pop_size: int = predator_pop_size
        self.prey_max_moves: int = prey_max_moves
        self.prob_prey_move: float = prob_prey_move
        self.offspring_place_attempts: int = offspring_place_attempts
        self.kill_interval: int = kill_interval
        self.max_rank: int = max_rank
        self.neighbourhood_radius: int = neighbourhood_radius
        self.lattice = Lattice(
            size_x=60,
            size_y=60,
            population=self.population,
            predator_pop_size=predator_pop_size,
            target_pop_size=self.target_pop_size,
            prob_prey_move=prob_prey_move,
            prey_max_moves=prey_max_moves,
            offspring_place_attempts=offspring_place_attempts,
            neighbourhood_radius=neighbourhood_radius,
        )

    def _next_gen(self):
        """Run one generation of PPGA.

        Intended to be used by next_iteration.

        Parameters
        ----------
        population: "Population"
            Population object
        """

        # Move prey and select neighbours for breeding
        mating_pop = self.lattice.move_prey()

        offspring = self.population.mate(mating_pop)

        # Try to place the offspring to lattice, add to population if successful
        placed_indices = self.lattice.place_offspring(len(offspring))

        # Remove from offsprings the ones that didn't get placed
        mask = np.ones(len(offspring), dtype=bool)
        mask[placed_indices] = False
        offspring = np.asarray(offspring)[~mask]

        # Add the successfully placed offspring to the population
        self.population.add(offspring)
        self._current_gen_count += 1
        self._gen_count_in_curr_iteration += 1
        self._function_evaluation_count += offspring.shape[0]

        # Kill bad individuals every n generations
        if self._current_gen_count % self.kill_interval == 0:
            selected = self.select(self.population, self.max_rank)
            self.lattice.update_lattice(selected)
            self.population.delete(selected)

        # Move predators
        self.lattice.move_predator()

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
        # Calculating fronts and ranks
        _, _, _, rank = nds(population.fitness)
        selection = np.nonzero(rank > max_rank)
        return selection[0]

    def manage_preferences(self, preference=None):
        return


class Lattice:
    """The 2-dimensional toroidal lattice in which the predators and prey are placed.

    Attributes
    ----------
    size_x : int
        Width of the lattice.
    size_y : int
        Height of the lattice.
    lattice : ndarray
        2d array for the lattice.
    predator_pop : ndarray
        The predator population.
    predators_loc : list
        Location (x, y) of predators on the lattice.
    preys_loc : list
        Location (x, y) of preys on the lattice.

    """

    def __init__(
        self,
        size_x,
        size_y,
        population,
        predator_pop_size,
        target_pop_size,
        prob_prey_move,
        prey_max_moves,
        offspring_place_attempts,
        neighbourhood_radius,
    ):

        self.size_x = size_x
        self.size_y = size_y
        self.population = population
        self.predator_pop_size = predator_pop_size
        self.target_pop_size = target_pop_size
        self.prob_prey_move = prob_prey_move
        self.prey_max_moves = prey_max_moves
        self.offspring_place_attempts = offspring_place_attempts
        self.neighbourhood_radius = neighbourhood_radius

        self.lattice = np.zeros((self.size_y, self.size_x), int)
        self.predator_pop = np.empty((0, 1))
        self.predators_loc = []
        self.preys_loc = []
        self.mating_pop = []
        self.init_predators()
        self.init_prey()

    def init_predators(self):
        """Initialize the predator population, linearly distributed in [0,1]
        and place them in the lattice randomly."""

        # Initialize the predator population
        self.predator_pop = np.linspace(0, 1, num=self.predator_pop_size)

        # Take random indices from free (==zero) lattice spaces
        free_space = np.transpose(np.nonzero(self.lattice == 0))
        indices = sample(free_space.tolist(), self.predator_pop.shape[0])

        for i in range(self.predator_pop.shape[0]):

            # Keep track of predator locations in a list
            self.predators_loc.append([indices[i][0], indices[i][1]])

            # +1 to offset zero index individual, set negative number to
            # identify predators from prey in the lattice
            self.lattice[indices[i][0]][indices[i][1]] = int(-1 * (i + 1))

    def init_prey(self):
        """Find an empty position in the lattice and place the prey."""

        # Take random indices from free (==zero) lattice spaces
        free_space = np.transpose(np.nonzero(self.lattice == 0))
        indices = sample(free_space.tolist(), len(self.population.individuals))

        for i in range(len(self.population.individuals)):

            # Keep track of preys in a list
            self.preys_loc.append([indices[i][0], indices[i][1]])

            # +1 to offset zero index individual
            self.lattice[indices[i][0]][indices[i][1]] = int(i + 1)

    def move_prey(self):
        """Find an empty position in prey neighbourhood for the prey to move in,
        and choose a mate for breeding if any available.

        Returns
        -------
        mating_pop : list
            List of parent indices to use for mating
        """
        mating_pop = []
        for prey, pos in enumerate(self.preys_loc):

            if np.random.random() < self.prob_prey_move:

                for i in range(self.prey_max_moves):

                    neighbours = self.neighbours(self.lattice, pos[0], pos[1])

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
                            (dest_y, dest_x), np.shape(self.lattice)
                        )

                    # Move prey, clear previous location
                    self.lattice[dest_y][dest_x] = int(prey + 1)
                    self.lattice[pos[0]][pos[1]] = 0

                    # Update prey location in the list
                    pos[0], pos[1] = dest_y, dest_x

            neighbours = self.neighbours(
                self.lattice, self.preys_loc[prey][0], self.preys_loc[prey][1]
            )
            mates = neighbours[(neighbours > 0) & (neighbours != prey + 1)]
            if len(mates) < 1:
                continue
            else:
                # -1 for lattice offset
                mate = int(choice(mates)) - 1
                mating_pop.append([prey, mate])
        if mating_pop ==[]:
            raise eaError("What's ahppening?!")
        return mating_pop

    def place_offspring(self, offspring):
        """Try to place the offsprings to the lattice. If no empty spot found within
        number of max attempts, do not place.

        Parameters
        ----------
        offspring : int
            number of offsprings

        Returns
        -------
        list
            Successfully placed offspring indices.
        """

        # Keep track of offspring in a list
        placed_offspring = []

        for i in range(offspring):

            y = np.random.randint(self.size_y)
            x = np.random.randint(self.size_x)

            for j in range(self.offspring_place_attempts):
                if self.lattice[y][x] != 0:
                    continue

                else:

                    if self.lattice[y][x] == 0:
                        # Append the offspring to the list of preys.
                        # len(self.preys_loc) is the index of the current
                        # last prey in the list
                        self.lattice[y][x] = int(len(self.preys_loc) + 1)
                        self.preys_loc.append([y, x])
                        placed_offspring.append(i)

        return placed_offspring

    def move_predator(self):
        """Find an empty position in the predator neighbourhood for the predators to move in,
        move the predator and kill the weakest prey in its neighbourhood, if any.
        Repeat until > predator_max_moves."""

        predator_max_moves = int(
            (len(self.population.individuals) - self.target_pop_size)
            / self.predator_pop_size
        )

        # Track killed preys in list and remove them at the end of the function
        to_be_killed = []

        for predator, pos in enumerate(self.predators_loc):

            for i in range(predator_max_moves):

                neighbours = self.neighbours(
                    self.lattice, pos[0], pos[1], n=self.neighbourhood_radius
                )
                targets = neighbours[neighbours > 0]

                # If preys found in the neighbourhood,
                # calculate their fitness and kill the weakest
                if len(targets) > 0:
                    fitness = []
                    weakest_prey = None
                    for target in targets:

                        obj1 = self.population.fitness[target - 1][0]
                        obj2 = self.population.fitness[target - 1][1]

                        fc = (
                            self.predator_pop[predator] * obj1
                            + (1 - self.predator_pop[predator]) * obj2
                        )

                        fitness.append((fc, target))
                        fitness.sort()
                        weakest_prey = fitness[-1][1] - 1

                    # Kill the weakest prey and move the predator to its place
                    self.lattice[self.preys_loc[weakest_prey][0]][
                        self.preys_loc[weakest_prey][1]
                    ] = -1 * (predator + 1)

                    # Set the old predator location to zero
                    self.lattice[pos[0]][pos[1]] = 0

                    # Update predator location in the list of predators
                    pos[0], pos[1] = (
                        self.preys_loc[weakest_prey][0],
                        self.preys_loc[weakest_prey][1],
                    )

                    # Set the killed prey as dead in the list of prey locations and
                    # end predator turn
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
                            (dest_y, dest_x), np.shape(self.lattice)
                        )

                    # Move predator, clear previous location
                    self.lattice[dest_y][dest_x] = -1 * (predator + 1)
                    self.lattice[pos[0]][pos[1]] = 0

                    # Update predator location in the list
                    pos[0], pos[1] = dest_y, dest_x

        # Remove killed prey from population
        self.population.delete(to_be_killed)
        self.update_lattice()

    def update_lattice(self, selected=None):
        """Update prey positions in the lattice.

        Parameters
        ----------
        selected : list
            Indices of preys to be removed from the lattice.

        """
        # Remove selected individuals from the lattice
        if selected is not None:
            for i in selected:
                self.lattice[self.preys_loc[i][0]][self.preys_loc[i][1]] = 0
                self.preys_loc[i] = None

        # Update the list of prey locations
        updated_preys = [x for x in self.preys_loc if x is not None]
        self.preys_loc = updated_preys

        # Update lattice
        for prey, pos in enumerate(self.preys_loc):
            self.lattice[pos[0]][pos[1]] = prey + 1

    @staticmethod
    def lattice_wrap_idx(index, lattice_shape):
        """Returns periodic lattice index
        for a given iterable index.

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
        """Given a 2D-array, returns an n*n array whose "center" element is arr[x,y]

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
        The neighbouring cells of x, y in radius n*n.
        Defaults to Moore neighbourhood (n=3).
        """

        arr = np.roll(np.roll(arr, shift=-x + 1, axis=0), shift=-y + 1, axis=1)
        return arr[:n, :n]
