import numpy as np
import random
from math import ceil
from pyDOE import lhs


def create_new_individuals(design, problem, pop_size=None):
    """Create new individuals to the population.

    The individuals can be created randomly, by LHS design, or can be passed by the
    user.

    Design does not apply in case of EvoNN and EvoDN2 problem, where neural networks
    are created as individuals.

    Parameters
    ----------
    design : str, optional
        Describe the method of creation of new individuals.
        "RandomDesign" creates individuals randomly.
        "LHSDesign" creates individuals using Latin hypercube sampling.
        "EvoNN" creates Artificial Neural Networks as individuals.
        "EvoDN2" creates Deep Neural Networks.
    problem : baseProblem
        An object of the class Problem
    pop_size : int, optional
        Number of individuals in the population. If none, some default population
        size based on number of objectives is chosen.

    Returns
    -------
    individuals : list
        A list of individuals.

    """

    if pop_size is None:
        pop_size_options = [50, 105, 120, 126, 132, 112, 156, 90, 275]
        pop_size = pop_size_options[problem.num_of_objectives - 2]

    if design == "RandomDesign":
        lower_limits = np.asarray(problem.get_variable_lower_bounds())
        upper_limits = np.asarray(problem.get_variable_upper_bounds())
        individuals = np.random.random((pop_size, problem.n_of_variables))
        # Scaling
        individuals = individuals * (upper_limits - lower_limits) + lower_limits

        return individuals

    elif design == "LHSDesign":
        lower_limits = np.asarray(problem.get_variable_lower_bounds())
        upper_limits = np.asarray(problem.get_variable_upper_bounds())
        individuals = lhs(problem.n_of_variables, samples=pop_size)
        # Scaling
        individuals = individuals * (upper_limits - lower_limits) + lower_limits

        return individuals

    elif design == "EvoNN":

        """Create a population of neural networks for the EvoNN algorithm.

        Individuals are 2d arrays representing the weight matrices of the NNs.
        One extra row is added for bias.

        """

        w_low = problem.params["w_low"]
        w_high = problem.params["w_high"]
        in_nodes = problem.num_of_variables
        num_nodes = problem.params["num_nodes"]
        prob_omit = problem.params["prob_omit"]

        individuals = np.random.uniform(
            w_low, w_high, size=(pop_size, in_nodes, num_nodes)
        )

        # Randomly set some weights to zero
        zeros = np.random.choice(
            np.arange(individuals.size), ceil(individuals.size * prob_omit)
        )
        individuals.ravel()[zeros] = 0

        # Set bias
        individuals = np.insert(individuals, 0, 1, axis=1)

        return individuals

    elif design == "EvoDN2":
        """Create a population of deep neural networks (DNNs) for the EvoDN2 algorithm.

        Each individual is a list of subnets, and each subnet contains a random amount
        of layers and
        nodes per layer. The subnets are evolved via evolutionary algorithms, and they
        converge
        on the final linear layer of the DNN.
        """

        individuals = []
        for i in range(problem.params["pop_size"]):
            nets = []
            for j in range(problem.params["num_subnets"]):

                layers = []
                num_layers = np.random.randint(1, problem.params["max_layers"])
                in_nodes = len(problem.subsets[j])

                for k in range(num_layers):
                    out_nodes = random.randint(2, problem.params["max_nodes"])
                    net = np.random.uniform(
                        problem.params["w_low"],
                        problem.params["w_high"],
                        size=(in_nodes, out_nodes),
                    )
                    # Randomly set some weights to zero
                    zeros = np.random.choice(
                        np.arange(net.size),
                        ceil(net.size * problem.params["prob_omit"]),
                    )
                    net.ravel()[zeros] = 0

                    # Add bias
                    net = np.insert(net, 0, 1, axis=0)
                    in_nodes = out_nodes
                    layers.append(net)

                nets.append(layers)

            individuals.append(nets)

        return individuals

    elif design == "BioGP":
        return problem.create_individuals()
