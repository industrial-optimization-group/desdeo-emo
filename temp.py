# Initialize the probabilities for the genetic algorithm
# These should go to the PPGA module
# self.prob_omit_max = 0.99
# self.prob_omit_min = 0.1
# self.prob_crossover = 0.8
# self.prob_mutation = 0.3
# self.mutation_strength = 0.7

# Eliminate some weights / set some weights to zero
# Also in the PPGA module
# flag = bn.rvs(p=1 - self.prob_omit_min, size=np.shape(individuals))
# random_numbers = np.zeros(np.shape(individuals))
# individuals[flag == 0] = random_numbers[flag == 0]

# Initialize Predator population
# predators = np.linspace(0, 1, num=pop_size)

# Scaling
# Taken from Population.create_new_individuals()
# individuals = (
#    individuals * (self.upper_limits - self.lower_limits)
#    + self.lower_limits
# )
