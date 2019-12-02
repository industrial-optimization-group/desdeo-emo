from desdeo_emo.population.Population import BasePopulation, Population


class SurrogatePopulation(Population, BasePopulation):
    def __init__(
        self, problem, pop_size: int, initial_pop, xover, mutation, recombination
    ):
        BasePopulation.__init__(self, problem, pop_size)
        self.add(initial_pop)
        self.xover = xover
        self.mutation = mutation
        self.recombination = recombination
