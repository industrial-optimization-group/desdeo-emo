from random import sample


def tour_select(individuals, tournament_size):
    aspirants = sample(list(enumerate(individuals)), tournament_size)
    aspirants.sort(key=lambda ind: ind[1].fitness)

    return aspirants[0][0]
