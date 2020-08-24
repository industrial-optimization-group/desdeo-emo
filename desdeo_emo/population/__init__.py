"""This module provides classes and methods which implement populations in an EA."""

__all__ = ["Population", "create_new_individuals", "SurrogatePopulation"]

from desdeo_emo.population.CreateIndividuals import create_new_individuals
from desdeo_emo.population.Population import Population
from desdeo_emo.population.SurrogatePopulation import SurrogatePopulation
