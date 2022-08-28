from desdeo_problem.testproblems.TestProblems import test_problem_builder
from desdeo_emo.EAs.RVEA import RVEA
from desdeo_emo.EAs.NSGAIII import NSGAIII
from desdeo_emo.EAs import MOEA_D
import plotly.express as ex


dtlz2 = test_problem_builder("DTLZ7", n_of_variables=12, n_of_objectives=3)
evolver = MOEA_D(dtlz2, n_iterations=10, n_gen_per_iter=30, save_non_dominated=True)
evolver.start()  # Note, this is important!!!
while evolver.continue_evolution():
    evolver.iterate()
    print(f"Running iteration {evolver._iteration_counter}")

# Non dominated archive
ex.scatter_3d(
    x=evolver.non_dominated["objectives"][:, 0],
    y=evolver.non_dominated["objectives"][:, 1],
    z=evolver.non_dominated["objectives"][:, 2],
).show()

print(f"Number of non-dominated solutions: {len(evolver.non_dominated['objectives'])}")

# Final population
ex.scatter_3d(
    x=evolver.population.objectives[:, 0],
    y=evolver.population.objectives[:, 1],
    z=evolver.population.objectives[:, 2],
).show()
