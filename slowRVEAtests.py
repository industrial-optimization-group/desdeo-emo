from pyrvea.Population.Population import Population
from pyrvea.Problem.testProblem import testProblem
from pyrvea.EAs.slowRVEA import slowRVEA
import altair as alt
import pandas as pd
import numpy as np


def main():
    problems = ["DTLZ2", "DTLZ3"]
    numobjs = [3, 6, 8, 10]
    algorithms = {"SmoothRVEA": smoothEvolve, "AbruptRVEA": abruptEvolve}
    foldername = "./results/"
    for problem_name in problems:
        for numobj in numobjs:
            name = problem_name
            k = 10
            numconst = 0
            numvar = numobj + k - 1
            problem = testProblem(name, numvar, numobj, numconst)
            orig_point = [1] * numobj
            first_ref = [1, 1] + [0] * (numobj - 2)
            second_ref = [0] * (numobj - 2) + [1, 1]
            for algo_name, evolve in algorithms.items():
                filename = foldername + name + "_" + str(numobj) + "_" + algo_name + "_"
                archive_df = evolve(problem, orig_point, first_ref, second_ref)

                objective_norms = archive_df["objective_values"].apply(
                    lambda x: np.linalg.norm(x)
                )
                angle_dev_from_1 = archive_df["objective_values"].apply(
                    lambda x: np.degrees(
                        np.arccos(
                            np.dot(x, first_ref)
                            / (np.linalg.norm(x) * np.linalg.norm(first_ref))
                        )
                    )
                )
                angle_dev_from_2 = archive_df["objective_values"].apply(
                    lambda x: np.degrees(
                        np.arccos(
                            np.dot(x, second_ref)
                            / (np.linalg.norm(x) * np.linalg.norm(second_ref))
                        )
                    )
                )
                archive_df["objective_norms"] = objective_norms
                archive_df["angle_1"] = angle_dev_from_1
                archive_df["angle_2"] = angle_dev_from_2
                x = alt.X("generation")
                y_obj = alt.Y(
                    "median(objective_norms)",
                    scale=alt.Scale(type="log", domain=(1, 10)),
                )
                y_angle_1 = alt.Y("median(angle_1)")
                y_angle_2 = alt.Y("median(angle_2)")
                magnitude = (
                    alt.Chart(archive_df)
                    .mark_line(clip=True)
                    .encode(x=x, y=y_obj)
                    .properties(title="Median Magnitude of objective vectors")
                )
                angle_1 = (
                    alt.Chart(archive_df)
                    .mark_line(clip=True)
                    .encode(x=x, y=y_angle_1)
                    .properties(
                        title="Median Angular Deviation of objective vectors from \
                        Reference Point 1"
                    )
                )
                angle_2 = (
                    alt.Chart(archive_df)
                    .mark_line(clip=True)
                    .encode(x=x, y=y_angle_2)
                    .properties(
                        title="Median Angular Deviation of objective vectors from \
                        Reference Point 2"
                    )
                )

                magnitude.save(filename + "magnitude.html")
                angle_1.save(filename + "angle_1.html")
                angle_2.save(filename + "angle_2.html")


def smoothEvolve(problem, orig_point, first_ref, second_ref):
    """Evolves using RVEA with abrupt change of reference vectors."""
    pop = Population(problem, assign_type="empty", plotting=False)
    try:
        pop.evolve(slowRVEA, {"generations_per_iteration": 200, "iterations": 15})
    except IndexError:
        return pop.archive
    try:
        pop.evolve(
            slowRVEA,
            {
                "generations_per_iteration": 10,
                "iterations": 20,
                "old_point": orig_point,
                "ref_point": first_ref,
            },
        )
    except IndexError:
        return pop.archive
    try:
        pop.evolve(
            slowRVEA,
            {
                "generations_per_iteration": 10,
                "iterations": 20,
                "old_point": first_ref,
                "ref_point": second_ref,
            },
        )
    except IndexError:
        return pop.archive
    return pop.archive


def abruptEvolve(problem, orig_point, first_ref, second_ref):
    """Evolves using RVEA with abrupt change of reference vectors."""
    pop = Population(problem, assign_type="empty", plotting=False)
    try:
        pop.evolve(slowRVEA, {"generations_per_iteration": 200, "iterations": 15})
    except IndexError:
        return pop.archive
    try:
        pop.evolve(
            slowRVEA,
            {
                "generations_per_iteration": 10,
                "iterations": 20,
                "old_point": first_ref,
                "ref_point": first_ref,
            },
        )
    except IndexError:
        return pop.archive
    try:
        pop.evolve(
            slowRVEA,
            {
                "generations_per_iteration": 10,
                "iterations": 20,
                "old_point": second_ref,
                "ref_point": second_ref,
            },
        )
    except IndexError:
        return pop.archive
    return pop.archive


if __name__ == "__main__":
    main()