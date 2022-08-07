# DBMOPP Generator

This DBMOPP generator is based on the original Distance-Based Multi-Objective Point Problem (DBMOPP) instance generator https://github.com/fieldsend/DBMOPP_generator.
Currently the usage and code is very similar to the original Matlab version.

## Example usage 

Example usage as notebook-tutorial at desdeo_problem/docs/notebooks/DBMOPP_tutorial.

## What works 
Combining DBMOPP with DESDEO's MOProblem instances for solving the generated
testproblems with DESDEO's methods is work in progress.

- Basics of original DBMOPP generator's functionality is done. Can create problems with different combinations from
  number of objectives, variables, local and global pareto fronts and dominance resistance regions, constrainted space, pareto set types and constraint types. 
  Possibility to plot the problem instance, plot pareto set memebers, plot landscape for single objective.
- MOProblem can be called to evaluate objective function values.
- MOProblem can be called to evaluate constraints.
- Constraints include:
    - Hard / Soft vertex constraints
    - Hard / Soft center constraints
    - Hard / Soft moat constraints
    - Hard / Soft extended checker constraints

  - Can we approximation of the Pareto set either by:
    - calling the plot_pareto_set_members() which returns the found members.
    - calling get_Pareto_set_member() which returns one pareto set member uniformly from the pareto regions.

## What does not work yet

- Dominance landscape not yet plotted
- Some parameters and combination of them is not tested yet with MOProblem and some are still under development.
These are ndo, vary_sol_density, vary_objective_scales, prop_neutral and nm and
are suggested to be left on their default values.
