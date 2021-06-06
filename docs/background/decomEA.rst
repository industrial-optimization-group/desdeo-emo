What are Decomposition-Based EAs?
=================================

Decomposition-based EAs are evolutionary algorithms which specialize in optimizing multiobjective optimization problems.
These algorithms work by breaking down, or "decomposing", the objective space into smaller sections along directional
vectors known as reference vectors (also known as weight vectors or reference points). These vectors are used along with
scalarizing functions to solve a single objective optimization problem in the local region of the vectors. Hence, the
multiobjective optimization problem is converted into multiple single objective optimization problems.

Non-interactive versions of such EAs usually utilize a set of reference vectors that is uniformly spread in the
objective space. This enables the EA to find solutions in all areas of the Pareto front. Interactivity is usually
achieved by manipulating the spread of the reference vectors in accordance to the preferences of a decision maker.

There are many examples of decomposition-based EAs, which mostly differ in the scalarizing function/s used for the
decomposition step. This package provides support for interactive and non-interactive versions of RVEA, NSGA-III, and
MOEA/D.