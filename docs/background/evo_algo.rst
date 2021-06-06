What are Evolutionary Algorithms?
=================================

Evolutionary algorithms (EAs) are optimization algorithms which emulate the process of
evolution via natural selection to find optimal solutions to single- or multiobjective
optimization problems (MOPs).
This is achieved by taking a *population* of candidate solutions, known as
*individuals*.
The individuals mix and match their properties with other individuals in a process
called *crossover* to form a new batch of candidate solutions, known as *offsprings*.
The process also involves a random change in the properties of the offsprings, which
occurs via a process called *mutation*.
Finally, there is a culling step, called *selection*, which kills the individuals which
are considered not optimal according to a *fitness* criteria.
The surviving members of the population then undergo the same steps as mentioned above,
and slowly converge towards optimality as determined by the fitness criteria used in the
selection step.

Different EAs differ in the way they handle the population; conduct crossover, mutation,
and selection; and calculate the fitness criteria.
The :code:`desdeo-emo` package currently provides implementations of 
**Decomposition based EAs**, which specialize in multiobjective optimization.
The package also provides the *EvoNN*, *BioGP*, and *EvoDN2* algorithms which can be used to
train surrogate models in an evolutionary manner.