Background concepts
===================


Evolutionary algorithms
-----------------------

Evolutionary algorithms (EAs) are optimization algorithms that emulate the process of evolution via 
natural selection to find optimal solutions to single- or multiobjective optimization problems (MOPs). 
This is achieved by taking a population of candidate solutions, known as individuals. The individuals mix
and match their properties with other individuals in a crossover process to form a new batch of candidate solutions,
known as offspring. The method also involves a random change in the properties of the offspring, which occurs via a process called mutation. 
Finally, during the selection step, the approach decides which individuals remain for the next generation based on a fitness criterion. 
The surviving population members then undergo the same steps as mentioned above and slowly converge towards optimality. 
The general structure of an EA is the following: 

::

    Step 1: t= 0 

    Step 2: Initialize P(t) 

    Step 3: Evaluate P(t) 

    Step 4: While not terminate do 

        P’(t) = variation [P(t)];------------->(Crossover and mutation) 

        evaluate [P’(t)]; 

        P(t+1) = select [P’(t) U P(t)]; 

        t = t + 1; 
    end 


Different EAs differ in the way they handle the population; conduct crossover, mutation,
and selection; and calculate the fitness criteria.
The :code:`desdeo-emo` package currently provides implementations of 
** Decomposition-based EAs**, which specialize in multiobjective optimization.
The package also provides the *EvoNN*, *BioGP*, and *EvoDN2* algorithms which can be used to
train surrogate models in an evolutionary manner.

Evolutionary operators 
----------------------

- **Selection:** Motivation is to preserve the best (make multiple copies) and eliminates the worst. 

    E.g., Roulette wheel, Tournament, steady-state, etc. 

- **Mutation:** Keep diversity in the population. 

    E.g., Polynomial mutation, random mutation, one-point mutation, etc. 

- **Crossover:** Create new solutions by considering more than one individual. 

    E.g., Simulated binary crossover, Linear crossover, blend crossover, uniform, one-point, etc. 

.. figure:: /images/EAs.png
   :scale: 60%
   :figclass: imgcenter

Multiobjective Evolutionary Algorithms
--------------------------------------

Multiobjective Evolutionary Algorithms (MOEAs) are Evolutionary algorithms employed to solve multiobjective optimization problems (MOPs). They use a population of solutions to obtain a diverse set of trade-off solutions close to the Pareto optimal front. There are mainly three types of MOEAs: 

 
- **Dominance-based MOEAs:** These approaches sort the individuals of the population using the dominance relationship, obtaining multiple convergence layers. In addition, an explicit diversity preservation scheme is employed in the last layer to maintain a diverse set of solutions. Dominance-based MOEAs have some advantages like a simple principle, easy understanding, and fewer parameters than other types of MOEAs. However, their ability to guarantee convergence degrades when the number of objectives exceeds three, mainly due to the loss of selection pressure. Among the most popular approaches following this principle are: NSGA-II and SPEA2 

- **Indicator-based MOEAs:** These algorithms employ a performance indicator (e.g., hypervolume) as a selection criterion to rank non-dominated solutions. However, they require a high computational complexity to compute the indicator values, especially in problems with more than three objectives. Some of the well-known indicator-based MOEAs are IBEA, SIBEA, etc. 

- **Decomposition-based MOEAs:** These techniques transform a MOP into a set of single-objective problems (or subMOPs) using scalarizing functions (e.g., Tchebycheff, PBI, etc.). These are solved simultaneously using an Evolutionary Algorithm. Most of these algorithms utilize reference vectors (also known as weight vectors) as a mechanism to maintain the population's diversity. The evolutionary operators for these techniques are similar to the ones employed in the rest of the EAs. However, during the selection process, they use an aggregated fitness value instead of a vector. Some of the most representative MOEAs in this category are MOEA/D, NSGA-III, and RVEA. 