# Evolutionary algorithms in pyRVEA

- [EvoNN](#what-is-evonn)
- [EvoDN2](#what-is-evodn2)
- [BioGP](#what-is-biogp)
- [PPGA](#what-is-ppga)
- [Notes](#notes)
- [References](#references)

## Introduction
The engineering and scientific data that one receives from experiments or simulations from any industry is often very noisy, one needs to have a model to process them effectively. It is, however, very difficult to build a model which captures only the correct trends in the data while devoid of any implicit noise. The models can easily be overfitted so that they capture every fluctuation in the data set, both significant and noisy, or underfitted where some major trends may remain undetected or unrepresented.

Evolutionary neural networks and genetic programming have succesfully been utilized to come around this problem and create balanced models. Thus, the purpose of these models is not to achieve the tightest fit, or the lowest training error, but to construct models that capture the correct trends in the data, while avoiding the problems of overfitting and underfitting.

Neural networks and genetic programming have efficiently been utilized for these purposes. This project introduces Python implementations of three evolutionary algrorithms, [EvoNN](#what-is-evonn), [EvoDN2](#what-is-evodn2) and [BioGP](#what-is-biogp). The models can be trained and optimized using the genetic algorithms available in pyRVEA. A predator-prey genetic algorithm ([PPGA](#what-is-ppga)) was also implemented in Python for this project. The user can choose to train and optimize the models with the existing reference vector (RVEA) algorithms, or using the PPGA.

These algorithms use a concept of an optimal tradeoff between two objectives, leading to a [Pareto frontier](https://en.wikipedia.org/wiki/Pareto_efficiency#Pareto_frontier). When there are conflicting requirements in a problem, it essentially leads to a situation where none of the objectives would be able to attain their individual best. The Pareto frontier consists of solutions where the optima are contained in a set of best possible tradeoffs between the objectives.

## What is EvoNN?
In **Evo**lutionary **N**eural **N**etwork [[1]](#1) [[2]](#2), the population consists of feed-forward neural networks. Each network consists of an input layer, a hidden layer and an output layer. The data is fed to the networks via input nodes, which pass the information to a hidden layer, where each active connection receives a nonzero weight. The number of connections increases the complexity of the model. The hidden layer also includes a bias term. Then through a transfer function the hidden layer passes the values onwards to the next level where the weights are optimized and passed to the final output of the network.

![EvoNN](https://raw.githubusercontent.com/delamorte/pyRVEA/master/docs/evonn.png "An example EvoNN model")

*Figure 1. An example EvoNN model. [[7]](#7)*

The evolution happens in the lower part of the network (i.e. between the input layer and the hidden layer), where the weights are optimized using a genetic algorithm (see available [algorithms](https://htmlpreview.github.io/?https://github.com/delamorte/pyRVEA/blob/master/docs/_build/html/pyrvea.EAs.html) in pyRVEA). The two objectives to optimize here are the model's accuracy and complexity. Thus, both the training error and the complexity need to be minimized. The weights are altered using crossover and mutation operations. The upper part of the network is then optimized using a Linear Least Square (LLSQ) approach, which ensures the mathematical convergence to the Pareto front. The evolutionary algorithm then chooses the fittest of the individuals (the ones with the best tradeoff between accuracy and complexity) to go on to the next generation.

## What is EvoDN2?
**Evo**lutionary **D**eep Neural **N**etwork [[3]](#3) is an extension to EvoNN using deep neural networks and has the capacity for deep learning. The principle of EvoDN2 is the same as EvoNN, but the structure of the networks is different. In EvoDN2, each network consists of multiple subnets, which take in subsets of the input variables so that each variable is used at least once. The subnets have an input layer and multiple hidden layers, and they are optimized just as EvoNN using a genetic algorithm. Finally, the subnets converge on a linear layer, optimized by LLSQ and are mapped to an output. Each subnet has a randomized number of layers and nodes on each layers based on the bounds set by the user. The number of subnets can also be set.

![EvoDN2](https://raw.githubusercontent.com/delamorte/pyRVEA/master/docs/evodn.png "An example EvoDN2 model")

*Figure 2. An example EvoDN2 model. [[3]](#3)*

The advantage of EvoDN2 over EvoNN is that it fares much better with larger data sets, which can have thousands of data points. With smaller data sets EvoDN2 performs similarly to EvoNN, although EvoDN2 models tend to have a tighter fit.

The complexity calculation of EvoDN2 is also different from EvoNN. With EvoDN2, the complexity is calculated as the product function of all weight matrices within a subnet, and these weights are the summed together to get the total complexity of the model. This way the connections with a larger magnitude of weight values contribute more to the final value, and inactive connections will be discounted.

## What is BioGP?
BioGP [[4]](#4) is a **Bi**-**o**bjective **g**enetic **p**rogramming technique, which similarly to EvoNN and EvoDN2 minimizes the model's training error and complexity using genetic algorithms. Whereas many conventional GP algorithms tend to only focus on minimizing the error, one advantage of BioGP is that by calculating the best possibly tradeoff between the error and the complexity, overfitting and underfitting can be addressed. BioGP is designed so that for the first number of generations, it minimizes the training error, after which it switches to bi-objective mode minimizing both error and complexity.

The benefit of genetic programming in general is the flexibility of allowing the user to select the mathematical operations (function set and terminal set) involved in building the model. BioGP is also designed to combat bloat, where the models may grow larger and larger in size without increasing the fitness.

The BioGP models are represented with a tree encoding. GP achieves its learning by utilizing a function set containing user-defined mathematical operations like division or square root, and a terminal set containing the variables and the constants. The advantage of GP is that it does not require any pre-defined configuration of weights, biases and transfer functions and thus, it can evolve any mathematical function representing the system being modeled.

The BioGP tree has a linear node at the top (code-wise, the linear node is at depth=0), from which a number of subtrees emerge. Each of these subtrees represent a nonlinear function, consisting of function nodes and terminal nodes. The linear node takes a weighted sum of the outputs of the subtrees, adds a bias value, and optimizes the weights using LLSQ to calculate the final output of the tree. To handle bloat, BioGP uses a parameter called error reduction ratio which provides a simple quantification of the contribution that a single subtree makes toward the performance of the model. If the contribution is less than the limit set by the user, that subtree is terminated and a new one is grown in its place.

![BioGP](https://raw.githubusercontent.com/delamorte/pyRVEA/master/docs/biogp.png "An example BioGP model")


*Figure 3. An example BioGP tree. [[4]](#4)*

The complexity of the model depends on the number of function nodes in the tree, and the tree's total depth. A scalar parameter is used in the complexity function allowing the user to control whether the depth or the number of function nodes should weigh more in the calculation.

## What is PPGA?
**P**redatory-**P**rey **G**enetic **A**lgorithm [[5]](#5) [[6]](#6)

A population of prey signify the various models or solutions to the problem at hand. Weaker prey, i.e. bad models or solutions, are killed by predators. The predators and prey are placed in a lattice, in which they are free to roam.

In each generation, each predator gets a certain number of turns to move about and hunt in its neighbourhood, killing the weaker prey, according to a fitness criteria. After this, each prey gets a certain number of moves to pursue a random walk and to reproduce with other prey. Each reproduction step generates two new prey from two parents, by crossing over their attributes and adding random mutations. After each prey has completed its move, the whole process starts again.

As the weaker individuals get eliminated in each generation, the population as a whole becomes more fit, i.e. the individuals get closer to the true pareto-optimal solutions.

## Notes

The algorithms presented here have been created earlier in MATLAB, and these Python implementations have been using that code as a basis. Python code has been written by Niko Rissanen under the supervision of professor Nirupam Chakraborti.

The source code of pyRVEA is implemented by Bhupinder Saini.

## Contact

If you have any questions about the code, please contact:

Bhupinder Saini: bhupinder.s.saini@jyu.fi

Project researcher at University of Jyväskylä.

Rissanen Niko: nijosari@student.jyu.fi

Research assistant at University of Jyväskylä.

## References

#### 1
Chakraborti, N. (2014). Strategies for evolutionary data driven modeling in chemical and metallurgical Systems. In Applications of Metaheuristics in Process Engineering (pp. 89-122). Springer, Cham.

#### 2
Pettersson, F., Chakraborti, N., & Saxén, H. (2007). A genetic algorithms based multi-objective neural net applied to noisy blast furnace data. Applied Soft Computing, 7(1), 387-397.

#### 3
Swagata R., Bhupinder S., Chakrabarti, N. and Chakraborti, N. (2019). A new Deep Neural Network algorithm employed in the study of mechanical properties of micro-alloyed steel. Department of Metallurgical and Materials Engineering, Indian Institute of Technology.

#### 4
Giri, B. K., Hakanen, J., Miettinen, K., & Chakraborti, N. (2013). Genetic programming through bi-objective genetic algorithms with a study of a simulated moving bed process involving multiple objectives. Applied Soft Computing, 13(5), 2613-2623.

#### 5
Laumanns, M., Rudolph, G., & Schwefel, H. P. (1998). A spatial predator-prey approach to multi-objective optimization: A preliminary study. In International Conference on Parallel Problem Solving from Nature (pp. 241-249). Springer, Berlin, Heidelberg.

#### 6
Li, X. (2003). A real-coded predator-prey genetic algorithm for multiobjective optimization. In International Conference on Evolutionary Multi-Criterion Optimization (pp. 207-221). Springer, Berlin, Heidelberg.

#### 7
Bhupinder S. Optimization of Vanadium Microalloyed Steel Composition Using Evolutionary Deep Learning Techniques. Master's thesis. Department of Metallurgical and Materials Engineering, Indian Institute of Technology.
