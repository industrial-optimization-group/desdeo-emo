from desdeo_problem.Problem import ProblemBase, EvaluationResults


class surrogateProblem(ProblemBase):
    def __init__(self, performance_evaluator):
        super().__init__()
        self.performance_evaluator = performance_evaluator
        self.n_of_constraints = 0
        self._max_multiplier = 1
        self.objective_names = ["error", "complexity"]

    def evaluate(self, model_parameters, use_surrogates=False):
        results = self.performance_evaluator(model_parameters)
        return EvaluationResults(results, results, None, None)

    def evaluate_constraint_values(self):
        pass

    def get_variable_bounds(self):
        pass

    def get_objective_names(self):
        return self.objective_names
