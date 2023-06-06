import pytest
from desdeo_emo.EAs.BaseEA import BaseEA
from desdeo_problem import MOProblem, Variable, ScalarObjective

def test_base_ea_initialization():
    # Initialize the BaseEA object
    ea = BaseEA(
        a_priori=True,
        interact=True,
        n_iterations=10,
        n_gen_per_iter=100,
        total_function_evaluations=0,
        use_surrogates=False,
        keep_archive=False,
        save_non_dominated=False
    )

    # Assert the attributes of the BaseEA object
    assert ea.a_priori == True
    assert ea.interact == True
    assert ea.n_iterations == 10
    assert ea.n_gen_per_iter == 100
    assert ea.total_gen_count == 1000
    assert ea.total_function_evaluations == 0
    assert ea.selection_operator == None
    assert ea.use_surrogates == False
    assert ea._iteration_counter == 0
    assert ea._gen_count_in_curr_iteration == 0
    assert ea._current_gen_count == 0
    assert ea._function_evaluation_count == 0
    assert ea._interaction_location == None
    assert ea.interaction_type_set_bool == False
    assert ea.allowable_interaction_types == None
    assert ea.population == None
    assert ea.keep_archive == False
    assert ea.archive == {}
    assert ea.save_non_dominated == False


