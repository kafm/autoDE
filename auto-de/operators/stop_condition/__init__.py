from core import StopCondition

def max_generation(max_gen: int)->StopCondition:
    return StopCondition(max_gen=max_gen)

def max_evaluations(max_evals: int)->StopCondition:
    return StopCondition(max_evals=max_evals)