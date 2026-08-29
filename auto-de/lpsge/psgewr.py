import sge.logger as logger
import sge.grammar as grammar
import numpy as np
import time, copy
from datetime import datetime
from dataclasses import dataclass
from sge.parameters import params, set_parameters, load_parameters
from joblib import Parallel, delayed, cpu_count
from typing import List, Callable, Tuple
from sge.operators.recombination import crossover
from sge.operators.mutation import mutate, mutate_level, mutation_prob_mutation
from sge.operators.selection import tournament
from sge.operators.update import independent_update

@dataclass
class EvalFn:
    fn: Callable[[str], List[float]]
    optimal: List[float]

class PSGEWR:

    def __init__(self, 
        evaluation_function: EvalFn,
        parameters_file: str,
        popsize: int, 
        pmin: int = 2,
        elitesize: int = 2,
        rmethod:str ="kruns", 
        nruns: int = 5):
        self._parameters_file = parameters_file
        self._rmethod = rmethod
        self._nruns = nruns
        self._pinit = popsize
        self._popsize = popsize
        self._pmin = pmin
        self._elitesize = elitesize
        self._g = 0
        self._max_g = 0
        self._numevals = 0
        self._evaluation_function = evaluation_function.fn
        self._gfit = evaluation_function.optimal

    def _setup(self):
        if self._parameters_file is not None:
            load_parameters(file_name=self._parameters_file)
        if params["SEED"] is None:
            params["SEED"] = int(datetime.now().microsecond)
        params["POPSIZE"] = self._popsize
        if  self._popsize <= 0:
            raise ValueError("popsize must be an integer greater than zero")
        if self._pmin > self._popsize:
            raise ValueError("pmin must be less than popsize")
        params["ELITISM"] = self._elitesize 
        params["EXPERIMENT_NAME"] += "/" + str(params["LEARNING_FACTOR"] * 100)
        logger.prepare_dumps()
        np.random.seed(int(params["SEED"]))
        grammar.set_path(params["GRAMMAR"])
        grammar.read_grammar()
        grammar.set_max_tree_depth(params["MAX_TREE_DEPTH"])
        grammar.set_min_init_tree_depth(params["MIN_TREE_DEPTH"])

    def _generate_random_individual(self):
        genotype = [[] for _ in grammar.get_non_terminals()]
        tree_depth = grammar.recursive_individual_creation(
            genotype, grammar.start_rule()[0], 0
        )
        if params["ADAPTIVE_MUTATION"]:
            return {
                "genotype": genotype,
                "fitness": None,
                "tree_depth": tree_depth,
                "mutation_probs": [params["PROB_MUTATION"] for _ in genotype],
            }
        else:
            return {"genotype": genotype, "fitness": None, "tree_depth": tree_depth}

    def _make_initial_population(self):
        for i in range(self._popsize):
            yield self._generate_random_individual()  
    
    def _get_candidate_individual(self, population):
        if np.random.uniform() < params["PROB_CROSSOVER"]:
            p1 = tournament(population, params["TSIZE"])
            p2 = tournament(population, params["TSIZE"])
            ni = crossover(p1, p2)
        else:
            ni = tournament(population, params["TSIZE"])
        if params["ADAPTIVE_MUTATION"]:
            # if we want to use Adaptive Facilitated Mutation
            ni = mutation_prob_mutation(ni)
            ni = mutate_level(ni)
        else:
            ni = mutate(ni, params["PROB_MUTATION"])
        return ni

    def run(self):
        self._setup()
        population = list(self._make_initial_population())
        flag = False  # alternate False - best overall
        best = None
        self._max_g = params["GENERATIONS"]
        self._g = 0
        self._evaluate_all(population)
        print(f"Will run GE in parallel mode over {cpu_count()-1} cores")
        while self._g < self._max_g:
            start_time = time.time()
            print(f"Started generation {self._g}")
            population.sort(key=lambda x: x["fitness"])
            popsize, elite_size = self._get_gen_pop_def() 
            if not best:
                best = copy.deepcopy(population[0])
            elif population[0]["fitness"] <= best["fitness"]:
                best = copy.deepcopy(population[0])
            best_fit = best["fitness"]
            print(f"Best fit is {best_fit}")
            if not flag:
                independent_update(best, params["LEARNING_FACTOR"])
            else:
                independent_update(best_gen, params["LEARNING_FACTOR"])
            flag = not flag
            if params["ADAPTIVE_LF"]:
                params["LEARNING_FACTOR"] += params["ADAPTIVE_INCREMENT"]

            logger.evolution_progress(self._g, population, best, grammar.get_pcfg())

            print(f"Going to evolve {popsize} individuals and overall array size is {len(population)} where elite size is {elite_size}")
            
            new_population = [self._get_candidate_individual(population) for _ in range(popsize)]
            self._evaluate_all(new_population)
            new_population.sort(key=lambda x: x["fitness"])
            best_gen = copy.deepcopy(new_population[0])
            new_population += population[:elite_size]
            population = new_population
            print(f"Generation {self._g} took {(time.time() - start_time)} seconds")
            best_gen_fit = best_gen["fitness"] 
            if best_gen_fit == 0:
                print(f"Will stop because best fit was found at generation {self._g} {best_gen_fit}")
                break
            self._g += 1
        res = min([best, best_gen], key=lambda x: x["fitness"])
        res["num_g"] = self._g
        res["num_evals"] = self._numevals
        return res

    def _get_gen_pop_def(self):
        new_popsize = max(self._pmin, round(self._pinit * ( 1 - self._g/self._max_g)))
        elite_size = min(self._elitesize , new_popsize)
        return new_popsize, elite_size

    def _evaluate_all(self, individuals):
        phenotypes = []
        tree_depths = []
        mappings = []
        for i in individuals:
            mapping_values = [0 for _ in i["genotype"]]
            phen, tree_depth = grammar.mapping(i["genotype"], mapping_values)
            phenotypes.append(phen)
            tree_depths.append(tree_depth)
            mappings.append(mapping_values)
        curr_evals = self._numevals
        results = Parallel(n_jobs=-2)(
            delayed(self._evaluate)(phen) for phen in phenotypes
        )
        fitness_all = self._reduce_fitness(results)
        for i in range(len(results)):
            curr_evals += self._nruns
            ind = individuals[i]
            ind["phenotype"] = phenotypes[i]
            ind["fitness"] = fitness_all[i]
            ind["other_info"] = {"generation": self._g, "evals": curr_evals}
            ind["mapping_values"] = mappings[i]
            ind["tree_depth"] = tree_depths[i]
        self._numevals = curr_evals

    def _evaluate(self, individual):
        start_time = time.time()
        res = [self._evaluation_function(individual) for _ in range(self._nruns)]
        print("ind run took %s seconds" % (time.time() - start_time))
        return res

    def _reduce_fitness(self, results):
        fits = []
        for result in results:
            error = 0
            N = len(result[0])
            R = self._nruns
            for f_result in result:
                error_f = 0
                for i in range(len(f_result)):
                    error_f += (f_result[i] - self._gfit[i])**2
                error += error_f
            fit = np.sqrt( (1 / (N*R)) *  error)
            fits.append(fit)
        #print(f"Fits={fits}")
        return fits