# Automatic Component-Wise Design of Differential Evolution Algorithms

This repository contains the code and experimental material associated with the paper **“Automatic Component-Wise Design of Differential Evolution Algorithms: A Validation Study.”**

The study evaluates a component-wise DE design space explored using **Probabilistic Structured Grammatical Evolution (PSGE)**. Candidate algorithms are assembled from reusable mechanisms including mutation, crossover, parameter adaptation, archive management, boundary repair, population resizing, and optional convergence mechanisms.

The repository contains the grammar and DE components, the automatic design procedure, the designed DE variants, and the scripts used for the CEC 2017 experiments and statistical analysis reported in the paper.

Dependencies can be installed with:

```bash
pip install -r requirements.txt
```

For the complete experimental configuration and methodology, please refer to the accompanying paper.

### Reusing evaluation seeds after training

`autode.lpsge.experiment_gen_algo.Experiment` accepts an optional
`test_seed_file`. Supply one JSON path with full problem keys, for example
`{"F1_30D": [10, 11], "F1_50D": [20, 21]}`, or use dimension-specific
competitor files:

```python
experiment = Experiment(
    name="experiment_autode",
    problems=training_sets,  # Dict[str, ExperimentProblems]
    parameters_file="parameters.yml",
    popsize=30,
    max_experiments=51,
    test_seed_file={
        30: "experiment_competitors_all_30d.json",
        50: "experiment_competitors_all_50d.json",
    },
)
```

Dimension-specific files may use bare problem names such as `F1`; set each
test `ProblemInstance.name` accordingly. Every requested problem must have at
least `max_experiments` seeds. Repetition order is preserved, and all trained
algorithms share the supplied evaluation schedule. Training seeds are unchanged.

The resolved schedule is saved beside each test CSV. Resuming checks both the
saved schedule and existing CSV seeds; incompatible results require a new
experiment name. Source files are read without modification. Omitting
`test_seed_file` retains automatic evaluation-seed generation.
