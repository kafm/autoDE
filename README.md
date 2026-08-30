# Automatic Component-Wise Design of Differential Evolution Algorithms

This repository contains the code and experimental material associated with the paper **“Automatic Component-Wise Design of Differential Evolution Algorithms: A Validation Study.”**

The study evaluates a component-wise DE design space explored using **Probabilistic Structured Grammatical Evolution (PSGE)**. Candidate algorithms are assembled from reusable mechanisms including mutation, crossover, parameter adaptation, archive management, boundary repair, population resizing, and optional convergence mechanisms.

The repository contains the grammar and DE components, the automatic design procedure, the designed DE variants, and the scripts used for the CEC 2017 experiments and statistical analysis reported in the paper.

Dependencies can be installed with:

```bash
pip install -r requirements.txt
```

For the complete experimental configuration and methodology, please refer to the accompanying paper.
