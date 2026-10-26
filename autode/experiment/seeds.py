"""Persist seed schedules shared by training and evaluation experiments."""
import csv
import json
import os
import secrets
from typing import List, Optional

import numpy as np
from numpy.random import Generator


def read_seed_schedule(filename):
    """Read a schedule without generating seeds or modifying its source."""
    with open(filename, encoding="utf-8") as file:
        seeds = json.load(file)
    if not isinstance(seeds, dict) or any(
        not isinstance(values, list) or any(type(seed) is not int or seed < 0 for seed in values)
        for values in seeds.values()
    ):
        raise ValueError(f"Invalid seed schedule in {filename}")
    return seeds


def save_external_seed_schedule(filename, seeds, problems, count):
    """Persist externally supplied seeds, rejecting incompatible prior results."""
    for problem in problems:
        if problem not in seeds or len(seeds[problem]) < count:
            raise ValueError(f"External seed schedule needs {count} seeds for {problem}")
    seed_filename = os.path.splitext(filename)[0] + ".json"
    saved = None
    if os.path.exists(seed_filename):
        saved = read_seed_schedule(seed_filename)
        if set(saved) != set(seeds) or any(
            values != seeds[key][:len(values)] for key, values in saved.items()
        ):
            raise ValueError(f"External seed schedule conflicts with {seed_filename}")
    if os.path.exists(filename):
        with open(filename, newline="") as file:
            for row in csv.DictReader(file):
                try:
                    index = int(row["no"]) - 1
                    if index < 0 or int(row["seed"]) != seeds[row["problem"]][index]:
                        raise ValueError
                except (KeyError, IndexError, TypeError, ValueError) as exc:
                    raise ValueError(f"Existing results conflict with external seeds in {filename}") from exc
    if saved != seeds:
        with open(seed_filename, "w", encoding="utf-8") as file:
            json.dump(seeds, file, indent=2)
            file.write("\n")
    return seeds


def load_seed_schedule(filename, problems, count, nbits=64):
    seed_filename = os.path.splitext(filename)[0] + ".json"
    exists = os.path.exists(seed_filename)
    if exists:
        seeds = read_seed_schedule(seed_filename)
    else:
        seeds = {}

    changed = not exists
    used_seeds = {seed for values in seeds.values() for seed in values}
    for problem in problems:
        if problem not in seeds:
            seeds[problem] = []
            changed = True
        while len(seeds[problem]) < count:
            seed = secrets.randbits(nbits)
            if seed not in used_seeds:
                seeds[problem].append(seed)
                used_seeds.add(seed)
                changed = True

    if changed:
        with open(seed_filename, "w", encoding="utf-8") as file:
            json.dump(seeds, file, indent=2)
            file.write("\n")
    return seeds

def generate_random_inner_seed(size: Optional[int] = None, rng: Optional[Generator] = None)->int | List[int]:
    if rng is None:
        return np.random.randint(low=0, high=2**32 -1, size=size)
    return rng.integers(low=0, high=2**32 - 1, size=size)
