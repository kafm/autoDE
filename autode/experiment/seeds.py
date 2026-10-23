"""Persist seed schedules shared by training and evaluation experiments."""
import json
import os
import secrets


def load_seed_schedule(filename, problems, count):
    seed_filename = os.path.splitext(filename)[0] + ".json"
    exists = os.path.exists(seed_filename)
    if exists:
        with open(seed_filename, encoding="utf-8") as file:
            seeds = json.load(file)
        if not isinstance(seeds, dict) or any(
            not isinstance(values, list) or any(
                type(seed) is not int or seed < 0 for seed in values
            )
            for values in seeds.values()
        ):
            raise ValueError(f"Invalid seed schedule in {seed_filename}")
    else:
        seeds = {}

    changed = not exists
    used_seeds = {seed for values in seeds.values() for seed in values}
    for problem in problems:
        if problem not in seeds:
            seeds[problem] = []
            changed = True
        while len(seeds[problem]) < count:
            seed = secrets.randbits(64)
            if seed not in used_seeds:
                seeds[problem].append(seed)
                used_seeds.add(seed)
                changed = True

    if changed:
        with open(seed_filename, "w", encoding="utf-8") as file:
            json.dump(seeds, file, indent=2)
            file.write("\n")
    return seeds

