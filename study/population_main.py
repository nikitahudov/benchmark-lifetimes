# -*- coding: utf-8 -*-
"""
Population candidates (236 units) from population_units.json.

A unit is a candidate if it belongs to the main stratum and passes the population rule of codebook v3:
reported in official release materials by >= 2 of the 9 labs, or by one lab in >= 3 releases.
Family rows without a version or domain (BFCL, tau3-bench) are not units: their reports belong to
the version units (BFCL v2/v3/v4) and domain units (tau3-bench banking/telecom).

Input:  population_units.json (written by population.py)
Output: population_main.json (read by reconcile_v3.py, which applies the refined report rule)
"""
import json

FAMILY_ROWS = {"BFCL", "tau3-bench"}

units = json.load(open("population_units.json", encoding="utf-8"))
main = {k: v for k, v in units.items()
        if v["stratum"] == "main" and (v["n_labs"] >= 2 or v["n_releases"] >= 3) and k not in FAMILY_ROWS}
json.dump(main, open("population_main.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(len(main), "candidate units")
