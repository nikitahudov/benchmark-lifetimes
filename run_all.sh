#!/usr/bin/env bash
# Rebuilds every derived file of the study from the coder outputs and the release-card matrix.
# Run from the repository root:  bash run_all.sh
set -euo pipefail
cd "$(dirname "$0")/study"
python3 population.py          # d3/*_benchmarks.csv  -> d3_matrix.csv, population_units.json
python3 population_main.py     # population_units.json -> population_main.json (236 candidates)
python3 reconcile_v3.py        # coder A/B outputs + overrides -> halflife_dataset_v3.csv, decisions_log_v3.csv, agreement_v3.json, d3_matrix_v3.csv
python3 cards_v3.py            # dead benchmarks in model cards -> cards_v3.json, cards_v3.csv
python3 analysis_v3.py         # survival analysis -> analysis_v3.json, headroom_v3.csv
(cd charts && python3 charts_v3.py 1 2 3 4 5 6 7 8 9)
python3 verify_article.py ../article/HABR_ARTICLE_v3.md
sha256sum halflife_dataset_v3.csv
