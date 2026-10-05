# Benchmark Lifetimes

When AI benchmarks are born and when they die: a survival analysis of 224 benchmarks that nine frontier labs reported in model release materials between June 2018 and September 2026.

Each benchmark has a birth date and, if it happened, a death date, coded under a codebook that was written before coding started. Benchmarks that were still alive at the cutoff (2026-09-30) or were abandoned are censored, not dropped.

- Article (Russian, Habr): link will be added on publication
- Earlier popular version (Russian, vc.ru): [Бенчмарки ИИ теперь живут пять месяцев](https://vc.ru/nikitahudov/3087090-benchmarki-ii-chto-izmeriat-vmesto-ustarevshikh-testov)
- Dataset on Hugging Face: [nikitahudov/benchmark-lifetimes](https://huggingface.co/datasets/nikitahudov/benchmark-lifetimes)

## Key results

| Quantity | Value |
| --- | --- |
| Median lifetime, all 224 benchmarks | 23 months (95% CI 19–27) |
| Born 2016–2022 / born 2023–2026 | 38 (27–69) / 19 (15–23) months |
| Hazard ratio per birth year, Cox | 1.22 (1.11–1.34) |
| Headroom at release, median, born ≤2022 / 2023+ | 2.02 / 2.14 logits to the threshold |
| Frontier speed on the same benchmarks (Epoch AI trajectories) | 0.75 → 1.56 logits per year |
| Born 2023+, dead of own defects or a corrected version within 24 months | 14% (Aalen–Johansen) |
| Agentic environments: deaths from defects or versions | 13 of 29 |

Every number of the article comes from `study/analysis_v3.json`, `study/cards_v3.json` and the dataset itself; `study/verify_article.py` checks the article text against them.

## Definitions

Birth is the earliest of: arXiv v1, the official announcement, the public dataset, a leaderboard launch with the items; for exams, the exam date. A benchmark dies at the earliest of five events:

| Code | Event | Applies when |
| --- | --- | --- |
| A | Best counted result ≥ human baseline | baseline ≥ 85% of the scale, measured by humans on the same split with a described protocol |
| B | Best counted result ≥ 90% of the main metric | no usable human baseline, or baseline below 85% |
| C | Maintainers withdraw the set, freeze submissions, deprecate it, or name saturation or defects as the reason for a replacement | public statement by the creators only |
| D | Organizer-declared target reached | e.g. 85% for ARC Prize, verified by the organizers |
| E | Independent compromise | answer leakage documented for most frontier models; a trivial method reaches the ceiling; an audit finds ≥ 20% defective items and at least one of the nine labs stops reporting |

Other outcomes: **abandoned** (12 months without lab reports and leaderboard activity before any event; censored at the last activity), **born dead** (threshold reached at release), **alive** at the cutoff. A new version (> 10% of items replaced, or a changed metric or protocol) is a separate unit.

The main mode counts any publicly reported system: a bare model, a model in a scaffold or agent, an ensemble, if the result was accepted by the official leaderboard or published by the system's developer with a protocol. The strict mode counts only single models with independent or organizer-verified runs. Full rules: [`docs/CODEBOOK_v3.md`](docs/CODEBOOK_v3.md).

**Population.** A public benchmark enters if its own score was reported in official release materials (blog, model card, system card, technical report) by at least two of nine labs (OpenAI, Anthropic, Google, Meta, xAI, DeepSeek, Alibaba/Qwen, Moonshot, Mistral), or by one lab in at least three releases. Before 2023 the sources are model papers: GPT-1, GPT-2, GPT-3, Codex, InstructGPT, BERT, RoBERTa, T5, Gopher, Chinchilla, PaLM, OPT and Flan-PaLM. Excluded: open-ended ratings (Elo arenas, agent revenue, perplexity), labs' internal evaluations without public items, behavioral evaluations without a "solution", suites (BIG-bench), speech and translation, exam batteries.

## Reproduce

```bash
pip install -r requirements.txt
bash run_all.sh
```

About 30 seconds. The script rebuilds every derived file from the coder outputs and the release-material extracts, renders the nine charts and checks every number of the article against the analysis outputs. Expected SHA-256 of `study/halflife_dataset_v3.csv`:

```
aca4ea7525fa066d0f43a67d8dd12938b22fd654958d4d23e25559f4369ec847
```

Charts use the Inter and JetBrains Mono fonts if they are installed. The Epoch-based trajectories in `study/d4/` are rebuilt separately: clone [epoch-research/benchmark-stitching](https://github.com/epoch-research/benchmark-stitching) and [evaleval/benchmark-saturation](https://github.com/evaleval/benchmark-saturation) into `study/` (or point `D4_BASE` to their parent folder) and run `python3 study/d4/d4_build.py`.

## Repository layout

```
run_all.sh                      full pipeline
docs/CODEBOOK_v3.md             birth and death rules (v1.1)
docs/CODER_INSTRUCTIONS.md      instructions given to both coders (52-column output)
article/HABR_ARTICLE_v3.md      article text checked by verify_article.py
study/
  population.py, canon.py       release-material rows -> canonical units (310 regular expressions)
  population_main.py            population rule -> 236 candidate units
  reconcile_v3.py               coder A + coder B + logged overrides -> dataset
  cards_v3.py                   benchmarks in model cards before and after death
  analysis_v3.py                Kaplan–Meier, Cox, Weibull/lognormal AFT, Aalen–Johansen, null model, sensitivity, headroom
  verify_article.py             article numbers vs analysis outputs
  halflife_dataset_v3.csv       main table, one row per unit
  decisions_log_v3.csv          21 reconciliation decisions with rule and source
  agreement_v3.json             double-coding agreement
  d3_matrix_v3.csv              6,652 rows release x benchmark x score from 230 releases
  cards_v3.csv, cards_v3.json   reporting before and after death, per unit and per year
  headroom_v3.csv               best result at release and headroom to the threshold
  analysis_v3.json              all results
  d1/                           coder inputs, coder A batches (+ repair of batch 1), coder B blind sample
  d2/                           extra fields for flagship benchmarks
  d3/                           per-lab extracts: releases, benchmark rows, maintainer and lab statements
  d4/                           frontier trajectories from Epoch AI and EvalEval data mapped to units
  charts/                       chart code and the nine charts of the article
```

## Main table: `study/halflife_dataset_v3.csv`

235 rows: 224 units of the main analysis (`stratum == "main"`) and 11 excluded units with the reason. Dates are `YYYY-MM`; durations are months.

| Group | Columns |
| --- | --- |
| Identity | `unit`, `stratum`, `exclusion_reason`, `canonical_name`, `lineage` |
| Birth | `release`, `release_source`, `birth_year`, `cohort` |
| Instrument | `metric`, `direction`, `scale_max`, `n_items`, `test_access`, `curation`, `format`, `modality`, `class` |
| Threshold | `human_baseline`, `human_type`, `criterion_ab`, `threshold` |
| Event A/B | `ab_date`, `ab_system`, `ab_score`, `ab_mode`, `ab_system_type` (single, scaffold, ensemble), `ab_source_type` (self, leaderboard, independent), `ab_source`, `ab_split_dev` |
| Strict mode | `strict_date`, `strict_system`, `strict_score`, `strict_source`, `strict_primary_date`, `T_strict`, `event_strict` |
| Event C | `c_date`, `c_reason` (saturation, flaws, administrative), `c_kind` (version, successor, freeze, withdrawal), `c_source`, `c_quote` |
| Events D, E | `d_date`, `d_source`, `e_date`, `e_type` (i leakage, ii trivial method, iii audit), `e_source`, `e_quote` |
| Abandonment | `abandoned_date`, `abandoned_evidence` |
| Outcome | `primary_date`, `primary_criterion`, `cause` (threshold, declared_saturation, defects), `cause3` (models, instrument, abandoned), `status_cutoff`, `T_months`, `event`, `outcome_cr` (0 censored, 1 models, 2 instrument, 3 abandoned) |
| Sensitivity flags | `T_replacement_censored`, `event_replacement_censored`, `born_saturated`, `born_saturated_calc`, `date_upper_bound`, `reissue` (annual exams) |
| Lab reporting | `first_report`, `last_report`, `entry_months`, `n_labs_v3`, `n_releases_v3` |
| Headroom | `score_at_release`, `score_at_release_system` |
| Audit trail | `best_by_cutoff`, `confidence` (high, medium, low), `recon_flags` (ids in `decisions_log_v3.csv`), `notes` |

## How the data was made

Coding was done by Claude agents following `docs/CODER_INSTRUCTIONS.md`: coder A (Claude Opus) coded all 15 batches, coder B (Claude Sonnet) blindly coded a sample of 24 units stratified by outcome. Every date carries a URL, every maintainer statement a quote, every doubt a note. Agreement on the status at cutoff: Cohen's κ = 0.71 (20 of 24); birth month agreed in 24 of 24; where both found a death, the month agreed within one month in 11 of 12. Every change to a coder A row is listed in `decisions_log_v3.csv` and applied in code in `reconcile_v3.py`.

## Limitations

- The 90% threshold is a convention; lifetimes from different studies are comparable only under the same threshold and the same rule on which results count.
- 51 of 82 deaths by criteria A and B rest on developers' self-reports. The strict mode raises the overall median from 23 to 39 months; the gap between eras remains (68 vs 26 months).
- Only benchmarks reported by frontier labs are included; a benchmark that died before becoming popular is not in the data.
- 11 rows are coded with low confidence, and 8 death dates are known only as an upper bound.
- Benchmarks born in 2026 are observed for at most eight months.

## Corrections

If you have an earlier date or see an error in a row, open an issue with the unit name and a source. Corrections are logged in the decisions log and in the release notes.

## Citation

```bibtex
@misc{khudov2026benchmarklifetimes,
  title  = {Benchmark Lifetimes: When AI Benchmarks Are Born and When They Die},
  author = {Khudov, Nikita},
  year   = {2026},
  url    = {https://github.com/nikitahudov/benchmark-lifetimes}
}
```

## License

Code: Apache License 2.0, see `LICENSE`. Data (CSV and JSON files under `study/` and the Hugging Face release): CC BY-SA 4.0, see `LICENSE-DATA`. Quotes in the `*_quote` fields are short excerpts from the cited sources, kept for verification. `study/d4/` contains values derived from MIT-licensed repositories of Epoch AI and EvalEval, see `third_party/`.
