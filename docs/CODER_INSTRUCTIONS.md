# Coding instructions: birth and death dates of AI benchmarks (codebook v3, operational version)

You are a coder in a research study on how long AI benchmarks stay useful. For each benchmark unit in your batch you must establish, from public sources, (1) when it was born, (2) whether and when it "died" under the rules below, and (3) a few structural properties. Another coder works on the same units independently; do not try to guess what anyone else wrote. Precision and honesty matter more than coverage: if something cannot be established, say so in `notes` and lower `confidence` — never invent a date, score or source.

Cutoff date for the whole study: **2026-09-30**. Ignore anything after it.

## 0. Inputs you get per unit
- `unit` — the benchmark unit to code (a specific benchmark, version, exam year or domain).
- `raw_names` — how frontier labs named it in model release materials.
- `lab_reports` — lines "date | lab | model | name [variant] | setting | score | source": self-reported scores from official release materials of 9 frontier labs (OpenAI, Anthropic, Google, Meta, xAI, DeepSeek, Alibaba/Qwen, Moonshot, Mistral). Use them as a starting point for the score trajectory, but they are NOT complete: leaderboards, papers and other developers may have crossed thresholds earlier. Treat parsing errors in these lines with suspicion.
- `metadata_hints` (sometimes) — notes from an earlier metadata pass. Verify before relying on them.

## 1. Is it in scope?
Set `stratum`:
- `main` — a publicly documented capability benchmark (public paper/dataset/website or public leaderboard that a third party can use or inspect) with a bounded score scale.
- `excluded:internal` — no public items/leaderboard (lab-internal eval);
- `excluded:no_ceiling` — Elo, $ earned, open-ended index without a ceiling;
- `excluded:not_benchmark` — not an evaluation benchmark;
- `propensity` — measures behaviour (refusals, bias, toxicity, honesty) rather than capability.
If excluded, fill only identification fields and the reason, then move on.

## 2. Birth (`release`, YYYY-MM)
Earliest of: arXiv v1 date, official announcement/blog, public dataset release, public leaderboard launch with the items. For annual exams (AIME, HMMT, USAMO, IMO, CNMO) — the exam date. A prize launch is not a birth (ARC-AGI-1 was born 2019-11, not 2024-06). Record `release_source` (URL).

## 3. Threshold (criterion A or B)
- Find the creators' human baseline. If there are several numbers (best/average/worst expert), use the **average** of the type the creators present as "human performance". Record `human_baseline`, `human_type` (average human | expert average | crowd aggregate | degenerate | none).
- **Criterion A** (human parity) applies if the human baseline is measured by people on the same split with a described protocol AND is ≥ 85% of the scale maximum. Then `threshold` = human baseline.
- Otherwise **criterion B**: `threshold` = 90% of the scale maximum.
- If the human baseline is an aggregate of many annotators or degenerate (e.g., 100% by construction) — do not use A; use B.
- Metric direction: if lower is better (error rate, attack success, WER), convert to a "higher is better" equivalent (e.g., 100 − error) and say so in `notes`.
- If the benchmark reports several sub-metrics, use its primary/leaderboard metric. For multi-domain benchmarks split into units (e.g., τ²-bench telecom), use that domain only.

## 4. Death events — record EACH applicable one separately
- **AB event**: the first public result (any system) at or above `threshold` on the official test split with the standard metric. 
  - What counts (main mode): any publicly reported system — a bare model, a model with a scaffold/agent harness, or an ensemble — if accepted on the official leaderboard or published by the system's developer with a described protocol (paper, tech report, blog, model card). Self-reports by labs count.
  - Tools: if the benchmark's standard protocol is without tools (e.g., AIME, HLE, GPQA, MMLU), tool-assisted results do NOT count for the main event (note them).
  - A self-report later refuted by an independent run by more than 5 points does not count; take the next valid result.
  - **First by time, not first found**: search backwards — leaderboard histories (Wayback Machine snapshots at web.archive.org are very useful), arXiv v1 dates, earlier papers, earlier model cards, community leaderboards (Papers with Code archive, HELM, Open LLM Leaderboard, MathArena, Epoch AI, Artificial Analysis, Vals.ai, Scale SEAL, official benchmark sites).
  - Record: date (YYYY-MM, the date of public disclosure), system, score, mode (pass@1, maj@k, best-of-N with verifier, etc.), `system_type` (single | scaffold | ensemble), `source_type` (self | leaderboard | independent), source URL.
- **Strict event**: first result at/above threshold that is (a) a single model, (b) no external verifier/reranker, at most maj@64, and (c) verified by organizers or run independently. Same fields, compact.
- **C event**: the maintainer/creator publicly withdraws/deprecates the benchmark, freezes submissions, declares it compromised/obsolete, or explicitly names saturation as the reason for a replacement. Date + URL + short quote.
- **D event**: only if organizers declared a target threshold (e.g., ARC Prize 85%): first organizer-verified result reaching it. Date + URL.
- **E event**: independent public evidence that the benchmark no longer measures its construct: (i) answer leakage/contamination affecting most frontier models; (ii) a trivial method reaching near-ceiling; (iii) an audit finding ≥ 20% defective items after which at least one of the 9 labs publicly stops reporting it. Date + URL + short quote.
- **Abandoned undefeated**: if before any event above the benchmark had ≥ 12 months with no reports by the 9 labs AND ≥ 12 months without leaderboard activity (or a closure announcement) — record `abandoned_date` (last activity) and evidence.
- `primary_date` = earliest of AB, C, D, E (by month); `primary_criterion` = which one.
- `status_cutoff`: `saturated` (an event occurred by 2026-09-30), `alive`, `abandoned`, or `excluded`.
- Flags: `born_saturated` = 1 if the threshold was already met at release; `date_upper_bound` = 1 if the event surely happened but only an upper-bound date can be documented; `reissue` = 1 for annual exam editions.

## 5. Structural properties (short)
`metric`, `scale_max`, `n_items` (official test size), `score_at_release` (best result reported by the creators at release, with system — for exams, the best publicly reported frontier result within ~2 weeks after the exam), `test_access` (public | hidden-labels | private-heldout | rolling), `curation` (exam | expert-written | crowdsourced | synthetic/templated | mined | mixed), `format` (multiple-choice | short-answer | free-form graded | code execution | interactive environment | pairwise expert comparison), `modality` (text | image+text | video | audio | GUI), `class` (static test | agentic environment | LLM-judge | economic/time), `lineage` (parents/successors with dates), `best_by_cutoff` (best known result by 2026-09-30: score, system, date, source).

## 6. Research tools — MANDATORY RULES (v1.1)
- **WebSearch is unavailable** (the session's search budget is exhausted). Work with WebFetch on known or derivable URLs: arXiv abs/html pages (https://arxiv.org/abs/<id>, https://arxiv.org/html/<id>v1), the arXiv API for finding papers (http://export.arxiv.org/api/query?search_query=ti:%22<title words>%22&max_results=5), official benchmark websites and leaderboards, the source URLs listed in `lab_reports`, GitHub READMEs and repositories.
- Direct `curl`/`git` work for github.com and raw.githubusercontent.com (useful for leaderboard repos and their git history: `git clone --depth 300 <repo>` then `git log -p` on the leaderboard file).
- **Do NOT use any browser tools** (no Claude in Chrome / mcp__claude-in-chrome__*, no built-in browser / mcp__remote-devices__Claude_Browser__*), even if they are available to you.
- **If a domain is blocked or fails to load (e.g., web.archive.org, huggingface.co), do NOT try to reach it or its content by other means** — no archive sites, cached copies, mirrors or proxies of blocked content. Record "source unreachable" in `notes`, use other allowed sources, and lower `confidence` or set `date_upper_bound = 1` where appropriate.
- When fetching, ask WebFetch for exact numbers and dates verbatim (it summarizes with a small model). Do not use memory for dates or numbers. Every date needs a URL.

## 6a. Clarifications to the rules (v1.1)
- **Split.** Use the official test split. If official test labels are hidden and the community de facto reports on a dev/validation split (e.g., MMMU val), results on that split count in the main mode — mark `split=dev` in `notes`; the strict event still requires the official test split or organizer verification.
- **E(iii) date** = the later of (audit publication, first time one of the 9 labs publicly stops reporting / switches away because of it).
- **Platform-wide leaderboard shutdowns** (e.g., a whole leaderboard platform closing) count as C only if that leaderboard was the benchmark's only official evaluation path (hidden test labels); otherwise mention in notes and do not code C.
- **C via a successor** counts only if the successor is published by the same maintainers/creators (substantial author overlap) and explicitly names saturation/flaws of the predecessor as a reason.

## 7. Output
Write a CSV with exactly these columns (Python csv module, UTF-8), one row per unit, appending as you go so partial work is never lost:

unit, stratum, exclusion_reason, canonical_name, release, release_source, metric, direction, scale_max, human_baseline, human_type, criterion_ab, threshold, ab_date, ab_system, ab_score, ab_mode, ab_system_type, ab_source_type, ab_source, strict_date, strict_system, strict_score, strict_source, c_date, c_source, c_quote, d_date, d_source, e_date, e_source, e_quote, abandoned_date, abandoned_evidence, primary_date, primary_criterion, status_cutoff, born_saturated, date_upper_bound, reissue, score_at_release, score_at_release_system, n_items, test_access, curation, format, modality, class, lineage, best_by_cutoff, confidence, notes

`confidence`: high (primary sources read for birth and for the event or for the current best) | medium (some dates from aggregators/secondary) | low.

Finish with a short report (≤ 400 words): units completed, the hardest judgement calls (name them), and anything that looked contradictory between sources.
