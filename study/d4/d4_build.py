#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
d4_build.py -- independent score trajectories for AI benchmarks from Epoch AI open data,
threshold-crossing dates, top-5 saturation index (EvalEval) and a cross-check against our
hand-coded saturation dates.

Run:  python3 d4_build.py          (about 5 seconds; no network)
Regenerates, in the directory of this script (override with D4_OUT):
    d4_points.csv     one row per (benchmark, model, setting), best score, model-publication date
    d4_mapping.csv    Epoch benchmark -> our unit mapping table with confidence flags
    d4_crossings.csv  per-benchmark running-best trajectory summary: first crossing of 80/90/95 % and of the
                      hand-coded threshold, S_index >= 0.7 date, logit slope per year (+ diagnostics)
    d4_compare.csv    Epoch crossing date vs hand-coded ab_date / primary_date for mapped units, with category,
                      disagreement size, rank and a reviewed explanation

Inputs (read only; override the base directory with D4_BASE):
    <base>/benchmark-stitching/data/*.csv             Epoch AI benchmark-stitching data (CC BY 4.0)
    <base>/benchmark-saturation/                      EvalEval code + data/manual_annotation_data.csv
    <base>/population_main.json                       our unit population
    <base>/d1/coderA/batch*.csv                       hand-coded thresholds and dates (reference)
    <base>/d1/coderA_repair/batch*_repair.csv         later re-coding of some units (supplementary columns only)

Data logic follows benchmark-stitching/data_loader.py: the score column used per benchmark, the parsing of
"xx.x%" strings, the best score per (model, benchmark), and the join of a result to data/model_versions.csv on
the "Model version" id and its "Version release date".  Differences, all deliberate and flagged in the output:
  * the loader's filters (models evaluated on > 3 benchmarks; date >= 2022-11-01) are NOT applied, because the
    full history of each benchmark is needed;
  * the loader's inner join discards results without a "Model version" or without a "Version release date";
    here such results are kept and dated by a fallback chain, and the origin of every date is recorded in
    `date_source`: (1) Version release date; (2) the same for the base id of an effort/thinking-budget variant
    (gemini-2.5-pro-preview-06-05_32K -> gemini-2.5-pro-preview-06-05); (3) publication date of the parent model
    in all_ai_models.csv; (4) the same via the result's Name; (5) a date column of the source file;
    otherwise the row stays undated (kept in d4_points.csv, left out of trajectories);
  * benchmarks that the loader skips (BoolQ, MCBench, METR, the two FrontierMath public sets) are processed too,
    as is the ARC "Easy score" column;
  * `report_month` (arXiv month of the cited source, or month of the Epoch run) is a diagnostic of when a result
    became public; it never replaces `date` in the trajectories, but `coverage_end` (last month with any result)
    and `cross_threshold_by_report_month` use it.

Only the standard library, numpy and pandas are used.
"""
import json
import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
BASE = Path(os.environ.get("D4_BASE", HERE.parent))
OUT = Path(os.environ.get("D4_OUT", HERE))
STITCH = BASE / "benchmark-stitching" / "data"
SAT = BASE / "benchmark-saturation"
POP_MAIN = BASE / "population_main.json"
CODER_A = BASE / "d1" / "coderA"
CODER_A_REPAIR = BASE / "d1" / "coderA_repair"

CUTOFF_STUDY = pd.Timestamp("2026-09-30")     # study cut-off (CODER_INSTRUCTIONS.md)

# --------------------------------------------------------------------------------------
# 1. model dates (repo logic: model_versions.csv  id -> "Version release date")
# --------------------------------------------------------------------------------------
# One explicit correction: the version release date of Kimi-K2-Instruct-0905 is 2024-09-05 in
# model_versions.csv (year typo: the Kimi K2 model itself was published 2025-07-11 and the id
# says 0905).  Corrected to 2025-09-05 and flagged in `date_source`.
DATE_FIXES = {"Kimi-K2-Instruct-0905": "2025-09-05"}


def norm_name(s):
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def load_model_dates():
    mv = pd.read_csv(STITCH / "model_versions.csv", encoding="utf-8-sig")
    am = pd.read_csv(STITCH / "all_ai_models.csv", low_memory=False, encoding="utf-8-sig")
    am = am[am["Publication date"].notna()].drop_duplicates("Model")
    pub_by_model = dict(zip(am["Model"].astype(str), pd.to_datetime(am["Publication date"], errors="coerce")))
    pub_by_norm = {}
    for m, d in zip(am["Model"].astype(str), pd.to_datetime(am["Publication date"], errors="coerce")):
        pub_by_norm.setdefault(norm_name(m), (m, d))
    vrd_by_id = {str(r["id"]).strip(): pd.to_datetime(r["Version release date"], errors="coerce")
                 for _, r in mv.iterrows()}
    info = {}
    for _, r in mv.iterrows():
        mid = str(r["id"]).strip()
        d, src = pd.NaT, None
        vrd = pd.to_datetime(r["Version release date"], errors="coerce")
        base = base_model(mid)
        if pd.notna(vrd):
            d, src = vrd, "model_versions.version_release_date"
        elif base != mid and pd.notna(vrd_by_id.get(base, pd.NaT)):
            # reasoning-effort / thinking-budget variant of a dated version, e.g. gemini-2.5-pro-preview-06-05_32K
            d, src = vrd_by_id[base], "model_versions.version_release_date(via base id)"
        elif pd.notna(r["Model"]) and str(r["Model"]) in pub_by_model and pd.notna(pub_by_model[str(r["Model"])]):
            d, src = pub_by_model[str(r["Model"])], "all_ai_models.publication_date(via model_versions.Model)"
        if mid in DATE_FIXES:
            d, src = pd.Timestamp(DATE_FIXES[mid]), "model_versions.version_release_date(corrected)"
        info[mid] = (d, src, None if pd.isna(r["Model"]) else str(r["Model"]))
    return info, pub_by_norm


# --------------------------------------------------------------------------------------
# 2. loaders
# --------------------------------------------------------------------------------------
def parse_pct(x, mult=1.0):
    """'71.8%' -> 71.8 ; 0.85 -> 0.85*mult ; '1,234' -> 1234 ; NaN -> NaN"""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return np.nan
    if isinstance(x, str):
        s = x.strip().rstrip("%").replace(",", "").strip()
        if s == "":
            return np.nan
        try:
            v = float(s)
        except ValueError:
            return np.nan
    else:
        v = float(x)
    return round(v * mult, 6)


# external benchmark files: stem -> spec
#   bench     benchmark name as in data_loader.py / benchmark_optim.csv
#   col       score column used by data_loader.py (the "primary" metric)
#   mult      factor that turns the raw number into a 0-100 score
#   rule      human-readable normalisation rule (written to d4_mapping.csv)
#   setting   columns that define a different "setting" of the same model (best score is kept)
#   names     columns used to label a result that has no "Model version"
#   datecol   fallback date column of the source file (day-first)
PCT = "string 'xx.x%' parsed to xx.x (already a 0-100 percentage; data_loader.py divides by 100)"
EXT = {
    "aider_polyglot": dict(bench="Aider polyglot", col="Percent correct", mult=1.0, setting=["Edit format"], names=[],
                           rule="'Percent correct' is numeric 0-100, used as is"),
    "anli": dict(bench="ANLI", col="Score", mult=1.0, setting=["Shots"], names=["Name"], rule="'Score' " + PCT),
    "arc_agi": dict(bench="ARC-AGI", col="Score", mult=1.0, setting=[], names=["Name"], rule="'Score' " + PCT),
    "arc_ai2": dict(bench="ARC AI2", col="Challenge score", mult=1.0, setting=["Shots"], names=["Name"],
                    rule="'Challenge score' " + PCT),
    "balrog": dict(bench="Balrog", col="Average progress", mult=1.0, setting=[], names=[], datecol="Date added",
                   rule="'Average progress' " + PCT),
    "bbh": dict(bench="BBH", col="Average", mult=1.0, setting=["Shots"], names=["Name"], rule="'Average' " + PCT),
    "boolq": dict(bench="BoolQ", col="Score", mult=1.0, setting=["Shots"], names=["Name"], rule="'Score' " + PCT),
    "cadeval": dict(bench="CadEval", col="Overall pass (%)", mult=1.0, setting=["Prompt"], names=["Name"],
                    rule="'Overall pass (%)' " + PCT),
    "csqa2": dict(bench="CSQA2", col="Score", mult=1.0, setting=["Shots"], names=["Name"], rule="'Score' " + PCT),
    "cybench": dict(bench="Cybench", col="Unguided % Solved", mult=1.0, setting=[], names=[],
                    rule="'Unguided % Solved' " + PCT),
    "deepresearch": dict(bench="DeepResearch Bench", col="Average score", mult=1.0, setting=["Agent"], names=[],
                         rule="'Average score' " + PCT),
    "factorio_learning_environment": dict(bench="Factorio learning environment", col="Lab Success %", mult=1.0,
                                          setting=["Tools"], names=[], datecol="Date added",
                                          rule="'Lab Success %' " + PCT),
    "fictionlivebench": dict(bench="Fiction.LiveBench", col="16k token score", mult=1.0, setting=[], names=[],
                             rule="'16k token score' " + PCT),
    "geobench": dict(bench="GeoBench", col="ACW Country %", mult=1.0, setting=["Tools"], names=[],
                     rule="'ACW Country %' " + PCT),
    "gsm8k": dict(bench="GSM8K", col="EM", mult=1.0, setting=["Shots"], names=["Name"], rule="'EM' " + PCT),
    "gso_bench": dict(bench="GSO-Bench", col="Score OPT@1", mult=1.0, setting=["Scaffold"], names=[],
                      rule="'Score OPT@1' " + PCT),
    "hellaswag": dict(bench="HellaSwag", col="Overall accuracy", mult=1.0, setting=["Shots"], names=["Name"],
                      rule="'Overall accuracy' " + PCT),
    "lambada": dict(bench="LAMBADA", col="Score", mult=1.0, setting=["Shots"], names=["Name"], rule="'Score' " + PCT),
    "lech_mazur_writing": dict(bench="Lech Mazur Writing", col="Mean score", mult=10.0, setting=[], names=[],
                               rule="'Mean score' is on a 0-10 scale; x10 (data_loader.py divides by 10)"),
    "livebench": dict(bench="LiveBench", col="Global average", mult=1.0, setting=["id"], names=["id"],
                      rule="'Global average' is numeric 0-100, used as is"),
    "mcbench": dict(bench="MCBench", col="Win rate", mult=1.0, setting=[], names=["Name"],
                    rule="'Win rate' " + PCT + "; arena win rate, NOT a fixed-ceiling accuracy "
                         "(loader skips it); the loader's selection is replicated here only for completeness"),
    "metr": dict(bench="METR", col="average_score", mult=100.0, setting=[], names=[],
                 rule="'average_score' (mean task success, 0-1) x100 as a bounded proxy; the headline METR metric "
                      "('Time horizon', minutes) is unbounded and cannot be put on a 0-100 scale"),
    "mmlu": dict(bench="MMLU", col="EM", mult=1.0, setting=["Shots"], names=["Name"], rule="'EM' " + PCT),
    "openbookqa": dict(bench="OpenBookQA", col="Accuracy", mult=1.0, setting=["Shots"], names=["Name"],
                       rule="'Accuracy' " + PCT),
    "os_world": dict(bench="OSWorld", col="Score", mult=1.0, setting=["Agent"], names=["Agent"], datecol="Date added",
                     rule="'Score' is numeric 0-100, used as is (data_loader.py divides by 100)"),
    "osuniverse": dict(bench="OSUniverse", col="Weighted Score", mult=1.0, setting=["Agent"], names=["Name"],
                       rule="'Weighted Score' " + PCT),
    "piqa": dict(bench="PIQA", col="Score", mult=1.0, setting=["Shots"], names=["Name"], rule="'Score' " + PCT),
    "scienceqa": dict(bench="ScienceQA", col="Score", mult=1.0, setting=["Shots"], names=["Name"],
                      rule="'Score' " + PCT),
    "simple_bench": dict(bench="SimpleBench", col="Score (AVG@5)", mult=1.0, setting=[], names=[],
                         rule="'Score (AVG@5)' " + PCT),
    "superglue": dict(bench="SuperGLUE", col="Score", mult=1.0, setting=[], names=["Name"], rule="'Score' " + PCT),
    "terminal_bench": dict(bench="Terminal Bench", col="Accuracy mean", mult=1.0, setting=["Agent"], names=["Agent"],
                           rule="'Accuracy mean' " + PCT),
    "the_agent_company": dict(bench="The Agent Company", col="% Resolved", mult=1.0, setting=["Model"],
                              names=["Model"], datecol="Date", rule="'% Resolved' " + PCT),
    "triviaqa": dict(bench="TriviaQA", col="EM", mult=1.0, setting=["Shots"], names=["Name"], rule="'EM' " + PCT),
    "videomme": dict(bench="VideoMME", col="Overall (no subtitles)", mult=1.0, setting=["Name", "Frames"],
                     names=["Name"], rule="'Overall (no subtitles)' " + PCT),
    "vpct": dict(bench="VPCT", col="Correct", mult=1.0, setting=[], names=[], rule="'Correct' " + PCT),
    "weirdml": dict(bench="WeirdML", col="Accuracy", mult=1.0, setting=[], names=[], rule="'Accuracy' " + PCT),
    "winogrande": dict(bench="Winogrande", col="Accuracy", mult=1.0, setting=["Shots"], names=["Name"],
                       rule="'Accuracy' " + PCT),
}
# extra metric read from an already-loaded file
EXT_EXTRA = {
    "arc_ai2": dict(bench="ARC AI2 Easy", col="Easy score", mult=1.0, setting=["Shots"], names=["Name"],
                    rule="'Easy score' " + PCT + " (not used by data_loader.py; added for the ARC-Easy unit)"),
}

INTERNAL_RULE = "'Best score (across scorers)' is a 0-1 fraction; x100 (data_loader.py keeps the fraction)"


def clean_str(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    s = str(x).strip()
    if re.fullmatch(r"-?\d+\.0", s):          # '5.0' (float-parsed 'Shots') -> '5'
        s = s[:-2]
    return s if s and s.lower() != "nan" else None


def classify_source(src):
    s = (src or "").lower()
    if not s:
        return "external_unspecified"
    if "helm" in s or "crfm" in s:
        return "external_HELM"
    if re.search(r"leaderboard|dashboard|website|experiments? results|papers?withcode|release notes|github repository",
                 s):
        return "external_leaderboard"
    return "external_paper_or_report"


def date_for(model_version, names, dateinfo, pub_by_norm, in_file_date):
    """date of a result: repo logic first, then flagged fallbacks"""
    if model_version is not None:
        d, src, _ = dateinfo.get(model_version, (pd.NaT, None, None))
        if pd.notna(d):
            return d, src
    for n in names:
        if n is None:
            continue
        hit = pub_by_norm.get(norm_name(n))
        if hit is not None and pd.notna(hit[1]):
            return hit[1], "all_ai_models.publication_date(via Name)"
    if in_file_date is not None and pd.notna(in_file_date):
        return in_file_date, "source_file_date_column"
    return pd.NaT, "none"


ARXIV_RE = re.compile(r"arxiv\.org/(?:abs|pdf|html)/(\d{2})(\d{2})\.\d{4,5}", re.I)


def arxiv_month(link):
    """first day of the month encoded in a new-style arXiv id (2404.14219 -> 2024-04-01); NaT otherwise.
    Used only as a diagnostic: when the result was made public (report month) vs the model's date."""
    m = ARXIV_RE.search(link or "")
    if not m or not 1 <= int(m.group(2)) <= 12:
        return pd.NaT
    return pd.Timestamp(2000 + int(m.group(1)), int(m.group(2)), 1)


BASE_RE = re.compile(r"(_(minimal|low|medium|high)|_\d+[kK]|\s\(\d+[kK] thinking\))$")


def base_model(mid):
    """strip reasoning-effort / thinking-budget suffixes: gpt-5-2025-08-07_high -> gpt-5-2025-08-07"""
    if mid is None:
        return None
    prev = None
    s = mid
    while prev != s:
        prev = s
        s = BASE_RE.sub("", s)
    return s


def load_points():
    dateinfo, pub_by_norm = load_model_dates()
    rows = []
    meta = {}   # benchmark -> dict(rule, kind, file)

    # ---- Epoch internal runs --------------------------------------------------------
    runs = pd.read_csv(STITCH / "benchmarks_runs.csv", encoding="utf-8-sig")
    for _, r in runs.iterrows():
        mid = clean_str(r["model"])
        sc = parse_pct(r["Best score (across scorers)"], 100.0)
        if np.isnan(sc):
            continue
        d, dsrc = date_for(mid, [], dateinfo, pub_by_norm, None)
        started = pd.to_datetime(r["started_at"], errors="coerce")
        rows.append(dict(benchmark=r["task"], model=mid, model_version_raw=mid, display_name=None, setting="",
                         score_0_100=sc, source="Epoch AI benchmarking hub run (Inspect)", run_type="epoch_run",
                         date=d, date_source=dsrc, eval_date=started, entity=base_model(mid),
                         source_link="",
                         report_month=(pd.Timestamp(started.year, started.month, 1) if pd.notna(started) else pd.NaT)))
    for t in runs["task"].unique():
        meta[t] = dict(rule=INTERNAL_RULE, kind="epoch_internal", file="benchmarks_runs.csv",
                       score_field="Best score (across scorers)")

    # ---- external files ------------------------------------------------------------
    def add_external(stem, spec, df):
        bench = spec["bench"]
        meta[bench] = dict(rule=spec["rule"], kind="epoch_external", file=f"external_benchmark_{stem}.csv",
                           score_field=spec["col"])
        link_col = next((c for c in df.columns if str(c).lower().startswith("source link")), None)
        for _, r in df.iterrows():
            sc = parse_pct(r[spec["col"]], spec["mult"])
            if np.isnan(sc):
                continue
            mv = clean_str(r["Model version"]) if "Model version" in df.columns else None
            names = [clean_str(r[c]) for c in spec["names"] if c in df.columns]
            names = [n for n in names if n]
            label = mv or (names[0] if names else None) or f"(unnamed:{clean_str(r.get('id')) or r.name})"
            sett = "; ".join(f"{c}={clean_str(r[c])}" for c in spec["setting"]
                             if c in df.columns and clean_str(r[c]) is not None)
            in_file = None
            if spec.get("datecol") and spec["datecol"] in df.columns and clean_str(r[spec["datecol"]]):
                in_file = pd.to_datetime(clean_str(r[spec["datecol"]]), dayfirst=True, errors="coerce")
            d, dsrc = date_for(mv, names, dateinfo, pub_by_norm, in_file)
            src = clean_str(r["Source"]) if "Source" in df.columns else None
            link = clean_str(r[link_col]) if link_col else None
            rows.append(dict(benchmark=bench, model=label, model_version_raw=mv,
                             display_name=names[0] if names else None, setting=sett, score_0_100=sc,
                             source=src or "", run_type=classify_source(src), date=d, date_source=dsrc,
                             eval_date=pd.NaT, entity=base_model(mv) if mv else label,
                             source_link=link or "", report_month=arxiv_month(link)))

    for stem, spec in EXT.items():
        df = pd.read_csv(STITCH / f"external_benchmark_{stem}.csv", encoding="utf-8-sig")
        add_external(stem, spec, df)
        if stem in EXT_EXTRA:
            add_external(stem, EXT_EXTRA[stem], df)

    pts = pd.DataFrame(rows)
    pts["date"] = pd.to_datetime(pts["date"])
    return pts, meta


def dedupe(pts):
    """one row per (benchmark, model, setting) with the best score"""
    pts = pts.copy()
    pts["setting"] = pts["setting"].fillna("")
    pts = pts.sort_values(["benchmark", "model", "setting", "score_0_100"], ascending=[True, True, True, False])
    out = pts.drop_duplicates(["benchmark", "model", "setting"], keep="first").copy()
    n_dups = len(pts) - len(out)
    return out.reset_index(drop=True), n_dups


# --------------------------------------------------------------------------------------
# 3. row-level data-quality flags (kept in d4_points.csv; excluded rows are not used in trajectories)
# --------------------------------------------------------------------------------------
# (benchmark, model, exclude_from_trajectory, flag)
ROW_FLAGS = [
    ("ANLI", "UNICORN", True,
     "EXCLUDED: 87.3 is UNICORN's abductive-NLI (alpha-NLI) score from the RAINBOW paper, not Adversarial NLI "
     "(hand-coded notes: the AI2 leaderboard 'anli' is alpha-NLI; every other ANLI row is 34-58)"),
    ("HellaSwag", "UNICORN", False,
     "SUSPECT: 86.6 equals UNICORN's WinoGrande AUC in the same paper; hand-coded notes give 93.85 for UNICORN on "
     "HellaSwag (kept as loaded)"),
    ("Winogrande", "UNICORN", False,
     "NOTE: 86.6 is UNICORN's AUC over training sizes; hand-coded notes give XL test accuracy 91.28 (kept as loaded)"),
]


DATE_AFTER_REPORT_MONTHS = 12     # flag rows whose model date is >= 12 months after the arXiv month of the source


def apply_flags(pts):
    pts = pts.copy()
    pts["flag"] = ""
    pts["use_in_trajectory"] = True
    for bench, model, excl, text in ROW_FLAGS:
        m = (pts["benchmark"] == bench) & (pts["model"] == model)
        pts.loc[m, "flag"] = text
        if excl:
            pts.loc[m, "use_in_trajectory"] = False
    # model date later than the cited paper: Epoch used a later model version as a proxy for the paper's model
    # (e.g. GPT-3 few-shot results -> text-davinci-001, 2022-01-27) or the source is mislabelled.  The row is
    # kept and dated as the repo does; the flag shows that the result was public earlier than its date.
    lag = (pts["date"].dt.year - pts["report_month"].dt.year) * 12 + (pts["date"].dt.month - pts["report_month"].dt.month)
    for i in pts.index[lag >= DATE_AFTER_REPORT_MONTHS]:
        txt = (f"DATE_AFTER_REPORT: model date is {int(lag[i])} months after the arXiv month "
               f"({pts.at[i, 'report_month']:%Y-%m}) of the cited source: either a later model version stands in for "
               f"the paper's model (result public earlier than dated) or the source is mislabelled")
        pts.at[i, "flag"] = (pts.at[i, "flag"] + " | " if pts.at[i, "flag"] else "") + txt
    return pts


# --------------------------------------------------------------------------------------
# 4. benchmark -> unit mapping  (confidence: high / medium / low / none)
# --------------------------------------------------------------------------------------
# epoch benchmark -> (unit or None, confidence, relation, note, related unit(s) NOT mapped)
MAPPING = {
    # --- Epoch internal runs -----------------------------------------------------------
    "GPQA diamond": ("GPQA Diamond", "high", "same benchmark / same split",
                     "Epoch-run, 198-question Diamond subset; our unit pools all GPQA variants into Diamond. "
                     "'Best score across scorers'. Epoch's own runs (Inspect 'choice' scorer); lab self-reports "
                     "(our hand-coded events) may use other settings.", ""),
    "SWE-Bench verified": ("SWE-bench Verified", "high", "same benchmark / same split",
                           "Epoch-run, 500-task Verified subset with Epoch's own harness; only 23 runs, none "
                           "before mid-2024; labs' self-reports use their own scaffolds/multiple trials.", ""),
    "FrontierMath-2025-02-28-Private": ("FrontierMath", "medium", "same benchmark (Tier 1-3), version v1",
                                        "Private Tier 1-3 set of 290 problems, snapshot 2025-02-28 (v1). Our unit "
                                        "'FrontierMath' pools Tier 1-3 raw names incl. python-tool variants and v2 "
                                        "reports; hand-coded notes treat v1 as the unit.", ""),
    "FrontierMath-2025-02-28-Public": (None, "none", "subset (10 public problems)",
                                       "Public sample of the Tier 1-3 set (10 problems, scores in steps of 0.1): "
                                       "not comparable with a full-set trajectory; kept separate.", "FrontierMath"),
    "FrontierMath-Tier-4-2025-07-01-Private": ("FrontierMath Tier 4", "medium",
                                               "same benchmark (Tier 4), version v1",
                                               "Private Tier 4 set (48 problems), snapshot 2025-07-01 (v1); "
                                               "hand-coded unit is Tier 4 incl. the corrected v2 set.", ""),
    "FrontierMath-Tier-4-2025-07-01-Public": (None, "none", "subset (public Tier 4 sample)",
                                              "Public Tier 4 sample; all scores 0.", "FrontierMath Tier 4"),
    "MATH level 5": (None, "none", "subset of MATH (Level 5 problems only)",
                     "Level-5 subset of the MATH test set, scored with Epoch's best-of-scorers (includes a "
                     "model-graded equivalence scorer that reads ~10 points above string match). NOT our 'MATH' "
                     "(all levels) nor 'MATH-500'; kept separate.", "MATH; MATH-500"),
    "OTIS Mock AIME 2024-2025": (None, "none", "different exam",
                                 "Mock AIME written by the OTIS course, not an official AIME edition; model-graded. "
                                 "Kept separate from AIME 2024 / AIME 2025.", "AIME 2024; AIME 2025"),
    # --- Epoch external files ----------------------------------------------------------
    "Aider polyglot": ("Aider Polyglot", "high", "same benchmark",
                       "Aider LLM leaderboard 'percent correct' (diff/whole/architect edit formats; best kept). "
                       "Leaderboard entries only (2-attempt harness); hand-coded event is an agentic scaffold "
                       "self-report.", ""),
    "ANLI": ("ANLI", "medium", "same benchmark; metric unclear",
             "Single 'Score' column (the round is not stated; the R1-R3 columns are empty); one row "
             "(UNICORN 87.3) is alpha-NLI and is excluded from the trajectory.", ""),
    "ARC-AGI": ("ARC-AGI-1", "medium", "same benchmark; split differs",
                "ARC Prize leaderboard scores (semi-private evaluation set, ARC 2-attempt scoring, with compute "
                "setting). Hand-coded unit codes the public evaluation set (split=dev). The o3-preview row has no "
                "model version and no date and is therefore undated.", "ARC-AGI-2"),
    "ARC AI2": ("ARC-Challenge", "high", "same benchmark / test (Challenge) split",
                "'Challenge score' column (the loader's choice); mostly vendor technical reports "
                "(25-shot/10-shot).", ""),
    "ARC AI2 Easy": ("ARC-Easy", "high", "same benchmark / test (Easy) split",
                     "'Easy score' column of the ARC AI2 file (not used by the loader).", ""),
    "Balrog": (None, "none", "no unit", "Game-agent benchmark; no unit in population_main.", ""),
    "BBH": ("BIG-Bench Hard", "high", "same benchmark",
            "'Average' column (mostly 3-shot, vendor reports); CoT vs direct not distinguished.", ""),
    "BoolQ": ("BoolQ", "high", "same benchmark",
              "Mix of fine-tuned (T5) and few-shot results (loader skips it as a duplicate of SuperGLUE).", ""),
    "CadEval": (None, "none", "no unit", "CAD code-generation benchmark; no unit.", ""),
    "CSQA2": (None, "none", "different benchmark",
              "CommonsenseQA 2.0 (CSQA2) is a different benchmark from our 'CommonsenseQA' (CSQA 1); no unit.",
              "CommonsenseQA"),
    "Cybench": ("Cybench", "medium", "same benchmark; set differs",
                "'Unguided % solved' on the 40-task set (HAL runs adjusted for a leak); data end 2025-02 "
                "(max 22.5%). Hand-coded events are lab self-reports 2026.", ""),
    "DeepResearch Bench": (None, "none", "no unit", "Research-agent benchmark; no unit.", ""),
    "Factorio learning environment": (None, "none", "no unit", "Game environment; no unit.", ""),
    "Fiction.LiveBench": (None, "none", "no unit", "Long-context fiction comprehension, 16k-token column; no unit.",
                          ""),
    "GeoBench": (None, "none", "no unit", "Geolocation benchmark; no unit.", ""),
    "GSM8K": ("GSM8K", "high", "same benchmark",
              "'EM' column; mixture of HELM, vendor reports; 5-8-shot.", ""),
    "GSO-Bench": (None, "none", "no unit", "Software-optimisation benchmark; no unit.", ""),
    "HellaSwag": ("HellaSwag", "high", "same benchmark",
                  "'Overall accuracy'; mostly vendor reports (0-10 shots, split not stated); the UNICORN row is "
                  "suspect (see d4_points.flag); hand-coded event is a fine-tuned leaderboard test submission.",
                  ""),
    "LAMBADA": ("LAMBADA", "high", "same benchmark", "'Score' accuracy column.", ""),
    "Lech Mazur Writing": (None, "none", "no unit", "LLM-judged creative-writing mean score on a 0-10 scale "
                           "(x10); population excludes writing scores as no-ceiling.", "Creative Writing; "
                           "WritingBench"),
    "LiveBench": ("LiveBench", "medium", "same benchmark; rolling version",
                  "Global average of the LiveBench-2024-11-25 release only; rolling benchmark whose items are "
                  "refreshed, so levels across releases are not comparable.", ""),
    "MCBench": (None, "none", "no unit", "Minecraft-build arena (win rate / Elo); no ceiling; loader skips it.", ""),
    "METR": (None, "none", "different benchmark",
             "METR time-horizon suite; headline metric is a time horizon in minutes (unbounded); "
             "bounded proxy = mean task success 'average_score' x100. Our 'RE-Bench' unit is a different "
             "METR benchmark.", "RE-Bench"),
    "MMLU": ("MMLU", "medium", "same benchmark; subsets possible",
             "'EM' column. 80 rows come from the Stanford CRFM (HELM Lite) leaderboard page, i.e. HELM's own "
             "prompts/sampling, the rest are vendor reports (5-shot); no CoT/majority-vote results (the hand-coded "
             "event is Gemini Ultra CoT@32), so levels may differ from full-MMLU numbers.", ""),
    "OpenBookQA": ("OpenBookQA", "high", "same benchmark",
                   "'Accuracy'; mostly vendor reports with 0-10 shots (split not stated); no fine-tuned or "
                   "self-consistency results, which the hand-coded event uses.", ""),
    "OSUniverse": (None, "none", "no unit", "GUI-agent benchmark; no unit.", ""),
    "OSWorld": ("OSWorld", "medium", "same benchmark; original (non-Verified) version",
                "OS World website leaderboard snapshot (entries added up to 2025-04, max 42.5%); dated by model "
                "publication date, else by name match, else the website's 'Date added'. Our 'OSWorld-Verified' "
                "is a separate unit.", "OSWorld-Verified"),
    "PIQA": ("PIQA", "high", "same benchmark", "'Score' column (vendor reports 0-5 shots plus the UNICORN "
                                                "leaderboard result).", ""),
    "ScienceQA": (None, "none", "no unit", "No ScienceQA unit in population_main; most rows are undated "
                  "leaderboard methods without a model version.", ""),
    "SimpleBench": (None, "none", "no unit", "No SimpleBench unit in population_main.", ""),
    "SuperGLUE": ("SuperGLUE", "high", "same benchmark",
                  "'Score' average; only 3 of 8 rows can be dated (T5-Base/Large/Small and Switch have no date); "
                  "the 'deBERTa' row has no score.", ""),
    "Terminal Bench": ("Terminal-Bench 1.0", "medium", "same benchmark; version inferred",
                       "Terminal-Bench leaderboard (Epoch snapshot Dec 2025). Scores and agents (Terminus, "
                       "Droid, Warp, OB-1, ...) match the 1.0 (terminal-bench-core) leaderboard; version is not "
                       "stated in the data. Results are agent+model systems, dated by the model's publication "
                       "date.", "Terminal-Bench 2.0"),
    "The Agent Company": (None, "none", "no unit", "Agentic office-task benchmark; no unit.", ""),
    "TriviaQA": ("TriviaQA", "high", "same benchmark",
                 "'EM'; vendor reports and HELM with 1-64 shots (split not stated; not official hidden-test "
                 "results).", ""),
    "VideoMME": ("Video-MME", "medium", "same benchmark; setting differs",
                 "'Overall (no subtitles)' column of the leaderboard (the loader's choice); our unit pools "
                 "with/without-subtitle raw names and the hand-coded event is the with-subtitles setting "
                 "(the file's with-subtitles column peaks at 81.3, Gemini 1.5 Pro, also below 90). Data end 2025-05.",
                 ""),
    "VPCT": (None, "none", "no unit", "Visual physics test; no unit.", ""),
    "WeirdML": (None, "none", "no unit", "ML-task benchmark; no unit.", ""),
    "Winogrande": ("WinoGrande", "high", "same benchmark",
                   "'Accuracy' (vendor reports, 0-5 shots); the UNICORN row is the AUC, not accuracy (see "
                   "d4_points.flag); hand-coded event is a fine-tuned UNICORN leaderboard result.", ""),
}

# --------------------------------------------------------------------------------------
# 5. test-set sizes for the saturation index
# --------------------------------------------------------------------------------------
# name of the matching row of benchmark-saturation/data/manual_annotation_data.csv
ANNOT_NAME = {
    "ANLI": "ANLI",
    "BBH": "Big Bench Hard (BBH)",
    "BoolQ": "BoolQ",
    "GSM8K": "GSM8K",
    "HellaSwag": "HellaSwag",
    "LAMBADA": "LAMBADA",
    "LiveBench": "LiveBench: A Challenging, Contamination-Free LLM Benchmark",
    "MMLU": "MMLU: Measuring Massive Multitask Language Understanding",
    "OpenBookQA": "OpenBookQA",
    "PIQA": "PIQA",
    "SuperGLUE": "SuperGLUE",
    "Terminal Bench": "Terminal Bench",
    "TriviaQA": "TriviaQA",
    "Winogrande": "Winogrande",
}
# split-specific sizes known from the benchmarks' documentation (NOT in the repo; flagged "assumed")
KNOWN_N = {
    "GPQA diamond": (198, "known: Diamond subset size (EvalEval lists 564 for the whole GPQA)"),
    "SWE-Bench verified": (500, "known: SWE-bench Verified size (EvalEval lists 2,294 for full SWE-bench)"),
    "FrontierMath-2025-02-28-Private": (290, "known: Epoch private Tier 1-3 set; consistent with the +-SE in "
                                             "benchmarks_runs.csv 'Scores'"),
    "FrontierMath-Tier-4-2025-07-01-Private": (48, "known: Epoch private Tier 4 set; consistent with the +-SE in "
                                                   "benchmarks_runs.csv 'Scores'"),
    "Aider polyglot": (225, "known: 225 Exercism exercises"),
    "ARC-AGI": (100, "known: ARC Prize semi-private evaluation set (100 tasks); EvalEval lists 1,000"),
    "Cybench": (40, "known: 40 CTF tasks"),
    "OSWorld": (369, "known: 369 tasks of the original benchmark"),
    "VideoMME": (2700, "known: 2,700 QA pairs (900 videos)"),
    "ARC AI2": (1172, "known: ARC-Challenge test split"),
    "ARC AI2 Easy": (2376, "known: ARC-Easy test split"),
}


def load_test_sizes():
    ann = pd.read_csv(SAT / "data" / "manual_annotation_data.csv")
    ann = ann[ann["Name"].notna()].copy()
    ann["_n"] = pd.to_numeric(ann["Quantity of test samples"].astype(str).str.replace(",", "").str.strip(),
                              errors="coerce")
    by_name = {" ".join(str(n).split()): n_ for n, n_ in zip(ann["Name"], ann["_n"])}
    sizes = {}
    for b, nm in ANNOT_NAME.items():
        v = by_name.get(" ".join(nm.split()))
        if v is not None and not np.isnan(v):
            sizes[b] = (int(v), f"EvalEval annotation: '{nm}'")
    for b, (v, txt) in KNOWN_N.items():
        sizes[b] = (int(v), txt)
    # CadEval: total tasks is a column of the Epoch file
    cad = pd.read_csv(STITCH / "external_benchmark_cadeval.csv", encoding="utf-8-sig")
    if "Total tasks" in cad.columns and cad["Total tasks"].notna().any():
        sizes["CadEval"] = (int(cad["Total tasks"].median()), "Epoch file column 'Total tasks'")
    return sizes


# --------------------------------------------------------------------------------------
# 6. trajectories: running best, threshold crossings, S_index, logit slope
# --------------------------------------------------------------------------------------
TOL = 1e-9
SINDEX_THRESHOLD = 0.7           # EvalEval: S_index >= 0.7 = "high" saturation (time_to_saturation)
TOP_N = 5                        # EvalEval default
ALPHA = 0.5                      # n_eff = n ** alpha
LOGIT_CLIP = (0.005, 0.995)      # applied to p = score/100 before the logit; exact 0 and 100 are dropped
MIN_POINTS_SLOPE = 6             # dated points required for a slope
MIN_FRONTIER_SLOPE = 3           # record-setting points required for a slope


def first_cross(use_sorted, x):
    """first record date at which score >= x; returns (date, model, setting, score, date_source, source)"""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    hit = use_sorted[use_sorted["score_0_100"] >= x - TOL]
    if hit.empty:
        return None
    d0 = hit["date"].min()
    r = hit[hit["date"] == d0].sort_values("score_0_100", ascending=False).iloc[0]
    return r


def s_index_trace(use, n_test):
    """EvalEval temporal logic: top-5 of everything dated up to now (one score per base model), evaluated
    after each distinct date; S = exp(-R^2), R = (s1-s5)/SE_delta, SE(s)=sqrt(s(1-s)/n_eff), n_eff = n**alpha"""
    ent = use.groupby("entity").agg(score=("score_0_100", "max"), date=("date", "min")).reset_index()
    ent = ent.sort_values("date")
    n_eff = n_test ** ALPHA
    out = []
    for d in sorted(ent["date"].unique()):
        sc = ent.loc[ent["date"] <= d, "score"].to_numpy(float)
        if len(sc) < TOP_N:
            continue
        top = np.sort(sc)[::-1][:TOP_N] / 100.0
        s1, s5 = top[0], top[-1]
        se = lambda s: math.sqrt(s * (1 - s) / n_eff) if 0 < s < 1 else 0.0
        se_delta = math.sqrt(se(s1) ** 2 + se(s5) ** 2)
        r_norm = 0.0 if se_delta == 0 else (s1 - s5) / se_delta
        out.append(dict(date=pd.Timestamp(d), s1=100 * s1, s5=100 * s5, mean_top=100 * float(top.mean()),
                        r_norm=r_norm, s_index=math.exp(-r_norm ** 2)))
    return pd.DataFrame(out)


def logit_slope(use_sorted):
    """OLS slope of logit(running best / 100) on time (years), fitted on the frontier points = days on which
    the running best strictly increases (daily maximum used)."""
    daily = use_sorted.groupby("date", as_index=False)["score_0_100"].max()
    daily["prev_best"] = daily["score_0_100"].cummax().shift(1)
    fr = daily[(daily["prev_best"].isna()) | (daily["score_0_100"] > daily["prev_best"] + TOL)]
    n_frontier_all = len(fr)
    fr = fr[(fr["score_0_100"] > 0) & (fr["score_0_100"] < 100)]
    if len(fr) < MIN_FRONTIER_SLOPE:
        return dict(slope=np.nan, r2=np.nan, n_frontier=len(fr), n_frontier_all=n_frontier_all, range="", span=np.nan)
    p = np.clip(fr["score_0_100"].to_numpy(float) / 100.0, *LOGIT_CLIP)
    y = np.log(p / (1 - p))
    t = (fr["date"] - fr["date"].min()).dt.days.to_numpy(float) / 365.25
    if np.ptp(t) == 0:
        return dict(slope=np.nan, r2=np.nan, n_frontier=len(fr), n_frontier_all=n_frontier_all, range="", span=0.0)
    slope, icpt = np.polyfit(t, y, 1)
    resid = y - (slope * t + icpt)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else np.nan
    rng = f"{fr['date'].min().date()}..{fr['date'].max().date()}"
    return dict(slope=float(slope), r2=r2, n_frontier=len(fr), n_frontier_all=n_frontier_all, range=rng,
                span=float(np.ptp(t)))


def fmt_date(x):
    return "" if x is None or (not isinstance(x, str) and pd.isna(x)) else str(pd.Timestamp(x).date())


def analyse_benchmark(bench, df, thr_pct, n_info):
    """df: all (deduplicated) rows of one benchmark"""
    use_all = df[df["use_in_trajectory"]]
    use = use_all[use_all["date"].notna()].sort_values(["date", "score_0_100"], ascending=[True, False])
    und = use_all[use_all["date"].isna()]
    res = dict(n_points=len(use), n_models=int(use["entity"].nunique()), n_undated=len(und),
               max_score_undated=float(und["score_0_100"].max()) if len(und) else np.nan,
               n_excluded=int((~df["use_in_trajectory"]).sum()))
    # last month in which the benchmark has any (trajectory-eligible) result: arXiv month of the source or month of
    # the Epoch run where known, else the model date; undated rows count when they carry a report month
    pub_all = use_all["report_month"].fillna(use_all["date"])
    res["coverage_end"] = pub_all.max() if pub_all.notna().any() else pd.NaT
    if len(use) == 0:
        return res, None
    res.update(first_date=use["date"].min(), last_date=use["date"].max(), max_score=float(use["score_0_100"].max()))
    mx = use.sort_values(["score_0_100", "date"], ascending=[False, True]).iloc[0]
    res.update(max_score_date=mx["date"], max_score_model=mx["model"])
    for key, x in (("cross80", 80.0), ("cross90", 90.0), ("cross95", 95.0), ("cross_threshold", thr_pct)):
        r = first_cross(use, x)
        res[key] = r["date"] if r is not None else pd.NaT
        if key == "cross_threshold":
            res["cross_threshold_model"] = r["model"] if r is not None else ""
            res["cross_threshold_score"] = float(r["score_0_100"]) if r is not None else np.nan
            res["cross_threshold_date_source"] = r["date_source"] if r is not None else ""
            res["cross_threshold_source"] = r["source"] if r is not None else ""
            res["cross_threshold_source_link"] = r["source_link"] if r is not None else ""
            res["cross_threshold_report_month"] = r["report_month"] if r is not None else pd.NaT
            res["cross_threshold_run_type"] = r["run_type"] if r is not None else ""
            # alternative date: earliest month in which any result >= threshold was publicly reported
            # (arXiv month / Epoch run month where known, else the model date)
            hit_all = use[use["score_0_100"] >= x - TOL] if x is not None else use.iloc[0:0]
            res["cross_threshold_by_report_month"] = (hit_all["report_month"].fillna(hit_all["date"]).min()
                                                      if len(hit_all) else pd.NaT)
    # saturation index
    res.update(n_test=np.nan, n_test_source="", sindex07_date=pd.NaT, sindex07_top1=np.nan,
               sindex07_date_top1_ge80=pd.NaT, sindex_last=np.nan, sindex_max=np.nan)
    if n_info is not None:
        res["n_test"], res["n_test_source"] = n_info
        tr = s_index_trace(use, n_info[0])
        if len(tr):
            hit = tr[tr["s_index"] >= SINDEX_THRESHOLD - TOL]
            if len(hit):
                res["sindex07_date"] = hit.iloc[0]["date"]
                res["sindex07_top1"] = float(hit.iloc[0]["s1"])
            hit80 = tr[(tr["s_index"] >= SINDEX_THRESHOLD - TOL) & (tr["s1"] >= 80)]
            if len(hit80):
                res["sindex07_date_top1_ge80"] = hit80.iloc[0]["date"]
            res["sindex_last"] = float(tr.iloc[-1]["s_index"])
            res["sindex_max"] = float(tr["s_index"].max())
    # slope
    res.update(logit_slope_per_year=np.nan, slope_range="", slope_n_frontier=0, slope_r2=np.nan,
               slope_span_years=np.nan, slope_note="")
    if len(use) >= MIN_POINTS_SLOPE:
        ls = logit_slope(use)
        res.update(logit_slope_per_year=ls["slope"], slope_range=ls["range"], slope_n_frontier=ls["n_frontier"],
                   slope_r2=ls["r2"], slope_span_years=ls["span"])
        if np.isnan(ls["slope"]):
            res["slope_note"] = (f"no slope: {ls['n_frontier']} frontier points with 0 < score < 100 "
                                 f"(need {MIN_FRONTIER_SLOPE})")
        else:
            warn = []
            if ls["n_frontier"] < 5:
                warn.append("few frontier points")
            if ls["span"] < 0.5:
                warn.append("frontier spans < 0.5 year")
            res["slope_note"] = "; ".join(warn)
    else:
        res["slope_note"] = f"no slope: {len(use)} dated points (need {MIN_POINTS_SLOPE})"
    return res, use


# --------------------------------------------------------------------------------------
# 7. hand-coded data and comparison
# --------------------------------------------------------------------------------------
def ym(s):
    """'2023-03' (or '2023-03-15') -> Timestamp of that month's first day; blank -> NaT"""
    if s is None or (isinstance(s, float) and np.isnan(s)) or str(s).strip() in ("", "nan"):
        return pd.NaT
    t = str(s).strip()[:7]
    try:
        return pd.Timestamp(t + "-01")
    except ValueError:
        return pd.NaT


def month_diff(a, b):
    """calendar-month difference a - b (Timestamps), NaN if either is missing"""
    if pd.isna(a) or pd.isna(b):
        return np.nan
    return (a.year - b.year) * 12 + (a.month - b.month)


def load_hand():
    A = pd.concat([pd.read_csv(f, encoding="utf-8-sig", dtype=str) for f in sorted(CODER_A.glob("batch*.csv"))],
                  ignore_index=True)
    reps = sorted(CODER_A_REPAIR.glob("batch*_repair.csv"))
    R = pd.concat([pd.read_csv(f, encoding="utf-8-sig", dtype=str) for f in reps], ignore_index=True) \
        if reps else pd.DataFrame(columns=["unit"])
    assert A["unit"].is_unique and R["unit"].is_unique, "duplicate unit in hand-coded files"
    return A.set_index("unit", drop=False), R.set_index("unit", drop=False)


def num(x):
    try:
        return float(str(x).replace(",", "").strip())
    except ValueError:
        return np.nan


# --------------------------------------------------------------------------------------
# 8. comparison rules and reviewed notes
# --------------------------------------------------------------------------------------
AGREE_TOL_MONTHS = 3     # |Epoch crossing - hand-coded AB date| <= 3 calendar months counts as agreement
DISAGREE = ("epoch_later", "epoch_earlier", "hand_only", "epoch_only")
CATEGORY_ORDER = {"epoch_later": 0, "epoch_earlier": 0, "hand_only": 0, "epoch_only": 0, "agree": 1,
                  "hand_only_no_coverage": 2, "both_neither": 3}

# Reviewed explanation per mapped unit, written after reading the numbers in d4_compare.csv / d4_points.csv and the
# hand-coded notes (d1/coderA).  "AB" = hand-coded first public result at/above the threshold (any system, self-
# reports and leaderboards count); "C" = maintainer withdrawal/replacement (not a score event).
ANALYST_NOTES = {
    "Aider Polyglot": (
        "Epoch (Aider leaderboard) never reaches 90: best 88.0 (gpt-5 high, 2025-08), data to 2025-09. The hand-coded "
        "event (2025-04) is Refact.ai's agent scaffold self-report (92.9; up to 30 steps, runs the exercise tests "
        "itself), which is not an entry of Aider's own leaderboard. Disagreement is about which systems count."),
    "ANLI": (
        "Neither source has a result at 90 (hand status abandoned). Epoch best 58.1 (gpt-3.5-turbo-1106, vendor report), "
        "data to 2024-04; Epoch's UNICORN 87.3 is alpha-NLI, not ANLI, and is excluded from the trajectory."),
    "ARC-AGI-1": (
        "Epoch (ARC Prize leaderboard, semi-private set, official compute settings) peaks at 73% (gpt-5.1 high, "
        "2025-11); o3-preview low (76%) has no model version and is undated. The hand-coded event (2024-12) is "
        "o3-preview's 91.5% on the public evaluation set with 1024 samples, a setting absent from Epoch's file; "
        "hand-coded notes put the first semi-private >=90 at 2025-12 (GPT-5.2 Pro), after Epoch's snapshot."),
    "ARC-Challenge": (
        "Agreement: GPT-4 (25-shot 96.3, GPT-4 technical report, 2023-03) is the first >=90 in both sources; Epoch "
        "dates the row by the model name (GPT-4, 2023-03-15)."),
    "ARC-Easy": (
        "Epoch is later. Its first >=90 is Mistral-7B-v0.1 (90.6, 10-shot, from the Phi-3 report; dated by the model's "
        "release, 2023-09; the earliest report with >=90 is Qwen-14B 90.3 in the Nemotron-4 report, 2024-02). The "
        "hand-coded event is a fine-tuned, retrieval-augmented UnifiedQA (92.7, 2021-02 per the original coder; 92.0, "
        "2020-05 per the repair coder), a system type that Epoch's file does not contain."),
    "BIG-Bench Hard": (
        "Epoch's BBH rows are paper/vendor reports only; best 87.5 (DeepSeek-V3, 2024-12), so no 90 within its data "
        "(to 2024-12). The hand-coded event (2024-06) is Claude 3.5 Sonnet's 93.1 (3-shot CoT, Anthropic model card), "
        "absent from Epoch's file, which does not separate CoT from direct answering."),
    "BoolQ": (
        "Agreement: T5-11B fine-tuned (91.2 in Epoch, 91.0 hand-coded; T5 paper, 2019-10) is the first >=90 in both. "
        "BoolQ is already above 90 at Epoch's first dated point, so the trajectory has no pre-crossing history "
        "(no slope); 73 of 177 rows are undated (HELM/method rows)."),
    "Cybench": (
        "Not testable: the hand-coded AB event is a 2026 lab self-report (Meta Muse Spark 1.1, 92.9%, 2026-07); "
        "Epoch's Cybench data end 2025-02 with a best of 22.5% (o3-mini medium)."),
    "FrontierMath": (
        "No AB crossing in either source (Epoch best 32.4%, gpt-5 high, v1 Tier 1-3). The hand-coded primary date "
        "2026-05 is a C event (Epoch's announcement of errors in about a third of problems; v2 released 2026-06-12), "
        "which score data cannot test."),
    "FrontierMath Tier 4": (
        "Not testable: the hand-coded AB event (2026-09, GPT-6 Astra 97.6% on the corrected Tier 4 v2 set, "
        "Epoch-verified) is after Epoch's data (Tier 4 v1: best 12.5%, gpt-5 high; data to 2025-11). The hand-coded "
        "primary date 2026-05 is a C event."),
    "GPQA Diamond": (
        "Not testable: the hand-coded event is Gemini 3 Pro 91.9% (vendor blog, 2025-11-18), in the last month of "
        "Epoch's data (best 87.6, gpt-5.1 high, 2025-11-13). Epoch shows no >=90 before its snapshot, consistent with "
        "the hand-coded notes (no earlier >=90 without tools)."),
    "GSM8K": (
        "Agreement: GPT-4 (5-shot CoT 92.0, GPT-4 report, 2023-03-14) is the first >=90 in both sources. 64 of 206 "
        "rows are undated; the highest, 'GPT-4 (8-shot)' 91.4 from the Qwen report, is the same model."),
    "HellaSwag": (
        "Epoch is later. Its first >=90 is GPT-4 (95.3, 10-shot, 2023-03). The hand-coded event is UNICORN (fine-tuned "
        "T5-11B, 93.85, AI2 leaderboard): dated 2020-07 by the leaderboard API (original coder) or 2021-03 by the paper "
        "(repair coder). Epoch's UNICORN row reads 86.6, equal to UNICORN's WinoGrande AUC (probably a copy error); "
        "with 93.9 Epoch would cross at 2021-03-24, matching the repair date."),
    "LAMBADA": (
        "Neither source has a result at 90 (hand status abandoned). Epoch best 87.2 (Megatron-Turing NLG 530B, "
        "2022-01; undated GLaM 86.6); data to 2023-11."),
    "LiveBench": (
        "Neither: hand status alive. Epoch has only the LiveBench-2024-11-25 release (best 82.35, "
        "gemini-2.5-pro-exp-03-25); the benchmark is rolling, so releases are not comparable."),
    "MMLU": (
        "Epoch's best is 88.1 (gpt-4o-2024-11-20, Phi-4 report; HELM rows peak at 87.3), so no 90 within its data "
        "(to 2025-02). The hand-coded event (2023-12) is Gemini Ultra's CoT@32 = 90.04 (5-shot greedy was 83.7); "
        "Epoch has no CoT@k / majority-vote results. Disagreement is about protocol."),
    "OSWorld": (
        "No AB crossing in either source (Epoch, original OSWorld leaderboard: best 42.5, UI-TARS-1.5, 2025-04; the "
        "hand-coded notes give about 45% as the maximum). The hand-coded primary date 2025-07 is a C event "
        "(OSWorld-Verified release, 2025-07-28), which score data cannot test."),
    "OpenBookQA": (
        "Epoch's rows are paper/vendor reports only (53 rows, no HELM) with best 88.0 (Phi-3-mini, 2024-04), so no "
        ">=91.7 within its data (to 2024-07). The hand-coded event (2022-10) is PaLM-540B after LMSI self-improvement "
        "(92.0 greedy, 94.4 maj@32); the hand-coded strict event is GPT-4 in HELM Lite (96.0, 2023-12). Neither is "
        "in Epoch's file."),
    "PIQA": (
        "Agreement: UNICORN (fine-tuned T5-11B, 90.1, AI2 leaderboard / RAINBOW paper, 2021-03) is the first >=90 in "
        "both sources; Epoch dates the row by the model name."),
    "SWE-bench Verified": (
        "Not testable: the hand-coded AB event is 2026-04 (Claude Mythos Preview 93.9%, Anthropic's own harness). "
        "Epoch's own-harness runs (23, to 2025-10) peak at 64.8 (claude-sonnet-4-5); the hand-coded notes list lab "
        "self-reports of 80.9 (Opus 4.5, 2025-11). The hand-coded primary date 2026-02 is a C event."),
    "SuperGLUE": (
        "Not testable (sparse data): Epoch has 11 rows, 3 datable (T5-11B 88.9, 2019-10; a GPT-3 row dated 2022-01 via "
        "text-davinci-001), Switch/T5 rows undated (best 84.7) and an empty DeBERTa row. The hand-coded event (2021-01: "
        "DeBERTa ensemble 90.3, T5+Meena 90.2, leaderboard submissions) is not recorded."),
    "Terminal-Bench 1.0": (
        "No AB crossing in either source (Epoch best 64.5, claude-sonnet-4-5 agent systems, 2025-09). The hand-coded "
        "primary date 2025-11 is a C event (Terminal-Bench 2.0 announcement), which score data cannot test."),
    "TriviaQA": (
        "Neither: hand status alive. Epoch best 87.6 (Llama-2-70b-hf, 2023-07); data to 2024-12."),
    "Video-MME": (
        "Not testable: the hand-coded AB event (2026-07, Kimi K3 90.0 with subtitles) is after Epoch's Video-MME data "
        "(to 2025-05; best 75.0 without and 81.3 with subtitles, Gemini 1.5 Pro). The hand-coded primary date 2026-04 "
        "is a C event (Video-MME-v2)."),
    "WinoGrande": (
        "Epoch has no >=90: best 89.2 (Llama-3.1-405B, DeepSeek-V3 report; data to 2024-12). The hand-coded event is "
        "UNICORN's XL test accuracy 91.28 (fine-tuned, AI2 leaderboard, 2021-03); Epoch's UNICORN row holds 86.6, the "
        "leaderboard AUC, not accuracy (metric mismatch). Vendor rows are 0-5-shot results."),
}


def month_floor(t):
    return pd.Timestamp(t.year, t.month, 1) if pd.notna(t) else pd.NaT


def fmt_ym(x):
    return "" if x is None or pd.isna(x) else f"{pd.Timestamp(x):%Y-%m}"


def txt(x):
    return x if isinstance(x, str) else ""


def categorise(ab, e_thr, cov_end):
    """category of one unit.  ab: hand-coded AB month, e_thr: Epoch first crossing of the hand-coded threshold,
    cov_end: last month with any Epoch result for the benchmark."""
    if pd.notna(ab) and pd.notna(e_thr):
        d = month_diff(e_thr, ab)
        return "agree" if abs(d) <= AGREE_TOL_MONTHS else ("epoch_later" if d > 0 else "epoch_earlier")
    if pd.notna(ab):
        # the absence of an Epoch crossing says something only if Epoch has results after the hand-coded event
        if pd.isna(cov_end) or month_diff(ab, cov_end) >= 0:
            return "hand_only_no_coverage"
        return "hand_only"
    if pd.notna(e_thr):
        return "epoch_only"
    return "both_neither"


def auto_reason(r):
    """rule-based first guess; the reviewed explanation is in ANALYST_NOTES / column likely_reason"""
    cat = r["category"]
    if cat == "agree":
        return "Epoch crossing and hand-coded AB event coincide (within %d months)." % AGREE_TOL_MONTHS
    if cat == "both_neither":
        return "No result at or above the threshold in either source."
    if cat == "hand_only_no_coverage":
        return ("Hand-coded AB event (%s) is in or after the last month of Epoch's data (%s): not testable."
                % (r["hand_ab_date"], r["epoch_coverage_end"]))
    if cat == "hand_only":
        return ("Epoch best is %.1f (%s), below the threshold, although its data run to %s: the hand-coded event rests "
                "on a system or setting that Epoch's data lack (%s; %s)." % (
                    r["epoch_max_score"], r["epoch_max_score_model"], r["epoch_coverage_end"],
                    r["hand_ab_source_type"], r["hand_ab_system"]))
    if cat == "epoch_only":
        return ("Epoch has a result at or above the threshold (%s, %.1f) that the hand-coded record does not count "
                "(other setting/split or status %s)." % (r["epoch_cross_model"], r["epoch_cross_score"],
                                                         r["hand_status"]))
    if cat == "epoch_later":
        return ("Epoch's first crossing (%s, %.1f) is later than the hand-coded event: the earlier hand-coded system "
                "(%s) is not in Epoch's data." % (r["epoch_cross_model"], r["epoch_cross_score"],
                                                  r["hand_ab_system"]))
    if cat == "epoch_earlier":
        return ("Epoch crosses earlier (%s, %.1f): model date precedes the public report, or another setting/split."
                % (r["epoch_cross_model"], r["epoch_cross_score"]))
    return ""


# --------------------------------------------------------------------------------------
# 9. main
# --------------------------------------------------------------------------------------
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pts_raw, meta = load_points()
    pts, n_dups = dedupe(pts_raw)
    pts = apply_flags(pts)
    pop = json.load(open(POP_MAIN, encoding="utf-8"))
    hand, repair = load_hand()
    sizes = load_test_sizes()
    optim = pd.read_csv(STITCH / "benchmark_optim.csv")
    optim["benchmark"] = optim["benchmark"].astype(str).str.strip()
    epoch_release = dict(zip(optim["benchmark"], pd.to_datetime(optim["benchmark_release_date"], errors="coerce")))
    horizon = max(pts["date"].max(), pts["report_month"].max(), pts["eval_date"].max())

    def unit_of(b):
        return MAPPING.get(b, (None,))[0]

    pts["unit"] = pts["benchmark"].map(unit_of)

    # ---- crossings -------------------------------------------------------------------
    cross_rows = []
    benches = sorted(pts["benchmark"].unique(), key=lambda s: s.lower())
    for b in benches:
        df = pts[pts["benchmark"] == b]
        unit = unit_of(b)
        thr_pct, thr_raw = np.nan, ""
        if unit is not None and unit in hand.index:
            h = hand.loc[unit]
            t, sm = num(h["threshold"]), num(h["scale_max"])
            if not np.isnan(t) and not np.isnan(sm) and sm > 0:
                thr_pct, thr_raw = 100.0 * t / sm, h["threshold"]
        res, use = analyse_benchmark(b, df, None if np.isnan(thr_pct) else thr_pct, sizes.get(b))
        row = dict(benchmark=b, unit=unit or "", n_points=res["n_points"],
                   first_date=fmt_date(res.get("first_date")), last_date=fmt_date(res.get("last_date")),
                   cross80=fmt_date(res.get("cross80")), cross90=fmt_date(res.get("cross90")),
                   cross95=fmt_date(res.get("cross95")), cross_threshold=fmt_date(res.get("cross_threshold")),
                   threshold=thr_raw, sindex07_date=fmt_date(res.get("sindex07_date")),
                   logit_slope_per_year=res.get("logit_slope_per_year"), slope_range=res.get("slope_range", ""))
        row.update(
            map_confidence=MAPPING.get(b, (None, "none"))[1], n_models=res["n_models"],
            n_undated=res["n_undated"], n_excluded=res["n_excluded"],
            max_score=res.get("max_score"), max_score_date=fmt_date(res.get("max_score_date")),
            max_score_model=res.get("max_score_model", ""), max_score_undated=res["max_score_undated"],
            coverage_end=fmt_ym(res.get("coverage_end")),
            cross_threshold_model=res.get("cross_threshold_model", ""),
            cross_threshold_score=res.get("cross_threshold_score"),
            cross_threshold_date_source=res.get("cross_threshold_date_source", ""),
            cross_threshold_run_type=res.get("cross_threshold_run_type", ""),
            cross_threshold_source=res.get("cross_threshold_source", ""),
            cross_threshold_source_link=res.get("cross_threshold_source_link", ""),
            cross_threshold_report_month=fmt_ym(res.get("cross_threshold_report_month")),
            cross_threshold_by_report_month=fmt_ym(res.get("cross_threshold_by_report_month")),
            n_test=res.get("n_test"), n_test_source=res.get("n_test_source", ""),
            sindex07_top1=res.get("sindex07_top1"),
            sindex07_date_top1_ge80=fmt_date(res.get("sindex07_date_top1_ge80")),
            sindex_last=res.get("sindex_last"), sindex_max=res.get("sindex_max"),
            slope_n_frontier=res.get("slope_n_frontier"), slope_r2=res.get("slope_r2"),
            slope_span_years=res.get("slope_span_years"), slope_note=res.get("slope_note", ""))
        cross_rows.append(row)
    cross = pd.DataFrame(cross_rows)
    cross["n_test"] = pd.array(cross["n_test"], dtype="Int64")

    # ---- mapping table ---------------------------------------------------------------
    map_rows = []
    cr = cross.set_index("benchmark")
    raw_counts = pts_raw.groupby("benchmark").size()
    for b in benches:
        unit, conf, rel, note, related = MAPPING.get(b, (None, "none", "", "unmapped Epoch benchmark", ""))
        m = meta[b]
        h = hand.loc[unit] if (unit is not None and unit in hand.index) else None
        n_info = sizes.get(b)
        map_rows.append(dict(
            epoch_benchmark=b, epoch_data=m["kind"], epoch_file=m["file"], score_field=m["score_field"],
            normalization_rule=m["rule"], unit=unit or "", confidence=conf, relation=rel, note=note,
            related_unit_not_mapped=related,
            unit_in_population_main=bool(unit in pop) if unit else "",
            unit_hand_coded=bool(h is not None) if unit else "",
            epoch_benchmark_release=fmt_date(epoch_release.get(b)),
            unit_release=(h["release"] if h is not None else ""),
            hand_threshold=(h["threshold"] if h is not None else ""),
            n_rows_loaded=int(raw_counts.get(b, 0)), n_points_dated=int(cr.loc[b, "n_points"]),
            n_undated=int(cr.loc[b, "n_undated"]), n_excluded=int(cr.loc[b, "n_excluded"]),
            first_date=cr.loc[b, "first_date"], last_date=cr.loc[b, "last_date"],
            coverage_end=cr.loc[b, "coverage_end"], max_score=cr.loc[b, "max_score"],
            test_size_n=(n_info[0] if n_info else ""), test_size_source=(n_info[1] if n_info else "")))
    mapping = pd.DataFrame(map_rows)

    # ---- comparison -----------------------------------------------------------------
    comp_rows = []
    for b in benches:
        unit = unit_of(b)
        if unit is None or unit not in hand.index:
            continue
        h, c = hand.loc[unit], cr.loc[b]
        rep = repair.loc[unit] if unit in repair.index else None
        ab, prim = ym(h["ab_date"]), ym(h["primary_date"])
        ab_rep = ym(rep["ab_date"]) if rep is not None else pd.NaT
        e_thr = pd.Timestamp(c["cross_threshold"]) if c["cross_threshold"] else pd.NaT
        e_rep = ym(c["cross_threshold_by_report_month"])
        e_sidx = pd.Timestamp(c["sindex07_date"]) if c["sindex07_date"] else pd.NaT
        cov_end = ym(c["coverage_end"])
        cat = categorise(ab, e_thr, cov_end)
        d_ab = month_diff(e_thr, ab)
        if cat in ("agree", "epoch_later", "epoch_earlier"):
            kind, dis = "measured", abs(d_ab)
        elif cat == "hand_only":          # Epoch has data for this many months after the event but no crossing
            kind, dis = "lower_bound", month_diff(cov_end, ab)
        elif cat == "epoch_only":         # hand-coded record has no AB event through the study cut-off
            kind, dis = "lower_bound", month_diff(CUTOFF_STUDY, e_thr)
        elif cat == "hand_only_no_coverage":
            kind, dis = "untestable", 0
        else:
            kind, dis = "none", 0
        row = dict(
            unit=unit, epoch_benchmark=b, map_confidence=c["map_confidence"],
            threshold=h["threshold"], scale_max=h["scale_max"], release=h["release"],
            hand_ab_date=txt(h["ab_date"]), hand_primary_date=txt(h["primary_date"]),
            hand_primary_criterion=txt(h["primary_criterion"]), hand_status=h["status_cutoff"],
            repair_ab_date=txt(rep["ab_date"]) if rep is not None else "",
            repair_primary_date=txt(rep["primary_date"]) if rep is not None else "",
            epoch_cross_threshold=c["cross_threshold"], epoch_cross_by_report_month=c["cross_threshold_by_report_month"],
            epoch_cross90=c["cross90"], epoch_sindex07_date=c["sindex07_date"],
            epoch_cross_model=c["cross_threshold_model"], epoch_cross_score=c["cross_threshold_score"],
            epoch_cross_date_source=c["cross_threshold_date_source"], epoch_cross_source=c["cross_threshold_source"],
            epoch_first_date=c["first_date"], epoch_last_date=c["last_date"], epoch_coverage_end=c["coverage_end"],
            epoch_n_points=c["n_points"], epoch_max_score=c["max_score"], epoch_max_score_model=c["max_score_model"],
            epoch_max_score_date=c["max_score_date"], epoch_max_undated=c["max_score_undated"],
            diff_ab_months=d_ab, diff_ab_months_repair=month_diff(e_thr, ab_rep),
            diff_primary_months=month_diff(e_thr, prim), diff_ab_months_by_report=month_diff(e_rep, ab),
            diff_sindex_ab_months=month_diff(e_sidx, ab),
            category=cat, disagreement_kind=kind, disagreement_months=dis,
            hand_ab_system=txt(h["ab_system"]), hand_ab_score=txt(h["ab_score"]),
            hand_ab_source_type=txt(h["ab_source_type"]), hand_ab_mode=txt(h["ab_mode"])[:160])
        row["likely_reason_auto"] = auto_reason(row)
        row["likely_reason"] = ANALYST_NOTES.get(unit, "")
        comp_rows.append(row)
    comp = pd.DataFrame(comp_rows)
    # rank: disagreement categories only, larger months first (measured differences before lower bounds on ties)
    is_dis = comp["category"].isin(DISAGREE) & (comp["disagreement_months"] > AGREE_TOL_MONTHS)
    order = comp[is_dis].assign(_k=lambda d: (d["disagreement_kind"] != "measured").astype(int)) \
        .sort_values(["disagreement_months", "_k", "unit"], ascending=[False, True, True]).index
    comp["rank_disagreement"] = pd.array([pd.NA] * len(comp), dtype="Int64")
    for i, ix in enumerate(order, 1):
        comp.loc[ix, "rank_disagreement"] = i
    comp["_cat"] = comp["category"].map(CATEGORY_ORDER)
    comp = comp.sort_values(["_cat", "rank_disagreement", "unit"], na_position="last").drop(columns="_cat") \
        .reset_index(drop=True)
    for col in ("diff_ab_months", "diff_ab_months_repair", "diff_primary_months", "diff_ab_months_by_report",
                "diff_sindex_ab_months", "disagreement_months"):
        comp[col] = pd.array(comp[col], dtype="Int64")
    first = ["rank_disagreement", "unit", "epoch_benchmark", "category", "disagreement_kind", "disagreement_months",
             "hand_ab_date", "epoch_cross_threshold", "diff_ab_months", "likely_reason"]
    comp = comp[first + [c for c in comp.columns if c not in first]]

    # ---- points ---------------------------------------------------------------------
    out_pts = pts.copy()
    out_pts["_b"] = out_pts["benchmark"].str.lower()
    out_pts = out_pts.sort_values(["_b", "date", "score_0_100", "model"], ascending=[True, True, False, True],
                                  na_position="last")
    out_pts["date"] = out_pts["date"].dt.date
    out_pts["eval_date"] = pd.to_datetime(out_pts["eval_date"]).dt.date
    out_pts["report_month"] = out_pts["report_month"].map(fmt_ym)
    cols = ["benchmark", "unit", "model", "date", "score_0_100", "source", "run_type", "setting", "date_source",
            "model_version_raw", "display_name", "entity", "eval_date", "report_month", "source_link",
            "use_in_trajectory", "flag"]
    out_pts = out_pts[cols].rename(columns={"entity": "base_model"})

    # ---- write ----------------------------------------------------------------------
    out_pts.to_csv(OUT / "d4_points.csv", index=False, encoding="utf-8")
    cross.to_csv(OUT / "d4_crossings.csv", index=False, encoding="utf-8", float_format="%.4f")
    mapping.to_csv(OUT / "d4_mapping.csv", index=False, encoding="utf-8")
    comp.to_csv(OUT / "d4_compare.csv", index=False, encoding="utf-8", float_format="%.2f")
    summary = dict(n_rows_loaded=len(pts_raw), n_rows_after_dedupe=len(pts), n_duplicates_removed=n_dups,
                   n_benchmarks=len(benches), n_mapped=int(mapping["unit"].ne("").sum()),
                   n_dated=int(pts["date"].notna().sum()), n_undated=int(pts["date"].isna().sum()),
                   n_excluded_from_trajectory=int((~pts["use_in_trajectory"]).sum()),
                   n_compare=len(comp), categories=comp["category"].value_counts().to_dict(),
                   epoch_horizon=str(pd.Timestamp(horizon).date()))
    print(json.dumps(summary))
    return pts, cross, mapping, comp


if __name__ == "__main__":
    main()
