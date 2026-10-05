# -*- coding: utf-8 -*-
"""
Д3: что происходит с бенчмарком в карточках моделей до и после смерти.
1) выпадение из карточек: последний отчёт, после которого прошло >=12 мес и вышло >=3 релизов лабораторий,
   раньше о нём отчитывавшихся (кодбук 8.1);
2) «посмертная жизнь»: сколько месяцев бенчмарк продолжают показывать после основного события;
3) доля отчётов на уже мёртвых бенчмарках по годам.
"""
import csv, glob, json, re, collections
import numpy as np, pandas as pd
from lifelines import KaplanMeierFitter

CUT = "2026-09"
def ym(s): return int(s[:4]) * 12 + int(s[5:7]) - 1

D = pd.read_csv("halflife_dataset_v3.csv")
M = D[D.stratum == "main"].set_index("unit")
rows = list(csv.DictReader(open("d3_matrix_v3.csv", encoding="utf-8")))   # единица после сведения: unit_v3
def relkey(lab, model):
    m = re.sub(r"\s*\((blog|technical report|tech report|paper|model card|system card|report)\)\s*$", "", model, flags=re.I)
    return (lab, m.strip())
for r in rows:
    r["unit"] = r["unit_v3"]
R = [r for r in rows if r["unit"] in M.index and r["is_report"] == "1" and r["release_date"]]
# релизы лабораторий
REL = []
for f in glob.glob("d3/*_releases.csv"):
    for r in csv.DictReader(open(f, encoding="utf-8-sig")):
        if r["release_date"]:
            REL.append((r["lab"], relkey(r["lab"], r["model"]), r["release_date"][:7]))
REL = sorted(set(REL), key=lambda x: x[2])

by_unit = collections.defaultdict(list)
for r in R:
    by_unit[r["unit"]].append((r["release_date"][:7], r["lab"], relkey(r["lab"], r["model"])))
out = []
for u, rep in by_unit.items():
    labs = {l for _, l, _ in rep}
    last = max(d for d, _, _ in rep); first = min(d for d, _, _ in rep)
    after = {k for (l, k, d) in REL if l in labs and d > last}
    gap = ym(CUT) - ym(last)
    dropped = int(gap >= 12 and len(after) >= 3)
    pdate = M.loc[u, "primary_date"] if isinstance(M.loc[u, "primary_date"], str) else ""
    n_after_death = sum(1 for d, _, _ in rep if pdate and d > pdate)
    rel_after_death = len({k for d, _, k in rep if pdate and d > pdate})
    out.append({"unit": u, "first_report": first, "last_report": last, "labs": len(labs), "releases_reporting": len({k for _, _, k in rep}),
                "releases_after_last_by_same_labs": len(after), "dropped_out": dropped,
                "primary_date": pdate, "status": M.loc[u, "status_cutoff"],
                "postmortem_months": (ym(last) - ym(pdate)) if pdate else None,
                "postmortem_censored": int(pdate != "" and not dropped),
                "releases_reporting_after_death": rel_after_death})
C = pd.DataFrame(out)
C.to_csv("cards_v3.csv", index=False)
res = {}
res["n_units"] = int(len(C))
res["dropped_out"] = int(C.dropped_out.sum())
res["dropped_by_status"] = C.groupby("status").dropped_out.agg(["sum", "count"]).to_dict()
dead = C[C.primary_date != ""].copy()
res["dead_n"] = int(len(dead))
res["dead_reported_after_death"] = int((dead.releases_reporting_after_death > 0).sum())
res["dead_reported_12m_after_death"] = int((dead.postmortem_months >= 12).sum())
res["dead_reported_24m_after_death"] = int((dead.postmortem_months >= 24).sum())
# KM «посмертной жизни»: от смерти до последнего отчёта (событие = выпал из карточек), отрицательные = выпал до смерти
pm = dead.copy()
pm["dur"] = pm.apply(lambda r: (ym(r.last_report) - ym(r.primary_date)) if r.dropped_out else (ym(CUT) - ym(r.primary_date)), axis=1)
pm["ev"] = pm.dropped_out
pm_pos = pm[pm.dur >= 0]
k = KaplanMeierFitter().fit(pm_pos.dur, pm_pos.ev)
res["postmortem_km_median_months"] = None if not np.isfinite(k.median_survival_time_) else float(k.median_survival_time_)
res["postmortem_km_S12"] = round(float(k.survival_function_at_times(12).iloc[0]), 3)
res["postmortem_km_S24"] = round(float(k.survival_function_at_times(24).iloc[0]), 3)
res["dropped_before_death"] = int((pm.dur < 0).sum())
res["examples_longest_postmortem"] = pm.sort_values("postmortem_months", ascending=False)[["unit", "primary_date", "last_report", "postmortem_months", "releases_reporting_after_death"]].head(12).to_dict("records")
# доля отчётов о мёртвых бенчмарках по году отчёта (единица учёта — пара «релиз × бенчмарк»)
pairs = {}
for r in R:
    key = (relkey(r["lab"], r["model"]), r["unit"])
    d = r["release_date"][:7]
    pairs[key] = min(pairs.get(key, d), d)
yr = collections.defaultdict(lambda: [0, 0])
for (rk, u), d in pairs.items():
    p = M.loc[u, "primary_date"]
    dead_flag = isinstance(p, str) and p != "" and p < d
    y = int(d[:4]); yr[y][0] += 1; yr[y][1] += int(dead_flag)
res["zombie_share_by_year"] = {y: {"pairs": v[0], "dead": v[1], "share": round(v[1] / v[0], 3)} for y, v in sorted(yr.items())}
# по лабораториям в 2026
lab26 = collections.defaultdict(lambda: [0, 0])
for (rk, u), d in pairs.items():
    if d >= "2026-01":
        p = M.loc[u, "primary_date"]
        lab26[rk[0]][0] += 1; lab26[rk[0]][1] += int(isinstance(p, str) and p != "" and p < d)
res["zombie_share_2026_by_lab"] = {l: {"pairs": v[0], "dead": v[1], "share": round(v[1] / v[0], 3)} for l, v in sorted(lab26.items())}
# время от смерти до первого «ухода» хотя бы одной лаборатории не считаем: слишком мало релизов у части лабораторий
json.dump(res, open("cards_v3.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(res, ensure_ascii=False, indent=1, default=str)[:6000])
