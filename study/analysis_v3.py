# -*- coding: utf-8 -*-
"""
Анализ выживаемости бенчмарков (Д1, кодбук v3). Вход: halflife_dataset_v3.csv. Выход: analysis_v3.json + печать.
Время — месяцы от рождения; событие — основное (A/B/C/D/E); живые цензурируются на 2026-09,
заброшенные — на дату последней активности (в конкурирующих рисках — отдельный исход).
"""
import json, warnings, math
import numpy as np, pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter, WeibullAFTFitter, LogNormalAFTFitter, AalenJohansenFitter
from lifelines.statistics import multivariate_logrank_test, logrank_test
from lifelines.utils import restricted_mean_survival_time, median_survival_times
warnings.filterwarnings("ignore")
rng = np.random.default_rng(42)

D = pd.read_csv("halflife_dataset_v3.csv")
M = D[D.stratum == "main"].copy()
M["T"] = M.T_months.astype(float)
M["E"] = M.event.astype(int)
M["cls"] = M["class"].replace({"economic/time": "static test"})
M["era"] = np.where(M.birth_year <= 2022, "≤2022", "2023+")
M["nonpublic"] = (M.test_access != "public").astype(int)
COH_ORDER = ["≤2019", "2020–2022", "2023", "2024", "2025", "2026"]
OUT = {"n_main": int(len(M)), "n_events": int(M.E.sum()),
       "status": M.status_cutoff.value_counts().to_dict(), "pending": int(D.stratum.str.startswith("pending").sum())}

def km_summary(T, E, label="", times=(6, 12, 24, 36), entry=None):
    k = KaplanMeierFitter().fit(T, E, entry=entry, label=label)
    med = k.median_survival_time_
    ci = median_survival_times(k.confidence_interval_)
    lo, hi = ci.iloc[0, 0], ci.iloc[0, 1]
    s = {f"S{t}": round(float(k.survival_function_at_times(t).iloc[0]), 3) for t in times}
    # 95% ДИ для S(t) по Гринвуду (лог-лог, как в lifelines)
    for t in times:
        ci_t = k.confidence_interval_.loc[:t].iloc[-1] if (k.confidence_interval_.index <= t).any() else None
        if ci_t is not None:
            s[f"S{t}_ci"] = [round(float(ci_t.iloc[0]), 3), round(float(ci_t.iloc[1]), 3)]
    q1 = k.percentile(0.75)   # время, к которому умерла четверть
    r24 = restricted_mean_survival_time(k, t=24)
    r36 = restricted_mean_survival_time(k, t=36)
    fin = lambda x: None if (x is None or not np.isfinite(x)) else round(float(x), 1)
    return {"n": int(len(T)), "events": int(np.sum(E)), "median": fin(med), "median_ci": [fin(lo), fin(hi)],
            "q25_dead_by": fin(q1), "RMST24": round(float(r24), 1), "RMST36": round(float(r36), 1), **s}, k

# ---------------------------------------------------------------- 1. вся популяция
OUT["km_all"], KM_ALL = km_summary(M["T"], M.E, "all")

# ---------------------------------------------------------------- 2. когорты
OUT["km_cohort"] = {}
for c in COH_ORDER:
    g = M[M.cohort == c]
    if len(g):
        OUT["km_cohort"][c] = km_summary(g["T"], g.E, c)[0]
        OUT["km_cohort"][c]["naive_median_dead"] = float(g.loc[g.E == 1, "T"].median()) if g.E.sum() else None
        OUT["km_cohort"][c]["alive"] = int((g.status_cutoff == "alive").sum())
        OUT["km_cohort"][c]["abandoned"] = int((g.status_cutoff == "abandoned").sum())
lr = multivariate_logrank_test(M["T"], M.cohort, M.E)
OUT["logrank_cohort"] = {"chi2": round(lr.test_statistic, 2), "df": int(lr.degrees_of_freedom), "p": float(lr.p_value)}
for e in ["≤2022", "2023+"]:
    g = M[M.era == e]
    OUT.setdefault("km_era", {})[e] = km_summary(g["T"], g.E, e)[0]
lr2 = logrank_test(M.loc[M.era == "≤2022", "T"], M.loc[M.era == "2023+", "T"], M.loc[M.era == "≤2022", "E"], M.loc[M.era == "2023+", "E"])
OUT["logrank_era"] = {"chi2": round(lr2.test_statistic, 2), "p": float(lr2.p_value)}
# по классу
OUT["km_class"] = {c: km_summary(g["T"], g.E, c)[0] for c, g in M.groupby("cls")}
lr3 = multivariate_logrank_test(M["T"], M.cls, M.E)
OUT["logrank_class"] = {"chi2": round(lr3.test_statistic, 2), "df": int(lr3.degrees_of_freedom), "p": float(lr3.p_value)}
# класс внутри эпохи 2023+
g = M[M.era == "2023+"]
OUT["km_class_2023plus"] = {c: km_summary(h["T"], h.E, c)[0] for c, h in g.groupby("cls")}
lr4 = multivariate_logrank_test(g["T"], g.cls, g.E)
OUT["logrank_class_2023plus"] = {"chi2": round(lr4.test_statistic, 2), "p": float(lr4.p_value)}

# ---------------------------------------------------------------- 3. Кокс и AFT
X = pd.DataFrame({"T": M["T"], "E": M.E,
                  "birth_year_c": M.birth_year - 2023,
                  "agentic": (M.cls == "agentic environment").astype(int),
                  "llm_judge": (M.cls == "LLM-judge").astype(int),
                  "nonpublic": M.nonpublic, "reissue": M.reissue.fillna(0).astype(int)})
cph = CoxPHFitter(penalizer=0.0).fit(X, "T", "E")
s = cph.summary
OUT["cox"] = {v: {"HR": round(float(s.loc[v, "exp(coef)"]), 2), "ci": [round(float(s.loc[v, "exp(coef) lower 95%"]), 2),
              round(float(s.loc[v, "exp(coef) upper 95%"]), 2)], "p": float(s.loc[v, "p"])} for v in s.index}
OUT["cox_concordance"] = round(float(cph.concordance_index_), 3)
try:
    from lifelines.statistics import proportional_hazard_test
    ph = proportional_hazard_test(cph, X, time_transform="rank")
    OUT["cox_ph_test"] = {v: round(float(p), 3) for v, p in zip(ph.summary.index, ph.summary.p)}
except Exception as ex:
    OUT["cox_ph_test"] = str(ex)
# только год рождения
c1 = CoxPHFitter().fit(X[["T", "E", "birth_year_c"]], "T", "E")
OUT["cox_birthyear_only"] = {"HR_per_year": round(float(c1.summary.loc["birth_year_c", "exp(coef)"]), 2),
                             "ci": [round(float(c1.summary.loc["birth_year_c", "exp(coef) lower 95%"]), 2),
                                    round(float(c1.summary.loc["birth_year_c", "exp(coef) upper 95%"]), 2)],
                             "p": float(c1.summary.loc["birth_year_c", "p"])}
# AFT (T+0.5, чтобы рождённые мёртвыми не обнуляли логарифм)
XA = X.copy(); XA["T"] = XA["T"] + 0.5
for name, F in [("weibull", WeibullAFTFitter), ("lognormal", LogNormalAFTFitter)]:
    f = F().fit(XA, "T", "E")
    sm = f.summary
    key = [i for i in sm.index if i[1] == "birth_year_c"][0]
    OUT[f"aft_{name}"] = {"time_ratio_per_year": round(float(np.exp(sm.loc[key, "coef"])), 2),
                          "ci": [round(float(np.exp(sm.loc[key, "coef lower 95%"])), 2), round(float(np.exp(sm.loc[key, "coef upper 95%"])), 2)],
                          "p": float(sm.loc[key, "p"]), "AIC": round(float(f.AIC_), 1)}
    # предсказанная медиана для «типичного» публичного статического теста по году рождения
    pr = {}
    for y in [2018, 2020, 2022, 2023, 2024, 2025, 2026]:
        row = pd.DataFrame({"birth_year_c": [y - 2023], "agentic": [0], "llm_judge": [0], "nonpublic": [0], "reissue": [0]})
        pr[y] = round(float(f.predict_median(row).iloc[0]) - 0.5, 1)
    OUT[f"aft_{name}"]["pred_median_static_public"] = pr

# ---------------------------------------------------------------- 4. конкурирующие риски
def aalen_johansen(T, code):
    """CIF по причинам без джиттера: CIF_k(t) = sum_{t_i<=t} S(t_i-) * d_k(t_i) / n(t_i)."""
    T = np.asarray(T, float); code = np.asarray(code, int)
    times = np.unique(T[code > 0])
    S = 1.0; cif = {k: [] for k in (1, 2, 3)}; grid = []
    for t in times:
        n = np.sum(T >= t)
        d = {k: np.sum((T == t) & (code == k)) for k in (1, 2, 3)}
        for k in (1, 2, 3):
            prev = cif[k][-1] if cif[k] else 0.0
            cif[k].append(prev + S * d[k] / n)
        S *= 1 - sum(d.values()) / n
        grid.append(t)
    grid = np.array(grid)
    def at(k, t):
        idx = np.where(grid <= t)[0]
        return float(cif[k][idx[-1]]) if len(idx) else 0.0
    return at, grid, cif
OUT["competing"] = {}
at, CR_GRID, CR_CIF = aalen_johansen(M["T"], M.outcome_cr.astype(int))
for code, name in [(1, "models"), (2, "instrument"), (3, "abandoned")]:
    OUT["competing"][name] = {f"CIF{t}": round(at(code, t), 3) for t in (6, 12, 24, 36, 60)}
for e in ["≤2022", "2023+"]:
    g = M[M.era == e]
    at_e, _, _ = aalen_johansen(g["T"], g.outcome_cr.astype(int))
    OUT["competing"][f"era_{e}"] = {name: {f"CIF{t}": round(at_e(code, t), 3) for t in (12, 24)} for code, name in [(1, "models"), (2, "instrument"), (3, "abandoned")]}
OUT["cause_counts"] = M[M.E == 1].cause.value_counts().to_dict()
OUT["cause_by_era"] = pd.crosstab(M[M.E == 1].era, M[M.E == 1].cause).to_dict()
OUT["cause_by_class"] = pd.crosstab(M[M.E == 1].cls, M[M.E == 1].cause).to_dict()
OUT["primary_criterion_counts"] = M[M.E == 1].primary_criterion.value_counts().to_dict()
OUT["ab_system_type_counts"] = M[(M.E == 1) & M.primary_criterion.str.contains("A|B", regex=True, na=False)].ab_system_type.value_counts().to_dict()
OUT["ab_source_type_counts"] = M[(M.E == 1) & M.primary_criterion.str.contains("A|B", regex=True, na=False)].ab_source_type.value_counts().to_dict()

# ---------------------------------------------------------------- 5. нулевая модель: одна и та же продолжительность жизни для всех когорт
# Берём распределение сроков из всей выборки (KM, «без учёта года»), разыгрываем сроки для каждой реальной даты рождения,
# цензурируем на срезе и считаем наивную медиану «среди умерших» по когортам.
sf = KM_ALL.survival_function_
tgrid = sf.index.values; Sv = sf.iloc[:, 0].values
def draw_life(n):
    u = rng.random(n)
    out = np.empty(n)
    for i, x in enumerate(u):
        idx = np.where(Sv <= x)[0]
        out[i] = tgrid[idx[0]] if len(idx) else np.inf   # «бессмертные» хвосты = не умерли в наблюдаемом диапазоне
    return out
cut = 2026 * 12 + 8
birth_m = M.release.map(lambda s: int(s[:4]) * 12 + int(s[5:7]) - 1).values
follow = cut - birth_m
sims = {c: [] for c in COH_ORDER}
for b in range(2000):
    life = draw_life(len(M))
    dead = life <= follow
    for c in COH_ORDER:
        mask = (M.cohort.values == c) & dead
        if mask.sum():
            sims[c].append(np.median(life[mask]))
OUT["null_model_naive_median_dead"] = {c: {"mean": round(float(np.mean(v)), 1), "p05": round(float(np.percentile(v, 5)), 1),
                                           "p95": round(float(np.percentile(v, 95)), 1)} for c, v in sims.items() if v}
OUT["observed_naive_median_dead"] = {c: OUT["km_cohort"][c]["naive_median_dead"] for c in COH_ORDER}
OUT["followup_months_by_cohort"] = {c: [int(follow[M.cohort.values == c].min()), int(follow[M.cohort.values == c].max())] for c in COH_ORDER}

# ---------------------------------------------------------------- 6. проверки устойчивости
def scen(df, Tcol, Ecol, entry=None):
    r, _ = km_summary(df[Tcol].astype(float), df[Ecol].astype(int), entry=entry)
    xx = pd.DataFrame({"T": df[Tcol].astype(float), "E": df[Ecol].astype(int), "b": df.birth_year - 2023})
    c = CoxPHFitter().fit(xx, "T", "E")
    era = {}
    for e in ["≤2022", "2023+"]:
        g = df[df.era == e]
        era[e] = km_summary(g[Tcol].astype(float), g[Ecol].astype(int))[0]
    return {"n": r["n"], "events": r["events"], "median": r["median"], "S12": r["S12"], "S24": r["S24"],
            "q25": r["q25_dead_by"], "median_le2022": era["≤2022"]["median"], "median_2023plus": era["2023+"]["median"],
            "S12_le2022": era["≤2022"]["S12"], "S12_2023plus": era["2023+"]["S12"],
            "HR_birth_year": round(float(c.summary.loc["b", "exp(coef)"]), 2), "p": float(c.summary.loc["b", "p"])}
SENS = {}
SENS["main"] = scen(M, "T", "E")
SENS["strict"] = scen(M, "T_strict", "event_strict")
SENS["no_reissues"] = scen(M[M.reissue.fillna(0) == 0], "T", "E")
SENS["replacement_is_censoring"] = scen(M, "T_replacement_censored", "event_replacement_censored")
# без dev-сплита: события, держащиеся только на dev, — цензура (на дату заброшенности по исходной разметке A, иначе на срез)
ND = M.copy()
orig_aband = {"CommonsenseQA": "2024-07", "QuALITY": "2025-01"}
def ym(s): return int(s[:4]) * 12 + int(s[5:7]) - 1
for i, r in ND.iterrows():
    if r.ab_split_dev == 1 and str(r.primary_criterion) in ("A", "B"):
        other = [x for x in [r.c_date, r.d_date, r.e_date] if isinstance(x, str) and x]
        if other:
            ND.at[i, "T"] = ym(min(other)) - ym(r.release); ND.at[i, "E"] = 1
        else:
            end = orig_aband.get(r.unit, "2026-09")
            ND.at[i, "T"] = ym(end) - ym(r.release); ND.at[i, "E"] = 0
SENS["no_dev_split"] = scen(ND, "T", "E")
SENS["no_dev_split"]["units_affected"] = M[(M.ab_split_dev == 1) & M.primary_criterion.isin(["A", "B"])].unit.tolist()
# отложенный вход: наблюдение с первого отчёта лаборатории; единицы, чьё событие раньше первого отчёта, выпадают
LT = M[M.entry_months.notna()].copy()
LT["entry"] = LT.entry_months.astype(float).clip(lower=0)
LT = LT[LT.entry <= LT["T"]]
SENS["delayed_entry"] = scen(LT, "T", "E", entry=LT.entry)
SENS["delayed_entry"]["dropped_event_before_first_report"] = int(len(M[M.entry_months.notna()]) - len(LT))
# точечные альтернативы
ALT = M.copy()
ALT.loc[ALT.unit == "ARC-AGI-2", "T"] = ym("2026-07") - ym("2025-03")
ALT.loc[ALT.unit == "GDPval", ["T", "E"]] = [ym("2025-12") - ym("2025-09"), 1]
ALT.loc[ALT.unit == "ARC-AGI-3", "T"] = ym("2026-09") - ym("2025-07")
SENS["alt_arc_gdpval"] = scen(ALT, "T", "E")
OUT["sensitivity"] = SENS

# ---------------------------------------------------------------- 7. запас на релизе
def to_num(x):
    try:
        return float(str(x).split()[0].replace(",", "."))
    except Exception:
        return np.nan
H = M.copy()
H["thr"] = H.threshold.map(to_num); H["scale"] = H.scale_max.map(to_num).fillna(100)
H["sar"] = H.score_at_release.map(to_num)
H = H[H.thr.notna() & H.sar.notna() & (H.scale > 0)]
def lg(p): p = min(max(p, 0.01), 0.99); return math.log(p / (1 - p))
H["headroom_logit"] = [lg(t / s) - lg(a / s) for t, s, a in zip(H.thr, H.scale, H.sar)]
H["gap_pp"] = (H.thr - H.sar) / H.scale * 100
H = H[np.isfinite(H.headroom_logit)]
OUT["headroom_n"] = int(len(H))
from scipy.stats import spearmanr
dead = H[H.E == 1]
rho, p = spearmanr(dead.headroom_logit, dead["T"])
OUT["headroom_vs_T_dead_spearman"] = {"n": int(len(dead)), "rho": round(float(rho), 2), "p": float(p)}
hx = pd.DataFrame({"T": H["T"], "E": H.E, "headroom": H.headroom_logit, "b": H.birth_year - 2023})
ch = CoxPHFitter().fit(hx, "T", "E")
OUT["cox_headroom"] = {v: {"HR": round(float(ch.summary.loc[v, "exp(coef)"]), 2), "p": float(ch.summary.loc[v, "p"])} for v in ["headroom", "b"]}
# запас по годам рождения
OUT["headroom_by_era"] = H.groupby("era").headroom_logit.median().round(2).to_dict()
OUT["gap_pp_by_era"] = H.groupby("era").gap_pp.median().round(1).to_dict()
OUT["headroom_by_cohort"] = H.groupby("cohort").headroom_logit.median().round(2).to_dict()
H[["unit", "cohort", "birth_year", "threshold", "score_at_release", "score_at_release_system", "headroom_logit", "gap_pp", "T", "E", "cause"]].to_csv("headroom_v3.csv", index=False)

json.dump(OUT, open("analysis_v3.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(json.dumps({k: OUT[k] for k in ["n_main", "n_events", "km_all", "logrank_cohort", "logrank_era", "cox_birthyear_only"]}, ensure_ascii=False, indent=1))
for c in COH_ORDER:
    print(c, OUT["km_cohort"][c])
print("era", json.dumps(OUT["km_era"], ensure_ascii=False))
print("class", json.dumps(OUT["km_class"], ensure_ascii=False))
print("logrank class", OUT["logrank_class"], "2023+", OUT["logrank_class_2023plus"])
print("cox", json.dumps(OUT["cox"], ensure_ascii=False), OUT["cox_concordance"], OUT["cox_ph_test"])
print("aft", json.dumps({k: OUT[k] for k in ["aft_weibull", "aft_lognormal"]}, ensure_ascii=False))
print("competing", json.dumps(OUT["competing"], ensure_ascii=False))
print("causes", OUT["cause_counts"], OUT["cause_by_era"])
print("null", json.dumps(OUT["null_model_naive_median_dead"], ensure_ascii=False), OUT["observed_naive_median_dead"])
for k, v in SENS.items():
    print("SENS", k, v)
print("headroom", OUT["headroom_n"], OUT["headroom_vs_T_dead_spearman"], OUT["cox_headroom"], OUT["headroom_by_cohort"], OUT["gap_pp_by_era"])

# ---------------------------------------------------------------- 8. скорость прогресса (Д4) и разложение «запас / скорость»
C4 = pd.read_csv("d4/d4_crossings.csv")
C4 = C4[C4.unit.isin(M.unit)].copy()
C4 = C4.merge(M[["unit", "birth_year", "era", "release", "T", "E"]], on="unit")
C4 = C4.merge(H[["unit", "headroom_logit"]], on="unit", how="left")
S4 = C4[C4.logit_slope_per_year.notna() & (C4.slope_n_frontier >= 3) & (C4.unit != "LAMBADA")].copy()
OUT["speed_n"] = int(len(S4))
OUT["speed_by_era"] = {e: {"n": int(len(g)), "median_logit_per_year": round(float(g.logit_slope_per_year.median()), 2),
                           "units": g.unit.tolist()} for e, g in S4.groupby("era")}
from scipy.stats import mannwhitneyu
a = S4[S4.era == "≤2022"].logit_slope_per_year; b = S4[S4.era == "2023+"].logit_slope_per_year
OUT["speed_mannwhitney_p"] = float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
S4b = S4[S4.headroom_logit.notna()].copy()
S4b["T_pred"] = S4b.headroom_logit / S4b.logit_slope_per_year * 12
dd = S4b[S4b.E == 1]
rho, p = spearmanr(dd.T_pred, dd["T"])
OUT["decomposition"] = {"n": int(len(S4b)), "n_dead": int(len(dd)), "spearman_pred_vs_actual": round(float(rho), 2), "p": float(p),
                        "median_ratio_actual_to_pred": round(float((dd["T"] / dd.T_pred).median()), 2),
                        "table": S4b[["unit", "era", "headroom_logit", "logit_slope_per_year", "T_pred", "T", "E"]].round(2).to_dict("records")}
# медианный бенчмарк эпохи: запас / скорость
for e in ["≤2022", "2023+"]:
    hr = float(H[H.era == e].headroom_logit.median()); sp = OUT["speed_by_era"][e]["median_logit_per_year"]
    OUT["decomposition"][f"typical_{e}"] = {"headroom": round(hr, 2), "speed": sp, "T_pred_months": round(hr / sp * 12, 1),
                                           "km_median": OUT["km_era"][e]["median"]}
# скорость по всем бенчмаркам Epoch с наклоном (в т. ч. вне популяции) — для прогноза
ALL4 = pd.read_csv("d4/d4_crossings.csv")
ALL4 = ALL4[ALL4.logit_slope_per_year.notna() & (ALL4.slope_n_frontier >= 3)]
recent = ALL4[ALL4.slope_range.str[-10:] >= "2025-01-01"]
OUT["speed_epoch_recent"] = {"n": int(len(recent)), "median": round(float(recent.logit_slope_per_year.median()), 2),
                             "q25": round(float(recent.logit_slope_per_year.quantile(.25)), 2),
                             "q75": round(float(recent.logit_slope_per_year.quantile(.75)), 2)}

# ---------------------------------------------------------------- 9. пороги 80 / 90 / 95 на подвыборке с траекториями Epoch
TH = {}
sub = C4.copy()
for th in ["cross80", "cross90", "cross95"]:
    T_ = []; E_ = []
    for _, r in sub.iterrows():
        end = r[th] if isinstance(r[th], str) else None
        cov = str(r.coverage_end)[:7] if isinstance(r.coverage_end, str) else "2025-11"
        if end:
            T_.append(max(0, ym(end[:7]) - ym(r.release))); E_.append(1)
        else:
            T_.append(max(0, ym(cov) - ym(r.release))); E_.append(0)
    k = KaplanMeierFitter().fit(T_, E_)
    TH[th] = {"n": len(T_), "events": int(sum(E_)), "median": None if not np.isfinite(k.median_survival_time_) else float(k.median_survival_time_),
              "S24": round(float(k.survival_function_at_times(24).iloc[0]), 3)}
OUT["threshold_sensitivity_epoch_subset"] = TH

# ---------------------------------------------------------------- 10. прогноз для нового бенчмарка (OenoBench)
# AFT Вейбулла с запасом и годом рождения, обучаем на всех основных единицах с известным запасом
XH = pd.DataFrame({"T": H["T"] + 0.5, "E": H.E, "headroom": H.headroom_logit.clip(upper=6.9), "b": H.birth_year - 2023,
                   "agentic": (H.cls == "agentic environment").astype(int)})
wf = WeibullAFTFitter().fit(XH, "T", "E")
OUT["aft_headroom"] = {"summary": {f"{i[0]}:{i[1]}": round(float(wf.summary.loc[i, "coef"]), 3) for i in wf.summary.index},
                       "time_ratio_per_logit_headroom": round(float(np.exp(wf.summary.loc[("lambda_", "headroom"), "coef"])), 2),
                       "time_ratio_per_birth_year": round(float(np.exp(wf.summary.loc[("lambda_", "b"), "coef"])), 2)}
def lgt(p): return math.log(p / (1 - p))
FC = {}
for label, best in [("best_83", 0.83), ("best_81", 0.81), ("closed_book_70", 0.70), ("closed_book_65", 0.65)]:
    hr = lgt(0.90) - lgt(best)
    row = pd.DataFrame({"headroom": [hr], "b": [2026 - 2023], "agentic": [0]})
    sfun = wf.predict_survival_function(row, times=np.arange(0.5, 60.5, 1.0))
    med = float(wf.predict_median(row).iloc[0]) - 0.5
    q = {f"S{t}": round(float(sfun.loc[t + 0.5].iloc[0]), 3) for t in (6, 12, 24)}
    # простая арифметика: запас / скорость (медиана и квартили скоростей Epoch за 2025)
    sp = OUT["speed_epoch_recent"]
    FC[label] = {"best": best, "headroom_logit": round(hr, 2), "aft_median_months": round(med, 1), **q,
                 "arith_months_median_speed": round(hr / sp["median"] * 12, 1),
                 "arith_months_slow_q25": round(hr / sp["q25"] * 12, 1), "arith_months_fast_q75": round(hr / sp["q75"] * 12, 1)}
OUT["forecast_new_benchmark_2026"] = FC
# какой лучший результат на релизе нужен статическому тесту (и агентной среде) 2026 года для заданной медианы жизни
def best_for_median(target, agentic=0):
    lo, hi = 0.005, 0.895
    for _ in range(60):
        mid = (lo + hi) / 2
        row = pd.DataFrame({"headroom": [lgt(0.90) - lgt(mid)], "b": [3], "agentic": [agentic]})
        if float(wf.predict_median(row).iloc[0]) - 0.5 > target: lo = mid
        else: hi = mid
    return round(mid, 3)
OUT["best_at_release_for_median"] = {"static_12": best_for_median(12), "static_24": best_for_median(24),
                                     "static_36": best_for_median(36), "agentic_24": best_for_median(24, 1)}
OUT["aft_headroom"]["n_train"] = int(len(XH))
# соседи OenoBench по запасу: 0,45–1 логит до порога
NB = H[(H.headroom_logit >= 0.45) & (H.headroom_logit <= 1.0)]
k_nb = KaplanMeierFitter().fit(NB["T"], NB.E)
nb_recent = NB[NB.birth_year >= 2025]
OUT["headroom_neighbors"] = {"n": int(len(NB)), "dead": int(NB.E.sum()), "km_median": float(k_nb.median_survival_time_),
                             "recent_n": int(len(nb_recent)), "recent_dead_within_6m": int(((nb_recent.E == 1) & (nb_recent["T"] <= 6)).sum()),
                             "units": NB.sort_values("headroom_logit")[["unit", "birth_year", "T", "E"]].to_dict("records")}

# ---------------------------------------------------------------- 11. ступенька или наклон; тренд внутри эпохи 2023+
EX = M.copy()
EX["bmc"] = EX.release.str[:4].astype(int) + (EX.release.str[5:7].astype(int) - 1) / 12 - 2023   # месяц рождения, в годах
EX["step"] = (EX.birth_year >= 2023).astype(int)
EX["agentic"] = (EX.cls == "agentic environment").astype(int)
EX["reissue"] = EX.reissue.fillna(0).astype(int)
def wfit(cols):
    X_ = EX[["T", "E"] + cols].copy(); X_["T"] = X_["T"] + 0.5
    return WeibullAFTFitter().fit(X_, "T", "E")
f_slope, f_step, f_both = wfit(["bmc"]), wfit(["step"]), wfit(["bmc", "step"])
EX["after"] = EX.bmc.clip(lower=0)
f_pw = wfit(["bmc", "after"])
b1 = f_pw.summary.loc[("lambda_", "bmc"), "coef"]; b2 = f_pw.summary.loc[("lambda_", "after"), "coef"]
OUT["step_vs_slope"] = {"AIC_slope": round(float(f_slope.AIC_), 1), "AIC_step": round(float(f_step.AIC_), 1),
                        "both_step_p": round(float(f_both.summary.loc[("lambda_", "step"), "p"]), 3),
                        "piecewise_time_ratio_before_2023": round(float(np.exp(b1)), 2),
                        "piecewise_time_ratio_after_2023": round(float(np.exp(b1 + b2)), 2)}
G23 = EX[EX.birth_year >= 2023]
c_a = CoxPHFitter().fit(G23[["T", "E", "bmc"]], "T", "E").summary.loc["bmc"]
c_b = CoxPHFitter().fit(G23[["T", "E", "bmc", "agentic", "reissue"]], "T", "E").summary.loc["bmc"]
OUT["cox_within_2023plus"] = {"HR": round(float(c_a["exp(coef)"]), 2), "p": round(float(c_a["p"]), 3),
                              "HR_adj_agentic_exams": round(float(c_b["exp(coef)"]), 2), "p_adj": round(float(c_b["p"]), 3)}
def died6(g):
    if not len(g): return None
    kk = KaplanMeierFitter().fit(g["T"], g.E)
    return {"n": int(len(g)), "dead6": round(1 - float(kk.survival_function_at_times(6).iloc[0]), 3), "at_risk_6": int((g["T"] >= 6).sum())}
OUT["died_within_6m_by_cohort"] = {c: {"static_no_exams": died6(EX[(EX.cohort == c) & (EX.cls == "static test") & (EX.reissue == 0)]),
                                       "agentic": died6(EX[(EX.cohort == c) & (EX.agentic == 1)]),
                                       "all": died6(EX[EX.cohort == c])} for c in ["2023", "2024", "2025", "2026"]}
json.dump(OUT, open("analysis_v3.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print("speed", json.dumps(OUT["speed_by_era"], ensure_ascii=False), OUT["speed_mannwhitney_p"])
print("decomp", {k: v for k, v in OUT["decomposition"].items() if k != "table"})
print("epoch recent speed", OUT["speed_epoch_recent"])
print("thresholds", OUT["threshold_sensitivity_epoch_subset"])
print("aft_headroom", OUT["aft_headroom"])
print("forecast", json.dumps(FC, ensure_ascii=False, indent=1))
print("best_for_median", OUT["best_at_release_for_median"], "neighbors", {k: v for k, v in OUT["headroom_neighbors"].items() if k != "units"})
print("step_vs_slope", OUT["step_vs_slope"], "cox 2023+", OUT["cox_within_2023plus"])
print("died6", json.dumps(OUT["died_within_6m_by_cohort"], ensure_ascii=False))
