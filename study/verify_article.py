# -*- coding: utf-8 -*-
"""
Сверка чисел статьи с результатами скриптов. Каждая проверка строит фразу из данных
(analysis_v3.json, halflife_dataset_v3.csv) и ищет её в тексте статьи дословно, а там,
где в тексте словесная оценка («около трети», «вдвое»), проверяет условие на данных.
Запуск: python3 verify_article.py [путь к статье .md]. Код выхода 1, если хоть одна проверка не прошла.
"""
import csv, hashlib, json, sys
import pandas as pd

ART = sys.argv[1] if len(sys.argv) > 1 else "HABR_ARTICLE_v3.md"
text = open(ART, encoding="utf-8").read()
A = json.load(open("analysis_v3.json", encoding="utf-8"))
D = pd.read_csv("halflife_dataset_v3.csv")
M = D[D.stratum == "main"]
LOG = list(csv.DictReader(open("decisions_log_v3.csv", encoding="utf-8")))

def f(x, d=0):
    """Число в русской записи: десятичная запятая."""
    return f"{x:.{d}f}".replace(".", ",")
def pct(x):
    """Проценты с округлением половины вверх по сохранённому значению (0,045 → 5%)."""
    from decimal import Decimal, ROUND_HALF_UP
    return f"{(Decimal(str(x)) * 100).quantize(Decimal('1'), rounding=ROUND_HALF_UP)}%"

fails, n = [], 0
def has(s, why=""):
    global n
    n += 1
    if s not in text:
        fails.append(f"нет фразы: «{s}» {why}")
def cond(ok, why):
    global n
    n += 1
    if not ok:
        fails.append(f"не выполнено: {why}")

N, EV = A["n_main"], A["n_events"]
k_all = A["km_all"]
e1, e2 = A["km_era"]["≤2022"], A["km_era"]["2023+"]
sp = A["speed_by_era"]
hb = A["headroom_by_era"]
cbc = A["cause_by_class"]
ag_dead = sum(cbc[c]["agentic environment"] for c in cbc)
st_dead = sum(cbc[c]["static test"] for c in cbc)
co = A["competing"]

# --- заголовок и анонс
has(f"# Период полураспада бенчмарка ИИ сократился вдвое: {int(e2['median'])} месяцев вместо {int(e1['median'])}")
cond(1.8 <= e1["median"] / e2["median"] <= 2.2, "«сократился вдвое»")
has(f"я собрал {N} бенчмарка, о которых в 2018–2026 годах отчитывались девять лабораторий")
has(f"**Период полураспада бенчмарка — {int(k_all['median'])} месяца.** У вышедших до 2023 года он был {int(e1['median'])} месяцев, у вышедших позже — {int(e2['median'])}.")
cond(abs(hb["2023+"] - hb["≤2022"]) < 0.3, "запас на старте прежний")
cond(sp["2023+"]["median_logit_per_year"] / sp["≤2022"]["median_logit_per_year"] >= 1.9, "модели вдвое быстрее")
cond(0.13 <= co["era_2023+"]["instrument"]["CIF24"] <= 0.155, "каждый седьмой новый бенчмарк — дефекты (≈14%)")
cond(0.4 <= cbc["defects"]["agentic environment"] / ag_dead < 0.5, "у агентных сред почти половина смертей — дефекты")
for gone in ["карточках моделей 2026", "мёртвых бенчмарках", "Жизнь после смерти", "Где это ломается", "Ступенька или наклон",
             "артефакт", "ДИ", "Журнал спорных решений", "Правила, дописанные на ходу", "Правилу популяции соответствовали",
             "первого найденного", "по памяти"]:
    cond(gone not in text, f"в тексте не должно быть: «{gone}»")
has("которые не требуют рассуждений (closed-book solvable)")

# --- выборка
has(f"В итоге осталось {N} бенчмарка.")
young = sum(A["km_cohort"][c]["n"] for c in ["2023", "2024", "2025", "2026"])
has(f"{young} бенчмарков из {N} вышли в 2023 году или позже")
LAB = {"≤2019": "2016–2019", "2020–2022": "2020–2022"}
for c, k in A["km_cohort"].items():
    lab = LAB.get(c, c)
    has(f"| {lab} | {k['n']} | {k['events']} | {k['alive']} | {k['abandoned']} |", "(таблица исходов)")
nm_rows = sum(1 for _ in open("d3_matrix_v3.csv", encoding="utf-8")) - 1
has(f"230 релизов и {nm_rows:,} строки".replace(",", " "))

# --- наивная медиана и нулевая модель
ob = A["observed_naive_median_dead"]
cells = " | ".join(f(v, 1) if v % 1 else str(int(v)) for v in ob.values())
has(f"| Медиана среди умерших, мес | {cells} |")
nm = A["null_model_naive_median_dead"]
cond(round(nm["≤2019"]["mean"]) == 19 and round(nm["2026"]["mean"]) == 2, "нулевая модель: с 19 месяцев до двух")
for c in ["2025", "2026"]:
    cond(nm[c]["p05"] <= ob[c] <= nm[c]["p95"], f"наивная медиана {c} внутри полосы нулевой модели")
cond(ob["≤2019"] >= nm["≤2019"]["p95"] and ob["2020–2022"] > nm["2020–2022"]["p95"], "старые когорты на границе и выше полосы")
cond(10 < ob["2020–2022"] / ob["2026"] < 11, "ускорение больше чем в десять раз")

# --- период полураспада
has(f"По всей выборке период полураспада — {int(k_all['median'])} месяца (95% CI {int(k_all['median_ci'][0])}–{int(k_all['median_ci'][1])})")
cond(0.24 <= 1 - k_all["S12"] <= 0.26 and k_all["S24"] < 0.5, "четверть умирает в первый год, к двум годам живых меньше половины")
has(f"период полураспада — {int(e1['median'])} месяцев (95% CI {int(e1['median_ci'][0])}–{int(e1['median_ci'][1])}), и за первый год из них умирало {pct(1 - e1['S12'])}")
has(f"У вышедших в 2023 году и позже — {int(e2['median'])} месяцев ({int(e2['median_ci'][0])}–{int(e2['median_ci'][1])})")
cond(0.29 <= 1 - e2["S12"] <= 0.35, "в первый год умирает около трети")
cx = A["cox_birthyear_only"]
has(f"риск смерти растёт примерно на {round((cx['HR_per_year'] - 1) * 100)}%")
has(f"# year ≈ {cx['HR_per_year']:.2f}")
pr = A["aft_weibull"]["pred_median_static_public"]
has(f"вышедший в 2018 году прожил бы около {round(pr['2018'])} месяцев, в 2022-м — {round(pr['2022'])}, в 2026-м — {round(pr['2026'])}")
for c, k_ in A["km_cohort"].items():
    lab = LAB.get(c, c)
    lo, hi = k_["median_ci"]
    med = "—" if c == "2026" else (f"{int(k_['median'])} ({int(lo)} и выше)" if hi is None else f"{int(k_['median'])} ({int(lo)}–{int(hi)})")
    has(f"| {lab} | {k_['n']} | {k_['events']} | {med} |", "(таблица периодов полураспада)")
d6 = A["died_within_6m_by_cohort"]
has(f"полгода наблюдений набралось только у {d6['2026']['all']['at_risk_6']} из {d6['2026']['all']['n']}")
st = A["sensitivity"]["strict"]
has(f"периоды полураспада вырастают до {int(st['median_le2022'])} и {int(st['median_2023plus'])} месяцев")
cond(st["median_le2022"] / st["median_2023plus"] > e1["median"] / e2["median"], "в строгом режиме разница эпох больше")

# --- запас и скорость
has(f"Для {A['headroom_n']} бенчмарков известен лучший результат на релизе")
gp = A["gap_pp_by_era"]
has(f"у старых бенчмарков — {f(hb['≤2022'], 2)} логита, у новых — {f(hb['2023+'], 2)}, или {round(gp['≤2022'])} и {round(gp['2023+'])} процентных пунктов")
has(f"для {sp['≤2022']['n'] + sp['2023+']['n']} бенчмарка из моей выборки")
has(f"фронтир рос на {f(sp['≤2022']['median_logit_per_year'], 2)} логита в год, на новых — на {f(sp['2023+']['median_logit_per_year'], 2)}")
t1, t2 = A["decomposition"]["typical_≤2022"], A["decomposition"]["typical_2023+"]
has(f"около {round(t1['T_pred_months'])} месяцев, новый — около {round(t2['T_pred_months'])}. Фактически они прожили {int(t1['km_median'])} и {int(t2['km_median'])}")
cond(1.4 <= A["aft_headroom"]["time_ratio_per_logit_headroom"] <= 1.65, "логит запаса ≈ в полтора раза")
cond(A["aft_headroom"]["time_ratio_per_birth_year"] < 1 and A["cox_headroom"]["b"]["HR"] > 1, "при равном запасе новый живёт меньше")

# --- причины
cc = A["cause_counts"]
has(f"Из {EV} смертей {cc['threshold']} — это порог")
has(f"Ещё {cc['declared_saturation']} раз создатели сами объявили")
has(f"Оставшиеся {cc['defects']} смертей — дефекты")
dead = M[M.event == 1]
decl = set(dead[dead.cause == "declared_saturation"].unit)
for u in ["GLUE", "SQuAD 1.1", "HealthBench", "Video-MME", "OSWorld-Verified", "Finance Agent v1"]:
    cond(u in decl, f"{u} в списке признанного насыщения")
has(f"от моделей умирали {pct(co['era_≤2022']['models']['CIF24'])}, от дефектов — ни один")
cond(co["era_≤2022"]["instrument"]["CIF24"] == 0, "до 2022 года смертей от дефектов к 24 мес нет")
has(f"от моделей умирают {pct(co['era_2023+']['models']['CIF24'])}, от дефектов и смены версий — {pct(co['era_2023+']['instrument']['CIF24'])}, "
    f"ещё {pct(co['era_2023+']['abandoned']['CIF24'])} забрасывают")
has(f"Из {ag_dead} умерших агентных сред {cbc['defects']['agentic environment']} сломались сами, а среди {st_dead} умершего статического теста таких шесть")
cond(cbc["defects"]["static test"] == 6, "у статических тестов шесть смертей от дефектов")
ce = A["cause_by_era"]["defects"]["2023+"]
cond(0.6 <= cbc["defects"]["agentic environment"] / ce <= 0.72, "две трети смертей от дефектов после 2022 года — агентные среды")
rc = A["sensitivity"]["replacement_is_censoring"]
has(f"период полураспада новых бенчмарков вырастает с {int(A['sensitivity']['main']['median_2023plus'])} до {int(rc['median_2023plus'])} месяцев")
cond(rc["median_le2022"] > rc["median_2023plus"] * 1.3, "разрыв с эпохой до 2023 года остаётся")
tb = {u: int(t) for u, t in zip(M.unit, M.T_months) if u.startswith("Terminal-Bench") and u[-3:] in ("1.0", "2.0", "2.1", "3.0")}
cond(len(tb) == 4 and max(tb.values()) <= 6, "Terminal-Bench: четыре версии, ни одна дольше полугода")

# --- что забрать и OenoBench
cond(A["cox"]["nonpublic"]["p"] > 0.05, "закрытый тест незначим")
fc = A["forecast_new_benchmark_2026"]
has(f"Та же модель выживаемости, обученная на {A['aft_headroom']['n_train']} бенчмарках, даёт ему около девяти месяцев")
cond(8.5 <= fc["best_83"]["aft_median_months"] <= 9.5 and 8.5 <= fc["best_81"]["aft_median_months"] <= 9.5, "около девяти месяцев")
cond(0.30 <= fc["best_83"]["S12"] <= 0.37 and 0.30 <= fc["best_81"]["S12"] <= 0.37, "до года — примерно треть")
has(f"до двух лет — {round(fc['best_83']['S24'] * 100)}–{round(fc['best_81']['S24'] * 100)}%")
has(f"проживёт {round(fc['closed_book_70']['aft_median_months'])}–{round(fc['closed_book_65']['aft_median_months'])} месяцев")
nb = A["headroom_neighbors"]
cond(nb["recent_n"] == 6 and nb["recent_dead_within_6m"] == 5, "пятеро из шести 2025–2026 умерли за полгода")
has("Из шести бенчмарков 2025–2026 годов с похожим запасом на старте пятеро умерли в первые полгода")
for u, rel, best in [("DeepSearchQA", "2025-12", 81.9), ("ExploitBench", "2026-05", 78.0)]:
    r = D[D.unit == u].iloc[0]
    cond(r.release == rel and abs(float(r.score_at_release) - best) < 0.05 and int(r.event) == 1, f"факты о {u}")
bm = A["best_at_release_for_median"]
cond(0.68 <= bm["static_12"] <= 0.72, "год — около 70%")
cond(0.34 <= bm["static_24"] <= 0.38, "два года — чуть больше трети")
cond(bm["static_36"] < 0.20, "три года — меньше 20%")
cond(0.08 <= bm["agentic_24"] <= 0.12, "агентная среда, два года — около 10%")

# --- данные
has(f"`halflife_dataset_v3.csv` — {len(D)} строк: {N} бенчмарка основного расчёта и {int((D.stratum != 'main').sum())} исключённых")
has(f"`decisions_log_v3.csv` — {len(LOG)} решение")
has(f"`d3_matrix_v3.csv` — {nm_rows:,} строки".replace(",", " "))
has(hashlib.sha256(open("halflife_dataset_v3.csv", "rb").read()).hexdigest())

print(f"проверок: {n}, не прошло: {len(fails)}")
for x in fails:
    print("  ✗", x)
sys.exit(1 if fails else 0)
