# -*- coding: utf-8 -*-
"""Графики статьи (v3). Запуск: python3 charts_v3.py [номер]."""
import sys, json, math
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from sai_style import *
from lifelines import KaplanMeierFitter

BASE = ".."
D = pd.read_csv(f"{BASE}/halflife_dataset_v3.csv")
M = D[D.stratum == "main"].copy()
M["T"] = M.T_months.astype(float); M["E"] = M.event.astype(int)
M["era"] = np.where(M.birth_year <= 2022, "≤2022", "2023+")
M["cls"] = M["class"].replace({"economic/time": "static test"})
A = json.load(open(f"{BASE}/analysis_v3.json", encoding="utf-8"))
CARDS = json.load(open(f"{BASE}/cards_v3.json", encoding="utf-8"))
def ru(x, d=0):
    s = f"{x:.{d}f}"
    return s.replace(".", ",")
def ym(s): return int(s[:4]) * 12 + int(s[5:7]) - 1

def check_overlaps(fig, ax, texts, points, r_px=13):
    """Печатает пересечения подписей друг с другом и с маркерами (радиус r_px в пикселях при dpi 200)."""
    fig.canvas.draw(); rr = fig.canvas.get_renderer()
    boxes = [(t.get_text(), t.get_window_extent(renderer=rr)) for t in texts]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i][1].overlaps(boxes[j][1]):
                print("  overlap text:", boxes[i][0], "×", boxes[j][0])
    for x, y in points:
        px, py = ax.transData.transform((x, y))
        for name, bb in boxes:
            if bb.x0 - r_px < px < bb.x1 + r_px and bb.y0 - r_px < py < bb.y1 + r_px:
                dx = max(bb.x0 - px, 0, px - bb.x1); dy = max(bb.y0 - py, 0, py - bb.y1)
                if (dx * dx + dy * dy) ** 0.5 < r_px:
                    print(f"  overlap marker ({x:.2f}, {y:.2f}) × {name}")

# ------------------------------------------------------------------ 1. KM по эпохам
def ch1():
    fig = new_fig()
    header(fig, f"Кривые выживания · оценка Каплана–Мейера · {len(M)} бенчмарков",
           [[("Период полураспада бенчмарка ", TXT)], [("сократился вдвое", OR)]],
           "Доля бенчмарков, ещё не умерших ни по одному критерию. Период полураспада — момент, когда кривая пересекает 50%.")
    ax = fig.add_axes([0.075, 0.15, 0.56, 0.56])
    style_ax(ax)
    cfg = [("≤2022", WHITE, "--", "Выпуск 2016–2022"), ("2023+", OR, "-", "Выпуск 2023–2026")]
    for era, col, ls, lab in cfg:
        g = M[M.era == era]
        k = KaplanMeierFitter().fit(g["T"], g.E)
        sf = k.survival_function_; ci = k.confidence_interval_
        xs = np.r_[0, sf.index.values]; ys = np.r_[1.0, sf.iloc[:, 0].values]
        ax.step(xs, ys, where="post", color=col, lw=2.6 if era == "2023+" else 1.8, ls=ls, zorder=3)
        ax.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1], step="post", color=col, alpha=0.10 if era == "2023+" else 0.06, lw=0)
        # цензурированные
        cens = g[g.E == 0]["T"].values
        ax.plot(cens, k.survival_function_at_times(cens).values, "|", color=col, ms=6, mew=1.0, alpha=0.8, zorder=4)
        med = A["km_era"][era]["median"]
        ax.plot([med, med], [0, 0.5], color=col, lw=0.9, ls=":", alpha=0.9)
        ax.plot(med, 0.5, "o", color=col, ms=6, zorder=5)
        ax.text(med + 1.2, 0.535 if era == "2023+" else 0.385, f"полураспад {int(med)} мес", color=col, family=MONO, fontsize=9.5,
                va="center")
    ax.axhline(0.5, color=BROWN, lw=0.8, ls="--", alpha=0.9, zorder=1)
    ax.set_xlim(0, 72); ax.set_ylim(0, 1.02)
    ax.set_xticks(range(0, 73, 12)); ax.set_yticks([0, .25, .5, .75, 1])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("месяцев с рождения бенчмарка", family=SANS, fontsize=10.5, color=MUTED, labelpad=10)
    ax.set_ylabel("доля ещё живых", family=SANS, fontsize=10.5, color=MUTED, labelpad=10)
    mono_ticks(ax)
    # первый год
    ax.axvspan(0, 12, color=OR, alpha=0.04, lw=0)
    ax.text(0.6, 0.025, "ПЕРВЫЙ ГОД", color=OR, family=MONO, fontsize=8.5, va="bottom")
    # легенда
    e1, e2 = A["km_era"]["≤2022"], A["km_era"]["2023+"]
    legend_item(fig, 0.685, 0.660, "Выпуск 2016–2022", [f"{e1['n']} шт · умерло {e1['events']}",
                f"полураспад {int(e1['median'])} мес (95% CI {int(e1['median_ci'][0])}–{int(e1['median_ci'][1])})",
                f"за первый год умерло {ru((1 - e1['S12']) * 100)}%"], WHITE, ls="--", lw=1.8)
    legend_item(fig, 0.685, 0.525, "Выпуск 2023–2026", [f"{e2['n']} шт · умерло {e2['events']}",
                f"полураспад {int(e2['median'])} мес (95% CI {int(e2['median_ci'][0])}–{int(e2['median_ci'][1])})",
                f"за первый год умерло {ru((1 - e2['S12']) * 100)}%"], OR, bold=True, accent_stats=True)
    hr = A["cox_birthyear_only"]
    callout(fig, 0.685, 0.13, 0.265, 0.27, "модель Кокса · год рождения",
            f"×{ru(hr['HR_per_year'], 2)} в год",
            f"во столько раз растёт риск смерти с каждым\nгодом рождения (95% CI {ru(hr['ci'][0], 2)}–{ru(hr['ci'][1], 2)}).\nРазница эпох: лог-ранговый тест, p < 0,001")
    footer(fig, "штрихи на кривых — живые и заброшенные (цензура) · тень — 95% CI · срез 30.09.2026 · ось обрезана на 72 мес")
    save(fig, "ch1_km_era.png")


# ------------------------------------------------------------------ 2. полосы жизни
# Цвета исходов (проверены validate_palette.js, тёмная тема, все пары): модели — фирменный оранжевый,
# прибор — розовый; «жив» — светло-серый сплошной, «заброшен» — тёмно-серый пунктир с крестиком (вторая кодировка).
PINK = "#d64f8f"      # дефекты и новые версии (сам прибор)
ALIVE = "#8c8c8c"     # жив на срез: светло-серый сплошной со стрелкой
GREY_AB = "#555555"   # заброшен непобеждённым: тёмно-серый пунктир с крестиком
OR2L = "#ffcfa8"      # признанное насыщение (подвид «моделей»)
COL_CAUSE3 = {"models": OR, "instrument": PINK}
def ch2():
    fig = new_fig()
    header(fig, f"Жизнь каждого бенчмарка · {len(M)} шт · рождение → смерть или срез",
           [[("Чем позже рождение, ", TXT), ("тем короче полоса", OR)]],
           "Каждая строка — бенчмарк, по оси — календарное время. Цвет — чем закончилась жизнь к срезу.")
    ax = fig.add_axes([0.06, 0.13, 0.66, 0.60])
    style_ax(ax, ygrid=False)
    g = M.sort_values(["release", "unit"], ascending=[True, True]).reset_index(drop=True)
    n = len(g)
    cut = 2026 + 8.5 / 12
    def dec(sv): return int(sv[:4]) + (int(sv[5:7]) - 0.5) / 12
    for i, r in g.iterrows():
        y = n - i
        x0 = dec(r.release)
        if r.E == 1:
            x1 = dec(r.primary_date); c = COL_CAUSE3[r.cause3]
            ax.plot([x0, x1], [y, y], color=c, lw=1.5, solid_capstyle="butt", alpha=0.95)
            ax.plot(x1, y, "o", color=c, ms=2.0, mew=0)
        elif r.status_cutoff == "abandoned":
            x1 = dec(r.abandoned_date)
            ax.plot([x0, x1], [y, y], color=GREY_AB, lw=1.4, ls=(0, (1.2, 1.2)), dash_capstyle="butt")
            ax.plot(x1, y, "x", color="#8a8a8a", ms=3.0, mew=0.9)
        else:
            ax.plot([x0, cut], [y, y], color=ALIVE, lw=1.3, solid_capstyle="butt", alpha=0.9)
            ax.plot(cut, y, ">", color=ALIVE, ms=2.2, mew=0)
    ax.axvline(cut, color=WHITE, lw=0.8, ls="--", alpha=0.5)
    ax.text(cut, n + 4, "срез\n30.09.2026", color=MUTED, family=MONO, fontsize=7.5, ha="center", va="bottom")
    ax.set_ylim(0, n + 2); ax.set_yticks([])
    ax.set_xlim(2016, 2027); ax.set_xticks(range(2016, 2027, 2))
    mono_ticks(ax)
    ax.spines["left"].set_visible(False)
    notes = {"SQuAD 1.1": "SQuAD 1.1 · 24 мес", "GLUE": "GLUE · 13", "MMLU": "MMLU · 39", "GSM8K": "GSM8K · 17",
             "GPQA Diamond": "GPQA Diamond · 24", "SWE-bench Verified": "SWE-bench Verified · 18",
             "Humanity's Last Exam": "HLE · 19, дефекты", "ARC-AGI-1": "ARC-AGI-1 · 61", "ARC-AGI-2": "ARC-AGI-2 · 13",
             "ARC-AGI-3": "ARC-AGI-3 · 6", "AIME 2025": "AIME 2025 · 0", "Terminal-Bench 3.0": "Terminal-Bench 3.0 · 1",
             "TriviaQA": "TriviaQA · 86", "WebArena": "WebArena · жив"}
    for i, r in g.iterrows():
        if r.unit in notes:
            y = n - i
            if r.E == 1: x1 = dec(r.primary_date)
            elif r.status_cutoff == "abandoned": x1 = dec(r.abandoned_date)
            else: x1 = cut
            right = x1 < 2025.3
            ax.annotate(notes[r.unit], (x1, y), xytext=((x1 + 0.25) if right else (dec(r.release) - 0.2), y),
                        color=WHITE if r.E == 1 else MUTED, fontsize=7.6, family=MONO, va="center",
                        ha="left" if right else "right",
                        bbox=dict(boxstyle="round,pad=0.15", facecolor=BG, edgecolor="none", alpha=0.85))
    # легенда справа: образец линии и конца, подпись, число
    cnt = M.assign(k=np.where(M.E == 1, M.cause3, M.status_cutoff)).k.value_counts().to_dict()
    items = [("побеждён моделями", "порог или насыщение", OR, "-", "o", cnt.get("models", 0)),
             ("дефекты и новые версии", "сломался сам прибор", PINK, "-", "o", cnt.get("instrument", 0)),
             ("жив на срез", "наблюдение идёт", ALIVE, "-", ">", cnt.get("alive", 0)),
             ("заброшен непобеждённым", "год без отчётов", GREY_AB, (0, (1.2, 1.2)), "x", cnt.get("abandoned", 0))]
    yy = 0.685
    for lab, sub, c, ls, mk, k in items:
        fig.add_artist(plt.Line2D([0.755, 0.785], [yy + 0.006, yy + 0.006], color=c, lw=2.4, ls=ls, transform=fig.transFigure))
        fig.add_artist(plt.Line2D([0.785], [yy + 0.006], marker=mk, color=c if mk != "x" else "#8a8a8a", ms=5.5,
                                  mew=1.4 if mk == "x" else 0, transform=fig.transFigure))
        fig.text(0.798, yy, lab, color=TXT, fontsize=10.5, va="baseline")
        fig.text(0.798, yy - 0.027, f"{k} · {sub}", color=MUTED, fontsize=8.4, family=MONO, va="baseline")
        yy -= 0.068
    e1, e2 = A["km_era"]["≤2022"], A["km_era"]["2023+"]
    callout(fig, 0.755, 0.13, 0.20, 0.25, "период полураспада",
            f"{int(e1['median'])} → {int(e2['median'])} мес", big_size=22,
            small="медиана по Каплану–Мейеру,\nвыпуск 2016–2022 и 2023–2026;\nживые учтены, а не выброшены")
    footer(fig, "числа у подписей — срок жизни в месяцах · срез 30.09.2026")
    save(fig, "ch2_lifelines.png")

# ------------------------------------------------------------------ 3. нулевая модель
COH = ["≤2019", "2020–2022", "2023", "2024", "2025", "2026"]
def ch3():
    fig = new_fig()
    header(fig, "Нулевая модель · одна и та же продолжительность жизни для всех когорт",
           [[("Наивная медиана падает ", TXT), ("и без всякого ускорения", OR)]],
           "Медиана, посчитанная только среди уже умерших, у свежих когорт мала всегда: длинные жизни ещё не успели закончиться.")
    ax = fig.add_axes([0.075, 0.15, 0.58, 0.55])
    style_ax(ax, xgrid=False)
    x = np.arange(len(COH))
    nm = A["null_model_naive_median_dead"]
    lo = [nm[c]["p05"] for c in COH]; hi = [nm[c]["p95"] for c in COH]; mu = [nm[c]["mean"] for c in COH]
    ax.fill_between(x, lo, hi, color="#4a4a4a", alpha=0.35, lw=0)
    ax.plot(x, mu, color=GREY, lw=1.4, ls="--")
    obs = [A["observed_naive_median_dead"][c] for c in COH]
    ax.plot(x, obs, "o", mfc=BG, mec=WHITE, mew=1.6, ms=9, zorder=4)
    for i, c in enumerate(COH):
        k = A["km_cohort"][c]
        med, (l, h) = k["median"], k["median_ci"]
        if med is not None:
            ax.plot([i + 0.12, i + 0.12], [l, h if h is not None else 110], color=OR, lw=1.4, alpha=0.8)
            if h is None:
                ax.annotate("", (i + 0.12, 112), (i + 0.12, 100), arrowprops=dict(arrowstyle="-|>", color=OR, lw=1.2))
            thin = (c == "2026")   # в 6 мес под риском мало бенчмарков: медиана — ориентир
            ax.plot(i + 0.12, med, "o", color=OR, mfc=BG if thin else OR, mew=1.6, ms=9, zorder=5)
            ax.text(i + 0.26, med, f"{int(med)}*" if thin else f"{int(med)}", color=OR, family=MONO, fontsize=10, va="center")
        else:
            ax.annotate("", (i + 0.12, l + 14), (i + 0.12, l), arrowprops=dict(arrowstyle="-|>", color=OR, lw=1.4))
            ax.text(i + 0.26, l + 4, f"> {int(l)}\nне достигнута", color=OR, family=MONO, fontsize=8.5, va="center")
        ax.text(i - 0.16, obs[i], f"{ru(obs[i], 1)}" if obs[i] % 1 else f"{int(obs[i])}", color=WHITE, family=MONO, fontsize=9.5,
                va="center", ha="right")
    ax.set_xticks(x); ax.set_xticklabels(COH)
    ax.set_ylim(0, 115); ax.set_xlim(-0.6, len(COH) - 0.3)
    ax.set_ylabel("месяцев", family=SANS, fontsize=10.5, color=MUTED, labelpad=8)
    ax.set_xlabel("когорта (год рождения)", family=SANS, fontsize=10.5, color=MUTED, labelpad=10)
    mono_ticks(ax)
    # легенда
    fig.add_artist(plt.Line2D([0.70], [0.665], marker="o", color=OR, ms=9, transform=fig.transFigure))
    fig.text(0.715, 0.66, "Период полураспада", color=TXT, fontsize=12, fontweight="bold")
    fig.text(0.715, 0.632, "медиана Каплана–Мейера, 95% CI", color=OR, fontsize=8.8, family=MONO)
    fig.add_artist(plt.Line2D([0.70], [0.575], marker="o", mfc=BG, mec=WHITE, mew=1.6, ms=9, transform=fig.transFigure))
    fig.text(0.715, 0.57, "Наивная медиана среди умерших", color=WHITE, fontsize=12)
    fig.text(0.715, 0.542, "так считают «на глаз»", color=MUTED, fontsize=8.8, family=MONO)
    fig.patches.append(Rectangle((0.69, 0.475), 0.02, 0.02, transform=fig.transFigure, color="#4a4a4a", alpha=0.6))
    fig.text(0.715, 0.48, "Нулевая модель", color=WHITE, fontsize=12)
    fig.text(0.715, 0.452, "наивная медиана, если сроки жизни\nне менялись (2000 симуляций, 90%)", color=MUTED, fontsize=8.8, family=MONO, va="top")
    kc = A["km_cohort"]; o26 = A["observed_naive_median_dead"]["2026"]
    mids = [int(kc[c]["median"]) for c in ["2023", "2024", "2025"] if kc[c]["median"] is not None]
    callout(fig, 0.69, 0.10, 0.26, 0.29, "как читать",
            "Это не срок жизни", big_size=19.5, small=
            f"{ru(o26, 1) if o26 % 1 else int(o26)} мес у выпуска 2026 года — внутри\nполосы нулевой модели. Сигнал —\nв периоде полураспада: {int(kc['≤2019']['median'])} и {int(kc['2020–2022']['median'])} мес\nу старых когорт, {min(mids)}–{max(mids)} у 2023–2025.")
    g26 = M[M.cohort == "2026"]
    footer(fig, f"нулевая модель: сроки из общей кривой Каплана–Мейера, реальные даты рождения · * 2026: в 6 мес под риском {int((g26['T'] >= 6).sum())} из {len(g26)}, ориентир")
    save(fig, "ch3_null_model.png")

# ------------------------------------------------------------------ 4. конкурирующие риски
def aj_curves(T, code, tmax=48):
    T = np.asarray(T, float); code = np.asarray(code, int)
    grid = np.arange(0, tmax + 1)
    S = 1.0; cif = {1: 0.0, 2: 0.0, 3: 0.0}; out = {k: [] for k in (1, 2, 3)}
    for t in grid:
        n = np.sum(T >= t)
        d = {k: np.sum((T == t) & (code == k)) for k in (1, 2, 3)}
        if n > 0:
            for k in (1, 2, 3):
                cif[k] += S * d[k] / n
            S *= 1 - sum(d.values()) / n
        for k in (1, 2, 3):
            out[k].append(cif[k])
    return grid, out
def ch4():
    fig = new_fig()
    header(fig, "Конкурирующие риски · оценка Аалена–Йохансена",
           [[("После 2023 года каждый седьмой бенчмарк ", TXT)], [("умирает не от моделей", OR)]],
           "Накопленная доля умерших по причинам: модели (порог или признанное насыщение), сам прибор, забвение.")
    for j, era in enumerate(["≤2022", "2023+"]):
        ax = fig.add_axes([0.075 + j * 0.31, 0.16, 0.26, 0.50])
        style_ax(ax)
        g = M[M.era == era]
        grid, c = aj_curves(g["T"], g.outcome_cr.astype(int), 48)
        y1 = np.array(c[1]); y2 = y1 + np.array(c[2]); y3 = y2 + np.array(c[3])
        ax.fill_between(grid, 0, y1, step="post", color=OR, alpha=0.85, lw=0)
        ax.fill_between(grid, y1, y2, step="post", color=PINK, alpha=0.9, lw=0)
        ax.fill_between(grid, y2, y3, step="post", color=GREY_AB, alpha=0.9, lw=0)
        ax.set_xlim(0, 48); ax.set_ylim(0, 0.8); ax.set_xticks([0, 12, 24, 36, 48])
        ax.set_yticks([0, .2, .4, .6, .8]); ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%"])
        mono_ticks(ax)
        ax.set_title(f"Выпуск {'2016–2022' if era == '≤2022' else '2023–2026'}", color=TXT if j else WHITE, fontsize=13,
                     fontweight="bold", loc="left", pad=12)
        at24 = {k: c[k][24] for k in (1, 2, 3)}
        ax.text(24.5, at24[1] / 2, f"модели\n{ru(at24[1] * 100)}%", color=BG, fontsize=9, family=MONO, va="center", fontweight="bold")
        if at24[2] > 0.03:
            ax.text(24.5, at24[1] + at24[2] / 2, f"прибор {ru(at24[2] * 100)}%", color=TXT, fontsize=8.5, family=MONO, va="center")
        ax.axvline(24, color=TXT, lw=0.6, ls=":", alpha=0.6)
        ax.set_xlabel("месяцев с рождения", family=SANS, fontsize=10, color=MUTED, labelpad=8)
    comp = A["competing"]
    items = [("модели: порог или насыщение", OR), ("прибор: дефекты и версии", PINK), ("заброшен непобеждённым", GREY_AB)]
    yy = 0.64
    for lab, col in items:
        fig.patches.append(Rectangle((0.71, yy - 0.004), 0.018, 0.022, transform=fig.transFigure, color=col))
        fig.text(0.735, yy, lab, color=WHITE, fontsize=10)
        yy -= 0.045
    e = comp["era_2023+"]
    callout(fig, 0.71, 0.16, 0.245, 0.24, "выпуск 2023–2026 · 24 месяца",
            f"{ru(e['instrument']['CIF24'] * 100)}%",
            f"умирают от дефектов, компрометации\nи смены версий. В выпуске 2016–2022 за\nте же 24 месяца таких смертей не было.")
    footer(fig, "основное событие — самое раннее из A–E; при равенстве дат причина — модели")
    save(fig, "ch4_competing.png")

# ------------------------------------------------------------------ 5. причины по классам
def ch5():
    fig = new_fig()
    header(fig, f"От чего умирают · {int(M.E.sum())} смертей по классам бенчмарков",
           [[("Агентные среды чаще ломаются сами, ", TXT)], [("статические тесты — побеждены", OR)]],
           "Основное событие по классу бенчмарка. Число в полосе — количество смертей.")
    ax = fig.add_axes([0.20, 0.22, 0.52, 0.42])
    style_ax(ax, ygrid=False)
    dd = M[M.E == 1]
    order = [("static test", "Статические тесты"), ("agentic environment", "Агентные среды"), ("LLM-judge", "Оценка LLM-судьёй")]
    causes = [("threshold", "порог", OR), ("declared_saturation", "насыщение признано", OR2L), ("defects", "дефекты и версии", PINK)]
    for i, (k, lab) in enumerate(order):
        g = dd[dd.cls == k]; tot = len(g); left = 0
        for c, cl, col in causes:
            v = int((g.cause == c).sum())
            if v:
                ax.barh(i, v / tot, left=left, color=col, height=0.62, edgecolor=BG, lw=1.5)
                ax.text(left + v / tot / 2, i, str(v), color=BG, ha="center", va="center", fontsize=11,
                        family=MONO, fontweight="bold")
            left += v / tot
        fig.text(0.19, 0.22 + (len(order) - 1 - i + 0.5) / len(order) * 0.42, lab, color=TXT, fontsize=12.5, ha="right", va="center",
                 fontweight="bold" if k == "agentic environment" else "medium")
        fig.text(0.19, 0.22 + (len(order) - 1 - i + 0.5) / len(order) * 0.42 - 0.032, f"{tot} смертей из {int((M.cls == k).sum())}",
                 color=MUTED, fontsize=8.5, family=MONO, ha="right", va="center")
    ax.set_ylim(len(order) - 0.5, -0.5); ax.set_xlim(0, 1)
    ax.set_yticks([]); ax.set_xticks([0, .25, .5, .75, 1]); ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.spines["left"].set_visible(False)
    mono_ticks(ax)
    yy = 0.155
    xx = 0.20
    for c, cl, col in causes:
        fig.patches.append(Rectangle((xx, yy - 0.004), 0.016, 0.02, transform=fig.transFigure, color=col))
        t = fig.text(xx + 0.022, yy, cl, color=WHITE, fontsize=10)
        xx += 0.18
    ag = dd[dd.cls == "agentic environment"]
    callout(fig, 0.75, 0.30, 0.205, 0.34, "агентные среды",
            f"{int((ag.cause == 'defects').sum())} из {len(ag)}",
            "умерли от дефектов или выхода\nисправленной версии:\nTerminal-Bench, τ-bench,\nOSWorld, SWE-bench,\nToolathlon, MCPMark и другие")
    footer(fig, "дефекты и версии — критерий C с причиной «ошибки в задачах или проверке» и критерий E")
    save(fig, "ch5_causes_class.png")

# ------------------------------------------------------------------ 6. запас и скорость
def ch6():
    fig = new_fig()
    header(fig, "Разложение срока жизни · запас на релизе ÷ скорость прогресса",
           [[("Запас на старте прежний, ", TXT), ("скорость — вдвое выше", OR)]],
           "Запас — расстояние от лучшего результата на релизе до порога, в логитах. Скорость — наклон фронтира по данным Epoch AI.")
    ax = fig.add_axes([0.075, 0.15, 0.56, 0.56])
    style_ax(ax)
    tab = pd.DataFrame(A["decomposition"]["table"])
    xs = np.linspace(0.3, 5.5, 50)
    for T, lab in [(6, "6 мес"), (12, "12 мес"), (24, "24 мес"), (48, "48 мес")]:
        ys = xs * 12 / T
        ax.plot(xs, ys, color="#3a3a3a", lw=0.9, ls="--")
        xl = 5.3 if T >= 24 else min(5.3, 4.3 * T / 12)
        yl = xl * 12 / T
        if yl > 4.4: xl = 4.3 * T / 12; yl = 4.3
        ax.text(xl, yl, lab, color=DIM, family=MONO, fontsize=8.5, ha="left", va="bottom", rotation=0)
    # подписи расставлены вручную: (dx, dy, выравнивание) в единицах осей; проверка пересечений — check_overlaps()
    LABEL_POS = {"Video-MME": (0.07, 0.06, "left"), "LiveBench": (0.0, -0.17, "right"), "GSM8K": (0.0, -0.19, "center"),
                 "GPQA Diamond": (0.0, 0.10, "center"), "HellaSwag": (0.07, 0.08, "left"), "OpenBookQA": (-0.06, -0.22, "right"),
                 "MMLU": (0.02, -0.20, "left"), "ANLI": (0.07, -0.10, "left"), "TriviaQA": (0.08, -0.13, "left"),
                 "SWE-bench Verified": (0.08, -0.15, "left"), "ARC-Challenge": (0.07, 0.08, "left"),
                 "BIG-Bench Hard": (0.0, -0.18, "center")}
    LABELS = []
    for _, r in tab.iterrows():
        col = OR if r.era == "2023+" else WHITE
        ax.plot(r.headroom_logit, r.logit_slope_per_year, "o", color=col if r.E == 1 else BG, mec=col, mew=1.4, ms=8, zorder=4)
        name = r.unit.replace("FrontierMath Tier 4", "FM Tier 4").replace("SWE-bench Verified", "SWE-bench V.")
        off = LABEL_POS.get(r.unit, (0.07, 0.06, "left"))
        LABELS.append(ax.text(r.headroom_logit + off[0], r.logit_slope_per_year + off[1], name, color=col, family=MONO,
                              fontsize=7.6, alpha=0.9, ha=off[2]))
    for era, col in [("≤2022", WHITE), ("2023+", OR)]:
        t = A["decomposition"][f"typical_{era}"]
        ax.plot(t["headroom"], t["speed"], "D", color=col, ms=12, mec=BG, mew=1.5, zorder=6)
    ax.set_xlim(0, 5.5); ax.set_ylim(0, 4.5)
    ax.set_xlabel("запас на релизе, логиты (больше — дальше от порога)", family=SANS, fontsize=10.5, color=MUTED, labelpad=10)
    ax.set_ylabel("скорость фронтира, логитов в год", family=SANS, fontsize=10.5, color=MUTED, labelpad=10)
    mono_ticks(ax)
    t1, t2 = A["decomposition"]["typical_≤2022"], A["decomposition"]["typical_2023+"]
    legend_item(fig, 0.685, 0.655, "Типичный бенчмарк 2016–2022", [f"запас {ru(t1['headroom'], 1)} · скорость {ru(t1['speed'], 2)}/год",
                f"прогноз {int(round(t1['T_pred_months']))} мес · факт (медиана) {int(t1['km_median'])}"], WHITE, lw=0)
    fig.add_artist(plt.Line2D([0.699], [0.661], marker="D", color=WHITE, ms=10, transform=fig.transFigure))
    legend_item(fig, 0.685, 0.53, "Типичный бенчмарк 2023–2026", [f"запас {ru(t2['headroom'], 1)} · скорость {ru(t2['speed'], 2)}/год",
                f"прогноз {int(round(t2['T_pred_months']))} мес · факт (медиана) {int(t2['km_median'])}"], OR, lw=0, bold=True, accent_stats=True)
    fig.add_artist(plt.Line2D([0.699], [0.536], marker="D", color=OR, ms=10, transform=fig.transFigure))
    sp = A["speed_by_era"]
    callout(fig, 0.685, 0.12, 0.265, 0.29, "скорость фронтира · медиана",
            f"{ru(sp['≤2022']['median_logit_per_year'], 2)} → {ru(sp['2023+']['median_logit_per_year'], 2)}",
            f"логита в год: {sp['≤2022']['n']} старых против {sp['2023+']['n']} новых\nбенчмарков (Манн–Уитни, p = {ru(A['speed_mannwhitney_p'], 3)}).\nПунктир — линии равного срока\nжизни «запас ÷ скорость».")
    footer(fig, "закрашенный круг — бенчмарк умер, полый — жив · наклоны по фронтиру, не меньше трёх точек")
    check_overlaps(fig, ax, LABELS, list(zip(tab.headroom_logit, tab.logit_slope_per_year)) +
                   [(A["decomposition"][f"typical_{e}"]["headroom"], A["decomposition"][f"typical_{e}"]["speed"]) for e in ["≤2022", "2023+"]])
    save(fig, "ch6_headroom_speed.png")

# ------------------------------------------------------------------ 7. мёртвые в карточках
def ch7():
    fig = new_fig()
    header(fig, "Посмертная жизнь · отчёты 9 лабораторий в материалах релизов",
           [[("Больше четверти результатов в карточках 2026 года — ", TXT)], [("на уже мёртвых бенчмарках", OR)]],
           "Доля пар «релиз × бенчмарк», где бенчмарк к дате релиза уже умер по нашей разметке.")
    ax = fig.add_axes([0.075, 0.17, 0.52, 0.50])
    style_ax(ax, xgrid=False)
    z = CARDS["zombie_share_by_year"]
    yrs = [y for y in sorted(z, key=int) if int(y) >= 2022]
    vals = [z[y]["share"] for y in yrs]; ns = [z[y]["pairs"] for y in yrs]
    bars = ax.bar(range(len(yrs)), vals, color=[OR if y == "2026" else "#6b3a1c" for y in yrs], width=0.62)
    for i, (v, nn) in enumerate(zip(vals, ns)):
        ax.text(i, v + 0.012, f"{ru(v * 100)}%", color=TXT, ha="center", fontsize=12, family=MONO, fontweight="bold")
        ax.text(i, 0.015, f"{nn} пар", color=TXT if yrs[i] == "2026" else MUTED, ha="center", fontsize=8.2, family=MONO)
    ax.set_xticks(range(len(yrs))); ax.set_xticklabels(yrs)
    ax.set_ylim(0, 0.55); ax.set_yticks([0, .1, .2, .3, .4, .5]); ax.set_yticklabels(["0%", "10%", "20%", "30%", "40%", "50%"])
    mono_ticks(ax)
    # справа: посмертная жизнь
    fig.text(0.655, 0.665, "// СКОЛЬКО ЖИВУТ ПОСЛЕ СМЕРТИ", color=MUTED, fontsize=8.5, family=MONO)
    rows = [(f"{CARDS['dead_reported_after_death']} из {CARDS['dead_n']}", "мёртвых бенчмарков\nпоказывали уже после смерти"),
            (f"{int(round(CARDS['postmortem_km_S12'] * 100))}%", "всё ещё в карточках\nчерез год после смерти"),
            (f"{int(CARDS['postmortem_km_median_months'])} мес", "медиана «посмертной жизни»\n(Каплан–Мейер)")]
    yy = 0.615
    for big, small in rows:
        fig.text(0.655, yy, big, color=OR, fontsize=20, fontweight="bold", va="top")
        fig.text(0.79, yy - 0.004, small, color=MUTED, fontsize=9.2, va="top", linespacing=1.35)
        yy -= 0.105
    lab = CARDS["zombie_share_2026_by_lab"]
    s_ = sorted(lab.items(), key=lambda kv: -kv[1]["share"])
    nm = lambda k: k.split(' (')[0].replace('Moonshot AI', 'Moonshot').replace('Alibaba', 'Qwen')
    fig.text(0.655, 0.275, "// ДОЛЯ В 2026 ПО ЛАБОРАТОРИЯМ", color=MUTED, fontsize=8.5, family=MONO)
    fig.text(0.655, 0.245, "\n".join(f"{nm(k)}: {int(round(v['share'] * 100))}%" for k, v in s_[:4]),
             color=WHITE, fontsize=8.6, family=MONO, va="top", linespacing=1.45)
    fig.text(0.80, 0.245, "\n".join(f"{nm(k)}: {int(round(v['share'] * 100))}%" for k, v in s_[4:]),
             color=WHITE, fontsize=8.6, family=MONO, va="top", linespacing=1.45)
    footer(fig, "пара «релиз × бенчмарк» учитывается один раз · смерть — основное событие по кодбуку v3")
    save(fig, "ch7_zombies.png")

# ------------------------------------------------------------------ 8. устойчивость
def ch8():
    fig = new_fig()
    header(fig, "Проверки устойчивости · период полураспада по эпохам",
           [[("Правила двигают числа, ", TXT), ("но не вывод", OR)]],
           "Каждая строка — тот же расчёт при другом спорном правиле. Белый — выпуск 2016–2022, оранжевый — 2023–2026.")
    ax = fig.add_axes([0.33, 0.14, 0.40, 0.58])
    style_ax(ax, ygrid=False)
    S = A["sensitivity"]
    rows = [("main", "Основной расчёт"), ("strict", "Строгий режим: только одиночные модели\nс независимой проверкой"),
            ("no_reissues", "Без ежегодных экзаменов\n(AIME, HMMT, USAMO, IMO, CNMO)"),
            ("replacement_is_censoring", "Новая версия — не смерть,\nа выбытие из наблюдения"),
            ("no_dev_split", "Без результатов на dev-сплите"),
            ("delayed_entry", "Наблюдение с первого отчёта\nлаборатории (отложенный вход)"),
            ("alt_arc_gdpval", "ARC-AGI-2 D → 2026-07, ARC-AGI-3 от превью,\nGDPval с порогом-паритетом")]
    for i, (k, lab) in enumerate(rows):
        r = S[k]; a, b = r["median_le2022"], r["median_2023plus"]
        ax.plot([b, a], [i, i], color="#4a4a4a", lw=2.2, zorder=1)
        ax.plot(a, i, "o", color=WHITE, ms=10, zorder=3); ax.plot(b, i, "o", color=OR, ms=10, zorder=3)
        ax.text(a + 1.5, i, f"{int(a)}", color=WHITE, family=MONO, fontsize=10, va="center")
        ax.text(b - 1.5, i, f"{int(b)}", color=OR, family=MONO, fontsize=10, va="center", ha="right")
        fig.text(0.315, 0.14 + (len(rows) - 1 - i + 0.5) / len(rows) * 0.58, lab, color=TXT if k == "main" else WHITE,
                 fontsize=10.5 if k == "main" else 9.6, ha="right", va="center", fontweight="bold" if k == "main" else "normal",
                 linespacing=1.3)
        fig.text(0.745, 0.14 + (len(rows) - 1 - i + 0.5) / len(rows) * 0.58, f"×{ru(r['HR_birth_year'], 2)}", color=MUTED,
                 fontsize=10, family=MONO, va="center")
    fig.text(0.745, 0.735, "РИСК/ГОД", color=DIM, fontsize=8, family=MONO)
    ax.set_ylim(len(rows) - 0.5, -0.5); ax.set_xlim(0, 80); ax.set_yticks([])
    ax.set_xticks([0, 12, 24, 36, 48, 60, 72])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("период полураспада, месяцев", family=SANS, fontsize=10.5, color=MUTED, labelpad=10)
    mono_ticks(ax)
    m0, st = S["main"], S["strict"]
    callout(fig, 0.80, 0.28, 0.16, 0.32, "сильнее всего",
            f"+{int(st['median'] - m0['median'])} мес", f"к общему периоду\nполураспада даёт\nстрогий режим: {int(st['median'])} мес\nвместо {int(m0['median'])}. Разрыв эпох\nне исчезает: {int(st['median_le2022'])} и {int(st['median_2023plus'])}")
    footer(fig, "риск/год — отношение рисков на год рождения в модели Кокса · все p < 0,002")
    save(fig, "ch8_sensitivity.png")

# ------------------------------------------------------------------ 9. прогноз для OenoBench
def ch9():
    from lifelines import WeibullAFTFitter
    fig = new_fig()
    header(fig, "Прогноз · модель Вейбулла с запасом, годом рождения и классом",
           [[("Модель даёт OenoBench около девяти месяцев. ", TXT)], [("Данные самого OenoBench с этим спорят", OR)]],
           "Вероятность, что бенчмарк 2026 года ещё жив, в зависимости от лучшего результата на релизе. Порог смерти — 90% (критерий B).")
    H = pd.read_csv(f"{BASE}/headroom_v3.csv")
    H = H.merge(M[["unit", "cls"]], on="unit")
    XH = pd.DataFrame({"T": H["T"] + 0.5, "E": H.E, "headroom": H.headroom_logit.clip(upper=6.9), "b": H.birth_year - 2023,
                       "agentic": (H.cls == "agentic environment").astype(int)})
    wf = WeibullAFTFitter().fit(XH, "T", "E")
    ax = fig.add_axes([0.075, 0.15, 0.56, 0.55])
    style_ax(ax)
    t = np.arange(0.5, 36.5, 0.5)
    def lg(p): return math.log(p / (1 - p))
    sets = [("весь набор: лидеры 81–83%", (0.81, 0.83), OR), ("трудная половина: 65–70%", (0.65, 0.70), WHITE)]
    for lab, (a, b), col in sets:
        curves = []
        for best in (a, b):
            row = pd.DataFrame({"headroom": [lg(0.9) - lg(best)], "b": [3], "agentic": [0]})
            curves.append(wf.predict_survival_function(row, times=t).iloc[:, 0].values)
        ax.fill_between(t - 0.5, curves[0], curves[1], color=col, alpha=0.25, lw=0)
        ax.plot(t - 0.5, (curves[0] + curves[1]) / 2, color=col, lw=2.2)
        mid = (curves[0] + curves[1]) / 2
        i50 = np.argmin(np.abs(mid - 0.5))
        ax.plot(t[i50] - 0.5, 0.5, "o", color=col, ms=7)
        if col == OR:
            ax.text(t[i50] - 1.0, 0.45, f"медиана ≈ {int(round(t[i50] - 0.5))} мес", color=col, family=MONO, fontsize=9.5, ha="right")
        else:
            ax.text(t[i50] + 0.6, 0.53, f"медиана ≈ {int(round(t[i50] - 0.5))} мес", color=col, family=MONO, fontsize=9.5)
    ax.axhline(0.5, color=BROWN, lw=0.8, ls="--")
    ax.axvline(12, color=GREY, lw=0.8, ls=":")
    ax.text(12.3, 0.95, "май 2027:\nпроверка", color=MUTED, family=MONO, fontsize=8.5, va="top")
    ax.set_xlim(0, 36); ax.set_ylim(0, 1.02); ax.set_xticks([0, 6, 12, 18, 24, 30, 36])
    ax.set_yticks([0, .25, .5, .75, 1]); ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("месяцев с релиза (май 2026)", family=SANS, fontsize=10.5, color=MUTED, labelpad=10)
    ax.set_ylabel("вероятность, что ещё жив", family=SANS, fontsize=10.5, color=MUTED, labelpad=10)
    mono_ticks(ax)
    sp = A["speed_epoch_recent"]["median"]
    h83, h81 = lg(0.9) - lg(0.83), lg(0.9) - lg(0.81)
    h70, h65 = lg(0.9) - lg(0.70), lg(0.9) - lg(0.65)
    fig.text(0.685, 0.66, "// АРИФМЕТИКА «ЗАПАС ÷ СКОРОСТЬ»", color=MUTED, fontsize=8.5, family=MONO)
    lines = [("лидеры 81–83%: запас", f"{ru(h83, 2)}–{ru(h81, 2)} логита"),
             (f"при медианной скорости ({ru(sp, 2)}/год)", f"{int(round(h83 / sp * 12))}–{int(round(h81 / sp * 12))} мес"),
             ("половина 65–70%: запас", f"{ru(h70, 2)}–{ru(h65, 2)} логита"),
             ("при той же скорости", f"{int(round(h70 / sp * 12))}–{int(round(h65 / sp * 12))} мес")]
    yy = 0.615
    for a_, b_ in lines:
        fig.text(0.685, yy, a_, color=WHITE, fontsize=9.4)
        fig.text(0.955, yy, b_, color=OR, fontsize=9.4, family=MONO, ha="right")
        yy -= 0.04
    callout(fig, 0.685, 0.09, 0.27, 0.33, "почему прогноз может не сбыться",
            "≈ 0 за год",
            "Opus 4.7 (апрель 2026) набрал на OenoBench\nменьше, чем o3 (апрель 2025). Модель\nобучена на бенчмарках, под которые\nлаборатории оптимизируют. Под вино\nне оптимизирует никто.")
    footer(fig, "AFT Вейбулла: запас, год рождения, класс · OenoBench как статический тест 2026 года без человеческого бейзлайна")
    save(fig, "ch9_oenobench.png")


if __name__ == "__main__":
    which = sys.argv[1:] or ["1"]
    for w in which:
        globals()[f"ch{w}"]()
        print("ok", w)
