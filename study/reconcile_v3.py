# -*- coding: utf-8 -*-
"""
Сведение разметки Д1 (кодбук v3, v1.1) в датасет halflife_dataset_v3.csv.

Вход:  d1/coderA/batch*.csv (основной кодировщик), d1/coderA_repair/batch01_repair.csv (перекодировка 9 единиц),
       d1/coderB/s*.csv (слепая подвыборка второго кодировщика), d3_matrix.csv (отчёты лабораторий),
       population_main.json (популяция до сведения).
Выход: halflife_dataset_v3.csv, decisions_log_v3.csv, agreement_v3.json.

Правило: каждое отступление от строки кодировщика A — запись в OVERRIDES с основанием (пункт кодбука) и источником.
Основное событие и время жизни пересчитываются из компонент, а не берутся из поля primary_date.
"""
import csv, glob, json, re, collections

CUTOFF = "2026-09"

# ---------------------------------------------------------------- загрузка
def load(files):
    out = {}
    for f in files:
        for r in csv.DictReader(open(f, encoding="utf-8")):
            r["_src"] = f.split("/")[-1]
            out[r["unit"]] = r
    return out

A = load(sorted(glob.glob("d1/coderA/batch*.csv")))
REP = load(["d1/coderA_repair/batch01_repair.csv"])
for k, v in REP.items():
    v["_repair"] = "1"
    A[k] = v
# Finance Agent разрезан на версии (кодбук 1.2): каждая версия сама проходит правило популяции
if "Finance Agent" in A:
    A["Finance Agent v2"] = A.pop("Finance Agent")
    A["Finance Agent v2"]["unit"] = "Finance Agent v2"
SPLIT_POP = {"Finance Agent": ["Finance Agent v1", "Finance Agent v1.1", "Finance Agent v2"]}
B = load(sorted(glob.glob("d1/coderB/s*.csv")))
POP = json.load(open("population_main.json", encoding="utf-8"))
MANUAL_POP_EXCL = {"ProgramBench (multi-agent)": "слит с ProgramBench",
                   "tau2-bench": "без домена, отчёты разнесены по доменам",
                   "Stanford SHP": "не бенчмарк способностей (датасет предпочтений)"}

# ---------------------------------------------------------------- популяция: пересчёт по уточнённому правилу
# Отчёт = собственный результат бенчмарка в материалах релиза. Компонент агрегата без своего числа — не отчёт.
# Блог и техотчёт одной модели — один релиз.
D3 = list(csv.DictReader(open("d3_matrix.csv", encoding="utf-8")))

def is_component(r):
    return "component of" in (r["variant"] + " " + r["setting"]).lower()

def relkey(r):
    m = re.sub(r"\s*\((blog|technical report|tech report|paper|model card|system card|report)\)\s*$", "",
               r["model"], flags=re.I)
    return (r["lab"], m.strip())

# строки Д3, которые кодировщики опознали как чужие бенчмарки (ошибки сопоставления) — не считаем отчётами
MISMAP = {
    "LongBench": lambda r: True,                       # все строки — LongBench-Chat и MMLongBench-Doc
    "M3Exam": lambda r: r["lab"] == "Alibaba (Qwen)",  # агрегат «Exams» Qwen1.5/2/2.5 без собственного числа
}
def d3_unit(r):
    """Единица наблюдения строки Д3 после сведения (разрез версий, которых не было в канонизации)."""
    u = r["unit"]
    if u == "Finance Agent":
        raw = (r["benchmark_raw"] + " " + r["variant"]).lower()
        if "v2" in raw:
            return "Finance Agent v2"
        if "v1.1" in raw:
            return "Finance Agent v1.1"
        return "Finance Agent v1" if r["release_date"][:10] < "2026-02-10" else "Finance Agent v1.1"
    return u

def is_report(r):
    return not is_component(r) and not (r["unit"] in MISMAP and MISMAP[r["unit"]](r))

with open("d3_matrix_v3.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(D3[0].keys()) + ["unit_v3", "is_report"])
    w.writeheader()
    for r in D3:
        w.writerow({**r, "unit_v3": d3_unit(r), "is_report": int(is_report(r))})

popstat = collections.defaultdict(lambda: {"labs": set(), "rel": set(), "dates": []})
for r in D3:
    if not is_report(r):
        continue
    u = d3_unit(r)
    s = popstat[u]
    s["labs"].add(r["lab"]); s["rel"].add(relkey(r))
    if r["release_date"]:
        s["dates"].append(r["release_date"][:7])

# ---------------------------------------------------------------- причины и вид событий C (по цитатам кодировщиков)
# reason: saturation | flaws | administrative ; kind: successor | version | freeze | withdrawal
C_INFO = {
    "APEX-Agents": ("flaws", "version"), "ARC-AGI-1": ("saturation", "successor"),
    "ArXivMath": ("saturation", "version"), "Arena-Hard": ("administrative", "version"),
    "BFCL v3": ("saturation", "version"), "C-Eval": ("administrative", "freeze"),
    "ChartQA": ("saturation", "successor"), "FrontierMath": ("flaws", "version"),
    "FrontierMath Tier 4": ("flaws", "version"), "GLUE": ("saturation", "successor"),
    "HealthBench": ("saturation", "successor"), "HellaSwag": ("administrative", "freeze"),
    "LAB-Bench": ("saturation", "successor"), "MCPMark": ("flaws", "version"),
    "MLE-bench": ("flaws", "freeze"), "MMLU": ("saturation", "successor"),
    "MMMU": ("flaws", "successor"), "OCRBench": ("saturation", "successor"),
    "OSWorld": ("flaws", "version"), "OSWorld-Verified": ("saturation", "successor"),
    "OmniDocBench": ("administrative", "version"), "SQuAD 1.1": ("saturation", "successor"),
    "SWE-bench": ("flaws", "version"), "SWE-bench Pro": ("flaws", "version"),
    "SWE-bench Verified": ("flaws", "withdrawal"), "StoryCloze": ("flaws", "version"),
    "Terminal-Bench 1.0": ("saturation", "version"), "Terminal-Bench 2.0": ("flaws", "version"),
    "Terminal-Bench 2.1": ("saturation", "version"), "Terminal-Bench 3.0": ("flaws", "version"),
    "Toolathlon": ("flaws", "version"), "Video-MME": ("saturation", "successor"),
    "tau-bench airline": ("flaws", "version"), "tau-bench retail": ("flaws", "version"),
    "tau2-bench airline": ("flaws", "version"), "tau2-bench retail": ("flaws", "version"),
    "tau2-bench telecom": ("saturation", "withdrawal"), "ruMMLU": ("administrative", "withdrawal"),
    "Finance Agent v1": ("saturation", "freeze"),
}
E_TYPE = {"AIME 2024": "i", "AlpacaEval": "ii", "Arena-Hard": "ii", "CritPt": "iii",
          "Humanity's Last Exam": "iii", "SWE-bench Verified": "i (пограничный)", "tau2-bench airline": "iii"}

# ---------------------------------------------------------------- решения при сведении
OVERRIDES = [
    # --- популяция
    dict(id="P1", units=["ruMMLU", "IndoMMLU", "Belebele", "XWinograd", "PAWS-X"],
         set={"stratum": "excluded:population_artifact"},
         rule="кодбук 2.1 (уточнение): компонент агрегата без собственного числа — не отчёт; блог и техотчёт одной модели — один релиз",
         why="Единицы попали в популяцию через строки Qwen «component of Multi-Exam / Multi-Understanding» без собственных результатов. "
             "После исключения таких строк: ruMMLU — 0 отчётов, IndoMMLU — 1 релиз (Qwen2.5), PAWS-X и XWinograd — 1 релиз (Qwen1.5), Belebele — 2 релиза Qwen.",
         src="d3_matrix.csv; заметки кодировщиков A и B (ruMMLU, IndoMMLU)"),
    dict(id="P2", units=["LongBench"], set={"stratum": "excluded:population_artifact"},
         rule="кодбук 2.1", why="Все три строки «LongBench» в Д3 — другие бенчмарки (LongBench-Chat, MMLongBench-Doc). Отчётов о LongBench v1 нет.",
         src="заметка кодировщика A (batch06)"),
    dict(id="P3", units=["M3Exam"], set={"stratum": "excluded:population_artifact"},
         rule="кодбук 2.1 (уточнение)",
         why="Самостоятельный отчёт один (график GPT-4o, 2024-05); у Qwen M3Exam входит только в агрегат «Exams» без своего числа.",
         src="d3_matrix.csv; заметка кодировщика A (batch01)"),
    # --- правило 6a (dev-сплит) для единиц волны 2a, размеченной до уточнений v1.1
    dict(id="S1", units=["CommonsenseQA"],
         set={"ab_date": "2021-12", "ab_system": "KEAR (DeBERTaV3-large + внешнее внимание к знаниям), Microsoft; одиночная модель (ансамбль 93,4)",
              "ab_score": "91.2", "ab_mode": "дообучение; dev-сплит (split=dev), метки теста скрыты", "ab_system_type": "scaffold",
              "ab_source_type": "self", "ab_source": "https://arxiv.org/abs/2112.03254 (v1 2021-12)", "abandoned_date": "",
              "abandoned_evidence": "", "confidence": "medium", "split_dev": "1"},
         rule="кодбук 6a (split=dev при скрытых метках теста)",
         why="Метки теста скрыты, сообщество отчитывается на dev. Первый результат ≥ 90 на dev — KEAR (2021-12). Решение совпадает с кодировщиком B; A размечал без 6a и записал «заброшен».",
         src="кодировщик B (s1/s2), arXiv 2112.03254"),
    dict(id="S2", units=["QuALITY"],
         set={"ab_date": "2024-03", "ab_system": "Claude 3 Opus", "ab_score": "90.5",
              "ab_mode": "1-shot, T=1; сплит не указан (вероятно dev, метки теста скрыты) — split=dev", "ab_system_type": "single",
              "ab_source_type": "self", "ab_source": "https://www-cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_Card_Claude_3.pdf (2024-03-04)",
              "abandoned_date": "", "abandoned_evidence": "", "confidence": "low", "split_dev": "1"},
         rule="кодбук 6a", why="Лаборатории отчитываются вне официального теста (метки скрыты). Сплит в карточке Claude 3 не назван, поэтому уверенность низкая. В строгом режиме событие не засчитывается (лучший на тесте — 88,0).",
         src="заметка кодировщика A (batch02)"),
    dict(id="S3", units=["TriviaQA"],
         set={"ab_date": "2024-07", "ab_system": "Llama 3.1 405B (base)", "ab_score": "91.8",
              "ab_mode": "5-shot, closed-book, Wiki validation (split=dev), EM", "ab_system_type": "single", "ab_source_type": "self",
              "ab_source": "https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/eval_details.md (2024-07-23)",
              "confidence": "medium", "split_dev": "1"},
         rule="кодбук 6a", why="Скрытые тесты: лучший 89,4 (Atlas, 2022-08). Закрытая книга на Wiki validation — де-факто стандарт лабораторий; первый ≥ 90 — Llama 3.1 405B.",
         src="заметка кодировщика A (batch02); eval_details Meta"),
    # --- правило 6a для E(iii): дата = позднейшая из (аудит, первый отказ лаборатории)
    dict(id="E1", units=["Humanity's Last Exam"],
         set={"e_date": "2026-08",
              "e_source": "Аудит: https://arxiv.org/abs/2602.13964 (HLE-Verified, Qwen Team, 2026-02); первый отказ лаборатории: Google, методология Gemini 3.7 Flash (2026-08-13) — отчёт только на 1 811 проверенных задачах"},
         rule="кодбук 6a: дата E(iii) = более поздняя из (публикация аудита, первый отказ лаборатории)",
         why="A поставил дату аудита (2026-02). Qwen в 2026-02 отчитался и по HLE, и по HLE-Verified, то есть HLE не бросил; первым перестал отчитываться Google (2026-08).",
         src="d3_matrix.csv (HLE, HLE-Verified по лабораториям)"),
    # --- C: первое по времени заявление мейнтейнера
    dict(id="C1", units=["HealthBench"],
         set={"c_date": "2026-07", "c_source": "https://deploymentsafety.openai.com/gpt-5-6/gpt-5-6.pdf (GPT-5.6 System Card, 2026-07-09)",
              "c_quote": "We believe HealthBench (now more than a year old) is approaching a noise ceiling for frontier models, and recommend the use of HealthBench Professional for measuring continued progress at the frontier."},
         rule="кодбук 4 (C), 6.1 «первый по времени»",
         why="A датировал C по карточке GPT-6 Astra (2026-09), но та же формулировка впервые появилась в карточке GPT-5.6 (2026-07-09). B не нашёл заявления и закодировал «жив».",
         src="d3/openai_statements.csv"),
    dict(id="C2", units=["BIG-Bench Hard"], set={"c_date": "", "c_source": "", "c_quote": ""},
         rule="кодбук 6a: C через преемника — только при существенном пересечении авторов",
         why="У BBEH из 11 авторов BBH только двое (Tay, Le), ведущие авторы другие. На основное событие не влияет (B 2024-06).",
         src="arXiv 2502.19187 vs 2210.09261"),
    # --- WMDP: метрика многосоставной единицы
    dict(id="M1", units=["WMDP"],
         set={"ab_date": "2026-04", "ab_system": "Grok 4.20 single-agent (xAI)", "ab_score": "90.9",
              "ab_mode": "zero-shot MCQ без инструментов; взвешенная по размеру точность по трём подмножествам (Bio 91 / Chem 90 / Cyber 91)",
              "ab_system_type": "single", "ab_source_type": "self", "ab_source": "https://data.x.ai/2026-04-07-grok-4-20-model-card.pdf",
              "status_cutoff": "saturated", "confidence": "medium"},
         rule="кодбук 3 (основная метрика); 6.2 (дата публичного раскрытия)",
         why="Создатели публикуют только подмножества. A считал общую точность, но не видел Chem/Cyber для Grok 4.20; в карточке 2026-04-07 все три ≥ 0,90, общая — 90,9. B пришёл к той же дате через WMDP-Bio. Результат deep research (90% Bio, 2025-02) — с браузингом, не засчитан.",
         src="d3_matrix.csv (xAI), кодировщик B"),
    # --- ARC: полузакрытый набор вместо публичного оценочного
    dict(id="A1", units=["ARC-AGI-1"],
         set={"ab_date": "2025-12", "ab_system": "GPT-5.2 Pro (ARC-AGI-1 Verified)", "ab_score": "90.5",
              "ab_mode": "полузакрытый оценочный набор, проверено ARC Prize", "ab_system_type": "single", "ab_source_type": "leaderboard",
              "ab_source": "https://openai.com/index/introducing-gpt-5-2/ (2025-12-11)"},
         rule="кодбук 6a не применяется: де-факто стандарт отчётности ARC — проверенный ARC Prize полузакрытый набор, а не публичный оценочный",
         why="Публичный оценочный набор ARC не dev-сплит в смысле 6a: лаборатории отчитываются «ARC-AGI (Verified)». Основное событие не меняется: C и D в 2024-12 (o3-preview, 87,5% при цели 85%).",
         src="заметка кодировщика A (batch11)"),
    dict(id="A2", units=["ARC-AGI-2"],
         set={"ab_date": "2026-07", "ab_system": "Claude Opus 5 (max effort), проверено ARC Prize", "ab_score": "90.42",
              "ab_mode": "полузакрытый набор, pass@2", "ab_system_type": "single", "ab_source_type": "leaderboard",
              "ab_source": "Claude Opus 5 System Card (2026-07-24)"},
         rule="как A1", why="Основное событие — D 2026-04 (GPT-5.5, «ARC-AGI-2 (Verified) 85.0%» по OpenAI). В карточке Gemini 3.5 Flash тот же результат указан как 84,6 — тогда D переносится на 2026-07; это вынесено в проверку устойчивости.",
         src="заметка кодировщика A (batch11)"),
    dict(id="A3", units=["ARC-AGI-3"],
         set={"release": "2026-03", "release_source": "https://arcprize.org/blog/arc-agi-3-launch (2026-03-25); превью 6 игр — 2025-07-17",
              "ab_date": "2026-09", "ab_system": "GPT-6 Astra (high), обвязка Provider Adapter; прогон ARC Prize", "ab_score": "99.9",
              "ab_mode": "полузакрытый набор; стандартная обвязка ARC — 62,7", "ab_system_type": "scaffold", "ab_source_type": "leaderboard",
              "ab_source": "https://arcprize.org/blog/astra (2026-09-03)"},
         rule="кодбук 1.2 (новая версия при замене >10% задач и метрики); A1 для сплита",
         why="Превью 2025-07 (6 игр) и полный бенчмарк 2026-03 (135 сред, метрика RHAE) — разные версии; лаборатории отчитываются по полному. Рождение 2026-03, событие 2026-09 — шесть месяцев жизни. С рождением от превью было бы 14.",
         src="заметка кодировщика A (batch11)"),
    # --- порог для попарного сравнения с экспертами
    dict(id="T1", units=["GDPval"],
         set={"criterion_ab": "B", "threshold": "90", "ab_date": "", "ab_system": "", "ab_score": "", "ab_mode": "",
              "ab_system_type": "", "ab_source_type": "", "ab_source": "", "strict_date": "", "strict_system": "",
              "strict_score": "", "strict_source": "", "status_cutoff": "alive"},
         rule="кодбук 4 (A только при бейзлайне ≥ 85% шкалы)",
         why="Паритет с экспертами (50% побед и ничьих) — не человеческий результат на шкале, а точка сравнения; 50 < 85, поэтому B (90). Лучший к срезу — 84,9 (GPT-5.5, 2026-04): жив. При пороге-паритете смерть была бы 2025-12 (GPT-5.2, 70,9) — это вынесено в проверку устойчивости.",
         src="заметка кодировщика A (batch13)"),
    # --- результат относится к другой версии
    dict(id="V1", units=["FrontierMath Tier 4"],
         set={"ab_date": "", "ab_system": "", "ab_score": "", "ab_mode": "", "ab_system_type": "", "ab_source_type": "",
              "ab_source": "", "strict_date": "", "strict_system": "", "strict_score": "", "strict_source": ""},
         rule="кодбук 1.2", why="97,6% GPT-6 Astra получено на Tier 4 v2 (43 задачи, 38% исправлено или удалено) — другой единице. Лучший результат на v1 — 35,4. Основное событие не меняется: C 2026-05.",
         src="заметка кодировщика A (batch04)"),
    # --- версии в batch12 (размечены отдельным агентом после остановки первого)
    dict(id="V2", units=["Finance Agent v1", "Finance Agent v1.1", "Finance Agent v2"],
         set={}, rule="кодбук 1.2 (новая версия — новая единица) и 2.1 (правило популяции для каждой версии)",
         why="Канонизация свела все версии Finance Agent (Vals AI) в одну единицу. Версии отличаются задачами, рубриками, инструментами и судьёй, и каждая сама проходит правило популяции: v1 — 3 релиза Anthropic, v1.1 — Anthropic и OpenAI, v2 — четыре лаборатории. v1 умер по C в 2026-02: Vals перевёл страницу в архив с формулировкой «performance on this benchmark has saturated, we no longer run this benchmark»; v1.1 и v2 живы.",
         src="d1/coderA/batch12b.csv; https://www.vals.ai/benchmarks/finance_agent_v1; github.com/vals-ai/finance-agent"),
    dict(id="V3", units=["Finance Agent v2"], set={"test_access": "private-heldout"}, rule="единообразие полей",
         why="Тестовые задачи Finance Agent закрыты у Vals AI для всех версий; у v1 и v1.1 стоит private-heldout, у v2 было hidden-labels.",
         src="d1/coderA/batch12b.csv"),
    dict(id="V4", units=["FrontierSWE", "DeepSWE", "FrontierCode"], set={},
         rule="кодбук 1.2 и 2.1",
         why="Размечена версия, о которой отчитывается большинство: FrontierSWE v2, DeepSWE v1.1, FrontierCode v1.1. DeepSWE v1.0 и FrontierCode v1.0 сами правило популяции не проходят (один отчёт или одна лаборатория). У FrontierSWE v1 основная метрика — попарное доминирование моделей, то есть шкала относительно меняющегося набора соперников; по правилу «без потолка» такая версия вне расчёта.",
         src="d1/coderA/batch12b.csv"),
    # --- техническое
    dict(id="X1", units=["MATH"], set={"ab_system_type": "scaffold"}, rule="кодбук 5.1",
         why="rm@256 — отбор из 256 решений моделью вознаграждения, это обвязка, а не одиночная модель.", src="заметка кодировщика A"),
    dict(id="X2", units=["SWE-bench Verified"], set={}, rule="кодбук 4 (E i)",
         why="E 2026-09 (Vals.ai: попытки найти решения в git-истории) — пограничный; оставлен в данных, на основное событие не влияет (C 2026-02).",
         src="заметка кодировщика A"),
    dict(id="X3", units=["ASDiv"], set={}, rule="кодбук 6.1 «первый по времени»",
         why="Расхождение A (2023-05, Faithful CoT + детерминированный решатель) и B (2023-09, GPT-4 CoT). Решатели — штатный протокол ASDiv, результат A допустим и раньше. Берём A.",
         src="кодировщики A и B"),
]

# ---------------------------------------------------------------- применение
for o in OVERRIDES:
    for u in o["units"]:
        if u not in A:
            raise SystemExit(f"нет единицы {u}")
        A[u].update(o["set"])
        A[u].setdefault("recon_flags", "")
        A[u]["recon_flags"] = (A[u].get("recon_flags", "") + " " + o["id"]).strip()

def ym(s):
    y, m = s.split("-"); return int(y) * 12 + int(m) - 1

def months(a, b):
    return ym(b) - ym(a)

rows_out = []
for u, r in sorted(A.items(), key=lambda x: x[0].lower()):
    r = dict(r)
    # --- C: причина и вид
    if r.get("c_date"):
        reason, kind = C_INFO.get(u, ("", ""))
        if not reason:
            raise SystemExit(f"нет классификации C для {u}")
        r["c_reason"], r["c_kind"] = reason, kind
    else:
        r["c_reason"] = r["c_kind"] = ""
    r["e_type"] = E_TYPE.get(u, "") if r.get("e_date") else ""
    if r.get("e_date") and not r["e_type"]:
        raise SystemExit(f"нет типа E для {u}")
    # --- основное событие из компонент
    comps = {"AB": r.get("ab_date", ""), "C": r.get("c_date", ""), "D": r.get("d_date", ""), "E": r.get("e_date", "")}
    if u == "SWE-bench Verified":
        comps["E"] = ""  # пограничное E не участвует (X2); на результат не влияет
    dated = {k: v for k, v in comps.items() if v}
    if r["stratum"] == "main" and dated:
        p = min(dated.values())
        crit = [k for k, v in dated.items() if v == p]
        crit = [r["criterion_ab"] if k == "AB" else k for k in crit]
        r["primary_date"], r["primary_criterion"] = p, "+".join(sorted(crit))
        r["status_cutoff"] = "saturated"
        r["abandoned_date"] = ""
    elif r["stratum"] == "main":
        r["primary_date"], r["primary_criterion"] = "", ""
        r["status_cutoff"] = "abandoned" if r.get("abandoned_date") else "alive"
    else:
        r["primary_date"] = r["primary_criterion"] = ""
        r["status_cutoff"] = "excluded"
    # --- причина смерти
    pc = r["primary_criterion"]
    cause = ""
    if pc:
        if any(x in pc.split("+") for x in ("A", "B", "D")):
            cause = "threshold"
        elif "C" in pc.split("+") and r["c_reason"] == "saturation":
            cause = "declared_saturation"
        elif "C" in pc.split("+") and r["c_reason"] == "flaws" or "E" in pc.split("+"):
            cause = "defects"
        else:
            cause = "administrative"
    r["cause"] = cause
    r["cause3"] = {"threshold": "models", "declared_saturation": "models", "defects": "instrument",
                   "administrative": "instrument", "": ""}[cause]
    if r["status_cutoff"] == "abandoned":
        r["cause3"] = "abandoned"
    # --- время
    if r["stratum"] == "main":
        end = r["primary_date"] or r.get("abandoned_date") or CUTOFF
        r["T_months"] = months(r["release"], end)
        r["event"] = 1 if r["primary_date"] else 0
        r["outcome_cr"] = 1 if r["cause3"] == "models" else 2 if r["cause3"] == "instrument" else 3 if r["status_cutoff"] == "abandoned" else 0
        r["born_saturated_calc"] = 1 if (r["primary_date"] and r["T_months"] == 0) else 0
        # строгий режим: AB заменяется строгим событием, C/D/E остаются
        sc = {k: v for k, v in {"S": r.get("strict_date", ""), "C": r.get("c_date", ""), "D": r.get("d_date", ""),
                                 "E": comps["E"]}.items() if v}
        if sc:
            r["strict_primary_date"] = min(sc.values())
            r["T_strict"] = months(r["release"], r["strict_primary_date"]); r["event_strict"] = 1
        else:
            r["strict_primary_date"] = ""
            r["T_strict"] = months(r["release"], r.get("abandoned_date") or CUTOFF); r["event_strict"] = 0
        # замена версией — не смерть, а цензурирование на дату замены
        if r["c_kind"] in ("successor", "version") and "C" in r["primary_criterion"].split("+") and len(r["primary_criterion"].split("+")) == 1:
            r["T_replacement_censored"] = months(r["release"], r["c_date"]); r["event_replacement_censored"] = 0
        else:
            r["T_replacement_censored"] = r["T_months"]; r["event_replacement_censored"] = r["event"]
        # dev-сплит не засчитывается: события, держащиеся только на dev, превращаются в цензурирование
        dev = r.get("split_dev") == "1" or "split=dev" in (r.get("ab_mode", "") + r.get("ab_system", "")).lower()
        r["ab_split_dev"] = 1 if dev and r.get("ab_date") else 0
        y = int(r["release"][:4])
        r["cohort"] = "≤2019" if y <= 2019 else "2020–2022" if y <= 2022 else str(y)
        r["birth_year"] = y
    # --- популяция по уточнённому правилу
    s = popstat.get(u, {"labs": set(), "rel": set(), "dates": []})
    r["n_labs_v3"] = len(s["labs"]); r["n_releases_v3"] = len(s["rel"])
    r["first_report"] = min(s["dates"]) if s["dates"] else ""
    r["last_report"] = max(s["dates"]) if s["dates"] else ""
    r["entry_months"] = months(r["release"], r["first_report"]) if (r["first_report"] and r.get("release")) else ""
    rows_out.append(r)

# ---------------------------------------------------------------- нераазмеченные единицы batch12
coded = {r["unit"] for r in rows_out}
for u, p in POP.items():
    if u in coded or u in MANUAL_POP_EXCL or any(x in coded for x in SPLIT_POP.get(u, [])):
        continue
    s = popstat.get(u, {"labs": set(), "rel": set(), "dates": []})
    rows_out.append({"unit": u, "stratum": "pending:batch12", "status_cutoff": "pending",
                     "n_labs_v3": len(s["labs"]), "n_releases_v3": len(s["rel"]),
                     "first_report": min(s["dates"]) if s["dates"] else "", "last_report": max(s["dates"]) if s["dates"] else "",
                     "notes": "Не размечено: агент batch12 остановлен по решению пользователя; ждёт решения о перезапуске."})

COLS = ["unit", "stratum", "exclusion_reason", "canonical_name", "cohort", "birth_year", "release", "release_source",
        "metric", "direction", "scale_max", "human_baseline", "human_type", "criterion_ab", "threshold",
        "ab_date", "ab_system", "ab_score", "ab_mode", "ab_system_type", "ab_source_type", "ab_source", "ab_split_dev",
        "strict_date", "strict_system", "strict_score", "strict_source",
        "c_date", "c_reason", "c_kind", "c_source", "c_quote", "d_date", "d_source", "e_date", "e_type", "e_source", "e_quote",
        "abandoned_date", "abandoned_evidence", "primary_date", "primary_criterion", "cause", "cause3", "status_cutoff",
        "T_months", "event", "outcome_cr", "T_strict", "event_strict", "strict_primary_date",
        "T_replacement_censored", "event_replacement_censored",
        "born_saturated", "born_saturated_calc", "date_upper_bound", "reissue",
        "first_report", "last_report", "entry_months", "n_labs_v3", "n_releases_v3",
        "score_at_release", "score_at_release_system", "n_items", "test_access", "curation", "format", "modality", "class",
        "lineage", "best_by_cutoff", "confidence", "recon_flags", "notes"]
with open("halflife_dataset_v3.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
    w.writeheader()
    for r in sorted(rows_out, key=lambda x: x["unit"].lower()):
        w.writerow(r)

with open("decisions_log_v3.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["id", "units", "rule", "decision", "source"])
    for o in OVERRIDES:
        w.writerow([o["id"], "; ".join(o["units"]), o["rule"], o["why"], o["src"]])

# ---------------------------------------------------------------- согласие кодировщиков (до арбитража, на исходных строках)
A0 = load(sorted(glob.glob("d1/coderA/batch*.csv")))
for k, v in REP.items():
    A0[k] = v
common = sorted(set(A0) & set(B))
cats = ["saturated", "alive", "abandoned", "excluded"]
pairs = [(A0[u]["status_cutoff"], B[u]["status_cutoff"]) for u in common]
agree = sum(a == b for a, b in pairs)
n = len(pairs)
pa = collections.Counter(a for a, _ in pairs); pb = collections.Counter(b for _, b in pairs)
pe = sum(pa[c] * pb[c] for c in cats) / n / n
kappa = (agree / n - pe) / (1 - pe)
dd = []
for u in common:
    a, b = A0[u]["primary_date"], B[u]["primary_date"]
    if a and b:
        dd.append(abs(months(a, b)))
rb = [A0[u]["release"] == B[u]["release"] for u in common if A0[u]["release"] and B[u]["release"]]
json.dump({"n_double_coded": n, "status_agreement": agree, "kappa_status": round(kappa, 3),
           "release_month_equal": f"{sum(rb)}/{len(rb)}",
           "primary_date_pairs": len(dd), "primary_date_abs_diff_median": sorted(dd)[len(dd) // 2] if dd else None,
           "primary_date_within_1m": round(sum(x <= 1 for x in dd) / len(dd), 3) if dd else None,
           "primary_date_max_diff": max(dd) if dd else None,
           "disagreements": [{"unit": u, "A": A0[u]["status_cutoff"], "B": B[u]["status_cutoff"]} for u in common
                             if A0[u]["status_cutoff"] != B[u]["status_cutoff"]]},
          open("agreement_v3.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

st = collections.Counter(r["status_cutoff"] for r in rows_out)
print("rows:", len(rows_out), dict(st))
main = [r for r in rows_out if r["stratum"] == "main"]
print("main:", len(main), collections.Counter(r["status_cutoff"] for r in main))
print("cause:", collections.Counter(r["cause"] for r in main if r["cause"]))
print("cause3:", collections.Counter(r["cause3"] for r in main if r["cause3"]))
print(open("agreement_v3.json").read()[:600])
