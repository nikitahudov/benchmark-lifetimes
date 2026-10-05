# -*- coding: utf-8 -*-
"""
Популяция бенчмарков по правилу кодбука v3:
публичный бенчмарк, о котором отчитались >=2 лаборатории или одна лаборатория в >=3 релизах.
Читает d3/*_benchmarks.csv, канонизирует (canon.py), выделяет единицы (версии, годы, домены),
классифицирует по страте и пишет population_units.json + d3_matrix.csv.
"""
import csv, glob, json, collections, re, sys
sys.path.insert(0, ".")
from canon import canon, version_tag, norm

STRATUM = {
    # компоненты наборов и датасеты до 2016 года — вне основного расчёта
    **{k: "excluded:component" for k in ["RTE", "CB", "WiC", "MultiRC", "ReCoRD"]},
    **{k: "excluded:pre2016" for k in ["WSC", "COPA"]},
    **{k: "excluded:suite" for k in ["BIG-bench", "AA Intelligence Index"]},
    **{k: "excluded:no_ceiling" for k in ["LMArena", "WebDev Arena", "Codeforces", "GDPval-AA", "AA-Briefcase",
                                          "Vending-Bench 2", "Vending-Bench", "Vending-Bench Arena",
                                          "Creative Writing", "EQ-Bench", "Petri"]},
    **{k: "excluded:not_benchmark" for k in ["WildChat", "RewardBench", "ProteinGym"]},
    **{k: "excluded:speech_translation" for k in ["LibriSpeech", "VoxPopuli", "FLEURS", "FLORES", "WMT24++",
                                                  "Full-Duplex-Bench", "VATEX"]},
    **{k: "excluded:exam_battery" for k in ["LSAT", "SAT Math", "AP exams", "AMC"]},
    **{k: "excluded:internal" for k in ["Natural2Code", "Monorepo-Bench", "KernelGen", "CursorBench"]},
    # поведенческие оценки (отказы, предвзятость, честность, безопасность агентов) — отдельный слой
    **{k: "propensity" for k in ["BBQ", "XSTest", "StrongREJECT", "RealToxicityPrompts", "WinoGender",
                                 "CrowS-Pairs", "MASK", "AgentHarm", "AbstentionBench", "SHADE-Arena",
                                 "AgentDojo"]},
}

# какие семейства режем на единицы и как
SPLIT = {"AIME", "HMMT", "Terminal-Bench", "tau-bench", "tau2-bench", "tau3-bench", "BFCL"}
MERGE = {"GPQA": "GPQA Diamond"}   # все варианты GPQA → единица GPQA Diamond (варианты в notes)

LABS_FRONTIER = {"OpenAI", "Anthropic", "Google", "Meta", "xAI", "DeepSeek", "Alibaba (Qwen)",
                 "Moonshot AI (Kimi)", "Mistral"}

def unit_of(fam, raw, variant, date):
    if fam in MERGE:
        return MERGE[fam]
    if fam in SPLIT:
        return version_tag(fam, raw, variant, date)
    if fam == "IMO":
        y = re.search(r"20(2[4-6])", raw + " " + variant)
        return f"IMO 20{y.group(1)}" if y else "IMO"
    if fam == "USAMO":
        y = re.search(r"20(2[4-6])", raw + " " + variant)
        return f"USAMO 20{y.group(1)}" if y else "USAMO"
    if fam == "CNMO":
        return "CNMO 2024"
    return fam

def num(s):
    s = (s or "").strip().replace(",", ".")
    m = re.match(r"^[~≈<>]?\s*(-?\d+(?:\.\d+)?)\s*%?", s)
    return float(m.group(1)) if m else None

def main():
    rows = []
    for f in sorted(glob.glob("d3/*_benchmarks.csv")):
        for r in csv.DictReader(open(f, encoding="utf-8-sig")):
            fam = canon(r["benchmark_raw"], r["variant"])
            r["family"] = fam
            r["unit"] = unit_of(fam, r["benchmark_raw"], r["variant"], r["release_date"]) if not fam.startswith("?") else fam
            r["score_num"] = num(r["score"])
            rows.append(r)
    # матрица для публикации
    with open("d3_matrix.csv", "w", newline="", encoding="utf-8") as fo:
        w = csv.writer(fo)
        w.writerow(["lab", "model", "release_date", "family", "unit", "benchmark_raw", "variant", "setting",
                    "score", "table_location", "source_type", "source_url"])
        for r in rows:
            w.writerow([r["lab"], r["model"], r["release_date"], r["family"], r["unit"], r["benchmark_raw"],
                        r["variant"], r["setting"], r["score"], r["table_location"], r["source_type"], r["source_url"]])
    # агрегаты по семействам (правило популяции считаем на уровне семейства)
    fam_stats = collections.defaultdict(lambda: {"labs": set(), "rel": set()})
    for r in rows:
        fam_stats[r["family"]]["labs"].add(r["lab"])
        fam_stats[r["family"]]["rel"].add((r["lab"], r["model"]))
    eligible = {f for f, s in fam_stats.items()
                if (len(s["labs"]) >= 2 or len(s["rel"]) >= 3) and not f.startswith("?")}
    units = collections.defaultdict(lambda: {"family": "", "labs": set(), "rel": set(), "dates": [],
                                             "raw": collections.Counter(), "reports": []})
    for r in rows:
        if r["family"] not in eligible:
            continue
        u = units[r["unit"]]
        u["family"] = r["family"]
        u["labs"].add(r["lab"]); u["rel"].add((r["lab"], r["model"]))
        if r["release_date"]:
            u["dates"].append(r["release_date"][:10])
        u["raw"][r["benchmark_raw"].strip()] += 1
        u["reports"].append({"date": r["release_date"][:10], "lab": r["lab"], "model": r["model"][:60],
                             "raw": r["benchmark_raw"][:60], "variant": r["variant"][:40],
                             "setting": r["setting"][:60], "score": r["score"][:20],
                             "loc": r["table_location"], "src": r["source_type"]})
    out = {}
    for name, u in units.items():
        fam = u["family"]
        out[name] = {
            "unit": name, "family": fam,
            "stratum": STRATUM.get(fam, "main"),
            "n_labs": len(u["labs"]), "labs": sorted(u["labs"]), "n_releases": len(u["rel"]),
            "first_report": min(u["dates"]) if u["dates"] else "", "last_report": max(u["dates"]) if u["dates"] else "",
            "raw_names": [k for k, _ in u["raw"].most_common(10)],
            "reports": sorted(u["reports"], key=lambda x: x["date"]),
        }
    json.dump(out, open("population_units.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    c = collections.Counter(v["stratum"] for v in out.values())
    print("eligible families:", len(eligible), "units:", len(out), dict(c))

if __name__ == "__main__":
    main()
