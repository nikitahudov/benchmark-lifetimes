# -*- coding: utf-8 -*-
"""
Канонизация названий бенчмарков из карточек моделей (Д3).
Правила применяются по порядку к строке "benchmark_raw | variant" в нижнем регистре.
Первое сработавшее правило задаёт каноническое имя. Если не сработало ни одно —
ключом служит нормализованное сырое имя (так неизвестные бенчмарки всё равно группируются).
"""
import re

def norm(s: str) -> str:
    s = (s or "").lower()
    s = s.replace("τ", "tau").replace("²", "2").replace("³", "3")
    s = re.sub(r"[‐-―\-_/]+", " ", s)
    s = re.sub(r"[()\[\],:;\.\*\+®™'’`\"]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def year_of(text):
    m = re.search(r"(?:20)?(2[3-6])\b", text)
    return int("20" + m.group(1)) if m else None

# (regex, canonical) — порядок важен: от частного к общему
RULES = [
    # --- знания ---
    (r"\bmmlu ?redux", "MMLU-Redux"),
    (r"\bmmlu ?prox", "MMLU-ProX"),
    (r"\bmmlu ?pro\b", "MMLU-Pro"),
    (r"global ?mmlu|\bgmmlu\b", "Global-MMLU"),
    (r"\bmmmlu\b|multilingual mmlu", "MMMLU"),
    (r"\bcmmlu\b", "CMMLU"),
    (r"\bkmmlu\b", "KMMLU"),
    (r"\bmmlu\b", "MMLU"),
    (r"supergpqa", "SuperGPQA"),
    (r"\bgpqa\b", "GPQA"),
    (r"hle ?verified", "HLE-Verified"),
    (r"humanity s last exam|humanitys last exam|\bhle\b", "Humanity's Last Exam"),
    (r"simpleqa ?verified", "SimpleQA Verified"),
    (r"chinese ?simpleqa", "Chinese SimpleQA"),
    (r"simpleqa", "SimpleQA"),
    (r"\btriviaqa\b|trivia qa", "TriviaQA"),
    (r"natural ?questions|\bnq\b", "Natural Questions"),
    (r"web ?questions", "WebQuestions"),
    (r"squad ?(v)?2|squad 2 0", "SQuAD 2.0"),
    (r"\bsquad\b", "SQuAD 1.1"),
    (r"\bquac\b", "QuAC"),
    (r"\bcoqa\b", "CoQA"),
    (r"\bdrop\b", "DROP"),
    (r"\brace\b", "RACE"),
    (r"\bboolq\b", "BoolQ"),
    (r"\bpiqa\b", "PIQA"),
    (r"\bsiqa\b|social ?iqa", "Social IQa"),
    (r"hellaswag", "HellaSwag"),
    (r"winogrande", "WinoGrande"),
    (r"arc ?agi ?3", "ARC-AGI-3"),
    (r"arc ?agi ?2", "ARC-AGI-2"),
    (r"arc ?agi", "ARC-AGI-1"),
    (r"\barc ?c\b|arc challenge|ai2 reasoning challenge \(?challenge|arc \(challenge", "ARC-Challenge"),
    (r"\barc ?e\b|arc easy", "ARC-Easy"),
    (r"openbook ?qa|open book qa", "OpenBookQA"),
    (r"commonsense ?qa|\bcsqa\b", "CommonsenseQA"),
    (r"lambada", "LAMBADA"),
    (r"story ?cloze", "StoryCloze"),
    (r"\bcopa\b", "COPA"),
    (r"\banli\b", "ANLI"),
    (r"agi ?eval", "AGIEval"),
    (r"\bc ?eval\b", "C-Eval"),
    (r"truthful ?qa", "TruthfulQA"),
    (r"\bbbq\b", "BBQ"),
    (r"superglue|super glue", "SuperGLUE"),
    (r"\bglue\b", "GLUE"),
    (r"\b(mnli|qqp|qnli|sst ?2|cola|sts ?b|mrpc|wnli)\b", "GLUE"),
    (r"big ?bench ?extra ?hard|\bbbeh\b", "BBEH"),
    (r"big ?bench ?hard|\bbbh\b", "BIG-Bench Hard"),
    (r"big ?bench", "BIG-bench"),
    (r"\bmusr\b", "MuSR"),
    (r"zebra ?logic", "ZebraLogic"),
    (r"live ?bench", "LiveBench"),
    # --- математика ---
    (r"frontier ?math.*tier ?4|frontier ?math t4", "FrontierMath Tier 4"),
    (r"frontier ?math", "FrontierMath"),
    (r"math ?500", "MATH-500"),
    (r"^(math|math level 5|math hendrycks|hendrycks math|math cot|math 4 shot|math 0 shot)( .*)?$", "MATH"),
    (r"gsm ?8k ?platinum", "GSM8K-Platinum"),
    (r"gsm ?8k|gsm8k", "GSM8K"),
    (r"\bmgsm\b", "MGSM"),
    (r"math ?vista", "MathVista"),
    (r"math ?verse", "MathVerse"),
    (r"math ?vision|math v\b", "MATH-Vision"),
    (r"omni ?math", "Omni-MATH"),
    (r"olympiad ?bench", "OlympiadBench"),
    (r"poly ?math", "PolyMath"),
    (r"\bcnmo\b", "CNMO"),
    (r"usamo", "USAMO"),
    (r"imo ?answer ?bench", "IMO-AnswerBench"),
    (r"hmmt", "HMMT"),
    (r"\bbrumo\b", "BRUMO"),
    (r"\baime\b|aime ?2[3-6]|aime 20", "AIME"),
    # --- код ---
    (r"livecodebench ?pro|live code bench pro", "LiveCodeBench Pro"),
    (r"live ?code ?bench|\blcb\b", "LiveCodeBench"),
    (r"humaneval ?\+|humaneval plus|humanevalplus", "HumanEval+"),
    (r"mbpp ?\+|mbpp plus|mbppplus", "MBPP+"),
    (r"evalplus", "EvalPlus"),
    (r"multi ?pl ?e\b|multipl e|multi lingual human ?eval|humaneval ?x|humaneval multilingual", "MultiPL-E"),
    (r"human ?eval", "HumanEval"),
    (r"\bmbpp\b", "MBPP"),
    (r"swe ?bench ?verified|swe bench v\b", "SWE-bench Verified"),
    (r"swe ?bench ?pro", "SWE-bench Pro"),
    (r"swe ?bench ?multiling", "SWE-bench Multilingual"),
    (r"swe ?bench ?multimodal", "SWE-bench Multimodal"),
    (r"swe ?bench ?lite", "SWE-bench Lite"),
    (r"swe ?lancer", "SWE-Lancer"),
    (r"swe ?bench", "SWE-bench"),
    (r"deep ?swe", "DeepSWE"),
    (r"aider", "Aider Polyglot"),
    (r"terminal ?bench ?science", "Terminal-Bench Science"),
    (r"terminal ?bench|terminalbench", "Terminal-Bench"),
    (r"codeforces", "Codeforces"),
    (r"sci ?code\b", "SciCode"),
    (r"big ?code ?bench", "BigCodeBench"),
    (r"cruxeval", "CRUXEval"),
    (r"\bds ?1000\b", "DS-1000"),
    (r"bird ?sql|\bbird\b", "BIRD-SQL"),
    (r"\bspider\b", "Spider"),
    (r"nl2repo", "NL2Repo"),
    (r"full ?stack ?bench", "FullStackBench"),
    # --- агенты и инструменты ---
    (r"osworld ?2 ?1", "OSWorld 2.1"),
    (r"osworld ?2", "OSWorld 2.0"),
    (r"osworld ?verified|osworld.*verified", "OSWorld-Verified"),
    (r"osworld", "OSWorld"),
    (r"tau ?3|tau ?bench ?3|tau3", "tau3-bench"),
    (r"tau ?voice|tau voice", "tau-voice"),
    (r"tau ?2|tau2", "tau2-bench"),
    (r"tau ?bench|tau1|taubench|\btau\b", "tau-bench"),
    (r"browse ?comp ?zh", "BrowseComp-ZH"),
    (r"browse ?comp ?plus", "BrowseComp-Plus"),
    (r"browse ?comp", "BrowseComp"),
    (r"\bgaia\b", "GAIA"),
    (r"visual ?web ?arena", "VisualWebArena"),
    (r"web ?arena", "WebArena"),
    (r"mind ?2 ?web", "Mind2Web"),
    (r"web ?voyager", "WebVoyager"),
    (r"mle ?bench", "MLE-bench"),
    (r"paper ?bench", "PaperBench"),
    (r"re ?bench|research engineering benchmark", "RE-Bench"),
    (r"mcp ?atlas", "MCP Atlas"),
    (r"mcp ?universe", "MCP-Universe"),
    (r"mcp ?mark", "MCPMark"),
    (r"toolathlon", "Toolathlon"),
    (r"\bbfcl\b|berkeley function", "BFCL"),
    (r"finance ?agent", "Finance Agent"),
    (r"vending ?bench ?arena", "Vending-Bench Arena"),
    (r"vending ?bench ?2", "Vending-Bench 2"),
    (r"vending ?bench", "Vending-Bench"),
    (r"gdpval ?aa", "GDPval-AA"),
    (r"gdpval", "GDPval"),
    (r"screen ?spot ?pro", "ScreenSpot-Pro"),
    (r"screen ?spot", "ScreenSpot"),
    (r"android ?world", "AndroidWorld"),
    (r"windows ?agent ?arena", "WindowsAgentArena"),
    (r"cy ?bench\b|cybench", "Cybench"),
    (r"cyber ?gym", "CyberGym"),
    (r"cve ?bench", "CVE-Bench"),
    (r"intercode", "InterCode-CTF"),
    (r"agent ?harm", "AgentHarm"),
    (r"agent ?dojo", "AgentDojo"),
    (r"harm ?bench", "HarmBench"),
    (r"strong ?reject", "StrongREJECT"),
    (r"xs ?test", "XSTest"),
    (r"\bwmdp\b", "WMDP"),
    (r"\bmask\b", "MASK"),
    (r"shade ?arena", "SHADE-Arena"),
    (r"lab ?bench", "LAB-Bench"),
    (r"virology capabilities|\bvct\b", "VCT"),
    (r"biolp", "BioLP-bench"),
    # --- мультимодальность ---
    (r"mmmu ?pro", "MMMU-Pro"),
    (r"video ?mmmu", "VideoMMMU"),
    (r"\bmmmu\b", "MMMU"),
    (r"chart ?qa ?pro", "ChartQA Pro"),
    (r"chart ?qa", "ChartQA"),
    (r"doc ?vqa", "DocVQA"),
    (r"info ?vqa|infographic ?vqa", "InfoVQA"),
    (r"\bai2d\b", "AI2D"),
    (r"text ?vqa", "TextVQA"),
    (r"ocr ?bench", "OCRBench"),
    (r"mm ?bench", "MMBench"),
    (r"mm ?star", "MMStar"),
    (r"real ?world ?qa", "RealWorldQA"),
    (r"char ?xiv", "CharXiv"),
    (r"video ?mme", "Video-MME"),
    (r"mv ?bench", "MVBench"),
    (r"ego ?schema", "EgoSchema"),
    (r"lv ?bench", "LVBench"),
    (r"\berqa\b", "ERQA"),
    (r"\bvqa ?v?2\b", "VQAv2"),
    (r"perception ?test", "Perception Test"),
    # --- длинный контекст ---
    (r"mrcr", "MRCR"),
    (r"\bruler\b", "RULER"),
    (r"infinite ?bench|\binf ?bench", "InfiniteBench"),
    (r"long ?bench ?v2", "LongBench v2"),
    (r"long ?bench", "LongBench"),
    (r"graph ?walks", "Graphwalks"),
    (r"\bloft\b", "LOFT"),
    (r"nolima", "NoLiMa"),
    (r"fiction ?live ?bench", "Fiction.LiveBench"),
    # --- следование инструкциям, чат ---
    (r"\bifbench\b|\bif bench\b", "IFBench"),
    (r"ifeval|if ?eval", "IFEval"),
    (r"multi ?challenge", "MultiChallenge"),
    (r"arena ?hard", "Arena-Hard"),
    (r"mt ?bench", "MT-Bench"),
    (r"alpaca ?eval", "AlpacaEval"),
    (r"wild ?bench", "WildBench"),
    (r"lm ?arena|chatbot ?arena|lmsys", "LMArena"),
    (r"creative ?writing", "Creative Writing"),
    (r"eq ?bench", "EQ-Bench"),
    (r"multi ?if\b", "Multi-IF"),
    (r"align ?bench", "AlignBench"),
    # --- добавлено во втором проходе ---
    (r"\bapps\b", "APPS"),
    (r"zero ?bench", "ZeroBench"),
    (r"exploit ?gym", "ExploitGym"),
    (r"exploit ?bench", "ExploitBench"),
    (r"multi agent program ?bench", "ProgramBench (multi-agent)"),
    (r"program ?bench", "ProgramBench"),
    (r"gdp ?pdf", "GDP-PDF"),
    (r"agents? ?s? last exam|agent s last exam", "Agent's Last Exam"),
    (r"real ?toxicity ?prompts", "RealToxicityPrompts"),
    (r"simple ?vqa", "SimpleVQA"),
    (r"baby ?vision", "BabyVision"),
    (r"winogender", "WinoGender"),
    (r"xwinograd", "XWinograd"),
    (r"winograd|\bwsc\b|wsc273", "WSC"),
    (r"needle in a haystack|\bniah\b", "Needle-in-a-Haystack"),
    (r"job ?bench", "JobBench"),
    (r"crit ?pt", "CritPt"),
    (r"omni ?doc ?bench", "OmniDocBench"),
    (r"apex ?agents", "APEX-Agents"),
    (r"\bapex\b", "APEX"),
    (r"\blsat\b", "LSAT"),
    (r"frontier ?swe", "FrontierSWE"),
    (r"frontier ?code", "FrontierCode"),
    (r"protocol ?qa", "ProtocolQA"),
    (r"bio ?mystery ?bench", "BioMysteryBench"),
    (r"legal agent benchmark|harvey ?lab", "Harvey LAB"),
    (r"\brte\b", "RTE"),
    (r"cy ?scenario ?bench", "CyScenarioBench"),
    (r"wild ?chat", "WildChat"),
    (r"\bxcopa\b", "XCOPA"),
    (r"swe ?verified", "SWE-bench Verified"),
    (r"\bcmath\b", "CMath"),
    (r"fleurs", "FLEURS"),
    (r"m3 ?exam", "M3Exam"),
    (r"make ?me ?say", "MakeMeSay"),
    (r"make ?me ?pay", "MakeMePay"),
    (r"bix ?bench", "BixBench"),
    (r"oj ?bench", "OJBench"),
    (r"aa ?lcr", "AA-LCR"),
    (r"wide ?search", "WideSearch"),
    (r"^cb$|commitmentbank", "CB"),
    (r"^wic\b", "WiC"),
    (r"multi ?rc", "MultiRC"),
    (r"^record\b", "ReCoRD"),
    (r"\bmmvu\b", "MMVU"),
    (r"cluewsc", "CLUEWSC"),
    (r"\bframes\b", "FRAMES"),
    (r"webdev ?arena", "WebDev Arena"),
    (r"crows ?pairs", "CrowS-Pairs"),
    (r"^quality\b", "QuALITY"),
    (r"tydi ?qa", "TyDiQA"),
    (r"fact ?score", "FActScore"),
    (r"auto ?logi", "AutoLogi"),
    (r"seal ?0|seal ?qa", "SealQA"),
    (r"world ?vqa", "WorldVQA"),
    (r"mls ?bench", "MLS-Bench"),
    (r"\bimo ?20(2[4-6])\b", "IMO"),
    (r"aa ?briefcase", "AA-Briefcase"),
    (r"cursor ?bench", "CursorBench"),
    (r"\basdiv\b", "ASDiv"),
    (r"\bsvamp\b", "SVAMP"),
    (r"\bmawps\b", "MAWPS"),
    (r"stanford ?shp", "Stanford SHP"),
    (r"sat ?math", "SAT Math"),
    (r"^ap ", "AP exams"),
    (r"librispeech", "LibriSpeech"),
    (r"voxpopuli", "VoxPopuli"),
    (r"abstention ?bench", "AbstentionBench"),
    (r"\bamc ?1[02]\b|\bamc\b", "AMC"),
    (r"^v ?bench|v\* ?bench|vstar", "V*Bench"),
    (r"spreadsheet ?bench", "SpreadsheetBench"),
    (r"reward ?bench", "RewardBench"),
    (r"\bblink\b", "BLINK"),
    (r"swe ?marathon", "SWE-Marathon"),
    (r"artificial analysis intelligence index", "AA Intelligence Index"),
    (r"blueprint ?bench", "Blueprint Bench"),
    (r"claw ?eval", "Claw-Eval"),
    (r"full ?duplex ?bench", "Full-Duplex-Bench"),
    (r"perception ?bench", "PerceptionBench"),
    (r"paws ?x", "PAWS-X"),
    (r"theorem ?qa", "TheoremQA"),
    (r"indo ?mmlu", "IndoMMLU"),
    (r"ru ?mmlu", "ruMMLU"),
    (r"writing ?bench", "WritingBench"),
    (r"\bmlvu\b", "MLVU"),
    (r"natural ?2 ?code", "Natural2Code"),
    (r"\bvatex\b", "VATEX"),
    (r"activity ?net ?qa", "ActivityNet-QA"),
    (r"eclektic", "ECLeKTic"),
    (r"odinw", "ODinW"),
    (r"lingo ?qa", "LingoQA"),
    (r"vita ?bench", "VitaBench"),
    (r"arxiv ?math", "ArXivMath"),
    (r"\bpetri\b", "Petri"),
    (r"linux ?arena|linux ?bench", "LinuxArena"),
    (r"biosecbench|bio sec bench", "BioSecBench"),
    (r"sec ?bench", "SEC-Bench"),
    (r"monorepo ?bench", "Monorepo-Bench"),
    (r"kernel ?gen", "KernelGen"),
    (r"post ?train ?bench", "PostTrainBench"),
    (r"proteingym", "ProteinGym"),
    (r"lifesci ?bench", "LifeSciBench"),
    # --- многоязычность, прочее ---
    (r"\binclude\b", "INCLUDE"),
    (r"\bmilu\b", "MILU"),
    (r"belebele", "Belebele"),
    (r"flores", "FLORES"),
    (r"wmt ?24", "WMT24++"),
    (r"health ?bench ?professional", "HealthBench Professional"),
    (r"health ?bench", "HealthBench"),
    (r"med ?qa|usmle", "MedQA"),
    (r"medmcqa", "MedMCQA"),
    (r"pubmed ?qa", "PubMedQA"),
    (r"medxpert", "MedXpertQA"),
    (r"facts ?grounding", "FACTS Grounding"),
    (r"aa ?omniscience", "AA-Omniscience"),
    (r"officeqa", "OfficeQA"),
    (r"deep ?search ?qa", "DeepSearchQA"),
    (r"automation ?bench", "AutomationBench"),
    (r"legal ?bench", "LegalBench"),
]
RULES = [(re.compile(p), c) for p, c in RULES]

def canon(raw: str, variant: str = "") -> str:
    """Каноническое имя семейства (без версии/года)."""
    n = norm(raw)
    for rx, c in RULES:
        if rx.search(n):
            return c
    return "?" + n

def version_tag(canon_name: str, raw: str, variant: str, release_date: str) -> str:
    """Уточнение единицы внутри семейства: год экзамена, версия, домен."""
    t = norm(raw + " " + (variant or ""))
    d = (release_date or "")[:7]
    if canon_name == "AIME":
        m = re.search(r"aime ?(?:20)?(2[3-6])\b|aime(2[3-6])\b", t)
        y = None
        if m:
            y = int("20" + (m.group(1) or m.group(2)))
        else:
            m2 = re.search(r"\b20(2[3-6])\b", t)
            if m2: y = int("20" + m2.group(1))
        if y is None:
            # год по дате отчёта: экзамен проходит в феврале
            if d >= "2026-02": y = 2026
            elif d >= "2025-02": y = 2025
            elif d >= "2024-02": y = 2024
            else: y = 2023
        return f"AIME {y}"
    if canon_name == "HMMT":
        m = re.search(r"(feb|nov)", t)
        mm = m.group(1) if m else "feb"
        y = re.search(r"(?:20)?(2[3-6])\b", t)
        yy = int("20" + y.group(1)) if y else None
        return f"HMMT {mm.capitalize()} {yy}" if yy else "HMMT"
    if canon_name == "Terminal-Bench":
        m = re.search(r"\b([1-4]) ?([0-9])\b", t.replace("terminal bench", ""))
        if m:
            return f"Terminal-Bench {m.group(1)}.{m.group(2)}"
        m = re.search(r"terminal ?bench ?([1-4])\b", t)
        if m:
            return f"Terminal-Bench {m.group(1)}.0"
        return "Terminal-Bench 1.0" if d < "2025-11" else "Terminal-Bench 2.0"
    if canon_name in ("tau-bench", "tau2-bench", "tau3-bench"):
        for dom in ("retail", "airline", "telecom", "banking"):
            if dom in t:
                return f"{canon_name} {dom}"
        return canon_name
    if canon_name == "GPQA":
        if "diamond" in t: return "GPQA Diamond"
        if "main" in t: return "GPQA main"
        if "extended" in t: return "GPQA extended"
        return "GPQA (unspecified)"
    if canon_name == "BFCL":
        m = re.search(r"\bv ?([1-4])\b", t)
        return f"BFCL v{m.group(1)}" if m else "BFCL"
    return canon_name
