# -*- coding: utf-8 -*-
"""Самодостаточный HTML-превью статьи: markdown → HTML, картинки встроены base64, вёрстка близка к Хабру."""
import base64, io, re, sys, os
import markdown
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "article", "HABR_ARTICLE_v3.md")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "article", "HABR_ARTICLE_v3_preview.html")
BASE = os.path.dirname(SRC)

md = open(SRC, encoding="utf-8").read()
md = re.sub(r"<!--.*?-->", "", md, flags=re.S)          # служебные комментарии не показываем

def embed(m):
    alt, path = m.group(1), m.group(2)
    full = os.path.join(BASE, path)
    im = Image.open(full).convert("RGB")
    if im.width > 1600:
        im = im.resize((1600, round(im.height * 1600 / im.width)), Image.LANCZOS)
    buf = io.BytesIO(); im.save(buf, "PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'<figure><img alt="{alt}" src="data:image/png;base64,{b64}"/></figure>'

md = re.sub(r"!\[([^\]]*)\]\(([^)]+\.png)\)", embed, md)
body = markdown.markdown(md, extensions=["tables", "fenced_code", "sane_lists"])
title = re.search(r"^# (.+)$", open(SRC, encoding="utf-8").read(), flags=re.M).group(1)
n_chars = len(re.sub(r"<[^>]+>", "", body))

html = f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{ --bg:#ffffff; --text:#222222; --muted:#6b6b6b; --line:#e4e6e8; --code:#f6f7f8; --link:#548eaa; --accent:#ff7a1f; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#16181a; --text:#e8e8e8; --muted:#a0a0a0; --line:#2c2f33; --code:#202326; --link:#7fb6d1; }} }}
:root[data-theme="dark"] {{ --bg:#16181a; --text:#e8e8e8; --muted:#a0a0a0; --line:#2c2f33; --code:#202326; --link:#7fb6d1; }}
html,body {{ background:var(--bg); color:var(--text); margin:0; }}
body {{ font-family:-apple-system, BlinkMacSystemFont, "Segoe UI", "Fira Sans", Roboto, Arial, sans-serif; font-size:17px; line-height:1.6; }}
main {{ max-width:780px; margin:0 auto; padding:32px 16px 80px; }}
.meta {{ color:var(--muted); font-size:13px; margin-bottom:8px; }}
h1 {{ font-size:30px; line-height:1.25; margin:8px 0 24px; }}
h2 {{ font-size:24px; line-height:1.3; margin:40px 0 12px; }}
h3 {{ font-size:19px; margin:28px 0 8px; }}
p {{ margin:0 0 16px; }}
a {{ color:var(--link); text-decoration:none; }} a:hover {{ text-decoration:underline; }}
figure {{ margin:24px 0; }} figure img {{ width:100%; height:auto; display:block; border-radius:4px; }}
table {{ border-collapse:collapse; width:100%; margin:16px 0 24px; font-size:14.5px; display:block; overflow-x:auto; }}
th,td {{ border:1px solid var(--line); padding:6px 10px; text-align:left; vertical-align:top; }}
th {{ background:var(--code); font-weight:600; }}
pre {{ background:var(--code); padding:14px 16px; border-radius:4px; overflow-x:auto; font-size:13.5px; line-height:1.45; }}
code {{ font-family:"JetBrains Mono", Menlo, Consolas, monospace; font-size:0.9em; background:var(--code); padding:1px 4px; border-radius:3px; }}
pre code {{ background:none; padding:0; }}
p, li {{ overflow-wrap:break-word; }} p code, li code {{ overflow-wrap:anywhere; word-break:break-all; }} a {{ overflow-wrap:anywhere; }}
ul,ol {{ padding-left:24px; margin:0 0 16px; }} li {{ margin:4px 0; }}
.badge {{ display:inline-block; font-size:12px; color:var(--muted); border:1px solid var(--line); border-radius:12px; padding:2px 10px; }}
</style></head>
<body><main>
<div class="meta"><span class="badge">черновик для Хабра · превью · {n_chars:,} знаков без разметки</span></div>
{body}
</main></body></html>"""
html = html.replace(f"{n_chars:,}", f"{n_chars:,}".replace(",", " "))
open(OUT, "w", encoding="utf-8").write(html)
print(OUT, round(len(html) / 1e6, 2), "MB,", n_chars, "chars")
