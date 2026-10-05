# -*- coding: utf-8 -*-
"""Тёмный стиль strategai insights для графиков статьи (16:10, 2400×1500 px)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, Rectangle

for f in ["/root/.fonts/JetBrainsMono-Regular.ttf", "/root/.fonts/JetBrainsMono-Medium.ttf", "/root/.fonts/JetBrainsMono-Bold.ttf"]:
    try:
        fm.fontManager.addfont(f)
    except Exception:
        pass

BG = "#0a0a0a"
PANEL = "#141414"
GRID = "#242424"
AX = "#3a3a3a"
TXT = "#f2f2f2"
MUTED = "#9b9b9b"
DIM = "#6b6b6b"
OR = "#ff7a1f"        # акцент
OR2 = "#ffb27a"       # светлый акцент
OR3 = "#c2410c"       # тёмный оранжевый
BROWN = "#5a3418"
RED = "#e0533d"
WHITE = "#e8e8e8"
GREY = "#8a8a8a"
SANS = "Inter"
MONO = "JetBrains Mono"

plt.rcParams.update({
    "font.family": SANS, "text.color": TXT, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": AX, "axes.facecolor": "none", "figure.facecolor": BG, "savefig.facecolor": BG,
    "axes.grid": False, "font.size": 11,
})

W, H = 12.0, 7.5   # дюймы; dpi 200 → 2400×1500

def new_fig(w=W, h=H, glow=True):
    fig = plt.figure(figsize=(w, h), dpi=200)
    fig.patch.set_facecolor(BG)
    if glow:
        # мягкое свечение в правом верхнем углу
        bg = fig.add_axes([0, 0, 1, 1], zorder=-10)
        nx, ny = 400, 250
        x = np.linspace(0, 1, nx)[None, :]; y = np.linspace(0, 1, ny)[:, None]
        d = np.sqrt(((x - 1.02) / 0.75) ** 2 + ((y - 1.05) / 0.85) ** 2)
        a = np.clip(1 - d, 0, 1) ** 2.2
        img = np.zeros((ny, nx, 4))
        img[..., 0] = 0.55; img[..., 1] = 0.22; img[..., 2] = 0.05; img[..., 3] = a * 0.22
        bg.imshow(img, extent=[0, 1, 0, 1], origin="lower", aspect="auto")
        bg.set_xlim(0, 1); bg.set_ylim(0, 1); bg.axis("off")
    return fig

def _width(fig, txt):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bb = txt.get_window_extent(renderer=r)
    return bb.width / fig.bbox.width

def rich_line(fig, x, y, segments, size=26, weight="bold", family=SANS, va="baseline"):
    """segments: [(text, color), ...] — одна строка с разноцветными фрагментами."""
    cx = x
    for text, color in segments:
        t = fig.text(cx, y, text, color=color, fontsize=size, fontweight=weight, family=family, va=va, ha="left")
        cx += _width(fig, t)
    return cx

def header(fig, kicker, title_lines, subtitle=None, x=0.045, top=0.935, title_size=25, kicker_box=True):
    """title_lines: список строк, каждая — список (текст, цвет)."""
    k = fig.text(x + 0.008, top, "// " + kicker.upper(), color=MUTED, fontsize=9.5, family=MONO, va="center",
                 ha="left")
    if kicker_box:
        fig.canvas.draw()
        bb = k.get_window_extent(renderer=fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
        fig.patches.append(FancyBboxPatch((bb.x0 - 0.008, bb.y0 - 0.010), bb.width + 0.016, bb.height + 0.020,
                                          boxstyle="round,pad=0,rounding_size=0.004", transform=fig.transFigure,
                                          facecolor="#1a1a1a", edgecolor="none", zorder=-1))
    y = top - 0.075
    lh = title_size / 72 * 1.22 / H
    for line in title_lines:
        rich_line(fig, x, y, line, size=title_size)
        y -= lh
    if subtitle:
        fig.text(x, y + lh * 0.30, subtitle, color=MUTED, fontsize=11.5, family=SANS, va="top", ha="left")
    return y

def footer(fig, note="", x=0.045):
    fig.add_artist(plt.Line2D([x, 1 - x], [0.062, 0.062], color="#262626", lw=0.8, transform=fig.transFigure))
    if note:
        fig.text(x, 0.034, "// " + note.upper(), color=DIM, fontsize=7.6, family=MONO, va="center", ha="left")
    t1 = fig.text(1 - x, 0.034, "insights", color=MUTED, fontsize=13, family=SANS, fontweight="light", va="center", ha="right")
    w1 = _width(fig, t1)
    t2 = fig.text(1 - x - w1 - 0.004, 0.034, "strategai", color=TXT, fontsize=13, family=SANS, fontweight="bold", va="center", ha="right")
    w2 = _width(fig, t2)
    fig.text(1 - x - w1 - w2 - 0.012, 0.034, "/", color=OR, fontsize=14, family=SANS, fontweight="bold", va="center", ha="right")

def style_ax(ax, xgrid=True, ygrid=True):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(AX); ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=9.5, length=0, pad=6)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_family(MONO)
    if xgrid:
        ax.grid(axis="x", color=GRID, lw=0.7)
    if ygrid:
        ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)

def mono_ticks(ax):
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_family(MONO); lab.set_color(MUTED)

def callout(fig, x, y, w, h, kicker, big, small, color=OR, big_size=24, small_size=9.3):
    fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.008", transform=fig.transFigure,
                                      facecolor="#1f1007", edgecolor=color, lw=1.0, zorder=1))
    k = fig.text(x + 0.016, y + h - 0.028, "// " + kicker.upper(), color=color, fontsize=8.3, family=MONO, va="top")
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    kb = k.get_window_extent(renderer=r).transformed(fig.transFigure.inverted())
    b = fig.text(x + 0.016, kb.y0 - 0.020, big, color=TXT, fontsize=big_size, family=SANS, fontweight="bold", va="top")
    bb = b.get_window_extent(renderer=r).transformed(fig.transFigure.inverted())
    fig.text(x + 0.016, bb.y0 - 0.018, small, color=MUTED, fontsize=small_size, family=SANS, va="top", linespacing=1.38)

def legend_item(fig, x, y, label, stats, color, ls="-", lw=2.4, bold=False, accent_stats=False):
    fig.add_artist(plt.Line2D([x, x + 0.028], [y + 0.006, y + 0.006], color=color, lw=lw, ls=ls, transform=fig.transFigure))
    fig.text(x + 0.036, y, label, color=TXT if bold else WHITE, fontsize=12, family=SANS,
             fontweight="bold" if bold else "medium", va="baseline")
    for i, s in enumerate(stats):
        fig.text(x + 0.036, y - 0.030 - i * 0.026, s, color=OR if accent_stats else MUTED, fontsize=8.8, family=MONO, va="baseline")

def save(fig, path):
    fig.savefig(path, dpi=200, facecolor=BG)
    plt.close(fig)
