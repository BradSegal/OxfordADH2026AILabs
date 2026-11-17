"""Figures for notebook 02 (how a language model reads and writes), drawn with matplotlib.

Each function draws one figure, shows it once in a notebook and returns it
outside one (``diagrams._show``), so the notebook calls it in one line. The
palette and style are notebook 01's (``diagrams``): ink, grey and light grey for
structure, one colour per part of the lab, orange for the thing that moves.
Where a figure can use the student's own model or function, it does.
"""

import numpy as np

from .diagrams import ACCENT, GREY, INK, PART_COLOURS, _arrow, _box, _canvas, _plt, _show

PARTS = {"1": PART_COLOURS["1"], "2": PART_COLOURS["2"], "3": "#0f7c86", "4": PART_COLOURS["3"],
         "5": PART_COLOURS["wrap"]}
GROUP_COLOURS = ["#2a78d6", "#d03b3b", "#12805a", "#7b4fd0", "#222222", "#0f7c86"]
TOKEN_SHADES = ["#dbe8f8", "#f6e3c8"]


def _clean(ax, keep=("bottom",)):
    for side in ("top", "right", "left", "bottom"):
        if side not in keep:
            ax.spines[side].set_visible(False)


def _visible(piece):
    """A token's text as it should be printed in a box: a leading space shown as a dot."""
    piece = piece.replace("\n", "\\n")
    return "\u00b7" + piece[1:] if piece.startswith(" ") else piece


# Where this is going ---------------------------------------------------------------

def roadmap():
    """Text, tokens, then two routes: vectors for search by meaning, and next-token probabilities in a loop."""
    fig, ax = _canvas(13, 4.4, xlim=(0, 13), ylim=(0, 4.4))
    # Text and tokens.
    _box(ax, 1.0, 2.2, 1.7, 0.9, "\"Sinus\nbradycardia\"", colour="white", edge=INK, size=10)
    ax.text(1.0, 1.55, "text", ha="center", fontsize=9.5, color=GREY)
    pieces = ["Sin", "us", "\u00b7br", "ad", "yc", "ard", "ia"]
    x = 2.45
    for i, piece in enumerate(pieces):
        _box(ax, x + 0.24, 2.2, 0.44, 0.5, piece, colour=TOKEN_SHADES[i % 2], edge=PARTS["1"], size=8.5)
        x += 0.5
    ax.text(4.2, 1.6, "tokens: each model cuts the\ntext into its own pieces (Part 1)", ha="center", va="top", fontsize=9.5,
            color=PARTS["1"], weight="bold")
    _arrow(ax, 1.9, 2.2, 2.4, 2.2)
    # Upper route: an embedding model makes one vector; search compares directions.
    _arrow(ax, 6.0, 2.45, 6.9, 3.35, colour=GREY)
    vax = fig.add_axes([0.535, 0.60, 0.10, 0.34])
    for angle, colour, label in [(25, PARTS["2"], "your query"), (38, INK, "a close report"),
                                 (85, GREY, "a distant report")]:
        a = np.radians(angle)
        vax.annotate("", xy=(np.cos(a), np.sin(a)), xytext=(0, 0),
                     arrowprops=dict(arrowstyle="-|>", color=colour, lw=1.6))
        vax.text(np.cos(a) * 1.05, np.sin(a) * 1.05, label, fontsize=8, color=colour, va="center")
    vax.set_xlim(-0.05, 1.9)
    vax.set_ylim(-0.05, 1.15)
    vax.axis("off")
    ax.text(9.35, 3.95, "Part 2: an embedding model turns the text\ninto one vector; similar meanings point\n"
            "the same way", fontsize=9.5, color=PARTS["2"], weight="bold", va="top")
    ax.text(9.35, 2.85, "Part 3: search by meaning finds the reports\nwhose vectors point closest to the query's",
            fontsize=9.5, color=PARTS["3"], weight="bold", va="top")
    # Lower route: the language model gives a probability for every next token, then the loop.
    _arrow(ax, 6.0, 1.95, 6.9, 1.1, colour=GREY)
    bax = fig.add_axes([0.565, 0.06, 0.09, 0.27])
    bax.patch.set_alpha(0)
    names, probs = ["is", "can", "refers"], [0.43, 0.08, 0.05]  # measured after "Sinus bradycardia"
    bax.barh(range(3), probs, color=PARTS["4"])
    bax.set_yticks(range(3), names, fontsize=8)
    bax.invert_yaxis()
    bax.set_xlim(0, 0.6)
    bax.set_xticks([])
    bax.set_title("probability of\neach next token", fontsize=8, loc="left", color=GREY)
    _clean(bax, keep=("left",))
    ax.text(9.35, 1.75, "Part 4: a language model gives every possible\nnext token a probability; pick one, append\n"
            "it to the end, and repeat", fontsize=9.5, color=PARTS["4"], weight="bold", va="top")
    ax.text(9.35, 0.65, "Part 5: a chat is one long text that the\nmodel continues in the same way", fontsize=9.5,
            color=PARTS["5"], weight="bold", va="top")
    ax.annotate("", xy=(5.95, 1.92), xytext=(7.0, 0.35),
                arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=1.4, connectionstyle="arc3,rad=-0.3"))
    ax.text(4.6, 0.75, "append the chosen token\nat the end and go again", fontsize=9, color=ACCENT, ha="center")
    ax.set_title("Where the lab is going: one piece of text, two kinds of model", fontsize=10.5, loc="left")
    return _show(fig)


# Part 1: tokens ----------------------------------------------------------------------

def token_boxes(pieces, ids, title=None, per_line=10, tokeniser="Qwen2.5"):
    """Each token as a coloured box with its id underneath; a leading space is shown as a dot.

    Lines break only before a token that starts a new word, so a word is never split across lines.
    """
    rows, row = [], []
    for i, piece in enumerate(pieces):
        if len(row) >= per_line and (piece.startswith(" ") or not piece.strip()):
            rows.append(row)
            row = []
        row.append(i)
    rows.append(row)
    width = 12
    fig, ax = _canvas(width, 0.55 + 1.15 * len(rows), xlim=(0, width), ylim=(0, 1.15 * len(rows) + 0.3))
    for r, row in enumerate(rows):
        y = 1.15 * (len(rows) - r) - 0.35
        x = 0.1
        for i in row:
            text = _visible(pieces[i])
            w = max(0.55, 0.16 * len(text) + 0.3)
            _box(ax, x + w / 2, y, w - 0.06, 0.5, text, colour=TOKEN_SHADES[i % 2], edge=PARTS["1"], size=10.5)
            ax.text(x + w / 2, y - 0.36, str(ids[i]), ha="center", va="top", fontsize=8.5, color=GREY)
            x += w
    words = len("".join(pieces).split())
    ax.set_title(title or f"The {tokeniser} tokeniser: {len(pieces)} tokens for {words} words. Grey numbers: each "
                 f"token's id, its position in the vocabulary (not a frequency or a meaning).\n\u00b7 marks a space at the start of a token (a box with only "
                 f"\u00b7 is a space on its own). The two colours only separate neighbouring tokens.",
                 fontsize=10, loc="left")
    return _show(fig)


# Part 2: vectors -------------------------------------------------------------------

# Three ECGs from the lab data as notebook 01 described them: ecg.rhythm_features (heart rate, RR
# irregularity), then scaled as in 01 (minus the mean, divided by the standard deviation; here over all
# 1,000 ECGs: 80.4 and 21.8 beats a minute, 0.071 and 0.101). Rows 39, 82 and 329 of metadata.csv, each the
# ECG nearest its group's median; measured 17 Nov 2026.
ECG_VECTORS = [("atrial fibrillation", 91, 0.20, (0.48, 1.26)),
               ("sinus bradycardia", 56, 0.02, (-1.13, -0.52)),
               ("sinus tachycardia", 113, 0.01, (1.51, -0.56))]


def ecg_vectors(examples=ECG_VECTORS):
    """Three ECGs from notebook 01 as two scaled numbers each: points, and arrows from (0, 0)."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    colours = [GROUP_COLOURS[0], GROUP_COLOURS[3], GROUP_COLOURS[2]]  # one per ECG, no meaning in the colour
    ax.axhline(0, color=GREY, lw=0.6, zorder=0)
    ax.axvline(0, color=GREY, lw=0.6, zorder=0)
    places = {"atrial fibrillation": (0.08, 0.10, "left", "bottom"), "sinus bradycardia": (-0.08, -0.12, "right", "top"),
              "sinus tachycardia": (0.0, -0.14, "center", "top")}
    for (name, rate, irregularity, (x, y)), colour in zip(examples, colours):
        ax.annotate("", xy=(x, y), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=colour, lw=2.2))
        ax.plot(x, y, "o", color=colour, ms=6)
        dx, dy, ha, va = places.get(name, (0.08, 0.08, "left", "bottom"))
        ax.text(x + dx, y + dy, f"an ECG with {name}\nheart rate {rate} a minute, RR irregularity {irregularity:.2f}\n"
                f"scaled: [{x:.1f}, {y:.1f}]", color=colour, fontsize=9.5, ha=ha, va=va)
    ax.plot(0, 0, "o", color=INK, ms=5)
    ax.text(-0.08, 0.1, "(0, 0): the average of all 1,000 ECGs\n(an average, not a normal ECG)", fontsize=9,
            color=INK, ha="right")
    ax.set_xlim(-3.3, 2.6)
    ax.set_ylim(-1.5, 1.9)
    ax.set_aspect("equal")
    ax.set_xlabel("heart rate, scaled: standard deviations from the average (right: faster)")
    ax.set_ylabel("RR irregularity, scaled: standard deviations\nfrom the average (up: more irregular)")
    _clean(ax, keep=("bottom", "left"))
    ax.set_title("Three ECGs from notebook 01, each as two numbers: a vector [heart rate, RR irregularity],\n"
                 "drawn here after scaling. As an arrow from (0, 0) it has a direction (which way the ECG differs\n"
                 "from average) and a length (how far it differs, not how severe it is).", fontsize=10, loc="left")
    fig.text(0.01, 0.01, "Scaled as in notebook 01: minus the average, divided by the standard deviation, so the "
             "two axes have the same scale.\nBelow 0 means below average (not a negative irregularity).\nOne ECG of "
             "each rhythm: other ECGs of the same rhythm spread around these.", fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    return _show(fig)


def cosine_arrows():
    """Pairs of arrows at 0, 60, 90 and 180 degrees, with their cosine similarity: direction, not length."""
    plt = _plt()
    fig, axes = plt.subplots(1, 4, figsize=(12, 2.6))
    cases = [(0, "same direction"), (60, "60 degrees apart"), (90, "at right angles"), (180, "opposite")]
    for ax, (angle, words) in zip(axes, cases):
        a = np.array([1.0, 0.0]) * 1.6
        t = np.radians(angle)
        b = np.array([np.cos(t), np.sin(t)]) * 0.9
        lift = np.array([0.0, 0.07]) if angle == 0 else np.zeros(2)  # so b is not hidden under a
        for vec, colour, name in [(a, INK, "a"), (b, ACCENT, "b")]:
            base = lift if name == "b" else np.zeros(2)
            ax.annotate("", xy=vec + base, xytext=base, arrowprops=dict(arrowstyle="-|>", color=colour, lw=2))
            offset = np.array([0.0, 0.12]) if name == "a" else vec / np.linalg.norm(vec) * 0.18
            if angle == 0 and name == "b":
                offset = np.array([0.0, 0.2])
            ax.text(*(vec + offset), name, color=colour, fontsize=11, weight="bold", ha="center", va="center")
        if 0 < angle:
            arc = np.linspace(0, t, 40)
            ax.plot(0.35 * np.cos(arc), 0.35 * np.sin(arc), color=GREY, lw=1)
        cos = np.cos(t)
        ax.set_title(f"{words}\ncosine similarity = {0 if abs(cos) < 1e-9 else cos:.1f}", fontsize=10)
        ax.set_xlim(-1.1, 1.9)
        ax.set_ylim(-0.25, 1.05)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.plot(0, 0, "o", color=INK, ms=3)
    fig.suptitle("Cosine similarity is the cosine of the angle between two vectors: it ignores their lengths "
                 "(a is longer than b in every panel)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    return _show(fig)


def _place_labels(ax, xy, texts, colours, size=8.5):
    """Put each label beside its point where it overlaps no other label (greedy, in display units)."""
    fig = ax.figure
    fig.canvas.draw()
    to_px = ax.transData.transform
    points = to_px(xy)
    width_px = size * 0.62 * fig.dpi / 72  # about one character's width
    height_px = size * 1.35 * fig.dpi / 72
    placed = [(px - 4, py - 4, px + 4, py + 4) for px, py in points]  # the dots themselves
    order = np.argsort(points[:, 0])
    candidates = [(dx, dy * height_px + (2 if dy >= 0 else -2), ha, "bottom")
                  for dy in (0, -1, 0.9, -1.9, 1.8, -2.8) for dx, ha in ((6, "left"), (-6, "right"))]

    def box(px, py, dx, dy, ha, n):
        w = n * width_px
        x0 = px + dx if ha == "left" else px + dx - w
        return x0, py + dy, x0 + w, py + dy + height_px

    def clash(b):
        return sum(max(0, min(b[2], o[2]) - max(b[0], o[0])) * max(0, min(b[3], o[3]) - max(b[1], o[1]))
                   for o in placed)

    for i in order:
        px, py = points[i]
        options = [(clash(box(px, py, dx, dy, ha, len(texts[i]))) + k * 2.0, k, dx, dy, ha)
                   for k, (dx, dy, ha, _) in enumerate(candidates)]
        _, _, dx, dy, ha = min(options)
        placed.append(box(px, py, dx, dy, ha, len(texts[i])))
        ax.annotate(texts[i], xy[i], xytext=(dx * 72 / fig.dpi, dy * 72 / fig.dpi), textcoords="offset points",
                    fontsize=size, color=colours[i], ha=ha, va="bottom")


def term_map(points, terms, groups, pairs=(), title=""):
    """Terms placed by the first two principal components of their vectors, coloured by group.

    ``points`` is one (x, y) per term, or a dict (panel title -> points) drawn side
    by side; dotted lines join each pair in ``pairs``, two names for the same thing.
    """
    plt = _plt()
    panels = points if isinstance(points, dict) else {title: points}
    fig, axes = plt.subplots(1, len(panels), figsize=(6.8 * len(panels), 6.6))
    axes = np.atleast_1d(axes)
    names = list(dict.fromkeys(groups))
    colours = {g: GROUP_COLOURS[i % len(GROUP_COLOURS)] for i, g in enumerate(names)}
    where = {t: i for i, t in enumerate(terms)}
    for ax, (name, xy) in zip(axes, panels.items()):
        xy = np.asarray(xy)
        for a, b in pairs:
            i, j = where[a], where[b]
            ax.plot(xy[[i, j], 0], xy[[i, j], 1], color=GREY, lw=0.9, ls=":", zorder=1)
        ax.scatter(xy[:, 0], xy[:, 1], color=[colours[g] for g in groups], s=26, zorder=2)
        ax.set_title(name, fontsize=10, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("first principal component")
        ax.set_ylabel("second principal component")
        span = xy.max(axis=0) - xy.min(axis=0)
        ax.set_xlim(xy[:, 0].min() - 0.3 * span[0], xy[:, 0].max() + 0.42 * span[0])
        ax.set_ylim(xy[:, 1].min() - 0.1 * span[1], xy[:, 1].max() + 0.1 * span[1])
        _clean(ax, keep=("bottom", "left"))
    handles = [plt.Line2D([], [], marker="o", ls="", color=colours[g], label=g) for g in names]
    handles.append(plt.Line2D([], [], color=GREY, ls=":", label="two names for the same thing"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.03))
    fig.text(0.01, 0.005, "Each panel has its own scale: compare distances within a panel, not between them. Two "
             "directions keep only part of the information in the vectors.", fontsize=9.5, color=INK)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    for ax, xy in zip(axes, panels.values()):
        _place_labels(ax, np.asarray(xy), terms, [colours[g] for g in groups])
    return _show(fig)


# Part 4: next-token probabilities --------------------------------------------------

def next_token_bars(top, text, title=None):
    """The most likely next tokens after ``text`` and their probabilities (``top``: (piece, probability) pairs)."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(top) + 1.3))
    names = [_visible(p) for p, _ in top]
    probs = [q for _, q in top]
    ax.barh(range(len(top)), probs, color=PARTS["4"])
    for i, q in enumerate(probs):
        ax.text(q + 0.01, i, f"{q:.2f}", va="center", fontsize=9.5)
    ax.set_yticks(range(len(top)), names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, max(1.0, max(probs) + 0.1))
    ax.set_xlabel("probability that the text continues with this token (not the probability of a diagnosis)")
    rest = 1 - sum(probs)
    ax.set_title(title or f"After \"{text}\", the {len(top)} most likely next tokens (\u00b7 marks the start of "
                 f"a word).\nEvery other token shares the remaining {rest:.2f}.", fontsize=10, loc="left")
    _clean(ax, keep=("bottom", "left"))
    fig.tight_layout()
    return _show(fig)


def temperature_bars(lm, text, softmax, temperatures=(0.2, 1.0, 2.0), k=6):
    """The same next-token scores turned into probabilities by ``softmax`` at three temperatures."""
    plt = _plt()
    from .checks.llm_applications import unfinished
    waiting = "\u23f3 The figure uses your softmax: finish TODO 4 first, then run this cell again."
    if unfinished(softmax):
        print(waiting)
        return None
    scores = lm.next_token_scores(text)
    top = np.argsort(-scores)[:k]
    names = [_visible(lm.decode([t])) for t in top]
    try:
        finished = softmax(scores, 1.0) is not ...
    except Exception as error:  # the student's softmax can raise anything
        print(f"\u274c The figure uses your softmax, which stopped with {type(error).__name__}: {error}. "
              "Fix TODO 4, then run this cell again.")
        return None
    if not finished:
        print(waiting)
        return None
    width = 12 if len(temperatures) > 1 else 8
    fig, axes = plt.subplots(1, len(temperatures), figsize=(width, 0.4 * k + 1.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, t in zip(axes, temperatures):
        probs = np.asarray(softmax(scores, t), dtype=float)[top]
        ax.barh(range(k), probs, color=PARTS["4"])
        for i, q in enumerate(probs):
            ax.text(q + 0.02, i, f"{q:.2f}" if q >= 0.005 else "<0.01", va="center", fontsize=9)
        ax.set_xlim(0, 1.15)
        ax.text(1.12, k - 0.5, f"these {k} together: {probs.sum():.2f}\nthe other {len(scores) - k:,} together: {1 - probs.sum():.2f}",
                ha="right", va="bottom", fontsize=10, color=INK)
        ax.set_title(f"temperature {t:g}" + (" (as in the figure above)" if t == 1 and len(temperatures) > 1
                                             else ""), fontsize=10, loc="left")
        ax.set_xlabel("probability")
        _clean(ax, keep=("bottom", "left"))
    axes[0].set_yticks(range(k), names, fontsize=10)
    axes[0].invert_yaxis()
    if len(temperatures) > 1:
        fig.suptitle(f"The same scores for the next token after \"{text}\", divided by the temperature before "
                     f"softmax.\nDividing never changes the order: the top token stays the most likely at every "
                     f"temperature.", fontsize=10, x=0.01, ha="left")
    else:
        fig.suptitle(f"The {k} most likely next tokens after \"{text}\".\nEach bar is the probability of the next "
                     f"piece of text, not of a diagnosis.\n\u00b7 marks a space at the start of a token (a bar "
                     f"labelled only \u00b7 is a space on its own).", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    return _show(fig)
