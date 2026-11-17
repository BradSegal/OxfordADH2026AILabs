"""Explanatory diagrams for the Intro to Deep Learning notebook, drawn with matplotlib.

Each function draws one figure, shows it in a notebook and returns it, so the
notebook calls it in one line and Colab needs no image files. Where a diagram
can use the real data or the numbers from the student's own run (an averaged
beat, the learnt weights, the path training took), it does. Colours match the
notebook's data plots: red for AF, blue for no AF, grey and black for structure.
"""

import numpy as np

from .data import FS, LEADS
from .ecg import AF_COLOUR, OTHER_COLOUR, r_peaks

INK, GREY, LIGHT = "#222222", "#6b6a66", "#e9e8e4"
PART_COLOURS = {"1": "#2a78d6", "2": "#12805a", "3": "#7b4fd0", "wrap": "#a45500"}
ACCENT = "#c26a00"  # orange: the moving filter, the training path


def _plt():
    import matplotlib.pyplot as plt
    return plt


def _show(fig):
    """Display a figure in a notebook and return None there; return the figure outside one.

    Returning the figure inside a notebook would show it twice when the call is the
    last line of a cell (once from display, once as the cell's result).
    """
    plt = _plt()
    try:
        from IPython import get_ipython
        from IPython.display import display
        if get_ipython() is not None:
            display(fig)
            plt.close(fig)
            return None
    except ImportError:
        pass
    plt.close(fig)
    return fig


def _box(ax, x, y, w, h, text, colour=LIGHT, text_colour=INK, size=10, weight="normal", edge=None):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=colour, ec=edge or colour, lw=1.2))
    ax.text(x, y, text, ha="center", va="center", fontsize=size, color=text_colour, weight=weight)


def _arrow(ax, x0, y0, x1, y1, colour=GREY, text=None, size=9, rad=0.0):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=colour, lw=1.4, shrinkA=2, shrinkB=2,
                                connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.06, text, ha="center", va="bottom", fontsize=size, color=colour)


def _canvas(width, height, xlim=(0, 10), ylim=(0, 1)):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    return fig, ax


def average_beat(signal, fs=FS, before=0.3, after=0.6):
    """The mean of all beats in one lead, aligned on their R peaks, and its time axis in seconds."""
    a, b = int(before * fs), int(after * fs)
    peaks = [r for r in r_peaks(signal, fs) if a <= r < len(signal) - b]
    beats = np.stack([signal[r - a:r + b] for r in peaks])
    return np.arange(-a, b) / fs, beats.mean(axis=0), len(peaks)


# 1. What an ECG records ---------------------------------------------------------

def annotated_beat(ecgs, row, fs=FS):
    """One heartbeat (the average of the beats in lead II of one ECG), with P wave, QRS complex and T wave."""
    plt = _plt()
    t, beat, n = average_beat(ecgs[row, 1], fs, after=1.0)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(t, beat, color=INK, lw=2)
    windows = {"P wave": (-0.25, -0.06), "QRS complex": (-0.05, 0.06), "T wave": (0.12, 0.45)}
    meaning = {"P wave": "the atria activate", "QRS complex": "the ventricles activate",
               "T wave": "the ventricles recover"}
    top = beat.max()
    for name, (lo, hi) in windows.items():
        sel = (t >= lo) & (t <= hi)
        peak_t = t[sel][np.argmax(beat[sel])]
        y = top * 1.08 if name == "QRS complex" else beat[sel].max() + 0.25 * top
        ax.annotate(f"{name}\n({meaning[name]})", xy=(peak_t, beat[sel].max()), xytext=(peak_t, y),
                    ha="center", va="bottom", fontsize=10,
                    arrowprops=dict(arrowstyle="-", color=GREY, lw=0.8))
    late = t > 0.45
    next_r = t[late][np.argmax(beat[late])]
    rr_y = beat.min() - 0.12 * top
    ax.annotate("", xy=(next_r, rr_y), xytext=(0, rr_y),
                arrowprops=dict(arrowstyle="<->", color=OTHER_COLOUR, lw=1.4, shrinkA=0, shrinkB=0))
    ax.text(next_r / 2, rr_y - 0.04 * top, f"RR interval: {next_r:.2f} s from one R peak to the next (on average)",
            ha="center", va="top", fontsize=10, color=OTHER_COLOUR)
    for x in (0, next_r):
        ax.plot([x, x], [rr_y, beat[np.argmin(np.abs(t - x))]], color=OTHER_COLOUR, lw=0.6, ls=":")
    ax.set_ylim(beat.min() - 0.35 * top, top * 1.55)
    ax.set_xlabel("Time from the R peak, the top of the QRS complex (s)")
    ax.set_yticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_title(f"One heartbeat in lead II, then the next one starting (ECG {row}, normal sinus rhythm: the average of "
                 f"{n} of its beats, lined up on their R peaks)", fontsize=10, loc="left")
    fig.tight_layout()
    return _show(fig)


def sinus_vs_af(ecgs, sinus_row, af_row, seconds=10, fs=FS):
    """Lead II in sinus rhythm and in AF, with P waves and the gaps between beats (RR intervals) marked."""
    plt = _plt()
    fig, axes = plt.subplots(2, 1, figsize=(12, 5.6))
    n = seconds * fs
    t = np.arange(n) / fs
    for ax, row, title, colour in [(axes[0], sinus_row, "Sinus rhythm", OTHER_COLOUR),
                                   (axes[1], af_row, "Atrial fibrillation", AF_COLOUR)]:
        signal = ecgs[row, 1, :n]
        peaks = r_peaks(ecgs[row, 1], fs)
        peaks = peaks[peaks < n]
        ax.plot(t, signal, color=INK, lw=0.9)
        top, bottom = signal.max(), signal.min()
        gap_y = bottom - 0.25 * (top - bottom)
        for left, right in zip(peaks[:-1], peaks[1:]):
            ax.annotate("", xy=(right / fs, gap_y), xytext=(left / fs, gap_y),
                        arrowprops=dict(arrowstyle="<->", color=colour, lw=1.2, shrinkA=0, shrinkB=0))
            ax.text((left + right) / 2 / fs, gap_y - 0.06 * (top - bottom), f"{(right - left) / fs:.2f}",
                    ha="center", va="top", fontsize=8.5, color=colour)
        if row == sinus_row:
            for r in peaks:
                window = signal[max(r - int(0.25 * fs), 0):max(r - int(0.06 * fs), 1)]
                if len(window) < 5:
                    continue
                p = max(r - int(0.25 * fs), 0) + int(np.argmax(window))
                ax.annotate("P", xy=(p / fs, signal[p]), xytext=(p / fs, signal[p] + 0.35 * (top - bottom)),
                            ha="center", fontsize=9, color=colour, weight="bold",
                            arrowprops=dict(arrowstyle="-", color=colour, lw=0.8))
            note = "a P wave before every beat; the gaps between beats are almost equal"
        else:
            note = "no P waves, just an uneven baseline; the gaps between beats vary with no pattern"
        ax.set_title(f"{title} (ECG {row}, lead II, all 10 s): {note}", fontsize=10, loc="left", color=colour)
        ax.set_xlim(0, seconds)
        ax.set_ylim(gap_y - 0.3 * (top - bottom), top + 0.5 * (top - bottom))
        ax.set_yticks([])
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
    axes[1].set_xlabel("Time (s). Arrows and numbers: the gaps between beats (RR intervals), in seconds")
    fig.tight_layout()
    return _show(fig)


# 2. An ECG as numbers -------------------------------------------------------------

def ecg_as_numbers(ecgs, row, fs=FS, start=1.0, width=0.2):
    """A strip of lead II, a zoom to its samples, and the whole dataset as an array of ECGs x leads x time."""
    plt = _plt()
    from matplotlib.patches import ConnectionPatch, Rectangle
    fig = plt.figure(figsize=(11, 6.2))
    strip = fig.add_axes([0.04, 0.70, 0.55, 0.24])
    zoom = fig.add_axes([0.67, 0.60, 0.31, 0.29])
    array = fig.add_axes([0.02, 0.0, 0.96, 0.52])
    signal = ecgs[row, 1]
    t = np.arange(len(signal)) / fs
    strip.plot(t, signal, color=INK, lw=0.7)
    strip.add_patch(Rectangle((start, signal.min()), width, signal.max() - signal.min(), fill=False,
                              ec=ACCENT, lw=1.5))
    strip.set_title(f"Lead II of ECG {row}: 10 seconds, drawn as a line", fontsize=10, loc="left")
    strip.set_xlabel("Time (s)")
    strip.set_yticks([])
    a, b = int(start * fs), int((start + width) * fs)
    zoom.plot(t[a:b], signal[a:b], color=GREY, lw=0.8)
    zoom.plot(t[a:b], signal[a:b], "o", color=ACCENT, ms=3.5)
    for k in (a + 3, b - 4):
        zoom.annotate(f"{signal[k]:.2f}", xy=(t[k], signal[k]), xytext=(0, 12), textcoords="offset points",
                      ha="center", fontsize=9)
    zoom.set_title(f"Zoomed in: {b - a} samples, one every 1/{fs} s\n(values are scaled, not millivolts)",
                   fontsize=10, loc="left")
    zoom.set_xlabel("Time (s)")
    zoom.set_yticks([])
    for corner in ((start + width, signal.max()), (start + width, signal.min())):
        fig.add_artist(ConnectionPatch(xyA=corner, coordsA=strip.transData,
                                       xyB=(t[a] if corner[1] == signal.max() else t[a], zoom.get_ylim()[1]
                                            if corner[1] == signal.max() else zoom.get_ylim()[0]),
                                       coordsB=zoom.transData, color=ACCENT, lw=0.8, ls=":"))
    # The array: a stack of ECGs, each a grid of 12 leads x time.
    ax = array
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    x0, y0, w, h = 0.75, 0.55, 5.3, 3.3
    for k in range(4, 0, -1):
        ax.add_patch(Rectangle((x0 + 0.18 * k, y0 + 0.14 * k), w, h, fc="white", ec=GREY, lw=0.8))
    ax.add_patch(Rectangle((x0, y0), w, h, fc="white", ec=INK, lw=1.2))
    row_h = h / 12
    for i, lead in enumerate(LEADS):
        y = y0 + h - (i + 1) * row_h
        if lead == "II":
            ax.add_patch(Rectangle((x0, y), w, row_h, fc=ACCENT, alpha=0.18, ec="none"))
        ax.text(x0 - 0.08, y + row_h / 2, lead, ha="right", va="center", fontsize=9)
        trace = ecgs[row, i, :1250]
        trace = (trace - trace.min()) / (np.ptp(trace) + 1e-9)
        ax.plot(x0 + np.linspace(0.05, w - 0.05, len(trace)), y + 0.12 * row_h + 0.76 * row_h * trace,
                color=INK, lw=0.4)
    ax.text(x0 + w / 2, y0 - 0.12, "2,500 time points per lead (10 seconds, 250 per second)", ha="center",
            va="top", fontsize=10)
    ax.text(x0 - 0.55, y0 + h / 2, "12 leads", ha="center", va="center", rotation=90, fontsize=11)
    ax.annotate("1,000 ECGs,\none behind the other", xy=(x0 + w + 0.72, y0 + h + 0.5),
                xytext=(x0 + w + 1.0, y0 + h - 0.1), fontsize=10, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=GREY, lw=0.8))
    ax.text(x0 + w + 1.0, y0 + h / 2 - 0.45,
            f"ecgs.shape is (1000, 12, 2500)\n\necgs[{row}, 1] is lead II\n(shaded) of ECG {row}\n\n"
            "12 leads: 12 views of the\nsame heartbeats",
            fontsize=10, ha="left", va="center", family="monospace")
    return _show(fig)


# 3. Where the lab goes ------------------------------------------------------------

def roadmap():
    """The lab in one line: from two hand-made features to the CNN's embedding."""
    fig, ax = _canvas(11, 2.3, xlim=(0, 11), ylim=(0, 1.6))
    steps = [("Two features\nmade by hand", "1"), ("A linear model\nlearns from\nthem", "1"),
             ("Test it fairly\non new patients\n(both models)", "2"), ("A CNN (neural\nnetwork) learns\nits own features", "3"),
             ("The CNN's\nembedding: a\nsummary of\neach ECG", "wrap")]
    xs = np.linspace(1.0, 10.0, len(steps))
    labels = {"1": "Part 1", "2": "Part 2", "3": "Part 3", "wrap": "Wrap-up"}
    for (text, part), x in zip(steps, xs):
        colour = PART_COLOURS.get(part, INK)
        _box(ax, x, 0.9, 1.7, 0.9, text, colour=colour if part else "white", edge=colour,
             text_colour="white" if part else INK, size=9.5)
        if part:
            ax.text(x, 0.28, labels[part], ha="center", fontsize=10.5, color=colour, weight="bold")
    for x0, x1 in zip(xs[:-1], xs[1:]):
        _arrow(ax, x0 + 0.86, 0.9, x1 - 0.86, 0.9)
    return _show(fig)


# 4. The model -------------------------------------------------------------------

def model_diagram(model=None, example=None, label=None):
    """The two-feature model: inputs, weights, sum plus bias, sigmoid, probability.

    With a trained ``model`` the diagram shows its learnt weights and bias; with
    ``example`` (the two scaled features of one ECG) it also works one ECG through.
    """
    fig, ax = _canvas(11, 3.4, xlim=(0, 11), ylim=(0, 3))
    if model is not None:
        w = model.weight.detach().cpu().view(-1).numpy()
        b = float(model.bias.detach().cpu())
        w_text, b_text = [f"\u00d7 {v:.2f}" for v in w], f"+ bias ({b:.2f})"
    else:
        w, b = None, None
        w_text, b_text = ["\u00d7 weight w1", "\u00d7 weight w2"], "+ bias b"
    inputs = ["scaled\nheart rate", "scaled RR\nirregularity"]
    ys = [2.2, 0.8]
    for name, y, wt, k in zip(inputs, ys, w_text, range(2)):
        value = f"\n= {example[k]:.2f}" if example is not None else ""
        _box(ax, 1.1, y, 1.8, 0.9, name + value, colour="white", edge=OTHER_COLOUR, size=9.5)
        _arrow(ax, 2.0, y, 4.0, 1.5 + (y - 1.5) * 0.25, colour=OTHER_COLOUR)
        ax.text(3.0, (y + 1.5 + (y - 1.5) * 0.25) / 2 + (0.18 if y > 1.5 else -0.32), wt, ha="center",
                fontsize=10, color=OTHER_COLOUR, weight="bold")
    score = ""
    if example is not None and w is not None:
        s = float(np.dot(w, example) + b)
        p = 1 / (1 + np.exp(-s))
        score = f" {s:.2f}"
    _box(ax, 4.75, 1.5, 1.5, 1.0, "add up\n" + b_text, colour=LIGHT, size=9.5)
    _arrow(ax, 5.5, 1.5, 6.5, 1.5, text="score" + score)
    # sigmoid box with a little curve
    _box(ax, 7.4, 1.5, 1.7, 1.2, "", colour="white", edge=INK)
    z = np.linspace(-6, 6, 50)
    ax.plot(7.4 + z / 6 * 0.6, 1.5 - 0.35 + 0.7 / (1 + np.exp(-z)), color=INK, lw=1.5)
    ax.text(7.4, 0.68, "sigmoid: squashes\nany score into 0 to 1", ha="center", va="top", fontsize=9)
    _arrow(ax, 8.25, 1.5, 9.1, 1.5)
    out = f"\n= {p:.2f}" if score else ""
    _box(ax, 10.0, 1.5, 1.7, 0.9, "probability\nof AF" + out, colour=AF_COLOUR, text_colour="white", size=9.5)
    title = ("The trained model, with its learnt numbers" if model is not None
             else "The model: score each ECG, then turn the score into a probability")
    if example is not None:
        title += f" worked through for {label or 'one ECG'}"
        ax.text(0.1, 0.05, "Numbers are rounded to two decimal places, so working them by hand can differ "
                "slightly.", fontsize=8.5, color=GREY)
    ax.text(0.1, 2.95, title, fontsize=10.5, va="top")
    return _show(fig)


# 5. The loss --------------------------------------------------------------------

def loss_curves():
    """Binary cross-entropy against the predicted probability of AF, for an AF and a non-AF ECG."""
    plt = _plt()
    p = np.linspace(0.005, 0.995, 400)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(p, -np.log(p), color=AF_COLOUR, lw=2.2, label="the ECG has AF: loss = -log(p)")
    ax.plot(p, -np.log(1 - p), color=OTHER_COLOUR, lw=2.2, ls="--", label="no AF: loss = -log(1 - p)")
    notes = {0.9: "confident\nand right", 0.5: "unsure", 0.1: "confident\nand wrong"}
    for q, note in notes.items():
        loss = -np.log(q)
        ax.plot(q, loss, "o", color=AF_COLOUR, ms=7)
        offset = {0.5: (10, -34), 0.9: (-60, 40), 0.1: (14, 10)}[q]
        ax.annotate(f"p = {q}: loss {loss:.2f}\n({note})", xy=(q, loss), xytext=offset,
                    textcoords="offset points", fontsize=9, color=AF_COLOUR,
                    arrowprops=dict(arrowstyle="-", color=AF_COLOUR, lw=0.8))
    ax.set_xlabel("The model's probability of AF, p")
    ax.set_ylabel("Loss (lower is better)")
    ax.set_ylim(0, 4.2)
    ax.set_xlim(0, 1)
    ax.legend(loc="upper center", fontsize=9, frameon=False)
    ax.set_title("Binary cross-entropy: the loss for one ECG (log is the natural logarithm)", fontsize=10.5, loc="left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return _show(fig)


# 6. Gradient descent --------------------------------------------------------------

def loss_landscape(x, y, path, bias):
    """The loss over the two weights, with the bias held at ``bias``, and the path training took.

    ``path`` is a list of (w1, w2) pairs recorded during training (``TrainingPlot.path``).
    """
    plt = _plt()
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    path = np.asarray(path, dtype=float)
    lo = path.min(axis=0) - 0.5
    hi = path.max(axis=0) + 0.5
    w1, w2 = np.meshgrid(np.linspace(lo[0], hi[0], 120), np.linspace(lo[1], hi[1], 120))
    scores = w1[..., None] * x[:, 0] + w2[..., None] * x[:, 1] + bias
    loss = np.mean(np.logaddexp(0, scores) - y * scores, axis=-1)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    import matplotlib.patheffects as pe
    filled = ax.contourf(w1, w2, loss, levels=20, cmap="cividis")
    ax.contour(w1, w2, loss, levels=20, colors="white", linewidths=0.3, alpha=0.5)
    fig.colorbar(filled, ax=ax, label="Loss on the training ECGs (darker is lower)")
    ax.plot(path[:, 0], path[:, 1], "-o", color="white", ms=4, lw=1.8, mfc=ACCENT, mec="white", mew=0.8)
    halo = [pe.withStroke(linewidth=3, foreground=INK)]
    ax.annotate("start: the weights PyTorch\npicked at random", xy=path[0], xytext=(10, -16), textcoords="offset points", color="white",
                fontsize=10, weight="bold", path_effects=halo)
    ax.annotate("after 100 steps", xy=path[-1], xytext=(-10, 12), textcoords="offset points", color="white",
                fontsize=10, weight="bold", ha="right", path_effects=halo)
    ax.set_xlabel("w1: weight on scaled heart rate")
    ax.set_ylabel("w2: weight on scaled RR irregularity")
    ax.set_title(f"Training walks the two weights downhill (a dot every 5 steps).\nThis is a slice: the bias "
                 f"also changed during training,\nbut here it is held at its final value, {bias:.2f}.",
                 fontsize=9.5, loc="left")
    fig.tight_layout()
    return _show(fig)


# 7. The training loop -------------------------------------------------------------

def training_loop():
    """The four steps every neural network repeats, as a cycle."""
    fig, ax = _canvas(9, 4.2, xlim=(0, 9), ylim=(0, 4.2))
    steps = [((2.3, 3.3), "1. Forward pass", "score the training ECGs\nwith the model"),
             ((6.7, 3.3), "2. Loss", "one number: how wrong\nthe scores are"),
             ((6.7, 0.9), "3. Backward pass", "loss.backward() works out\nthe gradient of every parameter"),
             ((2.3, 0.9), "4. Step", "optimiser.step() changes every\nparameter a little, to lower the loss")]
    for (x, y), name, text in steps:
        _box(ax, x, y, 3.3, 1.2, "", colour="white", edge=PART_COLOURS["1"])
        ax.text(x, y + 0.25, name, ha="center", va="center", fontsize=11, weight="bold", color=PART_COLOURS["1"])
        ax.text(x, y - 0.2, text, ha="center", va="center", fontsize=9)
    _arrow(ax, 3.95, 3.3, 5.05, 3.3)
    _arrow(ax, 6.7, 2.7, 6.7, 1.5)
    _arrow(ax, 5.05, 0.9, 3.95, 0.9)
    _arrow(ax, 2.3, 1.5, 2.3, 2.7)
    ax.text(4.5, 2.1, "repeat until the\nloss stops falling", ha="center", va="center", fontsize=10, color=GREY,
            style="italic")
    return _show(fig)


# 8. A convolution -----------------------------------------------------------------

def convolution_steps(signal, kernel, centre=None, fs=FS, half_window=22):
    """A short filter at three positions along a real ECG, and the output it builds."""
    plt = _plt()
    signal = np.asarray(signal, dtype=float)
    kernel = np.asarray(kernel, dtype=float)
    k = len(kernel)
    if centre is None:
        centre = int(r_peaks(signal, fs)[1])
    lo, hi = centre - half_window, centre + half_window
    output = np.array([np.dot(signal[i:i + k], kernel) for i in range(lo, hi - k)])
    positions = [lo + 2, centre - k - 1 - (k < 5) * 2, hi - k - 3]  # flat, rising edge, after the peak
    fig, axes = plt.subplots(2, 3, figsize=(12, 5), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
    t = (np.arange(lo, hi) - centre) / fs * 1000
    out_t = (np.arange(lo, hi - k) + k // 2 - centre) / fs * 1000
    for col, start in enumerate(positions):
        top, bottom = axes[0, col], axes[1, col]
        top.plot(t, signal[lo:hi], "o-", color=GREY, ms=2.5, lw=0.8)
        span = (np.arange(start, start + k) - centre) / fs * 1000
        top.axvspan(span[0] - 2, span[-1] + 2, color=ACCENT, alpha=0.18)
        for s_t, s_v, w in zip(span, signal[start:start + k], kernel):
            if w:
                top.plot(s_t, s_v, "^" if w > 0 else "v", color=ACCENT if w > 0 else "white", ms=8,
                         mec=INK, mew=0.8, zorder=3)
        value = float(np.dot(signal[start:start + k], kernel))
        top.set_title(f"Filter here: sum of (sample x filter number) = {value:.1f}", fontsize=9.5, loc="left")
        top.set_ylim(signal[lo:hi].min() - 0.5 * np.ptp(signal[lo:hi]), signal[lo:hi].max() + 0.1 * np.ptp(signal[lo:hi]))
        top.set_yticks([])
        done = np.arange(len(output)) <= start - lo
        bottom.plot(out_t[done], output[done], "o-", color=ACCENT, ms=2.5, lw=1)
        bottom.plot(out_t[start - lo], output[start - lo], "o", color=ACCENT, ms=8, mec=INK)
        bottom.set_ylim(output.min() - 1, output.max() + 1)
        bottom.set_xlim(t[0], t[-1])
        bottom.axhline(0, color=GREY, lw=0.8, zorder=0)
        bottom.set_xlabel("Time from the R peak (ms)")
    axes[0, 0].set_ylabel("Lead II samples", fontsize=9)
    axes[1, 0].set_ylabel("Output so far", fontsize=9)
    numbers = ", ".join(f"{w:+g}" if w else "0" for w in kernel)
    fig.suptitle(f"Filter [{numbers}] over {k} samples (shaded): white down-triangles are multiplied by -1, "
                 "orange up-triangles by +1, then all are added.\nIt slides one sample at a time; each position gives "
                 "one output value: large where the signal rises steeply, negative where it falls.\n(A hand-made "
                 "filter for illustration: a CNN learns its own filter numbers.)", fontsize=10,
                 x=0.01, ha="left")
    fig.tight_layout()
    return _show(fig)


# 9. The CNN -----------------------------------------------------------------------

def cnn_diagram():
    """The notebook's CNN: the shape after each block, how much ECG each value sees, and the final layer."""
    fig, ax = _canvas(13.5, 5, xlim=(0, 16), ylim=(0, 5.2))
    stages = [("input", 1, 2500, None), ("block 1", 16, 625, "0.05 s"), ("block 2", 32, 156, "0.2 s"),
              ("block 3", 64, 39, "0.9 s"), ("block 4", 64, 9, "3.7 s")]
    mid = 2.7
    xs = [0.9, 2.75, 4.6, 6.45, 8.3]
    for (name, channels, length, sees), x in zip(stages, xs):
        h = max(2.6 * channels / 64, 0.25)
        w = 0.35 + 0.9 * np.log10(length) / np.log10(2500)
        colour = PART_COLOURS["3"] if name != "input" else GREY
        _box(ax, x, mid, w, h, "", colour=colour, edge=colour)
        ax.text(x, 4.1, name, ha="center", va="bottom", fontsize=10, weight="bold")
        ax.text(x, 1.15, f"{channels} \u00d7 {length:,}", ha="center", va="top", fontsize=10)
        if sees:
            ax.text(x, 0.75, f"each value sees\n{sees} of ECG", ha="center", va="top", fontsize=9, color=INK)
        else:
            ax.text(x, 0.75, "lead II,\n10 s", ha="center", va="top", fontsize=9, color=GREY)
    for x0, x1 in zip(xs[:-1], xs[1:]):
        _arrow(ax, x0 + 0.7, mid, x1 - 0.7, mid)
    ax.text(0.2, 5.15, "Each block: filters (Conv1d), rescale (BatchNorm), keep positives (ReLU), keep the largest "
            "of every 4 values (MaxPool).\nShapes are channels \u00d7 length. Box height is proportional to the "
            "channels; box width only roughly shows the length.", fontsize=9.5, va="top", color=INK)
    # global max pooling into the embedding
    ex = 10.4
    _arrow(ax, 8.95, mid, ex - 0.25, mid)
    ax.text(9.5, mid + 0.25, "largest\nvalue over\ntime", ha="center", va="bottom", fontsize=8.5, color=GREY)
    cells = 16
    for i in range(cells):
        ax.add_patch(_rect(ex - 0.15, mid - 1.2 + i * 0.15, 0.3, 0.13, ACCENT))
    ax.text(ex, 4.15, "embedding", ha="center", va="bottom", fontsize=10, weight="bold", color=ACCENT)
    ax.text(ex, 1.15, "64 numbers", ha="center", va="top", fontsize=10)
    ax.text(ex, 0.75, "(16 drawn)", ha="center", va="top", fontsize=9, color=GREY)
    # final layer: the Part 1 model with 64 inputs
    sx = 12.3
    for i in range(cells):
        _arrow(ax, ex + 0.17, mid - 1.13 + i * 0.15, sx - 0.5, mid, colour=OTHER_COLOUR)
    _box(ax, sx, mid, 0.95, 0.8, "add up\n+ bias", colour=LIGHT, size=9)
    _arrow(ax, sx + 0.5, mid, sx + 0.85, mid)
    _box(ax, sx + 1.3, mid, 0.85, 0.8, "sigmoid", colour="white", edge=INK, size=9)
    _arrow(ax, sx + 1.75, mid, sx + 2.05, mid)
    _box(ax, sx + 2.7, mid, 1.2, 0.8, "probability\nof AF", colour=AF_COLOUR, text_colour="white", size=9)
    ax.text(sx + 1.3, 4.15, "final layer", ha="center", va="bottom", fontsize=10, weight="bold",
            color=OTHER_COLOUR)
    ax.text(sx + 1.3, 1.15, "the Part 1 model,\nwith 64 inputs\ninstead of 2", ha="center", va="top", fontsize=9.5,
            color=OTHER_COLOUR)
    return _show(fig)


def _rect(x, y, w, h, colour):
    from matplotlib.patches import Rectangle
    return Rectangle((x, y), w, h, fc=colour, ec="white", lw=0.5)


# 10. Hand-over to the Multimodal lab ------------------------------------------------

def handover(our_embedding, row, ecgfounder=None, n_train=641, label="labelled AF"):
    """One ECG's 64-number embedding from our CNN beside its 1,024-number ECGFounder embedding.

    ``ecgfounder`` is the (N, 1024) array from the lab zip; by default it is loaded from there.
    """
    plt = _plt()
    if ecgfounder is None:
        from .data import unpack
        ecgfounder = np.load(unpack() / "ecgfounder_embeddings.npy", mmap_mode="r")
    ours = np.asarray(our_embedding, dtype=float).reshape(1, -1)
    theirs = np.asarray(ecgfounder[row], dtype=float).reshape(1, -1)
    fig, axes = plt.subplots(2, 1, figsize=(11, 3.2), gridspec_kw={"height_ratios": [1, 1]})
    for ax, values, title in [
            (axes[0], ours, f"Our CNN: {ours.shape[1]} numbers per ECG, learnt today from {n_train} training ECGs"),
            (axes[1], theirs, f"ECGFounder (Multimodal lab), a large pretrained encoder: {theirs.shape[1]:,} numbers "
                              "per ECG, learnt from more than 10 million ECGs")]:
        scaled = (values - values.mean()) / (values.std() + 1e-9)
        image = ax.imshow(scaled, aspect="auto", cmap="cividis", vmin=-2.5, vmax=2.5)
        ax.set_yticks([])
        ax.set_xticks([0, values.shape[1] - 1], ["1", f"{values.shape[1]:,}"])
        ax.set_title(title, fontsize=10, loc="left")
    fig.suptitle(f"Two embeddings of the same ECG (ECG {row}, {label}). Each stripe is one number, coloured by its value. "
                 "The order of the numbers means nothing,\nand the two do not line up: each is simply a list of "
                 "features that a simple model can use.", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 0.9, 1))
    bar = fig.add_axes([0.92, 0.12, 0.015, 0.62])
    fig.colorbar(image, cax=bar, label="value\n(standardised)")
    return _show(fig)


# The convolution by hand ---------------------------------------------------------

def convolution_by_hand(signal, kernel):
    """A short list of numbers and a short filter: every position, its sum, and the output it builds."""
    signal = [float(v) for v in signal]
    kernel = [float(v) for v in kernel]
    n, k = len(signal), len(kernel)
    rows = n - k + 1
    fig, ax = _canvas(10.5, 0.62 * rows + 2.3, xlim=(0, 10.5), ylim=(-2.1, rows + 1.1))

    def cells(y, values, highlight=(), colour=LIGHT, label=None, fmt="{:g}"):
        for i, v in enumerate(values):
            on = i in highlight
            ax.add_patch(_rect_edge(0.9 + i * 0.62, y - 0.25, 0.56, 0.5, ACCENT if on else "white", INK))
            ax.text(0.9 + i * 0.62 + 0.28, y, fmt.format(v), ha="center", va="center", fontsize=10,
                    color="white" if on else INK, weight="bold" if on else "normal")
        if label:
            ax.text(0.8, y, label, ha="right", va="center", fontsize=9.5)

    cells(rows + 0.5, signal, label="signal")
    for i in range(n):
        ax.text(0.9 + i * 0.62 + 0.28, rows + 0.95, str(i), ha="center", va="bottom", fontsize=8, color=GREY)
    outputs = []
    for r in range(rows):
        y = rows - 0.5 - r
        cells(y, signal, highlight=range(r, r + k), label=f"position {r}")
        terms = " + ".join(f"({w:g} \u00d7 {v:g})" for w, v in zip(kernel, signal[r:r + k]))
        total = sum(w * v for w, v in zip(kernel, signal[r:r + k]))
        outputs.append(total)
        ax.text(0.9 + n * 0.62 + 0.3, y, f"{terms} = {total:g}", ha="left", va="center", fontsize=10,
                family="monospace")
    cells(-0.9, outputs, label="output")
    ax.text(0.9 + n * 0.62 + 0.3, -0.9, f"filter: [{', '.join(f'{w:g}' for w in kernel)}]; orange: the "
            "numbers under it", ha="left", va="center", fontsize=10)
    ax.text(0.15, -1.75, f"The filter fits {n} - {k} + 1 = {rows} times, so there are {rows} outputs, one per "
            "position (a position is where the filter's first number sits).", ha="left", va="center", fontsize=10)
    return _show(fig)


def _rect_edge(x, y, w, h, colour, edge):
    from matplotlib.patches import Rectangle
    return Rectangle((x, y), w, h, fc=colour, ec=edge, lw=0.8)


# One parameter, step by step ------------------------------------------------------

def _logistic_loss(x, y, w, b):
    z = x @ w + b
    return float(np.mean(np.logaddexp(0, z) - y * z))


def gradient_descent_1d(x, y, weight, bias, steps=6):
    """Gradient descent on one parameter: the weight on RR irregularity, with the other two held at their
    trained values. Three learning rates: a good one, one too small and one too large."""
    plt = _plt()
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    weight = np.asarray(weight, dtype=float).reshape(-1)

    def loss(w2):
        return _logistic_loss(x, y, np.array([weight[0], w2]), bias)

    def slope(w2, h=1e-4):
        return (loss(w2 + h) - loss(w2 - h)) / (2 * h)

    curvature = (slope(weight[1] + 1e-3) - slope(weight[1] - 1e-3)) / 2e-3
    good = 0.6 / curvature
    rates = [("a good learning rate", good), ("too small: tiny steps", good / 8),
             ("too large: overshoots and bounces", 2.6 / curvature)]
    start = weight[1] - 1.6
    grid = np.linspace(weight[1] - 2.4, weight[1] + 2.4, 200)
    curve = [loss(g) for g in grid]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, (name, lr) in zip(axes, rates):
        ax.plot(grid, curve, color=INK, lw=1.5)
        w = start
        for i in range(steps):
            g = slope(w)
            new = w - lr * g
            ax.plot(w, loss(w), "o", color=ACCENT, ms=6, mec=INK, mew=0.6, zorder=3)
            ax.annotate("", xy=(new, loss(new)), xytext=(w, loss(w)),
                        arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=1.3, shrinkA=4, shrinkB=4,
                                        connectionstyle="arc3,rad=-0.25"))
            if i == 0:
                t = np.linspace(w - 0.7, w + 0.7, 2)
                ax.plot(t, loss(w) + g * (t - w) + 0.015, color=OTHER_COLOUR, lw=2.2)
                ax.text(w + 0.1, loss(w) + 0.03, "slope here\n(the gradient)", color=OTHER_COLOUR,
                        fontsize=9, ha="left", va="bottom", weight="bold")
            ax.text(w, loss(w) - 0.012, str(i), ha="center", va="top", fontsize=8, color=GREY)
            w = new
            if abs(w - weight[1]) > 2.3:
                break
        ax.plot(w, loss(w), "o", color=ACCENT, ms=6, mec=INK, mew=0.6, zorder=3)
        ax.set_title(f"{name} ({lr:.1f})", fontsize=10, loc="left")
        ax.set_xlabel("w2: weight on scaled RR irregularity")
        ax.set_xlim(grid[0], grid[-1])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Loss on the training ECGs")
    fig.suptitle("Gradient descent on one weight: each arrow is one step downhill, of size learning rate \u00d7 "
                 f"slope; the small numbers count the steps ({steps} each).\nThe other weight and the bias are held "
                 "at their trained values; the number in brackets is the learning rate.", fontsize=10, x=0.01,
                 ha="left")
    fig.tight_layout()
    return _show(fig)


def full_batch_vs_mini_batch(x, y, bias, start=(0.0, 0.4), lr=0.5, steps=60, batch_size=32, seed=0):
    """Full-batch gradient descent against stochastic (mini-batch) gradient descent on the same landscape."""
    plt = _plt()
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    rng = np.random.default_rng(seed)

    def gradient(w, rows):
        p = 1 / (1 + np.exp(-(x[rows] @ w + bias)))
        return x[rows].T @ (p - y[rows]) / len(rows)

    paths = {}
    for name, size in [("full batch: all training ECGs every step", len(y)),
                       (f"mini-batch: {batch_size} random ECGs each step", batch_size)]:
        w = np.array(start, dtype=float)
        path = [w.copy()]
        for _ in range(steps):
            rows = np.arange(len(y)) if size == len(y) else rng.choice(len(y), size, replace=False)
            w = w - lr * gradient(w, rows)
            path.append(w.copy())
        paths[name] = np.array(path)
    allp = np.vstack(list(paths.values()))
    lo, hi = allp.min(axis=0) - 0.35, allp.max(axis=0) + 0.35
    w1, w2 = np.meshgrid(np.linspace(lo[0], hi[0], 100), np.linspace(lo[1], hi[1], 100))
    scores = w1[..., None] * x[:, 0] + w2[..., None] * x[:, 1] + bias
    loss = np.mean(np.logaddexp(0, scores) - y * scores, axis=-1)
    fig, ax = plt.subplots(figsize=(8, 5.2))
    filled = ax.contourf(w1, w2, loss, levels=20, cmap="cividis")
    fig.colorbar(filled, ax=ax, label="Loss on the training ECGs (darker is lower)")
    for (name, path), colour in zip(paths.items(), ["white", ACCENT]):
        ax.plot(path[:, 0], path[:, 1], "-o", color=colour, ms=2.5, lw=1.4, label=name)
        ax.plot(*path[-1], "*", color=colour, mec=INK, ms=14)
    ax.plot(*start, "s", color="white", mec=INK, ms=8)
    legend = ax.legend(loc="lower right", fontsize=9, facecolor=GREY, framealpha=1, labelcolor="white")
    legend.get_frame().set_edgecolor(INK)
    ax.set_xlabel("w1: weight on scaled heart rate")
    ax.set_ylabel("w2: weight on scaled RR irregularity")
    ax.set_title(f"{steps} steps from the same start (square) to the stars, learning rate {lr}, bias held fixed.\n"
                 f"Each mini-batch gives a slightly different slope, so the path wobbles, but each step uses {batch_size}\n"
                 f"ECGs instead of all {len(y)}. Both end near the bottom.", fontsize=9.5, loc="left")
    fig.tight_layout()
    return _show(fig)


# What torch.no_grad() changes ------------------------------------------------------

def no_grad_diagram():
    """Training keeps a record of each calculation for the gradients; prediction under no_grad does not."""
    fig, ax = _canvas(11, 3.6, xlim=(0, 11), ylim=(0, 3.6))
    for y, title, colour in [(2.45, "Training", PART_COLOURS["1"]), (0.75, "Prediction, inside torch.no_grad()",
                                                                        PART_COLOURS["2"])]:
        ax.text(0.1, y + 0.75 if y > 2 else y + 0.62, title, fontsize=10.5, weight="bold", color=colour)
        _box(ax, 1.0, y, 1.3, 0.6, "ECGs", colour="white", edge=colour)
        _arrow(ax, 1.65, y, 2.45, y)
        _box(ax, 3.1, y, 1.3, 0.6, "model", colour=LIGHT)
        _arrow(ax, 3.75, y, 4.55, y)
        _box(ax, 5.2, y, 1.3, 0.6, "scores", colour="white", edge=colour)
    _arrow(ax, 5.85, 2.45, 6.65, 2.45)
    _box(ax, 7.3, 2.45, 1.3, 0.6, "loss", colour="white", edge=PART_COLOURS["1"])
    ax.plot([1.0, 7.3], [2.0, 2.0], color=ACCENT, lw=6, alpha=0.35, solid_capstyle="round")
    ax.text(4.15, 1.88, "PyTorch keeps a record of every calculation (uses memory and time)", ha="center",
            va="top", fontsize=9, color=ACCENT)
    ax.annotate("", xy=(1.0, 2.95), xytext=(7.3, 2.95),
                arrowprops=dict(arrowstyle="-|>", color=PART_COLOURS["1"], lw=1.4, connectionstyle="arc3,rad=0.12"))
    ax.text(8.1, 2.45, "loss.backward() follows the\nrecord back to get the gradients;\noptimiser.step() then "
            "changes\nthe model", va="center", fontsize=9)
    ax.text(6.0, 0.75, "no record: less memory, faster.\nWe do not call backward() or\noptimiser.step() "
            "when making predictions", va="center", fontsize=9)
    return _show(fig)
