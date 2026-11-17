"""ECG helpers for the Intro to Deep Learning notebook: rhythm features and plots.

Students call these in one line each and do not need to read them. The rhythm
features are what a clinician reads off a rhythm strip: the heart rate and how
irregular the gaps between beats (the RR intervals) are. R peaks are found with
a band-pass filter and SciPy's peak finder, which is good enough on these
z-scored 10-second recordings; it is not a validated QRS detector.
"""

import numpy as np
import pandas as pd

from .data import FS

PEAK_BAND = (5, 20)  # Hz: keeps the sharp QRS complex, drops baseline wander and most T waves
PEAK_HEIGHT = 0.4  # a peak must reach this fraction of the recording's 99th percentile
REFRACTORY = 0.3  # seconds: no two beats closer than this (a heart rate of 200 a minute)
AF_COLOUR, OTHER_COLOUR = "#d63a3a", "#2a78d6"


def r_peaks(signal, fs=FS):
    """Sample positions of the R peaks in one lead."""
    from scipy.signal import butter, filtfilt, find_peaks
    b, a = butter(2, [PEAK_BAND[0] / (fs / 2), PEAK_BAND[1] / (fs / 2)], btype="band")
    energy = np.abs(filtfilt(b, a, np.asarray(signal, dtype=float)))
    peaks, _ = find_peaks(energy, height=PEAK_HEIGHT * np.percentile(energy, 99),
                          distance=int(REFRACTORY * fs))
    return peaks


def rhythm_features(ecgs, lead=1, fs=FS):
    """Heart rate (beats a minute) and RR irregularity for each ECG, from one lead (II by default).

    RR irregularity is the standard deviation of the gaps between beats divided by
    their mean: 0 for a perfectly regular rhythm, higher the more the gaps vary.
    """
    rows = []
    for signal in ecgs[:, lead]:
        rr = np.diff(r_peaks(signal, fs)) / fs
        if len(rr) < 2:
            rows.append((np.nan, np.nan))
        else:
            rows.append((60 / np.median(rr), np.std(rr) / np.mean(rr)))
    return pd.DataFrame(rows, columns=["heart_rate", "rr_irregularity"])


def _plt():
    import matplotlib.pyplot as plt
    return plt


def _show(fig, live=False):
    """Show a figure; a live figure replaces the previous one in the same output."""
    plt = _plt()
    try:
        from IPython import get_ipython
        from IPython.display import clear_output, display
        in_notebook = get_ipython() is not None
    except ImportError:
        in_notebook = False
    if in_notebook:
        if live:
            clear_output(wait=True)
        display(fig)
    plt.close(fig)


def plot_rhythm(ecgs, meta, rows, lead=1, seconds=10, fs=FS):
    """Lead II of a few ECGs, with each detected beat marked and the gaps between beats labelled."""
    plt = _plt()
    fig, axes = plt.subplots(len(rows), 1, figsize=(12, 2.3 * len(rows)), squeeze=False)
    for ax, i in zip(axes[:, 0], rows):
        signal = ecgs[i, lead, :seconds * fs]
        t = np.arange(len(signal)) / fs
        peaks = r_peaks(ecgs[i, lead], fs)
        peaks = peaks[peaks < len(signal)]
        ax.plot(t, signal, lw=0.8, color="#333333")
        ax.plot(t[peaks], signal[peaks], "o", color=AF_COLOUR, ms=4)
        for left, right in zip(peaks[:-1], peaks[1:]):
            ax.text((left + right) / 2 / fs, ax.get_ylim()[1] * 0.85, f"{(right - left) / fs:.2f}",
                    ha="center", fontsize=7, color="#52514e")
        report = meta["machine_report"].iloc[i]
        ax.set_title(f"ECG {i}, lead II. Report: {report[:110]}", fontsize=9, loc="left")
        ax.set_yticks([])
        ax.set_xlim(0, seconds)
    axes[-1, 0].set_xlabel("Time (s). Red dots: detected beats; numbers: seconds between beats")
    fig.tight_layout()
    _show(fig)


def plot_features(features, af, ax=None):
    """Scatter plot of heart rate against RR irregularity, coloured by AF."""
    plt = _plt()
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(6, 4.5))
    af = np.asarray(af).astype(bool)
    x, y = features["heart_rate"].to_numpy(), features["rr_irregularity"].to_numpy()
    ax.scatter(x[~af], y[~af], s=10, alpha=0.5, color=OTHER_COLOUR, label="no AF in the report")
    ax.scatter(x[af], y[af], s=14, alpha=0.8, color=AF_COLOUR, label="AF in the report")
    ax.set_xlabel("Heart rate (beats a minute)")
    ax.set_ylabel("RR irregularity")
    ax.set_xlim(30, 190)
    ax.set_ylim(-0.02, 0.62)
    ax.legend(loc="upper left", fontsize=8)
    if own:
        fig.tight_layout()
        _show(fig)
        return None
    return ax


class TrainingPlot:
    """The live picture of Part 1: the decision boundary moving, and the loss falling.

    The model sees each feature scaled (minus its mean, divided by its standard
    deviation); the plot draws the boundary back in heart rate and irregularity.
    """

    def __init__(self, features, af, mean, std, model=None):
        self.features, self.af = features, af
        self.mean, self.std = np.asarray(mean, dtype=float), np.asarray(std, dtype=float)
        # (w1, w2) after each update, starting from the untrained model if given: the path training took
        self.path = [] if model is None else [model.weight.detach().cpu().numpy().ravel().tolist()]

    def update(self, model, losses):
        plt = _plt()
        weight = model.weight.detach().cpu().numpy().ravel()
        bias = float(model.bias.detach().cpu().numpy().ravel()[0])
        self.path.append(weight.tolist())
        fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.3))
        plot_features(self.features, self.af, ax=left)
        # Probability 0.5 where w1 * (hr - m1) / s1 + w2 * (irr - m2) / s2 + b = 0
        hr = np.linspace(30, 190, 200)
        if abs(weight[1]) > 1e-8:
            irr = self.mean[1] - self.std[1] / weight[1] * (weight[0] * (hr - self.mean[0]) / self.std[0] + bias)
            left.plot(hr, irr, color="black", lw=2, label="probability of AF = 0.5")
            left.legend(loc="upper left", fontsize=8)
        left.set_title("The model's decision boundary", fontsize=10)
        right.plot(range(1, len(losses) + 1), losses, color="black")
        right.set_xlim(0, max(len(losses), 10))
        right.set_ylim(0, max(0.8, max(losses) * 1.05))
        right.set_xlabel("Training step")
        right.set_ylabel("Loss (lower is better)")
        right.set_title(f"Step {len(losses)}: loss {losses[-1]:.3f}. "
                        f"Weights {weight[0]:.2f}, {weight[1]:.2f}; bias {bias:.2f}", fontsize=10)
        fig.tight_layout()
        _show(fig, live=True)


def plot_learning_rates(curves):
    """Loss curves for several learning rates: {learning rate: [loss after each step]}."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4))
    for lr, losses in curves.items():
        ax.plot(range(1, len(losses) + 1), losses, label=f"learning rate {lr:g}")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss (lower is better)")
    ax.set_ylim(0, None)
    ax.legend()
    fig.tight_layout()
    _show(fig)


def plot_scores(scores, af, title):
    """Histogram of the model's scores, AF and not AF, to see how well they separate."""
    plt = _plt()
    af = np.asarray(af).astype(bool)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    bins = np.linspace(np.min(scores), np.max(scores), 30)
    ax.hist(scores[~af], bins=bins, alpha=0.6, color=OTHER_COLOUR, label="no AF in the report")
    ax.hist(scores[af], bins=bins, alpha=0.7, color=AF_COLOUR, label="AF in the report")
    ax.set_xlabel("Model's score (higher means more likely AF)")
    ax.set_ylabel("Number of ECGs")
    ax.set_title(title, fontsize=10)
    ax.legend()
    fig.tight_layout()
    _show(fig)


def plot_filter(signal, kernel, output, fs=FS, seconds=3):
    """A signal, the filter slid along it, and what comes out."""
    plt = _plt()
    n = seconds * fs
    t = np.arange(n) / fs
    fig, axes = plt.subplot_mosaic([["input", "input"], ["filter", "note"], ["output", "output"]],
                                   figsize=(12, 6), width_ratios=[1, 5], height_ratios=[2, 1.2, 2])
    axes["input"].plot(t, signal[:n], lw=0.8, color="#333333")
    axes["input"].set_title("Input: lead II", fontsize=10, loc="left")
    axes["filter"].stem(np.arange(len(kernel)), kernel, basefmt=" ")
    axes["filter"].set_title(f"The filter: {len(kernel)} numbers", fontsize=10, loc="left")
    axes["filter"].set_xlabel("Position in the filter")
    axes["note"].axis("off")
    axes["note"].text(0, 0.5, f"The filter covers {len(kernel)} samples ({len(kernel) / fs:.3f} s). It slides along the "
                      "input one sample at a time;\nat each position, multiply the filter by the signal under it "
                      "and add up: that is one output value.", va="center", fontsize=9)
    axes["output"].plot(t[:len(output[:n])], output[:n], lw=0.8, color=AF_COLOUR)
    axes["output"].set_title("Output: the filter's response at each point in time", fontsize=10, loc="left")
    axes["output"].set_xlabel("Time (s)")
    for name in ("input", "output"):
        axes[name].set_yticks([])
    fig.tight_layout()
    _show(fig)


class LossPlot:
    """The live picture of Part 3: training and validation loss after each epoch."""

    def __init__(self, epochs):
        self.epochs = epochs

    def update(self, train_losses, val_losses, seconds=None):
        plt = _plt()
        fig, ax = plt.subplots(figsize=(8, 4))
        epochs = range(1, len(train_losses) + 1)
        ax.plot(epochs, train_losses, marker="o", ms=3, label="training ECGs")
        ax.plot(epochs, val_losses, marker="o", ms=3, label="validation ECGs")
        ax.set_xlim(0.5, self.epochs + 0.5)
        ax.set_ylim(0, min(1.0, max(max(train_losses), max(val_losses)) * 1.1))
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Average loss (lower is better)")
        done = f"Epoch {len(train_losses)} of {self.epochs}"
        ax.set_title(done + (f", {seconds:.0f} s so far" if seconds is not None else ""), fontsize=10)
        ax.legend()
        fig.tight_layout()
        _show(fig, live=True)


def plot_embedding(points, af, title):
    """Two-dimensional map of the embeddings (for example from PCA), coloured by AF."""
    plt = _plt()
    af = np.asarray(af).astype(bool)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(points[~af, 0], points[~af, 1], s=10, alpha=0.5, color=OTHER_COLOUR, label="no AF in the report")
    ax.scatter(points[af, 0], points[af, 1], s=14, alpha=0.8, color=AF_COLOUR, label="AF in the report")
    ax.set_xlabel("First principal component")
    ax.set_ylabel("Second principal component")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _show(fig)


def show_aurocs(results):
    """Print test AUROCs side by side; a task not done yet shows as such."""
    for name, value in results.items():
        if value is ... or value is None:
            print(f"{name:32s} not computed yet")
        else:
            print(f"{name:32s} test AUROC {float(value):.3f}")


def plot_saliency(signal, saliency, report, fs=FS, seconds=10):
    """An ECG lead with its saliency map underneath: where a small change most moves the score."""
    plt = _plt()
    n = seconds * fs
    t = np.arange(n) / fs
    fig, axes = plt.subplots(2, 1, figsize=(12, 4), sharex=True)
    axes[0].plot(t, signal[:n], lw=0.8, color="#333333")
    axes[0].set_title(f"Lead II. Report: {report[:110]}", fontsize=9, loc="left")
    axes[1].fill_between(t, np.asarray(saliency)[:n], color=AF_COLOUR, lw=0)
    axes[1].set_title("Saliency: how much the score changes with a small change at each moment", fontsize=9,
                      loc="left")
    axes[1].set_xlabel("Time (s)")
    for ax in axes:
        ax.set_yticks([])
    fig.tight_layout()
    _show(fig)


def plot_calibration(curves):
    """Calibration curves: {name: (observed fraction, mean predicted probability)} from calibration_curve."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], ls="--", color="grey", label="perfectly calibrated")
    for name, (observed, predicted) in curves.items():
        ax.plot(predicted, observed, marker="o", label=name)
    ax.set_xlabel("Average predicted probability of AF")
    ax.set_ylabel("Fraction that had AF")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _show(fig)
