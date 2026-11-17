"""Checks for 01_intro_to_deep_learning.

Each checker prints one line through ``_core`` and returns a result that is
truthy when the answer is right (and does not echo True or False under the
cell). Nothing here is a worked answer: ``train_step`` runs the student's
function on a made-up problem whose loss and updated parameters are stored
below; ``test_auroc`` compares with the AUROC the notebook's seeded pipeline
gives (and those of the usual mistakes); ``best_epoch`` and ``saliency`` check
properties of the student's result.
"""

import math

import numpy as np

from . import _core

NOTEBOOK = "intro_to_deep_learning"
CODES = {"e2": "099d065227", "2a": "e155dd32da",
         "3a": "d5b3c90409", "3b": "3776711b13", "4a": "077642a3f7"}
EXPLANATIONS = {
    "e2": "Too large a step jumps across the valley instead of going down into it.",
    "2a": "AUROC is about ranking: it says nothing about any one cut-off or any one patient.",
    "3a": "Training loss alone would never show this: only data the model does not learn from can.",
    "3b": "Every choice made by looking at the test set makes its score a little less honest.",
    "4a": "The same idea as the CNN's 64 numbers, from a much bigger encoder trained on far more ECGs.",
}

# TODO 1: a 2-feature model with weights (0.5, -0.25) and bias 0.1, on four made-up rows,
# one step of SGD with lr=0.1 and BCEWithLogitsLoss.
TOY_X = [[1.0, 2.0], [-1.0, 0.5], [0.0, -1.0], [2.0, 1.0]]
TOY_Y = [[1.0], [0.0], [1.0], [0.0]]
TOY_WEIGHT, TOY_BIAS, TOY_LR = [[0.5, -0.25]], 0.1, 0.1
TOY_LOSS = 0.712089  # the loss before the step
TOY_WEIGHT_AFTER, TOY_BIAS_AFTER = [0.486139, -0.258744], 0.095404

# TODO 2: the seeded Part 1 model (SGD, lr 0.5, 100 steps) on the notebook's split.
# Measured on CPU with torch 2.15 and scikit-learn 1.9 (the weights move by under 0.01 between versions).
TRAINED_WEIGHT, TRAINED_BIAS = [0.484, 0.977], -2.580
TEST_AUROC = 0.890  # the right answer
TRAIN_AUROC = 0.911  # the same model scored on the training ECGs
THRESHOLD_AUROC = 0.548  # yes or no predictions: roc_auc_score(af_test, test_scores > 0)
AUROC_TOLERANCE = 0.015  # a correct answer within this passes, whatever the weights
MISTAKE_TOLERANCE = 0.005  # the training AUROC (0.911) sits 0.021 away, outside AUROC_TOLERANCE
WEIGHT_TOLERANCE = 0.25  # only for choosing the 'rerun Part 1' hint


class Result:
    """Pass or fail, without a stray True or False shown under the feedback."""

    __slots__ = ("passed",)

    def __init__(self, passed):
        self.passed = bool(passed)

    def __bool__(self):
        return self.passed

    def __repr__(self):
        return repr(self.passed)

    def _ipython_display_(self):
        pass


def check(question, answer):
    """Check a multiple-choice answer, for example check("1a", "c")."""
    return Result(_core.choice(NOTEBOOK, CODES, question, answer, EXPLANATIONS))


def _toy_problem():
    import torch
    import torch.nn as nn
    model = nn.Linear(2, 1)
    with torch.no_grad():
        model.weight.copy_(torch.tensor(TOY_WEIGHT))
        model.bias.fill_(TOY_BIAS)
    optimiser = torch.optim.SGD(model.parameters(), lr=TOY_LR)
    return model, optimiser, torch.tensor(TOY_X), torch.tensor(TOY_Y)


def train_step(fn):
    """TODO 1: one step of gradient descent that returns the loss before the step."""
    task = "TODO 1"
    model, optimiser, x, y = _toy_problem()
    try:
        returned = fn(model, optimiser, x, y)
    except NameError as error:
        return Result(_core.failed(task, f"Python does not know a name ({error}). Run the cells above this one "
                                         "first (loss_fn is defined in the cell about the loss)."))
    except Exception as error:  # a student's code can raise anything
        return Result(_core.failed(task, f"your function stopped with {type(error).__name__}: {error}. "
                                         "Check each of the three lines against the steps above."))
    try:
        returned = float(returned)
    except (TypeError, ValueError):
        return Result(_core.failed(task, "keep the last line, return loss.item(), so the function returns the loss."))
    weight = model.weight.detach().view(-1).tolist()
    bias = float(model.bias.detach())
    unchanged = np.allclose(weight, TOY_WEIGHT[0]) and math.isclose(bias, TOY_BIAS, abs_tol=1e-6)
    has_gradients = model.weight.grad is not None and float(model.weight.grad.abs().sum()) > 0
    if math.isnan(returned) and unchanged and not has_gradients:
        return Result(_core.waiting(task))
    if not math.isclose(returned, TOY_LOSS, abs_tol=1e-4):
        return Result(_core.failed(task, "the loss is not the one the model's scores give. Pass the scores "
                                         "model(x) straight to loss_fn (no sigmoid: the loss applies it), "
                                         "with the scores first and the labels y second."))
    if unchanged and not has_gradients:
        return Result(_core.failed(task, "the parameters did not change and no gradients were worked out. "
                                         "Call loss.backward() after computing the loss."))
    if unchanged:
        return Result(_core.failed(task, "the gradients were worked out but the parameters did not move. "
                                         "Call optimiser.step() after loss.backward(), not before."))
    if not (np.allclose(weight, TOY_WEIGHT_AFTER, atol=1e-4) and math.isclose(bias, TOY_BIAS_AFTER, abs_tol=1e-4)):
        return Result(_core.failed(task, "the parameters moved, but not by one step downhill. Call backward() "
                                         "and step() once each, in that order."))
    return Result(_core.passed(task, "correct: one forward pass, one backward pass and one step."))


def test_auroc(value, model):
    """TODO 2: the AUROC of the Part 1 model on the test patients.

    The right value passes on its own. The model's weights are only used to choose
    a hint when the value is wrong, so a correct answer never fails because the
    weights differ slightly on another machine or PyTorch version.
    """
    task = "TODO 2"
    if _core.is_placeholder(value):
        return Result(_core.waiting(task))
    try:
        value = float(value)
    except (TypeError, ValueError):
        return Result(_core.failed(task, "linear_auroc should be one number, from roc_auc_score(labels, scores)."))
    if not math.isfinite(value):
        return Result(_core.failed(task, "linear_auroc is not a number. Use roc_auc_score(af_test, test_scores)."))
    if math.isclose(value, TEST_AUROC, abs_tol=AUROC_TOLERANCE):
        return Result(_core.passed(task, f"correct: a test AUROC of {value:.3f}."))
    if math.isclose(value, TRAIN_AUROC, abs_tol=MISTAKE_TOLERANCE):
        return Result(_core.failed(task, "this is the AUROC on the training ECGs. Use the test patients' labels "
                                         "and scores: af_test and test_scores."))
    if math.isclose(value, THRESHOLD_AUROC, abs_tol=MISTAKE_TOLERANCE):
        return Result(_core.failed(task, "this looks like yes or no predictions rather than scores. AUROC needs the "
                                         "scores themselves: pass test_scores as they are."))
    if value < 0.5 and math.isclose(1 - value, TEST_AUROC, abs_tol=AUROC_TOLERANCE):
        return Result(_core.failed(task, "the scores are back to front (an AUROC below 0.5). Pass the labels first "
                                         "and the scores, not minus the scores, second."))
    weight = model.weight.detach().cpu().view(-1).numpy()
    bias = float(model.bias.detach().cpu())
    if not (np.allclose(weight, TRAINED_WEIGHT, atol=WEIGHT_TOLERANCE)
            and math.isclose(bias, TRAINED_BIAS, abs_tol=WEIGHT_TOLERANCE)):
        return Result(_core.failed(task, "the model is not the one Part 1 trains. Check that TODO 1 shows ✅, then "
                                         "run the 'Watch it learn' cell and the cells after it again."))
    return Result(_core.failed(task, "not the expected value. Use roc_auc_score with the test labels (af_test) "
                                     "first and the test scores (test_scores) second."))


def best_epoch(best, val_losses, train_losses):
    """TODO 3: the position of the lowest validation loss, as a whole number the notebook can index with."""
    task = "TODO 3"
    if _core.is_placeholder(best):
        return Result(_core.waiting(task))
    val_losses, train_losses = np.asarray(val_losses, dtype=float), np.asarray(train_losses, dtype=float)
    if np.isnan(train_losses).any():
        return Result(_core.waiting(task, "waiting for TODO 1: the CNN has not learnt anything yet. Finish TODO 1, "
                                          "run the training cell again, then this one."))
    if isinstance(best, (bool, np.bool_)) or not isinstance(best, (int, np.integer)):
        try:
            number = float(best)
        except (TypeError, ValueError):
            number = math.nan
        if math.isfinite(number) and np.isclose(number, val_losses.min()):
            return Result(_core.failed(task, "this is the lowest validation loss itself. We need its position in "
                                             "the list: np.argmin, not np.min."))
        return Result(_core.failed(task, "best should be a whole number, a position in val_losses, as np.argmin "
                                         "gives."))
    position = int(best)
    if not 0 <= position < len(val_losses):
        return Result(_core.failed(task, f"best should be a position from 0 to {len(val_losses) - 1}."))
    if val_losses[position] <= val_losses.min():
        return Result(_core.passed(task, f"correct: epoch {position + 1} has the lowest validation loss."))
    if position == len(val_losses) - 1:
        return Result(_core.waiting(task, "still the placeholder, which keeps the last epoch. Find the position "
                                          "of the lowest validation loss instead."))
    if train_losses[position] <= train_losses.min():
        return Result(_core.failed(task, "this is the epoch with the lowest training loss. The training loss "
                                         "keeps falling as the network overfits: use val_losses."))
    if position > 0 and val_losses[position - 1] <= val_losses.min():
        return Result(_core.failed(task, "one too far: this is the epoch number. Positions count from 0, so keep "
                                         "np.argmin's answer as it is."))
    if val_losses[position] >= val_losses.max():
        return Result(_core.failed(task, "this is the highest validation loss. We want the lowest: np.argmin."))
    return Result(_core.failed(task, "this is not the position of the lowest validation loss. Use "
                                     "np.argmin(val_losses)."))


SALIENCY_LENGTH = 2500


def saliency(values):
    """Extension 4: one finite, non-negative value for each of the 2,500 time points, not all zero."""
    task = "Extension 4"
    if _core.is_placeholder(values):
        return Result(_core.waiting(task))
    try:
        values = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return Result(_core.failed(task, "saliency should be an array of numbers, from ecg_in.grad."))
    if values.ndim != 1:
        return Result(_core.failed(task, f"expected one value per time point, got shape {values.shape}. "
                                         "Flatten it with .view(-1)."))
    if values.size != SALIENCY_LENGTH:
        return Result(_core.failed(task, f"expected {SALIENCY_LENGTH} values, one per time point, got "
                                         f"{values.size}. Keep every time point of ecg_in.grad."))
    if not np.isfinite(values).all():
        return Result(_core.failed(task, "some values are not numbers. Call .backward() on the CNN's score for "
                                         "ecg_in, then use ecg_in.grad."))
    if not np.any(values):
        return Result(_core.waiting(task))
    if np.any(values < 0):
        return Result(_core.failed(task, "some values are negative. Take the size of each gradient with .abs()."))
    return Result(_core.passed(task, "a saliency value for every time point."))
