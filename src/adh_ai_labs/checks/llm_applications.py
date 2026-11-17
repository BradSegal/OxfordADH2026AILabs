"""Checks for 02_llm_applications (how a language model reads and writes).

Each checker prints one line through ``_core`` and returns a result that is
truthy when the answer is right. Nothing here is a worked answer: ``cosine`` and
``softmax`` run the student's function on small made-up inputs whose answers are
stored below; ``search`` checks properties of the student's ranking (and two
stored numbers for one query); ``generation`` compares with the stored greedy
continuation of the notebook's prompt. ``queries`` checks only the shape: what
the student learns there is in what the searches return.
"""

import math

import numpy as np

from . import _core

NOTEBOOK = "llm_applications"
CODES = {"3a": "2a3a621860"}
EXPLANATIONS = {
    "3a": "The positive and negated phrases stay close in this model's embedding space, so similarity "
          "search ranks them together.",
}

# TODO 1: cosine similarity on made-up vectors (including different lengths, negatives and opposites).
COSINE_CASES = [([1, 2, 2], [2, 1, 2], 8 / 9),
                ([3, 4], [6, 8], 1.0),  # same direction, twice as long
                ([1, 0], [0, 2], 0.0),  # at right angles
                ([1, 2], [-2, -4], -1.0),  # opposite
                ([1, -1, 2, 0], [0, 3, -1, 1], -5 / math.sqrt(66))]

# TODO 2: the notebook's query for the check, with the biomedical model over the 691 unique reports
# (measured 17 Nov 2026, CPU; a GPU gives the same to three decimal places).
SEARCH_QUERY, SEARCH_K = "heart rate too slow", 10
SEARCH_EXACT = "Atrial fibrillation"  # a report text, so its best score is 1
SEARCH_REPORTS = 691
SEARCH_MAX, SEARCH_MEAN, SEARCH_TOLERANCE = 0.537, 0.296, 0.01

# TODO 4: softmax of the scores [2, 1, 0] at three temperatures, and of two large scores.
SOFTMAX_SCORES = [2.0, 1.0, 0.0]
SOFTMAX_EXPECTED = {1.0: [0.665241, 0.244728, 0.090031],
                    0.5: [0.866813, 0.117310, 0.015876],
                    2.0: [0.506480, 0.307196, 0.186324]}
LARGE_SCORES, LARGE_EXPECTED = [1000.0, 999.0], [0.731059, 0.268941]

# TODO 5: the greedy continuation of the notebook's prompt by Qwen2.5 0.5B Instruct on the CPU
# (" a rate of 180 beats per minute. The nurse should instruct the patient to take which").
GENERATION_PROMPT = "The patient's ECG shows atrial fibrillation with"
GENERATION_CHECK_STEPS = 3  # the check runs the first steps; the cell then runs all 20
GENERATION_PROMPT_TOKENS = [785, 8720, 594, 468, 8798, 4933, 29570, 530, 94135, 66084, 448]
GREEDY_CONTINUATION = [264, 4379, 315, 220, 16, 23, 15, 33327, 817, 9383, 13, 576, 28098, 1265, 21066, 279,
                       8720, 311, 1896, 892]


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
    """Check a multiple-choice answer, for example check("3a", "c")."""
    return Result(_core.choice(NOTEBOOK, CODES, question, answer, EXPLANATIONS))


def _number(value):
    """A plain float from a Python or NumPy number (or a one-element array), else None."""
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    return float(array.item()) if array.size == 1 else None


def unfinished(fn):
    """True if the source of ``fn`` still holds a ``...`` placeholder (comments and strings ignored).

    The partly written TODOs (for example ``scores = report_vectors @ ...``) can stop
    with an error before they return the placeholder; this tells "not done yet" from a
    real mistake. False when the source cannot be read.
    """
    import inspect
    import io
    import textwrap
    import tokenize
    try:
        source = textwrap.dedent(inspect.getsource(fn))
    except (OSError, TypeError):
        return False
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type == tokenize.OP and token.string == "...":
                return True
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    return False


def _crashed(task, error):
    return Result(_core.failed(task, f"your code stopped with {type(error).__name__}: {error}. "
                                     "Check it line by line against the steps above."))


# TODO 1 -----------------------------------------------------------------------------

def cosine(fn):
    """TODO 1: cosine(a, b) = a . b / (|a| |b|) for NumPy vectors."""
    task = "TODO 1"
    if unfinished(fn):
        return Result(_core.waiting(task))
    results = []
    for a, b, expected in COSINE_CASES:
        try:
            got = fn(np.array(a, dtype=float), np.array(b, dtype=float))
        except Exception as error:  # a student's code can raise anything
            return _crashed(task, error)
        if got is ...:
            return Result(_core.waiting(task))
        if got is None:
            return Result(_core.failed(task, "your function returned None. Check that it ends with return and the value you calculated."))
        results.append((a, b, expected, got))
    first = results[0][3]
    if np.ndim(first) != 0 and np.size(first) > 1:
        return Result(_core.failed(task, "your function returns several numbers, not one. a * b multiplies "
                                         "element by element; a @ b also adds them up (the dot product)."))
    values = [_number(got) for *_, got in results]
    if any(v is None or not math.isfinite(v) for v in values):
        return Result(_core.failed(task, "your function should return one finite number for each pair of vectors."))
    if all(math.isclose(v, e, abs_tol=1e-6) for v, (*_, e, _) in zip(values, results)):
        return Result(_core.passed(task, "correct: the dot product divided by the two lengths."))
    a, b = np.array(COSINE_CASES[1][0], float), np.array(COSINE_CASES[1][1], float)
    if math.isclose(values[1], a @ b, abs_tol=1e-6):
        return Result(_core.failed(task, "that is the dot product a @ b on its own. lengths should be the two "
                                         "lengths multiplied: np.linalg.norm(a) * np.linalg.norm(b)."))
    if math.isclose(values[1], a @ b / np.linalg.norm(a), abs_tol=1e-6) or math.isclose(
            values[1], a @ b / np.linalg.norm(b), abs_tol=1e-6):
        return Result(_core.failed(task, "you divided by only one of the lengths. lengths should be both "
                                         "multiplied: np.linalg.norm(a) * np.linalg.norm(b)."))
    if math.isclose(values[1], a @ b / (a @ a * (b @ b)), abs_tol=1e-6):
        return Result(_core.failed(task, "you divided by the squared lengths. np.linalg.norm(a) is the length "
                                         "itself (the square root of a @ a)."))
    if math.isclose(values[1], a @ b / np.linalg.norm(a) * np.linalg.norm(b), abs_tol=1e-6):
        return Result(_core.failed(task, "Python divides by the first length and then multiplies by the second. "
                                         "Put the two lengths in brackets: a @ b / (... * ...)."))
    return Result(_core.failed(task, f"for a = {COSINE_CASES[0][0]} and b = {COSINE_CASES[0][1]} the answer is "
                                     f"8 / 9 = 0.89, as worked above; your function gave {values[0]:.3g}."))


# TODO 2 -----------------------------------------------------------------------------

def search(fn):
    """TODO 2: search(query, k) returns the positions of the k highest scores, highest first, and every score."""
    task = "TODO 2"
    if unfinished(fn):
        return Result(_core.waiting(task))
    try:
        returned = fn(SEARCH_QUERY, SEARCH_K)
    except Exception as error:  # a student's code can raise anything
        return _crashed(task, error)
    if not isinstance(returned, tuple) or len(returned) != 2:
        return Result(_core.failed(task, "keep the last line, return top, scores."))
    top, scores = returned
    if _core.is_placeholder(top) or _core.is_placeholder(scores):
        return Result(_core.waiting(task))
    try:
        scores = np.asarray(scores, dtype=float) if np.ndim(scores) else np.array([scores], dtype=float)
        top = np.asarray(top)
    except (TypeError, ValueError) as error:
        return _crashed(task, error)
    if scores.shape != (SEARCH_REPORTS,):
        return Result(_core.failed(task, f"scores should hold one number per report ({SEARCH_REPORTS}), but its "
                                         f"shape is {scores.shape}. Multiply report_vectors (one row per report) "
                                         "by query_vector: report_vectors @ query_vector."))
    if not (math.isclose(scores.max(), SEARCH_MAX, abs_tol=SEARCH_TOLERANCE)
            and math.isclose(scores.mean(), SEARCH_MEAN, abs_tol=SEARCH_TOLERANCE)):
        return Result(_core.failed(task, "the scores are not the cosine similarities of the reports with the "
                                         "query. Use report_vectors @ query_vector (both come from the biomedical "
                                         "model and have length 1)."))
    if top.dtype.kind == "f" and np.all((top >= -1) & (top <= 1)):
        return Result(_core.failed(task, "top holds scores, not positions. np.sort gives the values; "
                                         "np.argsort gives the positions that would sort them."))
    if top.ndim != 1 or top.dtype.kind not in "iu":
        return Result(_core.failed(task, "top should be a list of report positions (whole numbers) from "
                                         "np.argsort."))
    if len(top) != SEARCH_K:
        return Result(_core.failed(task, f"search(query, k={SEARCH_K}) returned {len(top)} positions. "
                                         "Keep the first k with [:k]."))
    if np.any(top < 0) or np.any(top >= SEARCH_REPORTS) or len(set(top.tolist())) != SEARCH_K:
        return Result(_core.failed(task, "top should hold k different report positions."))
    chosen = scores[top]
    lowest = np.sort(scores)[:SEARCH_K]
    if np.allclose(np.sort(chosen), lowest):
        return Result(_core.failed(task, "these are the k lowest scores: np.argsort sorts from low to high. "
                                         "Reverse it with [::-1] before taking the first k with [:k]."))
    if np.any(np.diff(chosen) > 1e-9):
        return Result(_core.failed(task, "the closest report should come first: reverse the argsort with [::-1] "
                                         "before [:k]."))
    if np.delete(scores, top).max() > chosen.min() + 1e-9:
        return Result(_core.failed(task, "a report that was not returned scores higher than one that was. Take "
                                         "the first k of the reversed argsort of all the scores."))
    try:  # the query must be used: an exact report text scores 1 against itself
        other_top, other_scores = fn(SEARCH_EXACT, SEARCH_K)
        other_scores = np.asarray(other_scores, dtype=float)
    except Exception as error:  # a student's code can raise anything
        return _crashed(task, error)
    if other_scores.shape != (SEARCH_REPORTS,) or not math.isclose(float(other_scores.max()), 1.0, abs_tol=0.01):
        return Result(_core.failed(task, "search should embed the query it is given: query_vector comes from "
                                         "query, so a different query gives different scores."))
    for k in (3, SEARCH_REPORTS):  # k must be used, not a fixed 10 (Extension 7 asks for every report)
        try:
            other = np.asarray(fn(SEARCH_QUERY, k)[0])
        except Exception as error:  # a student's code can raise anything
            return _crashed(task, error)
        if other.ndim != 1 or len(other) != k:
            return Result(_core.failed(task, f"search(query, k={k}) returned {len(other) if other.ndim else 1} "
                                             "positions. Use the k you are given: [:k], not a fixed number."))
    return Result(_core.passed(task, "correct: every report scored, and the k highest returned, highest first."))


# TODO 3 -----------------------------------------------------------------------------

def queries(values):
    """TODO 3: three different, non-empty searches (the lesson is in what they return)."""
    task = "TODO 3"
    if _core.is_placeholder(values) or (isinstance(values, (list, tuple)) and any(v is ... for v in values)):
        return Result(_core.waiting(task, "not done yet. Replace each ... with a search of your own, in quotes."))
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        return Result(_core.failed(task, "QUERIES should be a list of three searches, each in quotes."))
    if not all(isinstance(v, str) and v.strip() for v in values):
        return Result(_core.failed(task, "each search should be some words in quotes, for example "
                                         "\"atrial fibrillation\"."))
    if len({v.strip().lower() for v in values}) != 3:
        return Result(_core.failed(task, "use three different wordings, so you can compare what each finds."))
    return Result(_core.passed(task, "three searches. Compare how many AF reports each finds below."))


# TODO 4 -----------------------------------------------------------------------------

def _softmax_output(fn, scores, temperature):
    out = fn(np.array(scores, dtype=float), temperature)
    if out is ...:
        return None
    if out is None:
        raise _ReturnedNone
    return np.asarray(out, dtype=float)


class _ReturnedNone(Exception):
    pass


def softmax(fn):
    """TODO 4: softmax(scores, temperature) = exp(scores / T) / sum(exp(scores / T)), computed safely."""
    task = "TODO 4"
    import warnings
    if unfinished(fn):
        return Result(_core.waiting(task))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            outputs = {t: _softmax_output(fn, SOFTMAX_SCORES, t) for t in SOFTMAX_EXPECTED}
            large = _softmax_output(fn, LARGE_SCORES, 1.0)
    except _ReturnedNone:
        return Result(_core.failed(task, "your function returned None. Check that it ends with return and the value you calculated."))
    except Exception as error:  # a student's code can raise anything
        return _crashed(task, error)
    if any(o is None for o in outputs.values()) or large is None:
        return Result(_core.waiting(task))
    if any(o.shape != (3,) for o in outputs.values()):
        return Result(_core.failed(task, "softmax should return one probability per score (3 scores in, 3 out)."))
    one = outputs[1.0]
    if not np.allclose(one, SOFTMAX_EXPECTED[1.0], atol=1e-5):
        if np.allclose(one, np.exp(SOFTMAX_SCORES), atol=1e-4):
            return Result(_core.failed(task, "these are the exponentials on their own. Divide by their sum, so "
                                             "the probabilities add up to 1."))
        if np.all(np.isfinite(one)) and not math.isclose(one.sum(), 1, abs_tol=1e-6):
            return Result(_core.failed(task, "the probabilities should add up to 1: divide each exponential by "
                                             "the sum of all of them, e.sum()."))
        return Result(_core.failed(task, "softmax([2, 1, 0]) at temperature 1 should be about [0.67, 0.24, 0.09], "
                                         "as worked above."))
    if np.allclose(outputs[0.5], SOFTMAX_EXPECTED[2.0], atol=1e-5):
        return Result(_core.failed(task, "you multiplied by the temperature. Divide the scores by it: a low "
                                         "temperature should sharpen the probabilities."))
    if not (np.allclose(outputs[0.5], SOFTMAX_EXPECTED[0.5], atol=1e-5)
            and np.allclose(outputs[2.0], SOFTMAX_EXPECTED[2.0], atol=1e-5)):
        return Result(_core.failed(task, "right at temperature 1 but not at 0.5 or 2. Divide the scores by the "
                                         "temperature inside the exponential, before the sum."))
    if not (np.all(np.isfinite(large)) and np.allclose(large, LARGE_EXPECTED, atol=1e-5)):
        return Result(_core.failed(task, "softmax of [1000, 999] gives nan: e to the power 1000 is too big for the "
                                         "computer. Subtract scores.max() from the scores first; it does not "
                                         "change the answer."))
    return Result(_core.passed(task, "correct, at every temperature and for very large scores."))


# TODO 5 -----------------------------------------------------------------------------

def generation(fn):
    """TODO 5: generate(steps) runs the greedy loop from the notebook's prompt and returns the token list.

    Checked on the first few steps only (each step is a full pass of the model on the CPU).
    """
    task = "TODO 5"
    steps = GENERATION_CHECK_STEPS
    if callable(fn):
        if unfinished(fn):
            return Result(_core.waiting(task))
        try:
            tokens = fn(steps)
        except Exception as error:  # a student's code can raise anything
            return _crashed(task, error)
    else:
        tokens, steps = fn, len(GREEDY_CONTINUATION)
    if not isinstance(tokens, list):
        return Result(_core.failed(task, "generate should return the list tokens: start from lm.tokens(start) "
                                         "and append to it."))
    start = len(GENERATION_PROMPT_TOKENS)
    if tokens[:start] != GENERATION_PROMPT_TOKENS:
        return Result(_core.failed(task, "tokens should start with the prompt's tokens. Run the whole cell again "
                                         "(it starts from lm.tokens(start))."))
    new = tokens[start:]
    if not new:
        return Result(_core.failed(task, "tokens has no new token. Set next_token, then add it to the end of "
                                         "tokens."))
    if any(isinstance(t, float) or isinstance(t, np.floating) for t in new):
        return Result(_core.failed(task, "you appended a probability (a decimal). next_token should be the "
                                         "position of the largest probability, which is the token's id."))
    if not all(isinstance(t, (int, np.integer)) for t in new):
        return Result(_core.failed(task, "append one token id (a whole number) at each step."))
    if len(new) != steps:
        return Result(_core.failed(task, f"the loop should add one token at each of its {steps} steps; "
                                         f"it added {len(new)}."))
    if [int(t) for t in new] != GREEDY_CONTINUATION[:steps]:
        if int(new[0]) != GREEDY_CONTINUATION[0]:
            return Result(_core.failed(task, "the first new token is not the most likely one. Pick it with "
                                             "np.argmax(probs) (the position of the largest probability)."))
        return Result(_core.failed(task, "the loop drifts from the most likely continuation. Append the token "
                                         "you picked at every step, so the next prediction sees it."))
    return Result(_core.passed(task, "correct: predict, pick the most likely, append, repeat."))
