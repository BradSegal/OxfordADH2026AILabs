"""How feedback is shown: one plain-English line per check, never an exception for a student.

Every checker in this package reports through ``passed``, ``failed`` and ``waiting``.
A student's mistake prints a hint and returns False; it raises only when
ADH_STRICT=1 (the worked copies run that way, so a broken answer key fails loudly).
Multiple-choice answers are stored as short hashes so reading the package does not
give them away.
"""

import hashlib
import os

PASS, FAIL, WAIT = "✅", "❌", "⏳"


class CheckFailed(AssertionError):
    """Raised instead of printing a hint when ADH_STRICT=1."""


def strict():
    return os.environ.get("ADH_STRICT") == "1"


def passed(task, message="correct."):
    print(f"{PASS} {task}: {message}")
    return True


def failed(task, hint):
    if strict():
        raise CheckFailed(f"{task}: {hint}")
    print(f"{FAIL} {task}: {hint}")
    return False


def waiting(task, message="not done yet. Replace the ... with your code, then run the cell again."):
    if strict():
        raise CheckFailed(f"{task}: {message}")
    print(f"{WAIT} {task}: {message}")
    return False


def is_placeholder(value):
    return value is ... or value is None


def answer_code(notebook, question, answer):
    """The stored form of a multiple-choice answer."""
    text = f"{notebook}:{question}:{str(answer).strip().lower()}"
    return hashlib.sha256(text.encode()).hexdigest()[:10]


def choice(notebook, codes, question, answer, explanations=None):
    """Check a multiple-choice answer against ``codes`` (question -> answer_code)."""
    task = f"Question {question}"
    answer = str(answer).strip().lower()
    if not answer:
        return waiting(task, "not answered yet. Put a letter between the quotes.")
    if answer_code(notebook, question, answer) == codes[question]:
        extra = (explanations or {}).get(question)
        return passed(task, "correct." + (f" {extra}" if extra else ""))
    return failed(task, "not quite. Have another look at the options and try again.")
