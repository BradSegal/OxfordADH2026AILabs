"""Checkers for notebook 03, LLM integrations (the AF stroke-prevention audit).

Every task is a decision whose result the student sees: fields of their own, verdicts on
disagreements with a keyword search, a revised definition, the calculator tool's
descriptions, a tool of their own on the MCP server, and which notes a clinician reads.
The checks test form (is everything filled in, of the right kind?) and consequence (do the
model's arguments now give each patient's correct score, stored with the notes?), never
the model's wording. If the student has filled everything in and the model still gets an
outcome wrong on a second try, the check says so plainly and lets them carry on: a small
model sometimes misses what a good description says. A check that needs an earlier
result waits (⏳) until it exists.

No worked answer lives here: the scores are stored with the notes, the apixaban and CURB-65
cases are stored values, and the planted CHA2DS2-VASc mistakes (extensions) are functions
that are wrong on purpose.
"""

import difflib
import re

from .. import extraction
from . import _core
from ._core import failed, passed, waiting

NOTEBOOK = "03_llm_integrations"
_ATTEMPTS = {}


def ready(*values, task="the task above"):
    """True when every value is filled in; otherwise say which earlier task to finish."""
    if any(_core.is_placeholder(v) for v in values):
        print(f"{_core.WAIT} This cell needs {task}: finish it, then run this cell again.")
        return False
    return True


def _placeholder_in(value):
    if _core.is_placeholder(value):
        return True
    if isinstance(value, dict):
        return any(_placeholder_in(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_placeholder_in(v) for v in value)
    return False


def filled(task, *values):
    """True when nothing in ``values`` is still a ...; otherwise show that the task is not done yet."""
    if any(_placeholder_in(v) for v in values):
        waiting(task, "not done yet. Replace each ... with your own text, then run the cell again.")
        return False
    return True


def _safe(task, check, *args):
    """Turn an unexpected error while checking into feedback instead of a traceback."""
    try:
        return check(*args)
    except _core.CheckFailed:
        raise
    except Exception as error:  # an unusual answer must not stop the notebook
        return failed(task, f"your answer could not be checked ({type(error).__name__}: {error}).")


def _words(text):
    return len(text.split()) if isinstance(text, str) else 0


def _model_outcome(task, hints, carry_on):
    """Fail with the hints; from the second try on, say plainly that the student can carry on."""
    _ATTEMPTS[task] = _ATTEMPTS.get(task, 0) + 1
    unique = []
    for hint in hints:  # one hint per argument: the second vignette with the same problem adds nothing
        key = hint.split(" the model gave ")[-1].split("=")[0]
        if key not in [u.split(" the model gave ")[-1].split("=")[0] for u in unique]:
            unique.append(hint)
    message = " ".join(unique[:2])
    from .. import llm
    if not llm.ON_GPU:  # the small CPU model's mistakes are not a broken answer, even in the worked copy
        print(f"{_core.FAIL} {task}: {message} {carry_on}" if _ATTEMPTS[task] >= 2 or _core.strict()
              else f"{_core.FAIL} {task}: {message}")
        return False
    if _ATTEMPTS[task] >= 2 and not _core.strict():
        message += f" {carry_on}"
    return failed(task, message)


# --- TODO 1: fields of the student's own --------------------------------------------------------

AUDIT = ("A diagnosis of atrial fibrillation of any type (paroxysmal, persistent or permanent), including a "
         "patient now in sinus rhythm. Not suspected atrial fibrillation that has not been confirmed, not a "
         "note that rules it out, and not atrial flutter.")
_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
GIVEN = ("atrial_fibrillation", "anticoagulant")


def _norm(text):
    return re.sub(r"\W+", " ", str(text).lower()).strip()


def _blank(value):
    return _core.is_placeholder(value) or (isinstance(value, str) and value.strip() in ("", "..."))


def own_fields(fields):
    """TODO 1: one or two fields of the student's own, each a usable name and a definition."""
    return _safe("TODO 1", _own_fields, fields)


def _own_fields(fields):
    task = "TODO 1"
    if not isinstance(fields, dict):
        return failed(task, 'MY_FIELDS should be a dictionary: {"a_field_name": "what the model should write"}.')
    if not fields or any(_blank(k) or _blank(v) for k, v in fields.items()):
        return waiting(task, 'not done yet. Replace each "..." with a field name and its definition.')
    for name, text in fields.items():
        if not isinstance(name, str) or not _NAME.match(name):
            return failed(task, f"{name!r} is not a usable field name. Use lower-case letters, digits and "
                                "underscores, such as \"date_of_diagnosis\".")
        if name in GIVEN or name in extraction.RESERVED:
            return failed(task, f"{name!r} is a name the notebook already uses. Choose another, such as "
                                "\"rate_control_drug\".")
        if _words(text) < 3:
            return failed(task, f"define {name} in a few words, and say what to write when the note does not say.")
    return passed(task, f"the model will fill in {', '.join(fields)} for eight notes.")


# --- TODO 2: verdicts on the disagreements ---------------------------------------------------------

VERDICT_OPTIONS = ("model", "keyword", "unclear")


def verdicts(verdicts, chosen):
    """TODO 2: a verdict for every note shown, from the options."""
    return _safe("TODO 2", _verdicts, verdicts, chosen)


def _verdicts(verdicts, chosen):
    task = "TODO 2"
    if chosen is None:
        return waiting(task, "this needs the disagreements above: run that cell first.")
    if _core.is_placeholder(verdicts) or (isinstance(verdicts, dict) and any(_core.is_placeholder(v)
                                                                             for v in verdicts.values())):
        return waiting(task, 'not done yet. Replace each ... with "model", "keyword" or "unclear".')
    if not isinstance(verdicts, dict):
        return failed(task, "VERDICTS should be the dictionary printed above, with each ... replaced.")
    missing = [i for i in chosen if i not in verdicts]
    if missing:
        return failed(task, f"add a verdict for {', '.join(missing)}: copy the line printed above again.")
    for note_id in chosen:
        if verdicts[note_id] not in VERDICT_OPTIONS:
            return failed(task, f'{note_id}: {verdicts[note_id]!r} is not "model", "keyword" or "unclear".')
    return passed(task, "every note has a verdict. The labels are below.")


# --- TODO 3: a revised definition --------------------------------------------------------------

def revision(definition, fields):
    """TODO 3: a revised atrial_fibrillation definition, in text, changed from the first draft."""
    return _safe("TODO 3", _revision, definition, fields)


def _revision(definition, fields):
    task = "TODO 3"
    if _blank(definition) or (isinstance(definition, str) and "..." in definition):
        return waiting(task, "not done yet. Replace the ... with what is true and what is false, then run the "
                             "cell again.")
    if not isinstance(definition, str):
        return failed(task, 'the definition should be text in quotes, such as "true if ...".')
    if _norm(definition) == _norm(fields["atrial_fibrillation"]):
        return failed(task, "that is the first draft unchanged. Say what to do with the kinds of note it got wrong.")
    if _words(definition) < 6:
        return failed(task, "the definition is very short. Say what counts as in the audit and what does not.")
    if difflib.SequenceMatcher(None, _norm(definition), _norm(AUDIT)).ratio() > 0.9:
        return failed(task, "that is the specification copied out. Write it as you would brief a colleague: "
                            "say true for which notes, false for which.")
    return passed(task, "a revised definition. The table shows what it does to the six hard notes.")


# --- TODO 4: the CHA2DS2-VASc tool description ----------------------------------------------------

_WHY = {
    "female": "Does your description say how a note shows that the patient is a woman (Mrs, Ms, \"79F\")?",
    "heart_failure": "In CHA2DS2-VASc, heart failure includes left ventricular systolic dysfunction such as a "
                     "reduced ejection fraction (LVEF). Does your description say what counts?",
    "hypertension": "High blood pressure counts even when it is controlled on treatment (for example on "
                    "amlodipine or ramipril). Does your description say so?",
    "diabetes": "Does your description say what counts as diabetes, however it is treated?",
    "stroke_tia": "A previous stroke (including CVA or cerebral infarction), transient ischaemic attack (TIA) or "
                  "thromboembolism counts. Does your description say so?",
    "vascular": "A documented previous myocardial infarction (including NSTEMI or STEMI), peripheral arterial "
                "disease, or aortic plaque counts as vascular disease. Does your description say what counts?",
    "age": "Does your description say the age is in years?",
}
_CARRY_ON = ("If your descriptions already say this and the model still gets it wrong, that is the limit of a "
             "small model, not your description: carry on with the next part.")


def tool_form(description, parameters):
    """TODO 4, before the model is asked: text for the description and for every argument. Silent when ready."""
    return _safe("TODO 4", _tool_form, description, parameters)


def _tool_form(description, parameters):
    task = "TODO 4"
    dots = isinstance(parameters, dict) and any(isinstance(v, str) and v.strip() == "..." for v in parameters.values())
    if _placeholder_in(description) or _placeholder_in(parameters) or dots:
        return waiting(task, "not done yet. Replace each ... with your description, then run the cell again.")
    if not isinstance(description, str) or _words(description) < 3:
        return failed(task, "DESCRIPTION should be text in quotes of a sentence or so: what the tool calculates "
                            "and when to use it.")
    if not isinstance(parameters, dict):
        return failed(task, "PARAMETERS should be a dictionary: each argument's name in quotes, then its "
                            "description in quotes.")
    missing = [name for name in _WHY if not isinstance(parameters.get(name), str) or not parameters[name].strip()]
    if missing:
        return failed(task, f"describe every argument in text, in quotes; missing or not text: {', '.join(missing)}.")
    return True


def tool_description(description, parameters, results):
    """TODO 4: the description and parameter descriptions let the model call the tool correctly."""
    return _safe("TODO 4", _tool_description, description, parameters, results)


def _misread(results):
    """Per live patient, the first argument the model got wrong (or that it did not call the tool, or a wrong score)."""
    wrong = []
    for result in [r for r in results if r.get("live")]:  # recorded rows show the worked description
        if not result["called"]:
            wrong.append((result, None, None))
            continue
        arguments = result["arguments"] or {}
        bad = [name for name, value in result["expected"].items() if arguments.get(name) != value]
        if bad:
            wrong.append((result, bad[0], arguments.get(bad[0])))
        elif result["score"] != result["right_score"]:
            wrong.append((result, "score", result["score"]))
    return wrong


def _recorded_comparison(results):
    """The worked description's recorded GPU result for the same patients, as one line."""
    from .. import recordings
    saved = (recordings.step("tool_described") or {}).get("rows", {})
    parts = []
    for result in results:
        row = saved.get(result["note"])
        if row and row.get("called"):
            mark = "right" if row.get("score") == result["right_score"] else "wrong"
            parts.append(f"{result['note']} scored {row.get('score')} ({mark}; correct {result['right_score']})")
    return ("For comparison, the recorded Qwen3 4B run with the worked descriptions, on a GPU: "
            + "; ".join(parts) + ".") if parts else ""


def _tool_description(description, parameters, results):
    task = "TODO 4"
    if not _tool_form(description, parameters):
        return False
    if not results or not any(r.get("live") for r in results):
        return waiting(task, "the model server is not running, so your description could not be tried.")
    wrong = _misread(results)
    from .. import llm
    if wrong and not llm.ON_GPU:  # every description is complete: the small model's misreading is not a failure
        passed(task, "every argument is described.")
        misread = "; ".join(
            f"{r['note']} did not call the tool" if name is None else
            f"{r['note']} returned {got!r} (correct {r['right_score']})" if name == "score" else
            f"{r['note']} {name}={got!r} (the note means {r['expected'][name]!r})" for r, name, got in wrong)
        print(f"Descriptions complete; the small model misread these arguments: {misread}.")
        comparison = _recorded_comparison([r for r, _, _ in wrong])
        if comparison:
            print(comparison)
        return True
    hints = []
    for result, name, got in wrong:
        if name is None:
            hints.append(f"For {result['note']} the model answered without the tool. Say in the tool's "
                         "description when to use it.")
        elif name == "score":
            hints.append(f"For {result['note']} the tool returned {got!r}; the correct score is "
                         f"{result['right_score']}.")
        else:
            hints.append(f"For {result['note']} the model gave {name}={got!r}; the note means "
                         f"{result['expected'][name]!r}. {_WHY.get(name, '')}")
    if not hints:
        live = sum(bool(r.get("live")) for r in results)
        return passed(task, "the model's arguments now give the correct score for "
                            + ("all three patients." if live == len(results) else
                               f"the {live} patient(s) it read live."))
    return _model_outcome(task, hints, _CARRY_ON)


# --- TODO 5: a tool of the student's own on the MCP server ------------------------------------------

# (age, weight in kg, creatinine in micromol/L, reduced dose?) at each threshold of the apixaban SmPC rule.
_APIXABAN_CASES = [
    ((84, 58, 96), True, "age 80 or over and weight 60 kg or less: two criteria, so the reduced dose."),
    ((81, 74, 102), False, "only one criterion (age 80 or over): the usual dose."),
    ((80, 60, 90), True, "exactly 80 years and exactly 60 kg both count: 'or over' and 'or less'."),
    ((79, 61, 132), False, "79 years, 61 kg and 132 micromol/L meet no criterion."),
    ((70, 55, 133), True, "weight 60 kg or less and creatinine exactly 133 micromol/L: two criteria."),
    ((85, 50, 150), True, "all three criteria: the reduced dose."),
    ((60, 90, 140), False, "only one criterion (creatinine 133 or more): the usual dose."),
]
_APIXABAN_WHY = {"age": "Does your docstring say the age is in years?",
                 "weight_kg": "Does your docstring say the weight is in kilograms?",
                 "creatinine_umol_l": "Does your docstring say the creatinine is in micromol/L?"}


def apixaban(function):
    """TODO 5, first part: a docstring that describes each argument, and the rule right at every threshold."""
    return _safe("TODO 5", _apixaban, function)


def _apixaban(function):
    task = "TODO 5"
    if not callable(function):
        return failed(task, "apixaban_reduced_dose should stay a function: fill in its docstring and its return line.")
    doc = function.__doc__ or ""
    try:
        first = function(84, 58, 96)
    except Exception as error:
        return failed(task, f"apixaban_reduced_dose raised {type(error).__name__}: {error}.")
    if _core.is_placeholder(first) or "..." in doc:
        return waiting(task, "not done yet. Replace each ... in the docstring and the return line.")
    for name in ("age", "weight_kg", "creatinine_umol_l"):
        line = next((ln for ln in doc.splitlines() if ln.strip().startswith(f"{name}:")), "")
        if _words(line) < 3:
            return failed(task, f"describe {name} in the docstring (\"{name}: ...\"), with its units: the docstring "
                                "is all the model will read about your tool.")
    for arguments, expected, why in _APIXABAN_CASES:
        value = function(*arguments)
        if not isinstance(value, (bool, int)) or bool(value) != expected:
            age, weight, creatinine = arguments
            return failed(task, f"for age {age}, {weight} kg and creatinine {creatinine}, the function gave "
                                f"{value!r}: {why}")
    print(f"{_core.PASS} {task}: the rule is right at every threshold. Now the server offers it to the model:")
    return True


def apixaban_tool(results):
    """TODO 5, second part: through the server, the model called the tool with each case's numbers."""
    return _safe("TODO 5", _apixaban_tool, results)


def _apixaban_tool(results):
    task = "TODO 5"
    from .. import llm
    if not llm.READY:
        return waiting(task, "the model server is not running, so the model could not try your tool.")
    hints = []
    for result in results:
        if not result["called"]:
            hints.append(f"In case {result['number']} the model answered without calling your tool. Does the "
                         "first line of your docstring say what the tool decides?")
            continue
        arguments = result["arguments"] or {}
        for name, value in result["expected"].items():
            got = arguments.get(name)
            if not (isinstance(got, (int, float)) and not isinstance(got, bool) and abs(got - value) < 1e-9):
                hints.append(f"In case {result['number']} the model gave {name}={got!r}; the case says {value!r}. "
                             f"{_APIXABAN_WHY.get(name, '')}")
                break
    if not hints:
        return passed(task, "the server described your tool from your docstring, and the model passed the right "
                            "numbers for both cases.")
    return _model_outcome(task, hints, _CARRY_ON.replace("descriptions already say", "docstring already says"))


# --- TODO 6: which notes a clinician reads -------------------------------------------------------

def review(rule, notes):
    """TODO 6: a True/False per note (a rule built from the model's and the keyword search's answers)."""
    return _safe("TODO 6", _review, rule, notes)


def _review(rule, notes):
    task = "TODO 6"
    import pandas as pd
    if _core.is_placeholder(rule):
        return waiting(task, "not done yet. Replace the ... with one of the rules listed above the cell.")
    if not isinstance(rule, pd.Series) or not rule.index.equals(notes.index):
        return failed(task, "REVIEW should be built from model_says_af and keyword_says_af with | (or), & (and), "
                            "!= (they disagree) or ~ (not), so it has a True or False for every note.")
    if rule.dtype != bool:
        return failed(task, "REVIEW should hold True or False for every note: combine model_says_af and "
                            "keyword_says_af with |, &, != or ~.")
    return passed(task, f"your rule sends {int(rule.sum())} of {len(notes)} notes to a clinician. The consequence:")


# --- Extension D: tests that catch a wrong CHA2DS2-VASc function --------------------------------

def _correct(age, female, heart_failure, hypertension, diabetes, stroke_tia, vascular):
    """The notebook's own given calculator (Part 3), used to check that a student's expected scores are right."""
    return _age(age) + female + heart_failure + hypertension + diabetes + vascular + 2 * stroke_tia


def _age(age, start=65):
    return 2 if age >= 75 else 1 if age >= start else 0


# Wrong on purpose. The first is the colleague's: right for no risk factors and for every one.
PLANTED = {
    "doubles diabetes and ignores vascular disease": lambda age, female, heart_failure, hypertension, diabetes,
    stroke_tia, vascular: _age(age) + female + heart_failure + hypertension + 2 * diabetes + 2 * stroke_tia,
    "scores heart failure 2": lambda age, female, heart_failure, hypertension, diabetes, stroke_tia, vascular:
    _age(age) + female + 2 * heart_failure + hypertension + diabetes + 2 * stroke_tia + vascular,
    "scores stroke or TIA 1": lambda age, female, heart_failure, hypertension, diabetes, stroke_tia, vascular:
    _age(age) + female + heart_failure + hypertension + diabetes + stroke_tia + vascular,
    "gives the sex point to men": lambda age, female, heart_failure, hypertension, diabetes, stroke_tia, vascular:
    _age(age) + (not female) + heart_failure + hypertension + diabetes + 2 * stroke_tia + vascular,
    "starts the age point at 66": lambda age, female, heart_failure, hypertension, diabetes, stroke_tia, vascular:
    _age(age, 66) + female + heart_failure + hypertension + diabetes + 2 * stroke_tia + vascular,
}
_COLLEAGUE = next(iter(PLANTED))
_ARGUMENTS = ("age", "female", "heart_failure", "hypertension", "diabetes", "stroke_tia", "vascular")


def _fails(function, cases):
    return any(function(**{k: case[k] for k in _ARGUMENTS}) != case["expected"] for case in cases)


def tests(cases):
    """Extension D: the student's cases pass a correct function and fail the colleague's wrong one."""
    return _safe("Extension D", _tests, cases)


def _tests(cases):
    task = "Extension D"
    if _core.is_placeholder(cases) or (isinstance(cases, list) and any(_core.is_placeholder(c) for c in cases)):
        return waiting(task)
    if not isinstance(cases, list) or not cases or not all(isinstance(c, dict) and "expected" in c for c in cases):
        return failed(task, "my_tests should be a list of cases, each made with bench.case(...).")
    patients = {tuple(case[k] for k in _ARGUMENTS) for case in cases}
    if len(patients) < 3:
        return failed(task, "add at least one more case with a different patient: you have "
                            f"{len(patients)} different patient{'s' if len(patients) != 1 else ''}.")
    for case in cases:
        if _correct(**{k: case[k] for k in _ARGUMENTS}) != case["expected"]:
            return failed(task, f"a correct function fails your case '{case['case']}': check its expected score "
                                "against the rules.")
    caught = [name for name, function in PLANTED.items() if _fails(function, cases)]
    if _COLLEAGUE not in caught:
        return failed(task, "your tests pass the colleague's wrong function. It scores 0 for a patient with no "
                            "risk factors and 9 for one with all of them, so test risk factors one at a time.")
    others = [name for name in PLANTED if name != _COLLEAGUE]
    found = [name for name in others if name in caught]
    note = (f" They also catch a version that {' and one that '.join(found)}" if found else "")
    missed = [name for name in others if name not in caught]
    note += (f"; they would miss one that {' or one that '.join(missed)}." if missed else ".")
    return passed(task, f"your tests pass a correct function and catch the colleague's.{note}")


# --- Extension F: a CURB-65 tool on the MCP server -------------------------------------------------

# (arguments, the CURB-65 score) at each threshold of NICE NG250 box 2, and the mistake each catches.
_NORMAL = {"confusion": False, "urea": 5.0, "respiratory_rate": 18, "systolic_bp": 120, "diastolic_bp": 75, "age": 50}
_CURB_CASES = [
    ({}, 0, "a patient with nothing abnormal should score 0."),
    ({"confusion": True}, 1, "new confusion scores 1 point."),
    ({"urea": 7.0}, 0, "a urea of exactly 7 mmol/L scores 0: the point is for urea over 7."),
    ({"urea": 7.1}, 1, "a urea over 7 mmol/L scores 1 point."),
    ({"respiratory_rate": 30}, 1, "a respiratory rate of exactly 30 scores 1 point: the point is for 30 or more."),
    ({"respiratory_rate": 29}, 0, "a respiratory rate of 29 scores 0: the point is for 30 or more."),
    ({"systolic_bp": 85}, 1, "a systolic pressure under 90 scores 1 point on its own, even when the diastolic "
                             "pressure is normal."),
    ({"systolic_bp": 90}, 0, "a systolic pressure of exactly 90 scores 0: the point is for systolic under 90."),
    ({"diastolic_bp": 60}, 1, "a diastolic pressure of 60 or less scores 1 point on its own, even when the "
                              "systolic pressure is normal."),
    ({"diastolic_bp": 61}, 0, "a diastolic pressure of 61 scores 0: the point is for 60 or less."),
    ({"systolic_bp": 85, "diastolic_bp": 50}, 1, "low blood pressure is one point, even when both the systolic "
                                                 "and the diastolic pressure are low."),
    ({"age": 64}, 0, "an age of 64 scores 0: the point is for 65 or more."),
    ({"age": 65}, 1, "an age of exactly 65 scores 1 point: the point is for 65 or more."),
    ({"confusion": True, "urea": 9.0, "respiratory_rate": 32, "systolic_bp": 85, "diastolic_bp": 50, "age": 80}, 5,
     "a patient with every feature should score 5."),
]
_CURB_WHY = {
    "systolic_bp": "Does your docstring say which number of a reading such as 104/58 is the systolic pressure? "
                   "An example helps a small model.",
    "diastolic_bp": "Does your docstring say which number of a reading such as 104/58 is the diastolic pressure? "
                    "An example helps a small model.",
    "urea": "Does your docstring give the units of urea (mmol/L)?",
    "confusion": "Does your docstring say what counts as confusion?",
    "respiratory_rate": "Does your docstring give the units of the respiratory rate (breaths per minute)?",
    "age": "Does your docstring say the age is in years?",
}


def curb_65_function(function):
    """Extension F, first part: the function has a docstring and scores CURB-65 correctly at every threshold."""
    return _safe("Extension F", _curb_65_function, function)


def _curb_65_function(function):
    task = "Extension F"
    if not callable(function):
        return failed(task, "curb_65 should stay a function: fill in its docstring and body.")
    try:
        first = function(**_NORMAL)
    except TypeError as error:
        if "ellipsis" in str(error):  # a `score += ...` line still waiting to be filled in
            return waiting(task)
        return failed(task, f"curb_65 raised TypeError on an ordinary patient: {error}.")
    except Exception as error:
        return failed(task, f"curb_65 raised {type(error).__name__} on an ordinary patient: {error}.")
    if _core.is_placeholder(first):
        return waiting(task)
    if _words(function.__doc__) < 10:
        return failed(task, "write a docstring of a sentence or two: it is all the model will read about your "
                            "tool, so say what the score is for and what each argument means, with units.")
    for changes, score, why in _CURB_CASES:
        value = function(**{**_NORMAL, **changes})
        if value != score:
            return failed(task, f"for a patient with {changes or 'nothing abnormal'}, curb_65 gave {value!r}: {why}")
    print(f"{_core.PASS} Extension F: curb_65 scores every threshold correctly. Now the server offers it to the model:")
    return True


def curb_65_tool(results):
    """Extension F, second part: through the server, the model called curb_65 with the right arguments."""
    return _safe("Extension F", _curb_65_tool, results)


def _curb_65_tool(results):
    task = "Extension F"
    from .. import llm
    if not llm.READY:
        return waiting(task, "the model server is not running, so the model could not try your tool.")
    hints = []
    for result in results:
        if not result.get("listed", True):
            hints.append("The server did not list curb_65. Run the cell again.")
        elif not result["called"]:
            hints.append(f"In vignette {result['number']} the model answered without calling curb_65. Does your "
                         "docstring say what the score is for?")
        else:
            arguments = result["arguments"] or {}
            for name, value in result["expected"].items():
                got = arguments.get(name)
                same = got == value or (isinstance(got, (int, float)) and not isinstance(got, bool)
                                        and abs(got - value) < 1e-9)
                if not same:
                    hints.append(f"In vignette {result['number']} the model gave {name}={got!r}; it should be "
                                 f"{value!r}. {_CURB_WHY.get(name, '')}")
                    break
    if not hints:
        return passed(task, "the server described your tool from your docstring, and the model used it correctly "
                            "for both patients.")
    return _model_outcome(task, hints, _CARRY_ON.replace("descriptions already say", "docstring already says"))
