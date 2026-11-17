"""Helpers for the extensions of notebook 03 (LLM integrations).

- ``accuracy`` prints a share with its 95% interval (Wilson's, from SciPy).
- ``show_case`` prints one marked reply as a readable case.
- ``final_letter`` reads a multiple-choice reply's final ``Answer: X`` line.
- ``json_reply`` reads a structured (JSON) reply, or gives None if it was cut off or malformed.
- ``run_with_tests`` runs a function that a model wrote against test cases, contained but
  not sandboxed (see its docstring), and decides pass or fail itself.
- ``case`` writes one test case for a CHA2DS2-VASc function (the student writes the cases:
  no suite of cases lives in this package); ``ask_for_code`` and ``test_code`` ask the
  model for that function and run each version against the student's cases.
- ``answer_questions`` runs the MedMCQA questions with exact match (Extension E).
- ``judge_explanations`` has a judge mark explanations of a note under the student's rubric
  (Extension A).
"""

import json
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

from .checks._core import WAIT

DRIVE = Path("/content/drive")



def _unfinished(values):
    return values is ... or values is None or any(v is ... or v is None for v in values)


def accuracy(correct, label="Accuracy"):
    """Print and return the share of True values with its 95% interval."""
    from scipy.stats import binomtest
    if _unfinished(correct):
        print(f"{WAIT} {label}: waiting for the task above.")
        return None
    correct = [bool(v) for v in correct]
    k, n = sum(correct), len(correct)
    interval = binomtest(k, n).proportion_ci(confidence_level=0.95, method="wilson")
    print(f"{label}: {k} of {n}, {k / n:.0%} (95% interval {interval.low:.0%} to {interval.high:.0%})")
    return {"correct": k, "n": n, "share": k / n, "low": interval.low, "high": interval.high}


_ANSWER_LINE = re.compile(r"^\W*answer\W*:\W*([ABCD])\b", re.IGNORECASE)


def final_letter(reply):
    """The letter on the reply's last non-empty line if that line reads "Answer: X", else ""."""
    lines = [line for line in str(reply).splitlines() if line.strip()]
    found = _ANSWER_LINE.match(lines[-1]) if lines else None
    return found.group(1).upper() if found else ""


def json_reply(response):
    """The parsed JSON of a structured reply, or None if it was cut off or is not valid JSON."""
    choice = response.choices[0]
    if choice.finish_reason != "stop":
        return None
    try:
        return json.loads(choice.message.content)
    except (TypeError, ValueError):
        return None


def show_case(question, reply, exact, judge):
    """Print one marked reply as a short case: the question, the answer key, the reply and both marks."""
    mark = {True: "right", False: "wrong"}
    print(question["question"])
    for letter, option in question["options"].items():
        print(f"  {letter}) {option}")
    print(f"Correct answer: {question['answer']}) {question['options'][question['answer']]}")
    text = str(reply).strip()
    if len(text) > 700:  # keep the opening and the end, where the final choice is
        text = text[:350].rstrip() + "\n[...]\n" + text[-300:].lstrip()
    print("Reply:")
    print("  " + text.replace("\n", "\n  "))
    print(f"Letter on its final Answer: line: {final_letter(reply) or 'none'}")
    print(f"Exact match marked it {mark[bool(exact)]}; the judge marked it {mark[bool(judge)]}.")
    print("-" * 60)


def code_block(reply):
    """The code inside the first ```python block of a reply (or the whole reply if there is none)."""
    found = re.search(r"```(?:python|py)?\s*\n(.*?)```", reply, re.DOTALL)
    return found.group(1) if found else reply


def unmount_drive():
    """Disconnect Google Drive (Colab only), so code run later in this session cannot reach it."""
    if not os.path.ismount(DRIVE):
        return False
    from google.colab import drive
    drive.flush_and_unmount()
    print("Google Drive is disconnected for the rest of this session.")
    return True


# --- CHA2DS2-VASc test cases ------------------------------------------------------------------

CONDITIONS = ["heart_failure", "hypertension", "diabetes", "stroke_tia", "vascular"]


def case(name, expected, age, female=False, **conditions):
    """One test case for a CHA2DS2-VASc function: a patient and the score a correct function gives.

    ``conditions`` are any of heart_failure, hypertension, diabetes, stroke_tia and vascular set
    to True; the rest are False. For example: case("diabetes alone", 1, age=50, diabetes=True).
    """
    unknown = [key for key in conditions if key not in CONDITIONS]
    if unknown:
        raise ValueError(f"case() does not know {', '.join(unknown)}; the conditions are {', '.join(CONDITIONS)}.")
    return {"case": name, "age": age, "female": bool(female),
            **{condition: bool(conditions.get(condition, False)) for condition in CONDITIONS},
            "expected": expected}


REQUEST = ("Write a Python function cha2ds2_vasc(age, female, heart_failure, hypertension, diabetes, stroke_tia, "
           "vascular) that returns a patient's CHA2DS2-VASc score as an integer. age is in years; the other "
           "arguments are True or False. Reply with the code only, in one Python code block.")
RULES = ("\n\nThe score adds: congestive heart failure 1, hypertension 1, age 75 or over 2, diabetes 1, "
         "previous stroke or TIA 2, vascular disease 1, age 65 to 74 1, female sex 1.")


def ask_for_code(client, model, request, attempts=None):
    """The functions the model writes for ``request``, ``attempts`` times at temperature 1 (3 on a GPU, 1 on the CPU)."""
    from . import llm
    attempts = attempts or (3 if llm.ON_GPU else 1)
    codes = []
    for _ in range(attempts):
        response = client.chat.completions.create(
            model=model, temperature=1, max_tokens=1000, messages=[{"role": "user", "content": request}])
        codes.append(code_block(response.choices[0].message.content or ""))
    return codes


def test_code(codes, cases, function_name="cha2ds2_vasc"):
    """Run each piece of model code against ``cases`` with ``run_with_tests``; a table of outcomes."""
    import pandas as pd
    print(f"Containment here: {containment()}.\n")
    rows = []
    for name, group in codes.items():
        for number, code in enumerate(group, start=1):
            passed, outcome = run_with_tests(code, function_name, cases)
            print(f"{name}, attempt {number}: {'passed' if passed else 'failed'} ({outcome})")
            rows.append({"request": name, "attempt": number, "passed": passed, "outcome": outcome, "code": code})
    return pd.DataFrame(rows)


# --- Exam questions, exact match and a judge ----------------------------------------------------

EXAM_INSTRUCTIONS = (
    "Answer the multiple-choice question. Give your reasoning in one or two sentences, "
    "then give your final answer on its own last line as 'Answer: X', where X is A, B, C or D.")
JUDGE_SCHEMA = {"type": "object", "properties": {"correct": {"type": "boolean"}}, "required": ["correct"]}


def questions():
    """The 20 bundled MedMCQA questions (Apache 2.0; see assets/SOURCES.md)."""
    from importlib.resources import files
    return json.loads(files("adh_ai_labs").joinpath("assets/medmcqa_20.json").read_text())


def as_text(q):
    return q["question"] + "\n" + "\n".join(f"{letter}) {option}" for letter, option in q["options"].items())


def models_to_compare():
    """Extension E: the two models to compare, or one when a comparison would not be real.

    With no model server the replies are the worked copy's recording, which holds one model's
    answers, so there is nothing to compare.
    """
    from . import llm
    if not llm.READY:
        print(f"No model server: the replies below are the run recorded on a GPU with {llm.GPU_MODEL}. The "
              "recording holds one model's answers, so this run makes no comparison between models.")
        return [llm.GPU_MODEL]
    if llm.MODEL == llm.SMALL_MODEL:
        return [llm.MODEL]
    try:
        return [llm.MODEL, llm.pull(llm.SMALL_MODEL)]
    except Exception as error:  # the download failed or ran out of time: compare nothing rather than stop
        print(f"Could not download {llm.SMALL_MODEL} ({type(error).__name__}), so only {llm.MODEL} answers.")
        return [llm.MODEL]


def answer_questions(client, models, qs=None):
    """Each model answers every question; a table of replies with exact match, and its accuracy per model."""
    import pandas as pd

    from . import llm
    if qs is None:
        qs = questions() if llm.ON_GPU else questions()[:3]
        if not llm.ON_GPU:
            print("On the CPU this asks the first 3 questions only (about 15 seconds each).")
    rows = []
    for model in models:
        for q in qs:
            response = client.chat.completions.create(
                model=model, temperature=0, max_tokens=300,
                messages=[{"role": "system", "content": EXAM_INSTRUCTIONS}, {"role": "user", "content": as_text(q)}])
            reply = response.choices[0].message.content or ""
            rows.append({"model": model, "id": q["id"], "correct": q["answer"], "reply": reply,
                         "exact": final_letter(reply) == q["answer"]})
    replies = pd.DataFrame(rows)
    for model, group in replies.groupby("model", sort=False):
        accuracy(group["exact"], f"{model}, exact match")
    return replies


# --- Extension A: a judge for patient-friendly explanations of a note ---------------------------

EXPLAIN_NOTE = "N03"  # Mrs Okafor: paroxysmal AF on edoxaban, ramipril for blood pressure
# Written for the course, each with one known property, so the student can mark them under their
# own rubric before the judge does. No verdicts are stored: the rubric is the student's.
EXPLANATIONS = [
    "Your heart sometimes beats in an irregular way that comes and goes (paroxysmal atrial fibrillation). It "
    "was found in 2021 on a heart monitor you wore for a few days. You take edoxaban, a medicine that thins the "
    "blood, and ramipril for your blood pressure, which is well controlled. You have had no bleeding problems.",
    "You have PAF, diagnosed on Holter in 2021, and you are anticoagulated with a DOAC (edoxaban 60 mg OD). "
    "Your HTN is well controlled on an ACE inhibitor.",
    "Your heart sometimes beats irregularly. You had a small stroke in 2021, so you now take edoxaban, a "
    "blood thinner, to stop it happening again, and ramipril for your blood pressure.",
    "Your heart sometimes beats in an irregular way that comes and goes. You take edoxaban, a blood thinner, "
    "and ramipril for your blood pressure. If you feel well and have no palpitations, you can stop the "
    "edoxaban.",
]
MARK_SCHEMA = {"type": "object", "properties": {"reason": {"type": "string"}, "acceptable": {"type": "boolean"}},
               "required": ["reason", "acceptable"]}


def judge_explanations(client, model, rubric, note, explanations):
    """The judge's mark (True, False or None) and one-line reason for each explanation, under ``rubric``."""
    marks = []
    for number, text in enumerate(explanations, start=1):
        response = client.chat.completions.create(
            model=model, temperature=0, max_tokens=200,
            messages=[{"role": "system", "content": rubric},
                      {"role": "user", "content": f"Clinic note:\n{note}\n\nExplanation to mark:\n{text}"}],
            response_format={"type": "json_schema", "json_schema": {"name": "mark", "schema": MARK_SCHEMA}})
        mark = json_reply(response) or {}
        verdict = mark.get("acceptable") if isinstance(mark.get("acceptable"), bool) else None
        marks.append(verdict)
        print(f"{number}. the judge says {'acceptable' if verdict else 'not acceptable' if verdict is False else 'nothing usable'}"
              f": {mark.get('reason', '')}")
    return marks


# --- Running a model's function against test cases ----------------------------------------------

OUTPUT_BYTES = 200_000
MEMORY_BYTES = 1 << 30
_NOBODY = 65534

# The harness runs in the child process. It reads its one-time marker and the test arguments
# before loading the model's code, limits its own resources, blocks Python's sockets, drops root
# if it has it, then reports the function's return values on a line starting with the marker.
_HARNESS = r'''
import importlib.util, json, os, resource, socket, sys
marker = sys.stdin.readline().strip()
name = sys.stdin.readline().strip()
cases = json.loads(sys.stdin.readline())
memory, output, seconds = (int(v) for v in sys.argv[1:4])
resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
resource.setrlimit(resource.RLIMIT_FSIZE, (output, output))
resource.setrlimit(resource.RLIMIT_CPU, (seconds, seconds))
def _blocked(*args, **kwargs):
    raise OSError("network access is blocked for model code")
socket.socket = socket.create_connection = socket.socketpair = _blocked
if os.geteuid() == 0:
    try:
        os.setgroups([]); os.setgid(%(nobody)d); os.setuid(%(nobody)d)
    except OSError:
        pass
def report(**result):
    sys.stdout.flush()
    print("\n" + marker, json.dumps(result), flush=True)
try:
    spec = importlib.util.spec_from_file_location("model_code", "model_code.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
except BaseException as error:
    report(error=f"the code did not load ({type(error).__name__}: {error})")
    os._exit(0)
function = getattr(module, name, None)
if not callable(function):
    report(error=f"the code has no function called {name}")
    os._exit(0)
results = []
for arguments in cases:
    try:
        value = function(**arguments)
        results.append(value if type(value) in (int, float) else repr(value))
    except BaseException as error:
        results.append(f"raised {type(error).__name__}: {error}")
report(results=results)
os._exit(0)
''' % {"nobody": _NOBODY}

_PREFIX = None


def _network_prefix():
    """A command prefix that gives the child no network at all, if this machine allows one."""
    global _PREFIX
    if _PREFIX is None:
        _PREFIX = []
        unshare = shutil.which("unshare")
        options = ["-n"] if os.geteuid() == 0 else ["-rn"]
        if unshare:
            try:
                probe = subprocess.run([unshare, *options, sys.executable, "-I", "-c", "print('ok')"],
                                       capture_output=True, text=True, timeout=10)
                if probe.stdout.strip() == "ok":
                    _PREFIX = [unshare, *options]
            except (OSError, subprocess.SubprocessError):
                pass
    return _PREFIX


def containment():
    """What the runner can do on this machine, in one line."""
    network = ("the network is isolated" if _network_prefix()
               else "Python's network access is blocked, but the network itself is not isolated")
    user = "code runs as the unprivileged user nobody" if os.geteuid() == 0 else "code runs as your own user"
    return f"{network}; {user}; Google Drive is {'mounted' if os.path.ismount(DRIVE) else 'not mounted'}"


def run_with_tests(code, function_name, cases, timeout=10):
    """Run ``function_name`` from ``code`` on each case; return (passed, what happened).

    The code runs in a new Python process, in an empty temporary folder, with a minimal
    environment (no inherited keys or tokens), no network (a network namespace where the
    machine allows one, and Python's sockets blocked in any case), as the unprivileged user
    ``nobody`` when the notebook runs as root (as in Colab), with memory, output and CPU time
    capped, and the whole process group killed at the end. This is containment, not a sandbox:
    the code still runs on this machine and could read files that every user can read. It
    refuses to run while Google Drive is mounted.

    The harness, not the model's code, decides the result: the function must exist and return
    the expected value for every case, reported after all cases have run.
    """
    if os.path.ismount(DRIVE):
        return False, "not run: Google Drive is still mounted (run bench.unmount_drive() first)"
    marker = secrets.token_hex(16)
    arguments = [{k: v for k, v in case.items() if k not in ("case", "expected")} for case in cases]
    with tempfile.TemporaryDirectory(prefix="adh-model-code-") as folder:
        folder = Path(folder)
        (folder / "model_code.py").write_text(code)
        (folder / "harness.py").write_text(_HARNESS)
        if os.geteuid() == 0:  # the child drops to nobody, who must be able to work here
            folder.chmod(0o777)
        environment = {"PATH": os.defpath, "HOME": str(folder), "TMPDIR": str(folder), "LANG": "C.UTF-8"}
        command = [*_network_prefix(), sys.executable, "-I", "harness.py",
                   str(MEMORY_BYTES), str(OUTPUT_BYTES), str(timeout + 1)]
        request = f"{marker}\n{function_name}\n{json.dumps(arguments)}\n".encode()
        with open(folder / "output.txt", "wb") as sink:
            process = subprocess.Popen(command, cwd=folder, env=environment, stdin=subprocess.PIPE,
                                       stdout=sink, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                process.communicate(request, timeout=timeout)
                timed_out = False
            except subprocess.TimeoutExpired:
                timed_out = True
            finally:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                process.wait()
        with open(folder / "output.txt", "rb") as output:
            text = output.read(OUTPUT_BYTES).decode(errors="replace")
    if timed_out:
        return False, f"stopped after {timeout} seconds"
    reports = [line[len(marker):].strip() for line in text.splitlines() if line.startswith(marker)]
    if not reports:
        return False, "stopped before the tests finished"
    result = json.loads(reports[-1])
    if "error" in result:
        return False, result["error"]
    for case, value in zip(cases, result["results"]):
        if value != case["expected"]:
            return False, f"{case['case']}: expected {case['expected']}, got {value}"
    return True, f"passed all {len(cases)} cases"
