"""The synthetic clinic notes of notebook 03's audit (``assets/clinic_notes.json``).

Each note was written for the course (no real patient) with its labels: ``af_status``,
``in_audit`` (the audit's definition: a diagnosis of atrial fibrillation of any type,
including in sinus rhythm now; not suspected, ruled out or flutter), the CHA2DS2-VASc
inputs and score, and flags for the three patients the tool is tried on
(``tool_example``) and the ten notes run live on the CPU (``cpu_subset``).
"""

import hashlib
import json
from importlib.resources import files

PATH = files("adh_ai_labs") / "assets" / "clinic_notes.json"
SCORE_INPUTS = ("age", "female", "heart_failure", "hypertension", "diabetes", "stroke_tia", "vascular")


def records():
    return json.loads(PATH.read_text())


def load():
    """The notes as a table indexed by note id, with ``female`` (True or False) beside ``sex``."""
    import pandas as pd
    table = pd.DataFrame(records()).set_index("id")
    table["female"] = table["sex"] == "F"
    return table


def sha():
    """A hash of the notes' ids and text: what a recorded model run depends on (not the labels)."""
    text = json.dumps([(r["id"], r["note"]) for r in records()])
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def score_inputs(row):
    """The arguments a CHA2DS2-VASc calculator should receive for one note's patient."""
    return {name: (int(row[name]) if name == "age" else bool(row[name])) for name in SCORE_INPUTS}


def show(table, ids):
    """Print the given notes in full, one after another."""
    for note_id in ids:
        print(f"{note_id}\n{table.loc[note_id, 'note']}\n")
