"""Notes to a table with a language model, for notebook 03's audit, and how to check the table.

The notebook gives first-draft field definitions; these helpers turn definitions into a
system message and a JSON schema, send notes through the OpenAI SDK (exactly the call the
notebook makes once by hand in Part 1), and show the results as tables a student can read:

- ``extract_all``: every note's fields (``show`` draws the table);
- ``try_definition``: one definition of ``atrial_fibrillation`` on the six hard notes, with
  the label beside each answer (fast enough on the CPU to try again and again);
- ``try_fields``: the student's own fields on eight notes, with what the model wrote;
- ``disagreements`` and ``reveal``: where the model and the keyword search differ, and the
  labels beside the student's verdicts;
- ``before_after``: the revised definition against the first draft;
- ``counts`` and ``review``: Part 5's two-by-two counts and the consequence of a review rule.

On a GPU every note runs live. On the CPU the first ``CPU_LIVE`` notes run live (about two
minutes on Colab's two cores) and the rest come from the worked copy's run recorded on a GPU
(see ``recordings``); with no model server at all, every row is recorded. Each row says
which in its ``source`` column. A recorded row shows a field's value only when it was
recorded with the same definition; otherwise it says it was not read.
"""

import json
import time

import pandas as pd

from . import integration_tables as tables
from . import llm, recordings

INSTRUCTIONS = "Fill in each field for the patient this clinic note is about, using only what the note says."
BOOLEAN = ("atrial_fibrillation",)  # true or false; every other field is text
LIVE, RECORDED = "live", "recorded on a GPU"
NOT_READ = "(recorded row: not read)"  # a field whose definition differs from the recorded run's
CPU_LIVE = 16  # notes read live on the CPU: about 70 s on two cores here, so about two minutes on Colab
MAX_TOKENS = 120  # enough for a few short fields; a cap keeps the CPU path quick
# Columns the helpers put beside the student's fields: a field with one of these names would clash.
RESERVED = ("source", "id", "note", "note_id", "in_audit", "keyword", "model", "label", "kind")
# The six hard notes the experiment helper tries a definition on, and what makes each hard.
HARD = {"N10": "AF ruled out", "N06": "family history", "N04": "suspected AF (?AF)", "N16": "atrial flutter",
        "N11": "paroxysmal AF, sinus rhythm today", "N03": "PAF (abbreviation)"}
OWN_NOTES = ["N01", "N03", "N05", "N09", "N13", "N19", "N28", "N41"]  # TODO 1's eight notes
KINDS = {"current": "AF", "paroxysmal": "paroxysmal AF", "persistent_or_permanent": "persistent or permanent AF",
         "history_in_sinus_rhythm": "AF, now in sinus rhythm", "ruled_out": "AF ruled out",
         "suspected": "suspected AF", "flutter": "atrial flutter", "none": "no AF"}


def system_prompt(fields):
    """What the model reads before each note: the instructions, then each field's definition."""
    return "\n".join([INSTRUCTIONS] + [f"- {name}: {text}" for name, text in fields.items()])


def schema(fields):
    """The JSON schema the server enforces: true/false for ``atrial_fibrillation``, text for the rest."""
    properties = {name: {"type": "boolean" if name in BOOLEAN else "string"} for name in fields}
    return {"type": "object", "properties": properties, "required": list(fields)}


def response_format(fields, strict=False):
    """The ``response_format`` that asks the server to follow the schema.

    ``strict=True`` is the form OpenAI's own API needs to enforce it (no other keys allowed);
    Ollama follows the schema either way.
    """
    if not strict:
        return {"type": "json_schema", "json_schema": {"name": "audit", "schema": schema(fields)}}
    return {"type": "json_schema",
            "json_schema": {"name": "audit", "strict": True,
                            "schema": {**schema(fields), "additionalProperties": False}}}


def messages(note, fields):
    """The messages for one note: the system message, then the note."""
    return [{"role": "system", "content": system_prompt(fields)}, {"role": "user", "content": note}]


def extract_one(client, note, fields):
    """One note's fields as a dictionary; every field None if the reply was unusable."""
    try:
        response = client.chat.completions.create(
            model=llm.MODEL, temperature=0, max_tokens=MAX_TOKENS, messages=messages(note, fields),
            response_format=response_format(fields))
        choice = response.choices[0]
        answer = json.loads(choice.message.content) if choice.finish_reason == "stop" else {}
    except Exception as error:  # an error or JSON that does not parse: no answer, not a guess
        if llm.server_failed(error):  # the server itself stopped: the caller stops the batch
            raise
        answer = {}
    if not isinstance(answer, dict):
        answer = {}
    return {name: answer.get(name) for name in fields}


def live_ids(notes):
    """The notes read live: all of them on a GPU, the first ``CPU_LIVE`` on the CPU, none with no server."""
    if not llm.READY:
        return []
    return list(notes.index if llm.ON_GPU else notes.index[:CPU_LIVE])


def _run(client, notes, ids, fields):
    """Read ``ids`` live; stop at the first server failure. Returns (rows, seconds)."""
    start, rows = time.time(), {}
    for note_id in ids:
        try:
            rows[note_id] = extract_one(client, notes.loc[note_id, "note"], fields)
        except Exception as error:
            if not llm.server_failed(error):
                raise
            print(f"The model server stopped answering after {len(rows)} note(s), so no more are sent in this "
                  "step and the rest show the run recorded on a GPU.")
            break
    return rows, time.time() - start


def extract_all(client, notes, fields, step="first"):
    """Every note's fields as a table (one row per note), with a ``source`` column: live or recorded."""
    rows, seconds = _run(client, notes, live_ids(notes), fields)
    live = list(rows)
    rows = {i: {**row, "source": LIVE} for i, row in rows.items()}
    saved = recordings.step(f"extraction_{step}") or {}
    recorded_lines = set((saved.get("inputs") or {}).get("system", "").splitlines())
    for note_id in notes.index.difference(live):
        row = saved.get("rows", {}).get(note_id, {})
        rows[note_id] = {name: row.get(name) if name in BOOLEAN or f"- {name}: {text}" in recorded_lines
                         else NOT_READ for name, text in fields.items()}
        rows[note_id]["source"] = RECORDED
    table = pd.DataFrame.from_dict(rows, orient="index").reindex(notes.index)[list(fields) + ["source"]]
    if not live:
        print("No model server: every row is the worked copy's run recorded on a GPU.")
    elif len(live) < len(notes):
        print(f"{len(live)} notes read live in {seconds:.0f} s; the other {len(notes) - len(live)} rows are the "
              "worked copy's run recorded on a GPU (source column).")
    else:
        print(f"{len(live)} notes read live in {seconds:.0f} s.")
    missing = int(table.loc[live, list(fields)].isna().all(axis=1).sum()) if live else 0
    if missing:
        print(f"{missing} live note(s) got no usable answer; they count as wrong below.")
    if recordings.recording() and len(live) == len(notes):
        recordings.save_step(f"extraction_{step}", {"system": system_prompt(fields)},
                             table.drop(columns="source").to_dict(orient="index"))
    return table


def said(value):
    """True, False or "no answer", for a model's atrial_fibrillation value."""
    return "no answer" if value is None or value != value or value == NOT_READ else bool(value)


def _yes(value):
    value = said(value)
    return value if value == "no answer" else ("yes" if value else "no")


def show(table, notes, keyword=None):
    """The extraction table: each note, highlighted, beside the model's fields (and the keyword search)."""
    fields = [c for c in table.columns if c != "source"]
    frame = pd.DataFrame({"note": [tables.highlight(t) for t in notes["note"]]}, index=notes.index)
    for name in fields:
        frame[name] = table[name].map(_yes) if name in BOOLEAN else table[name]
    if keyword is not None:
        frame["keyword search"] = keyword.map(_yes)
    if (table["source"] != LIVE).any():
        frame["source"] = table["source"]
    says = table["atrial_fibrillation"].map(said) == True  # noqa: E712 (the value may be "no answer")
    tables.table(frame, f"The model says yes to atrial_fibrillation for {int(says.sum())} of {len(notes)} notes. "
                        "Scroll to read them all.", html_columns=["note"], height=420)


def _try_rows(client, notes, ids, fields):
    if not llm.READY:  # the replay server answers only the requests the worked copy made
        return {i: extract_one(client, notes.loc[i, "note"], fields) for i in ids}
    rows, _ = _run(client, notes, ids, fields)
    return rows


def try_definition(client, notes, definition):
    """Try a definition of atrial_fibrillation on the six hard notes; show each answer beside the label."""
    if not isinstance(definition, str) or not definition.strip():
        print(f"{tables.WRONG} The definition should be text in quotes.")
        return None
    start = time.time()
    rows = _try_rows(client, notes, list(HARD), {"atrial_fibrillation": definition})
    seconds = time.time() - start
    frame, right = [], 0
    for note_id, kind in HARD.items():
        answer = said(rows.get(note_id, {}).get("atrial_fibrillation"))
        label = bool(notes.loc[note_id, "in_audit"])
        ok = answer == label
        right += ok
        frame.append({"note": note_id, "what makes it hard": kind, "text": tables.highlight(notes.loc[note_id, "note"]),
                      "model": tables.mark(ok, _yes(answer)), "label: in the audit?": "yes" if label else "no"})
    summary = f"This definition gets {right} of {len(HARD)} hard notes right ({seconds:.0f} s)."
    if not llm.READY and right == 0:
        summary += " No model server: only the worked copy's definitions have recorded answers."
    tables.table(pd.DataFrame(frame).set_index("note"), summary, html_columns=["text"])


def try_fields(client, notes, fields):
    """TODO 1: the student's own fields on eight notes; show what the model wrote and count each value."""
    rows = _try_rows(client, notes, OWN_NOTES, fields)
    if not llm.READY and not any(any(v for v in row.values()) for row in rows.values()):
        print("No model server: your fields could not be tried here.")
        return None
    frame = pd.DataFrame({"note": [tables.highlight(notes.loc[i, "note"]) for i in OWN_NOTES]}, index=OWN_NOTES)
    for name in fields:
        frame[name] = [rows.get(i, {}).get(name) for i in OWN_NOTES]
    for name in fields:
        counts = frame[name].map(lambda v: "no answer" if v is None else str(v)).value_counts()
        print(f"{name}: " + "; ".join(f"{value!r} x{count}" for value, count in counts.items()))
    tables.table(frame, None, html_columns=["note"])
    return frame


def _kinds(notes, ids):
    """Up to one note of each kind first, then the rest, keeping the notes' order within a kind."""
    kinds = notes.loc[ids, "af_status"]
    return list(kinds.groupby(kinds, sort=False).cumcount().sort_values(kind="stable").index)


def disagreements(table, notes, keyword, n=5):
    """Show up to ``n`` notes (different kinds first) where the model and the keyword search differ.

    Prints a ``VERDICTS`` line to copy, and returns the ids shown.
    """
    model = table["atrial_fibrillation"].map(said)
    differ = notes.index[model.map(str) != keyword.map(str)]
    chosen = _kinds(notes, differ)[:n]
    frame = pd.DataFrame({"text": [tables.highlight(notes.loc[i, "note"]) for i in chosen],
                          "model": [_yes(model[i]) for i in chosen],
                          "keyword search": [_yes(keyword[i]) for i in chosen]}, index=pd.Index(chosen, name="note"))
    if (table.loc[chosen, "source"] != LIVE).any():
        frame["source"] = table.loc[chosen, "source"]
    agree = len(notes) - len(differ)
    tables.table(frame, f"The model and the keyword search agree on {agree} of {len(notes)} notes. "
                        f"Here are {len(chosen)} of the {len(differ)} they disagree on.", html_columns=["text"])
    if chosen:
        print("Copy this line into the next cell:\n\nVERDICTS = {" + ", ".join(f'"{i}": ...' for i in chosen) + "}")
    return chosen


def reveal(chosen, verdicts, table, notes, keyword):
    """The label beside each verdict: who was right, and whether the student agreed."""
    model = table["atrial_fibrillation"].map(said)
    frame, agreed = [], 0
    for note_id in chosen:
        label = bool(notes.loc[note_id, "in_audit"])
        right = "model" if model[note_id] == label else "keyword" if keyword[note_id] == label else "neither"
        mine = verdicts[note_id]
        ok = mine == right
        agreed += ok
        frame.append({"note": note_id, "kind of note": KINDS[notes.loc[note_id, "af_status"]],
                      "label: in the audit?": "yes" if label else "no", "so the right one is": right,
                      "your verdict": mine if mine == "unclear" else tables.mark(ok, mine)})
    tables.table(pd.DataFrame(frame).set_index("note"),
                 f"Your verdict matches the label on {agreed} of {len(chosen)} notes.")


def correct(table, notes):
    """True where the model's atrial_fibrillation matches the label (no answer counts as wrong)."""
    return table["atrial_fibrillation"].map(said).map(str) == notes["in_audit"].map(str)


def before_after(first, second, notes):
    """The match with the labels before and after a revision, and every note whose answer changed."""
    was, now = first["atrial_fibrillation"].map(said), second["atrial_fibrillation"].map(said)
    live = notes.index[(first["source"] == LIVE) & (second["source"] == LIVE)]
    if not len(live):
        print("No model server: both runs are the worked copy's recordings, not your definitions.")
        live = notes.index
    changed = [i for i in live if str(was[i]) != str(now[i])]
    before, after = int(correct(first.loc[live], notes.loc[live]).sum()), int(correct(second.loc[live], notes.loc[live]).sum())
    where = "" if len(live) == len(notes) else " read live"
    summary = (f"Right before your revision: {before} of {len(live)} notes{where}; after: {after} of {len(live)}. "
               f"{len(changed)} answer(s) changed" + (":" if changed else "."))
    labels = notes["in_audit"].astype(bool)
    frame = pd.DataFrame({"text": [tables.highlight(notes.loc[i, "note"]) for i in changed],
                          "before": [tables.mark(was[i] == labels[i], _yes(was[i])) for i in changed],
                          "after": [tables.mark(now[i] == labels[i], _yes(now[i])) for i in changed],
                          "label": ["yes" if labels[i] else "no" for i in changed]},
                         index=pd.Index(changed, name="note"))
    if changed:
        tables.table(frame, summary, html_columns=["text"])
    else:
        print(summary)


def says_af(table):
    """True where the model said yes to atrial_fibrillation (no answer counts as no)."""
    return table["atrial_fibrillation"].map(lambda v: said(v) is True)


def sources(table):
    """One line on where a table's rows came from, for a figure's footnote."""
    shown = table["source"].value_counts()
    return (f"All {len(table)} notes ({', '.join(f'{n} {source}' for source, n in shown.items())}). Labels: the "
            "course's clinical review. The definition was revised on these same notes, so the model looks better here "
            "than it would on new notes.")


def counts(says, notes):
    """The two-by-two counts of a method's yes/no answers against the labels (no answer counts as no)."""
    says = says.map(lambda v: said(v) is True)
    labels = notes["in_audit"].astype(bool)
    return {"found": int((says & labels).sum()), "missed": int((~says & labels).sum()),
            "wrongly included": int((says & ~labels).sum()), "correctly left out": int((~says & ~labels).sum())}


def review(rule, model_says, notes):
    """Part 5: the notes a clinician reads under ``rule``, and the patients still missed or wrongly included.

    Notes the rule sends for review are decided by the clinician (right every time); the rest
    keep the model's answer. Compared with reading every note and with trusting the model alone.
    """
    from scipy.stats import binomtest
    labels = notes["in_audit"].astype(bool)
    model_says = model_says.map(lambda v: said(v) is True)

    def outcome(name, reviewed):
        kept = model_says & ~reviewed
        found = (labels & (reviewed | kept)).sum()
        return {"process": name, "notes a clinician reads": int(reviewed.sum()),
                "patients still missed": int((labels & ~reviewed & ~model_says).sum()),
                "wrongly included": int((~labels & kept).sum()), "found": int(found)}

    rows = [outcome("your rule", rule.astype(bool)),
            outcome("read every note", pd.Series(True, index=notes.index)),
            outcome("trust the model alone", pd.Series(False, index=notes.index))]
    frame = pd.DataFrame(rows).set_index("process")
    mine = rows[0]
    in_audit = int(labels.sum())
    interval = binomtest(mine["found"], in_audit).proportion_ci(method="wilson")
    tables.table(frame.drop(columns="found"),
                 f"Your rule: a clinician reads {mine['notes a clinician reads']} of {len(notes)} notes and "
                 f"{mine['patients still missed']} of {in_audit} patients in the audit are still missed.")
    print(f"Sensitivity of your process: {mine['found']} of {in_audit} found ({mine['found'] / in_audit:.0%}; "
          f"95% interval {interval.low:.0%} to {interval.high:.0%}, illustrative only: these notes were written "
          "for the course and the definition was tuned on them).")
    return mine
