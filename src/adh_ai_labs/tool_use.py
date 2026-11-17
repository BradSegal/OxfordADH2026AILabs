"""Tool use for notebook 03: describe a Python function as a tool, and try it on the audit's patients.

- ``describe`` builds the JSON schema of a tool from a function's name and type hints and
  the descriptions the student writes.
- ``try_patients`` sends each patient's note with that tool, runs the call the model makes,
  and shows a table: the model's arguments beside the correct ones (stored with the notes),
  wrong ones marked, and the score against the correct score.
- ``without_tool`` (Extension C) asks for each score with no calculator.

These run live on the three patients, on a GPU or the CPU; with no model server, all three
are the worked copy's run recorded on a GPU. Each result says whether it ran live.
"""

import inspect
import json
import re

import pandas as pd

from . import integration_tables as tables
from . import llm, notes, recordings

SYSTEM = ("You help a GP practice with an audit of stroke prevention in atrial fibrillation. Use the calculator "
          "tool to work out scores; do not calculate them yourself.")
QUESTION = "\n\nWhat is this patient's CHA2DS2-VASc score?"
_TYPES = {int: "integer", float: "number", bool: "boolean", str: "string"}


def describe(function, description, parameters=None):
    """The JSON schema that tells a model about ``function``: its name, what it does, and its arguments.

    The argument names and types come from the function itself; ``description`` and the
    ``parameters`` descriptions (argument name -> text) are what the model reads to decide
    when to call it and what to pass.
    """
    properties = {}
    for name, parameter in inspect.signature(function).parameters.items():
        properties[name] = {"type": _TYPES.get(parameter.annotation, "string")}
        text = (parameters or {}).get(name)
        if isinstance(text, str) and text.strip():
            properties[name]["description"] = text.strip()
    return {"type": "function",
            "function": {"name": function.__name__,
                         "description": description if isinstance(description, str) else "",
                         "parameters": {"type": "object", "properties": properties,
                                        "required": list(properties)}}}


def patients(table):
    """The three audit patients the tool is tried on."""
    return table[table["tool_example"]]


def _ask(client, tool, function, note):
    response = client.chat.completions.create(
        model=llm.MODEL, temperature=0, max_tokens=200, tools=[tool],
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": note + QUESTION}])
    message = response.choices[0].message
    if not message.tool_calls:
        return {"called": False, "arguments": None, "score": None}
    try:
        arguments = json.loads(message.tool_calls[0].function.arguments)
    except ValueError:
        arguments = {}
    try:
        score = function(**arguments)
    except Exception as error:  # the model's arguments did not fit the function
        score = f"error ({type(error).__name__})"
    return {"called": True, "arguments": arguments, "score": score}


def show(results, function_name="cha2ds2_vasc"):
    """One table for the patients: the model's arguments beside the correct ones, and the scores."""
    columns, right = {}, 0
    for result in results:
        head = result["note"] + ("" if result["live"] else " (recorded)")
        expected, got = result["expected"], result["arguments"] or {}
        if not result["called"]:
            columns[f"{head}: model"] = {name: "" for name in expected} | {"score": "❌ no tool call"}
        else:
            columns[f"{head}: model"] = {name: (str(got.get(name)) if got.get(name) == value
                                                else tables.mark(False, str(got.get(name))))
                                         for name, value in expected.items()}
            ok = result["score"] == result["right_score"]
            right += ok
            columns[f"{head}: model"]["score"] = tables.mark(ok, str(result["score"]))
        columns[f"{head}: correct"] = {**{name: str(value) for name, value in expected.items()},
                                       "score": str(result["right_score"])}
    frame = pd.DataFrame(columns)
    frame.index.name = "argument"
    tables.table(frame, f"{function_name}: the right score for {right} of {len(results)} patients. "
                        "Wrong arguments are marked ❌.")


def try_patients(client, tool, function, table, step="described"):
    """Ask about each patient with the tool available, run the call the model makes, and show the table.

    Returns one result per patient: whether the model called the tool, its arguments, the
    score they give, the arguments and score that are correct, and whether it ran live.
    With no model server every patient shows the worked copy's run recorded on a GPU.
    """
    chosen = patients(table)
    live = list(chosen.index) if llm.READY else []
    saved = (recordings.step(f"tool_{step}") or {}).get("rows", {})
    results, stopped = [], False
    for note_id, row in chosen.iterrows():
        if note_id in live and not stopped:
            try:
                result = _ask(client, tool, function, row["note"])
            except Exception as error:
                if not llm.server_failed(error):
                    raise
                print("The model server stopped answering, so no more patients are sent in this step and the rest "
                      "show the run recorded on a GPU.")
                stopped = True
                live = [i for i in live if i in {r["note"] for r in results}]
        if note_id not in live:
            result = saved.get(note_id, {"called": False, "arguments": None, "score": None})
        results.append({**result, "note": note_id, "expected": notes.score_inputs(row),
                        "right_score": int(row["cha2ds2_vasc"]), "live": note_id in live})
    if not live:
        print("No model server: these are the worked copy's runs recorded on a GPU.")
    show(results, function.__name__)
    if recordings.recording() and len(live) == len(chosen):
        recordings.save_step(f"tool_{step}", {"tool": tool},
                             {r["note"]: {k: r[k] for k in ("called", "arguments", "score")} for r in results})
    return results


def _last_number(text):
    numbers = re.findall(r"\d+", text or "")
    return numbers[-1] if numbers else "no answer"


def without_tool(client, table):
    """Extension C: ask for each score with no calculator; print the model's number beside the right one.

    On the CPU only the first patient is asked (the working is long to write).
    """
    replies = []
    chosen = patients(table) if llm.ON_GPU else patients(table).head(1)
    for note_id, row in chosen.iterrows():
        response = client.chat.completions.create(
            model=llm.MODEL, temperature=0, max_tokens=500,
            messages=[{"role": "system", "content": "Show your working in at most eight short lines. End with the score."},
                      {"role": "user", "content": row["note"] + QUESTION}])
        reply = response.choices[0].message.content or ""
        replies.append(reply)
        finished = response.choices[0].finish_reason == "stop"
        print(f"{note_id}: the model says {_last_number(reply) if finished else 'nothing (cut off)'}; "
              f"the correct score is {row['cha2ds2_vasc']}.")
    print(f"\nThe model's working for {chosen.index[0]}:\n{replies[0]}")
    return replies
