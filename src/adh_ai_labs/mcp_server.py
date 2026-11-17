"""A small MCP server for notebook 03: the CHA2DS2-VASc calculator (and a QTc one), served over MCP.

The notebook starts it as a separate program and talks to it over MCP, the way desktop
chat apps, coding assistants and agent frameworks start a local tool server:

    async with mcp_server.connect() as client:
        listed = await client.list_tools()
        result = await client.call_tool("cha2ds2_vasc", {...})

TODO 5 (and Extension F) add a tool written in the notebook:

    async with mcp_server.connect(extra_tools=[apixaban_reduced_dose]) as client:
        listed = await client.list_tools()
        result = await client.call_tool("apixaban_reduced_dose", {...})

``extra_tools`` are functions written in the notebook: their source is copied into a
temporary file that the server process loads at start, so the server describes them
from their name, type hints and docstring exactly as it does its own tool. Each
``connect`` starts a fresh server process and stops it when the block ends, so running
a cell again (or after editing the function) always serves the current version.
``ask`` is the tool loop of Part 3 with the server's tools; ``try_vignettes`` asks about short
written cases through the server and shows the model's arguments beside the right ones.
"""

import contextlib
import importlib.util
import inspect
import json
import math
import os
import sys
import tempfile
import textwrap
from pathlib import Path

import pandas as pd
from mcp import Client, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.server.mcpserver import MCPServer

# log_level: MCPServer sets up Python's logging when it is created; at INFO, every request the
# notebook sends to the model would be logged under its cell.
server = MCPServer("clinical-calculators", instructions="Clinical calculators for the course.",
                   log_level="WARNING")


@server.tool()
def cha2ds2_vasc(age: int, female: bool, heart_failure: bool, hypertension: bool, diabetes: bool,
                 stroke_tia: bool, vascular: bool) -> int:
    """CHA2DS2-VASc stroke risk score (0 to 9) for a patient with atrial fibrillation (NICE NG196).

    age: the patient's age in years.
    female: True if the patient is a woman (Mrs, Ms, "woman", "F"), False if a man.
    heart_failure: heart failure, or left ventricular systolic dysfunction (for example an ejection
        fraction of 40% or less), whether or not it is treated.
    hypertension: high blood pressure, including when it is controlled on treatment.
    diabetes: diabetes mellitus, type 1 or type 2, however it is treated.
    stroke_tia: a previous stroke (including CVA or cerebral infarction), transient ischaemic attack
        (TIA) or thromboembolism.
    vascular: a documented previous myocardial infarction (including NSTEMI or STEMI), peripheral
        arterial disease, or aortic plaque.
    """
    score = 2 if age >= 75 else 1 if age >= 65 else 0
    score += female + heart_failure + hypertension + diabetes + vascular + 2 * stroke_tia
    return int(score)


@server.tool()
def qtc_bazett(qt_ms: float, heart_rate_bpm: float) -> int:
    """QT interval corrected for heart rate (QTc) by Bazett's formula, in milliseconds.

    qt_ms: the QT interval measured on the ECG, in milliseconds.
    heart_rate_bpm: the heart rate, in beats per minute.
    """
    rr_seconds = 60 / heart_rate_bpm
    return round(qt_ms / math.sqrt(rr_seconds))


_HEADER = "import math\nfrom typing import *\n\n"


def _add_tools_from(path, names):
    spec = importlib.util.spec_from_file_location("notebook_tools", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in names:
        server.add_tool(getattr(module, name))


def source_file(functions, folder):
    """Write the functions' source to a file the server can load; return its path."""
    path = Path(folder) / "notebook_tools.py"
    path.write_text(_HEADER + "\n\n".join(textwrap.dedent(inspect.getsource(f)) for f in functions))
    return path


def parameters(extra_tools=(), folder=None):
    """How a client starts this server: the same Python running this module, plus any notebook tools."""
    args = ["-m", "adh_ai_labs.mcp_server"]
    if extra_tools:
        args += [str(source_file(extra_tools, folder)), *[f.__name__ for f in extra_tools]]
    return StdioServerParameters(command=sys.executable, args=args)


@contextlib.asynccontextmanager
async def connect(extra_tools=()):
    """Start the server as a separate program, with any notebook tools, and yield a client connected to it.

    The same as ``Client(StdioServerParameters(...))`` from the mcp package, except that the
    server's own log lines are discarded instead of being written into the notebook. The
    server process stops when the block ends.
    """
    with tempfile.TemporaryDirectory(prefix="adh-mcp-") as folder, open(os.devnull, "w") as log:
        async with Client(stdio_client(parameters(extra_tools, folder), errlog=log)) as client:
            yield client


def openai_tools(tools):
    """MCP tool listings in the format the openai package's ``tools=`` expects."""
    return [{"type": "function",
             "function": {"name": tool.name, "description": tool.description, "parameters": tool.input_schema}}
            for tool in tools]


async def ask(client, model, question, extra_tools=(), max_rounds=5, show_tools=True, system=None):
    """Answer a question, running any tool the model asks for on this server (the loop of Part 3).

    Returns (the model's answer, the list of (tool name, arguments, result) the server ran).
    """
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": question}]
    calls = []
    async with connect(extra_tools) as mcp_client:
        listed = (await mcp_client.list_tools()).tools
        if show_tools:
            print("The server offers:", ", ".join(tool.name for tool in listed))
        tools = openai_tools(listed)
        for _ in range(max_rounds):
            response = client.chat.completions.create(
                model=model, temperature=0, max_tokens=300, messages=messages, tools=tools)
            message = response.choices[0].message
            if not message.tool_calls:
                return message.content, calls
            messages.append(message)
            for call in message.tool_calls:
                try:
                    arguments = json.loads(call.function.arguments)
                    result = await mcp_client.call_tool(call.function.name, arguments)
                    text = result.content[0].text
                except (ValueError, IndexError) as error:
                    arguments, text = call.function.arguments, f"Error: {error}"
                if show_tools is not None:
                    shown = ", ".join(f"{k}={v!r}" for k, v in arguments.items()) if isinstance(arguments, dict) \
                        else arguments
                    print(f"Tool call: the server ran {call.function.name}({shown}) and returned {text}")
                calls.append((call.function.name, arguments, text))
                messages.append({"role": "tool", "tool_call_id": call.id, "content": text})
    return "No answer: the model kept asking for tools.", calls


CURB_VIGNETTES = [
    {"text": "A 72-year-old man is admitted with community-acquired pneumonia. He is newly confused. His urea "
             "is 8.4 mmol/L, respiratory rate 24 breaths per minute and blood pressure 104/58 mmHg. What is "
             "his CURB-65 score?",
     "arguments": {"confusion": True, "urea": 8.4, "respiratory_rate": 24, "systolic_bp": 104,
                   "diastolic_bp": 58, "age": 72},
     "score": 4},
    {"text": "A 58-year-old woman is admitted with community-acquired pneumonia. She is alert and orientated. "
             "Her urea is 7.6 mmol/L, respiratory rate 32 breaths per minute and blood pressure 126/74 mmHg. "
             "What is her CURB-65 score?",
     "arguments": {"confusion": False, "urea": 7.6, "respiratory_rate": 32, "systolic_bp": 126,
                   "diastolic_bp": 74, "age": 58},
     "score": 2},
]  # written for the course (Extension F); not real patients


APIXABAN_VIGNETTES = [
    {"text": "Mrs Price, 84, has atrial fibrillation and is starting apixaban. She weighs 58 kg and her serum "
             "creatinine is 96 micromol/L. Should she take the reduced dose?",
     "arguments": {"age": 84, "weight_kg": 58, "creatinine_umol_l": 96}, "result": True},
    {"text": "Mr Hughes, 81, has atrial fibrillation and is starting apixaban. He weighs 74 kg and his serum "
             "creatinine is 102 micromol/L. Should he take the reduced dose?",
     "arguments": {"age": 81, "weight_kg": 74, "creatinine_umol_l": 102}, "result": False},
]  # written for the course (TODO 5); not real patients
VIGNETTE_SYSTEM = ("You help a GP practice. When a tool fits the question, use it; do not work the answer out "
                   "yourself. Answer in one sentence.")


def _same(got, value):
    if isinstance(value, bool) or isinstance(got, bool):
        return got is value
    return isinstance(got, (int, float)) and abs(got - value) < 1e-9


def _shown(results, function_name):
    from . import integration_tables as tables
    columns = {}
    for result in results:
        head = f"case {result['number']}"
        got = result["arguments"] or {}
        column = {}
        for name, value in result["expected"].items():
            column[name] = (tables.mark(False, "not called") if not result["called"] else
                            f"{got.get(name)!r}" if _same(got.get(name), value) else
                            tables.mark(False, f"{got.get(name)!r} (should be {value!r})"))
        column["the tool returned"] = (tables.mark(str(result["score"]).lower() == str(result["right_score"]).lower(),
                                                   str(result["score"])) if result["called"] else "")
        column["the model's answer"] = (result["answer"] or "")[:160]
        columns[head] = column
    right = sum(r["called"] and str(r["score"]).lower() == str(r["right_score"]).lower() for r in results)
    tables.table(pd.DataFrame(columns), f"{function_name} through the server: the right result for {right} of "
                                        f"{len(results)} cases. Wrong arguments are marked ❌.")


async def try_vignettes(client, model, extra_tools, vignettes=None):
    """Show how the server describes the notebook's tool, then ask about each case through the server.

    Returns one result per case: whether the model called the notebook's tool, with which
    arguments, what the server returned, and the right arguments and result.
    """
    names = [function.__name__ for function in extra_tools]
    if vignettes is None:
        vignettes = APIXABAN_VIGNETTES if "apixaban_reduced_dose" in names else CURB_VIGNETTES
    async with connect(extra_tools) as mcp_client:
        listed = {tool.name: tool for tool in (await mcp_client.list_tools()).tools}
    for name in names:
        if name in listed:
            types = ", ".join(f"{key} ({value.get('type')})"
                              for key, value in listed[name].input_schema["properties"].items())
            print(f"What the model reads about {name}, built by MCP from your function:\n"
                  f"  description: {' '.join(listed[name].description.split())}\n  arguments: {types}\n")
    results = []
    for number, vignette in enumerate(vignettes, start=1):
        print(f"Case {number}: {vignette['text']}")
        answer, calls = await ask(client, model, vignette["text"], extra_tools, show_tools=None, system=VIGNETTE_SYSTEM)
        mine = [(arguments, text) for name, arguments, text in calls if name in names]
        results.append({"number": number, "listed": all(name in listed for name in names), "called": bool(mine),
                        "arguments": mine[-1][0] if mine else None, "score": mine[-1][1] if mine else None,
                        "expected": vignette["arguments"], "right_score": vignette.get("score", vignette.get("result")),
                        "answer": answer})
    print()
    _shown(results, names[0])
    return results


if __name__ == "__main__":
    if len(sys.argv) > 2:
        _add_tools_from(sys.argv[1], sys.argv[2:])
    server.run()
