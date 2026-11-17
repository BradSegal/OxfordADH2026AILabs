"""Figures for notebook 03 (LLM integrations), drawn with matplotlib.

Two sequence diagrams in the same layout, one column per party and time running down,
so the second differs from the first only where MCP changes who does what:

- ``tool_loop``: the model asks for a tool; your code runs the function and sends the
  result back; the model answers. The example is an audit patient's CHA2DS2-VASc score.
- ``mcp_loop``: your code (the MCP client) gets the tool descriptions from a separate
  server program, which builds them from each function's name, type hints and
  docstring, and runs the tool the model picks. The example is the same calculator.

And three more:

- ``request_response``: Part 1's call, filled with the student's own messages and response:
  the messages list, the server applying the chat template, the model, and the response
  object (content, ``finish_reason``, ``usage``).
- ``extraction_step``: how definitions become a system message and a JSON schema, how the
  server keeps the model's next tokens inside the schema (constrained decoding), and how each
  note becomes a row.
- ``two_by_two``: Part 5's counts for the model and the keyword search against the labels.

Each function shows its figure in a notebook and returns it elsewhere, as in
``diagrams.py`` (notebook 01), whose colours and default font these follow.
"""

INK, GREY, LIGHT = "#222222", "#6b6a66", "#f1f0ec"
MODEL, CODE, TOOL = "#2a78d6", "#12805a", "#a45500"


def _plt():
    import matplotlib.pyplot as plt
    return plt


def _show(fig):
    plt = _plt()
    try:
        from IPython import get_ipython
        from IPython.display import display
        if get_ipython() is not None:
            display(fig)
            plt.close(fig)
            return None
    except ImportError:
        pass
    plt.close(fig)
    return fig


def _canvas(width, height):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")
    return fig, ax


def _head(ax, x, y, title, subtitle, colour):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x - 1.45, y - 0.42), 2.9, 0.84, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=colour, ec=colour))
    ax.text(x, y + 0.12, title, ha="center", va="center", fontsize=11, color="white", weight="bold")
    ax.text(x, y - 0.2, subtitle, ha="center", va="center", fontsize=8.5, color="white")


def _lifeline(ax, x, top, bottom, colour):
    ax.plot([x, x], [top, bottom], color=colour, lw=1.2, ls=(0, (4, 3)), alpha=0.6)


def _message(ax, x0, x1, y, number, text, colour=INK):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", color=colour, lw=1.5, shrinkA=0, shrinkB=0))
    ax.text((x0 + x1) / 2, y + 0.08, f"{number}. {text}", ha="center", va="bottom", fontsize=9.5, color=colour,
            zorder=4, bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none"))


def _code(ax, x, y, text):
    ax.text(x, y, text, ha="left", va="top", fontsize=8.5, color=MODEL, family="monospace", zorder=4,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))


def _action(ax, x, y, text, colour, width=2.9):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x - width / 2, y - 0.28), width, 0.56, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc="white", ec=colour, lw=1.3, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=9, color=INK, zorder=4)


def _title(ax, width, height, text):
    ax.text(width / 2, height - 0.05, text, ha="center", va="top", fontsize=12.5, color=INK, weight="bold")
    ax.text(0.05, 0.2, "time runs\ndown the page", ha="left", va="bottom", fontsize=8, color=GREY)


def tool_loop():
    """Part 3: the model asks; your code runs the tool and sends the result back; the model answers."""
    width, height = 10, 6.8
    fig, ax = _canvas(width, height)
    _title(ax, width, height, "The model only asks for the tool; your code runs it")
    columns = {"model": 1.7, "code": 5.0, "tool": 8.3}
    _head(ax, columns["model"], 5.75, "The model", "Ollama, on this Colab machine", MODEL)
    _head(ax, columns["code"], 5.75, "Your code", "in this notebook", CODE)
    _head(ax, columns["tool"], 5.75, "The tool", "a Python function, also here", TOOL)
    for name, colour in [("model", MODEL), ("code", CODE), ("tool", TOOL)]:
        _lifeline(ax, columns[name], 5.3, 0.15, colour)
    _message(ax, columns["code"], columns["model"], 4.85, 1, "a note + the tool's description")
    ax.text(columns["model"] + 0.15, 4.78, '"Mrs Kowalski, 76. AF confirmed ..."', ha="left", va="top",
            fontsize=8.5, color=GREY, style="italic")
    _message(ax, columns["model"], columns["code"], 4.0, 2, "a tool call, not an answer")
    _code(ax, columns["model"] + 0.1, 3.92, "cha2ds2_vasc(age=76, female=True, ...)")
    ax.text(columns["model"] + 0.15, 3.6, "the model chose these values from the note", ha="left", va="top",
            fontsize=8.5, color=GREY, style="italic")
    _message(ax, columns["code"], columns["tool"], 3.2, 3, "your code runs the function")
    _action(ax, columns["tool"], 2.65, "it calculates: 5", TOOL)
    _message(ax, columns["tool"], columns["code"], 1.95, 4, "the result: 5")
    _message(ax, columns["code"], columns["model"], 1.4, 5, "the messages so far + the result")
    _message(ax, columns["model"], columns["code"], 0.85, 6, "the answer, in words")
    _action(ax, columns["code"], 0.35, 'the model\'s answer: "Her score is 5."', CODE, width=3.2)
    return _show(fig)


def mcp_loop():
    """Part 4: a separate server describes its tools (built from each function) and runs the one the model picks."""
    width, height = 10, 8.6
    fig, ax = _canvas(width, height)
    _title(ax, width, height, "With MCP, a tool server describes the tool and runs it")
    columns = {"model": 1.7, "code": 5.0, "server": 8.3}
    _head(ax, columns["model"], 7.55, "The model", "Ollama, on this Colab machine", MODEL)
    _head(ax, columns["code"], 7.55, "Your code", "the MCP client", CODE)
    _head(ax, columns["server"], 7.55, "The MCP server", "a separate program, on this machine", TOOL)
    for name, colour in [("model", MODEL), ("code", CODE), ("server", TOOL)]:
        _lifeline(ax, columns[name], 7.1, 0.15, colour)
    _message(ax, columns["code"], columns["server"], 6.7, 1, "which tools do you have?", TOOL)
    ax.text(columns["server"] - 1.35, 6.42, "inside the server:", ha="left", va="bottom", fontsize=8, color=GREY)
    ax.text(columns["server"] - 1.35, 6.35, 'def cha2ds2_vasc(age: int,\n                 female: bool, ...):\n'
            '    """CHA2DS2-VASc score ...\n    age: in years ..."""',
            ha="left", va="top", fontsize=7.5, color=INK, family="monospace", multialignment="left", zorder=4,
            bbox=dict(boxstyle="round,pad=0.4", fc=LIGHT, ec=TOOL, lw=1))
    ax.text(columns["server"], 5.45, "it builds each description from the\nfunction's name, type hints and docstring",
            ha="center", va="top", fontsize=8.5, color=TOOL, zorder=4,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
    _message(ax, columns["server"], columns["code"], 4.6, 2, "each tool's name + description", TOOL)
    _message(ax, columns["code"], columns["model"], 3.95, 3, "a note + the descriptions")
    _message(ax, columns["model"], columns["code"], 3.3, 4, "a tool call")
    _code(ax, columns["model"] + 0.1, 3.22, "cha2ds2_vasc(age=76, ...)")
    ax.text(columns["model"] + 0.15, 2.92, "values chosen by the model", ha="left", va="top", fontsize=8.5,
            color=GREY, style="italic")
    _message(ax, columns["code"], columns["server"], 2.55, 5, "run it with these values", TOOL)
    _action(ax, columns["server"], 2.0, "the server runs it: 5", TOOL)
    _message(ax, columns["server"], columns["code"], 1.4, 6, "the result: 5", TOOL)
    _message(ax, columns["code"], columns["model"], 0.95, 7, "the messages so far + the result")
    _message(ax, columns["model"], columns["code"], 0.45, 8, "the answer, in words")
    ax.text(columns["code"] + 0.2, 0.1, "New with MCP: steps 1, 2, 5 and 6 (orange).\nThe rest is the tool diagram of Part 3.",
            ha="left", va="bottom", fontsize=8.5, color=GREY, zorder=4,
            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none"))
    return _show(fig)


def _panel(ax, x, y, w, h, title, colour, fc="white"):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=colour, lw=1.6))
    ax.text(x + 0.15, y + h - 0.12, title, ha="left", va="top", fontsize=10.5, color=colour, weight="bold")


def _mono(ax, x, y, text, size=8.3, colour=INK):
    ax.text(x, y, text, ha="left", va="top", fontsize=size, color=colour, family="monospace", linespacing=1.35)


def _arrow(ax, x0, y0, x1, y1, colour=GREY, text=None):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=colour, lw=1.6))
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.12, text, ha="center", va="bottom", fontsize=8.5, color=colour)


def _short(text, n):
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n - 3] + "..."


def request_response(messages, response):
    """Part 1: the messages you send, what the server does with them, and the response object you get back."""
    width, height = 12, 5.4
    fig, ax = _canvas(width, height)
    ax.text(width / 2, height - 0.05, "One call: a list of messages goes in, a response object comes out",
            ha="center", va="top", fontsize=12.5, color=INK, weight="bold")
    # 1. what your code sends
    _panel(ax, 0.1, 0.5, 3.7, 4.2, "1. Your code sends: messages", CODE)
    lines = []
    for message in messages:
        lines += [f'{{"role": "{message["role"]}",', f' "content": "{_short(message["content"], 20)}"}}']
    _mono(ax, 0.3, 4.1, "[\n" + "\n".join(" " + line for line in lines) + "\n]")
    ax.text(0.3, 1.0, "system: the application's instructions\nuser: the question, with the note",
            ha="left", va="bottom", fontsize=8.5, color=GREY)
    # 2. the server
    _panel(ax, 4.3, 0.5, 3.5, 4.2, "2. The server (Ollama)", MODEL)
    ax.text(4.45, 4.05, "a. applies the chat template\n    (as in the LLM applications lab)", ha="left", va="top",
            fontsize=9, color=INK)
    _mono(ax, 4.5, 3.4, "<|im_start|>system\n" + _short(messages[0]["content"], 22)
          + "<|im_end|>\n<|im_start|>user\n" + _short(messages[-1]["content"], 20) + "\n<|im_end|>\n"
          "<|im_start|>assistant", size=7.6, colour=MODEL)
    ax.text(4.45, 1.85, "b. the model predicts the next\n    token, again and again, until\n    it writes <|im_end|> (end of\n"
            "    turn) or reaches max_tokens", ha="left", va="top", fontsize=9, color=INK)
    # 3. the response
    choice = response.choices[0]
    usage = response.usage
    _panel(ax, 8.3, 0.5, 3.6, 4.2, "3. Your code gets: response", TOOL)
    rows = [("response.choices[0]", ""), (".message.content", repr(_short(choice.message.content or "", 18))),
            (".finish_reason", repr(choice.finish_reason)), ("response.usage", ""),
            (".prompt_tokens", str(usage.prompt_tokens if usage else "")),
            (".completion_tokens", str(usage.completion_tokens if usage else ""))]
    y = 4.05
    for name, value in rows:
        indent = 0.2 if name.startswith(".") else 0
        _mono(ax, 8.45 + indent, y, name, size=8.6)
        if value:
            ax.text(11.75, y, value, ha="right", va="top", fontsize=9.5, color=TOOL, weight="bold",
                    family="monospace")
        y -= 0.42
    ax.text(8.45, 0.75, '"stop": the model ended its turn\n"length": it reached max_tokens\nprompt tokens include the template\'s\n'
            'markers; a hosted provider bills tokens',
            ha="left", va="bottom", fontsize=8.5, color=GREY)
    _arrow(ax, 3.85, 2.6, 4.25, 2.6, INK)
    _arrow(ax, 7.85, 2.6, 8.25, 2.6, INK)
    ax.text(width / 2, 0.15, "Values on the right are from your own call above.", ha="center", va="bottom",
            fontsize=8.5, color=GREY)
    return _show(fig)


def extraction_step():
    """Part 2: definitions become a system message and a JSON schema; the schema limits the next tokens."""
    width, height = 12, 5.6
    fig, ax = _canvas(width, height)
    ax.text(width / 2, height - 0.05, "From your definitions to a table: the schema decides which tokens are allowed",
            ha="center", va="top", fontsize=12.5, color=INK, weight="bold")
    # A. definitions -> system message + schema
    _panel(ax, 0.1, 3.0, 3.6, 1.95, "Your definitions: FIELDS", CODE)
    _mono(ax, 0.25, 4.45, '{"atrial_fibrillation":\n    "true if the patient has AF",\n "anticoagulant":\n'
          '    "the anticoagulant ..."}', size=8)
    _panel(ax, 0.1, 0.35, 1.75, 2.3, "system\nmessage", MODEL)
    ax.text(0.2, 1.8, "Fill in each\nfield ...\n- atrial_\n  fibrillation:\n  true if ...\n- anticoagulant:\n  the ...",
            ha="left", va="top", fontsize=7.2, color=INK, family="monospace")
    _panel(ax, 1.95, 0.35, 1.75, 2.3, "JSON\nschema", TOOL)
    ax.text(2.05, 1.8, "atrial_fibrillation:\n  true or false\nanticoagulant:\n  text\nboth required",
            ha="left", va="top", fontsize=7.2, color=INK, family="monospace")
    _arrow(ax, 1.2, 2.95, 1.0, 2.7, GREY)
    _arrow(ax, 2.6, 2.95, 2.8, 2.7, GREY)
    # B. constrained decoding
    _panel(ax, 4.1, 0.35, 4.5, 4.6, "For each note, token by token", MODEL)
    _mono(ax, 4.25, 4.5, 'Note N06: "Mother had AF ..."\nThe model has written so far:\n{"atrial_fibrillation": ', size=8.0)
    ax.text(4.25, 3.72, "Next-token probabilities (made-up numbers)", ha="left", va="top", fontsize=8.8, color=INK)
    tokens = [("true", 0.46, True), ("false", 0.31, True), ("yes", 0.15, False), ("maybe", 0.08, False)]
    allowed = sum(p for _, p, ok in tokens if ok)
    for k, (token, p, ok) in enumerate(tokens):
        y = 3.05 - k * 0.45
        ax.text(4.95, y, token, ha="right", va="center", fontsize=9, family="monospace", color=INK)
        ax.add_patch(__import__("matplotlib.patches", fromlist=["Rectangle"]).Rectangle(
            (5.05, y - 0.15), 1.4 * p / 0.6, 0.3, fc=LIGHT, ec=GREY))
        ax.text(5.1 + 1.4 * p / 0.6, y, f"{p:.2f}", ha="left", va="center", fontsize=8, color=GREY)
        after = p / allowed if ok else 0
        if after:
            ax.add_patch(__import__("matplotlib.patches", fromlist=["Rectangle"]).Rectangle(
                (6.75, y - 0.15), 1.4 * after / 0.6, 0.3, fc=MODEL, ec=MODEL))
        ax.text(6.8 + 1.4 * after / 0.6, y, f"{after:.2f}" if ok else "0: not allowed", ha="left", va="center",
                fontsize=8, color=MODEL if ok else "#b3261e")
    ax.text(5.05, 3.3, "the model", ha="left", va="bottom", fontsize=8.3, color=GREY)
    ax.text(6.75, 3.3, "after the schema", ha="left", va="bottom", fontsize=8.3, color=MODEL)
    ax.text(4.25, 0.45, "The server sets every token that would break the\nJSON to zero (constrained decoding), so the "
            "reply\nalways parses. At temperature 0 the model takes the\nlikeliest allowed token. It can still be wrong:\nhere true, though it is her mother's AF.", ha="left", va="bottom", fontsize=8.3, color=INK)
    # C. rows
    _panel(ax, 9.0, 0.35, 2.9, 4.6, "One row per note", CODE)
    rows = [("note", "atrial_fib.", "anticoag."), ("N01", "true", "apixaban"), ("N02", "false", "none"),
            ("N03", "true", "edoxaban"), ("...", "...", "...")]
    for k, (a, b, c) in enumerate(rows):
        y = 4.3 - k * 0.5
        weight = "bold" if k == 0 else "normal"
        for x, value in [(9.15, a), (9.75, b), (10.85, c)]:
            ax.text(x, y, value, ha="left", va="top", fontsize=8.3, color=INK, weight=weight, family="monospace")
    ax.text(9.15, 0.5, "pandas collects the replies\ninto one table: a row per note", ha="left", va="bottom", fontsize=8.3,
            color=GREY)
    _arrow(ax, 3.75, 2.6, 4.05, 2.6, INK)
    _arrow(ax, 8.65, 2.6, 8.95, 2.6, INK)
    return _show(fig)


def two_by_two(results, note=None):
    """Part 5: per method, how many in-audit patients it found or missed and how many others it included."""
    from matplotlib.patches import Rectangle
    width, height = 11, 4.7
    lift = 0.3  # room under the boxes for the note
    fig, ax = _canvas(width, height)
    ax.text(width / 2, height - 0.05, "Who does each method put in the audit? Counts against the labels",
            ha="center", va="top", fontsize=12.5, color=INK, weight="bold")
    cells = [("found", 0, 0, "#cfe8dc"), ("missed", 1, 0, "#f6c9c4"),
             ("wrongly included", 0, 1, "#fde3bf"), ("correctly left out", 1, 1, LIGHT)]
    for k, (name, counts) in enumerate(results.items()):
        x0 = 2.0 + k * 4.6
        ax.text(x0 + 1.6, 3.55 + lift, name, ha="center", va="bottom", fontsize=11, color=INK, weight="bold")
        ax.text(x0 + 0.8, 3.3 + lift, "says yes", ha="center", va="bottom", fontsize=9, color=GREY)
        ax.text(x0 + 2.4, 3.3 + lift, "says no", ha="center", va="bottom", fontsize=9, color=GREY)
        for label, col, row, colour in cells:
            x, y = x0 + col * 1.6, 1.75 + lift - row * 1.4
            ax.add_patch(Rectangle((x, y), 1.55, 1.35, fc=colour, ec="white", lw=2))
            ax.text(x + 0.78, y + 0.82, str(counts[label]), ha="center", va="center", fontsize=18, color=INK,
                    weight="bold")
            ax.text(x + 0.78, y + 0.3, label, ha="center", va="center", fontsize=8.8, color=INK)
        if k == 0:
            ax.text(x0 - 0.1, 2.42 + lift, "label: in\nthe audit", ha="right", va="center", fontsize=9, color=GREY)
            ax.text(x0 - 0.1, 1.02 + lift, "label: not\nin the audit", ha="right", va="center", fontsize=9, color=GREY)
    if note:
        import textwrap
        ax.text(width / 2, 0.0, textwrap.fill(note, 130), ha="center", va="bottom", fontsize=8.5, color=GREY)
    return _show(fig)
