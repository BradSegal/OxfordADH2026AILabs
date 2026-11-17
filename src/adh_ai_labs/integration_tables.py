"""How notebook 03 shows notes and results: short cards and styled tables instead of raw text.

- ``cards``: notes as small cards, the words that matter for AF highlighted.
- ``table``: a pandas table with wrapped text, ✅/❌ cells coloured, and an optional
  one-line summary above it; long tables scroll inside a box.
- ``highlight``: a note's text as HTML with AF-related words marked.

Outside a notebook (the tests), each function prints plain text instead.
"""

import html
import re

RIGHT, WRONG = "✅", "❌"
GOOD_BG, BAD_BG, MARK_BG = "#e3f1ea", "#fbe3e1", "#fff1b8"
# Words that decide whether a note is in the audit: AF and its abbreviations, flutter, suspicion,
# negation, family and rhythm.
KEY = re.compile(
    r"(\?AF|\bPAF\b|\bAFL\b|\bAF\b|atrial fibrillation|paroxysmal atrial fibrillation|atrial flutter|flutter|"
    r"[Pp]aroxysmal AF|[Pp]ersistent AF|[Pp]ermanent AF|no evidence of [a-z ]+|no AF\b[a-z ]*|not AF|"
    r"[Pp]ossible [a-z ]+|irregularly irregular|irregular pulse|[Pp]ulse irregular\w*|sinus rhythm|"
    r"[Mm]other had AF|[Ff]ather [a-z ]+|excluded|ablated|cardioverted|cardioversion)")


def _ipython():
    try:
        from IPython import get_ipython
        from IPython.display import HTML, display
    except ImportError:
        return None
    return (display, HTML) if get_ipython() is not None else None


def highlight(text):
    """A note as HTML: escaped, line breaks kept, AF-related words marked."""
    out, last = [], 0
    for match in KEY.finditer(text):
        out.append(html.escape(text[last:match.start()]))
        out.append(f'<mark style="background:{MARK_BG};padding:0 2px">{html.escape(match.group(0))}</mark>')
        last = match.end()
    out.append(html.escape(text[last:]))
    return "".join(out).replace("\n", "<br>")


def show_html(markup, plain):
    """Show ``markup`` in a notebook, or print ``plain`` elsewhere."""
    shown = _ipython()
    if shown:
        display, HTML = shown
        display(HTML(markup))
    else:
        print(plain)


def cards(notes, ids, captions=None):
    """Show the given notes as cards, side by side where there is room."""
    captions = captions or {}
    items = []
    for note_id in ids:
        caption = captions.get(note_id, "")
        items.append(
            '<div style="flex:1 1 260px;max-width:360px;border:1px solid #d6d4cf;border-radius:8px;'
            'padding:8px 10px;font-size:13px;line-height:1.4;background:white;color:#222">'
            f'<div style="font-weight:bold;margin-bottom:4px">{note_id}'
            f'<span style="font-weight:normal;color:#6b6a66"> {html.escape(caption)}</span></div>'
            f'{highlight(notes.loc[note_id, "note"])}</div>')
    plain = "\n\n".join(f"{i} {captions.get(i, '')}\n{notes.loc[i, 'note']}" for i in ids)
    show_html(f'<div style="display:flex;flex-wrap:wrap;gap:8px">{"".join(items)}</div>', plain)


def _cell_colour(value):
    text = str(value)
    if text.startswith(RIGHT):
        return f"background-color:{GOOD_BG}"
    if text.startswith(WRONG):
        return f"background-color:{BAD_BG}"
    return ""


def table(frame, summary=None, html_columns=(), height=None, colour=None):
    """Show a table: wrapped left-aligned text, ✅/❌ cells coloured, ``summary`` in bold above it.

    ``html_columns`` hold HTML already (a highlighted note); ``colour`` is a function of the
    frame returning a same-shaped frame of CSS (for cells that are wrong without a mark);
    ``height`` (pixels) makes a long table scroll inside a box.
    """
    shown = _ipython()
    if not shown:
        if summary:
            print(summary)
        plain = frame.copy()
        for column in html_columns:
            plain[column] = plain[column].map(lambda v: re.sub(r"<[^>]+>", "", str(v).replace("<br>", " / ")))
        print(plain.to_string())
        return None
    display, HTML = shown
    styler = frame.style
    styler = styler.map(_cell_colour)
    if colour is not None:
        styler = styler.apply(colour, axis=None)
    styler = styler.set_properties(**{"text-align": "left", "white-space": "normal", "vertical-align": "top",
                                      "font-size": "13px"})
    styler = styler.set_table_styles([{"selector": "th", "props": "text-align:left;font-size:13px"},
                                      {"selector": "td", "props": "max-width:420px"}])
    escape = [c for c in frame.columns if c not in html_columns]
    styler = styler.format(lambda v: html.escape(str(v)) if v is not None else "", subset=escape)
    body = styler.to_html()
    if height:
        body = f'<div style="max-height:{height}px;overflow-y:auto;border:1px solid #e9e8e4">{body}</div>'
    head = f'<p style="font-weight:bold;margin:4px 0">{html.escape(summary)}</p>' if summary else ""
    display(HTML(head + body))
    return None


def mark(ok, text=""):
    """A cell that says right or wrong: ✅ or ❌, then the text."""
    return f"{RIGHT if ok else WRONG} {text}".strip()
