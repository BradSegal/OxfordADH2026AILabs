"""The three small open models notebook 02 runs on the CPU, behind a few plain functions.

Notebook 02 (how a language model reads and writes) opens the language model up
instead of calling it through a chat API: students see the tokens it reads, the
vectors an embedding model makes, and the probability it gives every possible
next token. Everything runs locally with ``transformers`` and
``sentence-transformers`` so the real reports never leave the Colab machine and
no GPU is needed.

- ``load()`` downloads the three models once and prints one line per model.
- ``embed(texts, model)`` returns one unit-length vector per text from the general
  (all-MiniLM-L6-v2) or the biomedical (NeuML pubmedbert-base-embeddings) model.
- ``LanguageModel`` wraps Qwen2.5 0.5B Instruct: ``tokens``, ``pieces``, ``decode``,
  ``next_token_scores``, ``next_token_probabilities``, ``top_next_tokens``,
  ``chat_text``, ``reply`` and ``samples``. Token lists are plain Python lists of
  integers and scores and probabilities are NumPy arrays, so the notebook works in NumPy.
- ``table(rows)`` and ``show_search(...)`` print results as small left-aligned
  tables instead of walls of text.
"""

import os

import numpy as np

LANGUAGE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
EMBEDDING_MODELS = {"general": "sentence-transformers/all-MiniLM-L6-v2",
                    "biomedical": "NeuML/pubmedbert-base-embeddings"}

# Notebook 02's clinical terms (vocabulary from the course's 2025 embeddings notebook, trimmed to four body
# systems and some medicines) and the pairs that name the same thing.
CLINICAL_TERMS = {
    "myocardial infarction": "heart", "heart attack": "heart", "MI": "heart", "atrial fibrillation": "heart",
    "AF": "heart", "hypertension": "heart", "high blood pressure": "heart", "heart failure": "heart",
    "pneumonia": "lungs", "chronic obstructive pulmonary disease": "lungs", "COPD": "lungs", "asthma": "lungs",
    "shortness of breath": "lungs", "dyspnoea": "lungs",
    "type 2 diabetes": "hormones", "hyperglycaemia": "hormones", "high blood sugar": "hormones",
    "hypoglycaemia": "hormones", "low blood sugar": "hormones", "hypothyroidism": "hormones",
    "underactive thyroid": "hormones",
    "chronic kidney disease": "kidneys", "CKD": "kidneys", "acute kidney injury": "kidneys", "AKI": "kidneys",
    "urinary tract infection": "kidneys", "UTI": "kidneys",
    "paracetamol": "medicines", "acetaminophen": "medicines", "atorvastatin": "medicines", "Lipitor": "medicines",
    "apixaban": "medicines", "Eliquis": "medicines",
}
SAME_MEANING = [("myocardial infarction", "heart attack"), ("myocardial infarction", "MI"),
        ("atrial fibrillation", "AF"), ("hypertension", "high blood pressure"),
        ("chronic obstructive pulmonary disease", "COPD"), ("shortness of breath", "dyspnoea"),
        ("hyperglycaemia", "high blood sugar"), ("hypoglycaemia", "low blood sugar"),
        ("hypothyroidism", "underactive thyroid"), ("chronic kidney disease", "CKD"),
        ("acute kidney injury", "AKI"), ("urinary tract infection", "UTI"),
        ("paracetamol", "acetaminophen"), ("atorvastatin", "Lipitor"), ("apixaban", "Eliquis")]

_embedders = {}
_language_model = None


def _quiet():
    """Hide download progress bars and library advice that would bury the notebook's own output."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    try:
        from huggingface_hub.utils import disable_progress_bars
        disable_progress_bars()
    except ImportError:
        pass
    try:
        from transformers.utils import logging
        logging.set_verbosity_error()
        logging.disable_progress_bar()
    except ImportError:
        pass


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def embedder(model="general"):
    """The sentence-transformers model called ``model`` ("general" or "biomedical"), loaded once."""
    if model not in EMBEDDING_MODELS:
        raise ValueError(f"model must be one of {sorted(EMBEDDING_MODELS)}, not {model!r}.")
    if model not in _embedders:
        _quiet()
        from sentence_transformers import SentenceTransformer
        _embedders[model] = SentenceTransformer(EMBEDDING_MODELS[model], device=_device())
    return _embedders[model]


def embed(texts, model="general"):
    """One vector per text, each scaled to length 1, as a NumPy array of shape (texts, dimensions).

    Because every vector has length 1, the dot product of two of them is their
    cosine similarity. A GPU is used if there is one; the numbers are the same to
    about three decimal places.
    """
    if isinstance(texts, str):
        raise TypeError("embed takes a list of texts; for one text write embed([text]).")
    vectors = embedder(model).encode(list(texts), batch_size=32, convert_to_numpy=True,
                                     normalize_embeddings=True, show_progress_bar=False)
    return vectors.astype(np.float32)


class LanguageModel:
    """Qwen2.5 0.5B Instruct on the CPU: tokens in, a probability for every possible next token out."""

    def __init__(self, name=LANGUAGE_MODEL):
        _quiet()
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.name = name
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        # Full precision on the CPU. transformers 4.56 renamed torch_dtype to dtype (Colab may have either).
        version = tuple(int(part) for part in transformers.__version__.split(".")[:2])
        key = "dtype" if version >= (4, 56) else "torch_dtype"
        self.model = AutoModelForCausalLM.from_pretrained(name, **{key: torch.float32})
        self.model.eval()
        self.end_of_turn = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.vocabulary_size = len(self.tokenizer)

    def tokens(self, text):
        """The token ids for ``text``, as a list of integers."""
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def pieces(self, text):
        """The pieces of text the tokens stand for, in order (spaces kept)."""
        return [self.decode([t]) for t in self.tokens(text)]

    def decode(self, tokens):
        """The text a list of token ids stands for."""
        return self.tokenizer.decode([int(t) for t in tokens])

    def next_token_scores(self, tokens):
        """The model's score (logit) for every token in its vocabulary coming next after ``tokens``.

        ``tokens`` is a list of ids or a text. Returns a NumPy array with one number per
        vocabulary entry. The scores are not probabilities: softmax turns them into ones.
        """
        import torch
        if isinstance(tokens, str):
            tokens = self.tokens(tokens)
        if any(t is ... for t in tokens):
            raise ValueError("tokens holds a ... placeholder: set next_token = ... to a token id before you "
                             "append it.")
        with torch.no_grad():
            logits = self.model(torch.tensor([list(tokens)])).logits[0, -1]
        return logits[: len(self.tokenizer)].double().numpy()

    def next_token_probabilities(self, tokens):
        """The probability of every token in the vocabulary coming next (sums to 1)."""
        import torch
        return torch.softmax(torch.from_numpy(self.next_token_scores(tokens)), dim=0).numpy()

    def top_next_tokens(self, text, k=5):
        """The ``k`` most likely next tokens after ``text``, as (piece of text, probability) pairs."""
        probabilities = self.next_token_probabilities(text)
        top = np.argsort(probabilities)[::-1][:k]
        return [(self.decode([t]), float(probabilities[t])) for t in top]

    def greedy(self, tokens, steps):
        """``steps`` greedy next tokens after ``tokens`` from ``transformers``' ``generate`` (with its cache)."""
        import torch
        ids = torch.tensor([list(tokens)])
        with torch.no_grad():
            out = self.model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=steps,
                                      min_new_tokens=steps, do_sample=False, pad_token_id=self.end_of_turn,
                                      temperature=None, top_p=None, top_k=None,
                                      repetition_penalty=1.0)  # the model's default (1.1) changes the pick
        return [int(t) for t in out[0]]

    def chat_text(self, messages):
        """The one long text the model actually reads for a list of chat messages.

        This is the model's chat template, with the marker that starts the
        assistant's turn added at the end, so the next token is the reply's first.
        """
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def reply(self, messages, max_new_tokens=60):
        """The model's reply to a list of chat messages, greedy (always the most likely next token).

        Uses ``transformers``' ``generate``, which runs the same predict, pick and
        append loop as the notebook, and stops at the end-of-turn token or after
        ``max_new_tokens``.
        """
        import torch
        ids = self.tokens(self.chat_text(messages))
        with torch.no_grad():
            out = self.model.generate(torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long),
                                      max_new_tokens=max_new_tokens, do_sample=False,
                                      eos_token_id=self.end_of_turn, pad_token_id=self.end_of_turn,
                                      temperature=None, top_p=None, top_k=None,
                                      repetition_penalty=1.0)  # the model's default (1.1) changes the pick
        return self.tokenizer.decode(out[0, len(ids):], skip_special_tokens=True).strip()

    def samples(self, text, temperature=1.0, steps=12, n=3, seed=None):
        """``n`` continuations of ``text``, each ``steps`` tokens sampled at ``temperature`` from every token.

        Uses ``generate`` with sampling and nothing else switched on (no top-k, top-p or
        repetition penalty), so each token is drawn from softmax(scores / temperature).
        """
        import torch
        if seed is not None:
            torch.manual_seed(seed)
        ids = torch.tensor([self.tokens(text)] * n)
        with torch.no_grad():
            out = self.model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=steps,
                                      min_new_tokens=steps, do_sample=True, temperature=float(temperature),
                                      top_k=0, top_p=1.0, repetition_penalty=1.0,
                                      pad_token_id=self.end_of_turn)
        return [self.decode(row[ids.shape[1]:]) for row in out]


def language_model():
    """Qwen2.5 0.5B Instruct, loaded once."""
    global _language_model
    if _language_model is None:
        _language_model = LanguageModel()
    return _language_model


def load():
    """Download and load the three models (about 1.5 GB the first time), one line per model."""
    import time
    _quiet()
    start = time.monotonic()
    if _language_model is None:
        print("Loading three models (the first time, this downloads about 1.5 GB: one to two minutes)...")
    lm = language_model()
    print(f"Language model ready: {LANGUAGE_MODEL} ({sum(p.numel() for p in lm.model.parameters()) / 1e6:.0f} "
          f"million parameters, {lm.vocabulary_size:,} possible tokens).")
    for model, name in EMBEDDING_MODELS.items():
        size = embed(["ECG"], model).shape[1]
        print(f"Embedding model ready ({model}): {name} ({size} numbers per text).")
    print(f"All three models loaded in {time.monotonic() - start:.0f} s.")
    return lm


def table(rows, columns=None):
    """A small table of text, left-aligned, without the row numbers (a plain DataFrame outside a notebook)."""
    import pandas as pd
    frame = pd.DataFrame(rows, columns=columns)
    try:
        from IPython import get_ipython
        if get_ipython() is None:
            return frame
        styles = [{"selector": "th", "props": [("text-align", "left")]},
                  {"selector": "td", "props": [("text-align", "left"), ("white-space", "pre-wrap")]}]
        return frame.style.hide(axis="index").set_table_styles(styles).format(precision=2)
    except ImportError:
        return frame


def similarities(pairs):
    """A table of the cosine similarity of each pair of phrases, from the general and the biomedical model.

    ``embed`` gives vectors of length 1, so the dot product of two of them is their cosine similarity.
    """
    rows = []
    for first, second in pairs:
        row = {"first": first, "second": second}
        for model in EMBEDDING_MODELS:
            a, b = embed([first, second], model)
            row[model] = round(float(a @ b), 2)
        rows.append(row)
    return table(rows)


def show_search(search, query, reports, is_af, n=5, using=None):
    """Show how many of the top 10 for ``query`` are AF reports, then the first ``n`` as a table.

    ``search`` is the student's TODO 2 function: until it is finished this prints a ⏳ line
    instead. ``using`` is another search function to run instead (Extension 4), built on it.
    """
    from .checks.llm_applications import unfinished
    if unfinished(search):
        print("\u23f3 This uses your search: complete TODO 2 first, then run this cell again.")
        return None
    top, scores = (using or search)(query)
    if top is ... or scores is ...:
        print("\u23f3 This uses your search: complete TODO 2 first, then run this cell again.")
        return None
    top = np.asarray(top)
    print(f"Search: \"{query}\". AF reports in the top {len(top)}: {int(np.asarray(is_af)[top].sum())}")
    rows = [{"rank": rank, "score": round(float(scores[i]), 2), "AF report?": "yes" if is_af[i] else "",
             "report": reports[i]} for rank, i in enumerate(top[:n], start=1)]
    return _display(table(rows))


def _display(obj):
    """Show ``obj`` in a notebook (so several tables can come out of one cell); return it outside one."""
    try:
        from IPython import get_ipython
        from IPython.display import display
        if get_ipython() is not None:
            display(obj)
            return None
    except ImportError:
        pass
    return obj


def unique_reports(meta):
    """The distinct machine reports in the metadata table, in order of first appearance."""
    return list(dict.fromkeys(meta["machine_report"].astype(str)))
