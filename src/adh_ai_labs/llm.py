"""Run an open LLM inside Colab and talk to it with the OpenAI Python SDK.

The notebooks never send course data to a hosted service: Ollama serves an open
model on the Colab machine, and its OpenAI-compatible endpoint lets students use
the same SDK calls they would use with a hosted provider.

    llm.start()   # installs and starts Ollama, downloads the model
    client = OpenAI(base_url=llm.URL, api_key="ollama")   # the student's own cell
    client.chat.completions.create(model=llm.MODEL, messages=[...])

Which model depends on the machine:

- with a GPU (a Colab T4): Qwen3-4B-Instruct-2507 at 4 bits (Apache 2.0, 2.5 GB), which
  answers in about a second, follows a JSON schema and calls tools;
- without one: Qwen2.5-1.5B-Instruct (Apache 2.0, 1 GB), which takes a few seconds per
  answer on two CPU cores and is less reliable. ``ON_GPU`` says which.

If the server cannot start at all (for example, no network to install it), ``start``
prints one plain line and runs a replay server at ``URL`` instead (``recordings``): the
student's client works unchanged, a request the worked copy made gets the response it
recorded on a GPU and any other gets an empty reply, and ``READY`` is False so the
helpers show recordings and the checks wait.

Environment variables, for testing outside Colab:

- ``ADH_LLM_CPU=1`` takes the CPU path even when a GPU is present;
- ``ADH_LLM_URL`` is the server's address (default ``http://localhost:11434``); a
  server already running there is used as it is;
- ``ADH_LLM_MODEL`` overrides the model;
- ``ADH_LLM_OFFLINE=1`` behaves as if the server could not start;
- ``ADH_RECORD=1`` on a GPU puts a recording proxy at ``URL`` that saves every response
  to the recordings (see ``recordings``).
"""

import json
import os
import shutil
import subprocess
import time
import urllib.request
from urllib.parse import urlparse

GPU_MODEL = "qwen3:4b-instruct-2507-q4_K_M"
CPU_MODEL = "qwen2.5:1.5b"
SMALL_MODEL = os.environ.get("ADH_LLM_SMALL_MODEL", CPU_MODEL)
BASE_URL = os.environ.get("ADH_LLM_URL", "http://localhost:11434").rstrip("/")
DEFAULT_MODEL = GPU_MODEL  # kept for older callers

# Standard prices in US dollars per million tokens, read on 15 November 2026 from
# developers.openai.com/api/docs/pricing and ai.google.dev/gemini-api/docs/pricing.
# An illustration: prices change often.
PRICES = [
    {"model": "gpt-6-luna", "provider": "OpenAI", "input_per_million": 0.10, "output_per_million": 0.50},
    {"model": "gemini-3.1-flash-lite", "provider": "Google", "input_per_million": 0.25, "output_per_million": 1.50},
    {"model": "gemini-3.8-flash", "provider": "Google", "input_per_million": 0.75, "output_per_million": 3.75},
    {"model": "gpt-6.1-terra", "provider": "OpenAI", "input_per_million": 1.00, "output_per_million": 5.00},
    {"model": "gemini-3.1-pro-preview", "provider": "Google", "input_per_million": 2.00, "output_per_million": 12.00},
    {"model": "gpt-6-astra", "provider": "OpenAI", "input_per_million": 10.00, "output_per_million": 50.00},
]


def _has_gpu():
    if os.environ.get("ADH_LLM_CPU") == "1":
        return False
    smi = shutil.which("nvidia-smi") or ("/usr/lib/wsl/lib/nvidia-smi"
                                          if os.path.exists("/usr/lib/wsl/lib/nvidia-smi") else None)
    if not smi:
        return False
    try:
        return subprocess.run([smi, "-L"], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


ON_GPU = _has_gpu()
MODEL = os.environ.get("ADH_LLM_MODEL") or (GPU_MODEL if ON_GPU else CPU_MODEL)
READY = False  # True once a server answers; False before start() and when it could not start
URL = f"{BASE_URL}/v1"  # where the notebook's OpenAI client sends requests (a replay server when READY is False)


class Unavailable(RuntimeError):
    """The model server could not be started."""


def _server_up():
    try:
        with urllib.request.urlopen(f"{BASE_URL}/api/version", timeout=2):
            return True
    except OSError:
        return False


INSTALL_SECONDS, PULL_SECONDS = 600, 900  # generous deadlines; past them start() falls back to the recordings


def _run(command, timeout=INSTALL_SECONDS, **kwargs):
    done = subprocess.run(command, capture_output=True, text=True, timeout=timeout, **kwargs)
    if done.returncode:
        raise Unavailable(f"{command} failed: {(done.stderr or done.stdout)[-500:]}")


def _install():
    if shutil.which("ollama"):
        return
    print("Installing Ollama (about a minute)...", flush=True)
    deadline = time.monotonic() + INSTALL_SECONDS  # one deadline for the whole install

    def left():
        return max(1, deadline - time.monotonic())

    if not shutil.which("zstd") and shutil.which("apt-get"):  # the installer unpacks .tar.zst
        _run("apt-get -qq update && apt-get -qq install -y zstd", shell=True, timeout=left())
    # Download, then run: in "curl | sh" a failed download can still exit 0.
    _run(["curl", "-fsSL", "https://ollama.com/install.sh", "-o", "/tmp/ollama-install.sh"], timeout=left())
    _run(["sh", "/tmp/ollama-install.sh"], timeout=left())
    if not shutil.which("ollama"):
        raise Unavailable("Ollama did not install.")


def _serve():
    if _server_up():
        return
    # A context window of 8,192 tokens: ample for the labs, and small enough for a T4's memory.
    env = {**os.environ, "OLLAMA_CONTEXT_LENGTH": os.environ.get("OLLAMA_CONTEXT_LENGTH", "8192"),
           "OLLAMA_HOST": urlparse(BASE_URL).netloc}
    subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     start_new_session=True, env=env)
    for _ in range(60):
        if _server_up():
            return
        time.sleep(1)
    raise Unavailable("The local model server did not start.")


def _has_model(model):
    try:
        with urllib.request.urlopen(f"{BASE_URL}/api/tags", timeout=5) as reply:
            names = [m["name"] for m in json.load(reply)["models"]]
    except (OSError, ValueError, KeyError):
        return False
    return model in names or f"{model}:latest" in names


def pull(model):
    """Download a model to the local server (quick if it is already there)."""
    if _has_model(model):
        return model
    if not READY:
        return model  # no server: the offline client answers every model the same way
    print(f"Downloading {model} (a minute or two the first time)...", flush=True)
    env = {**os.environ, "OLLAMA_HOST": urlparse(BASE_URL).netloc}
    _run(["ollama", "pull", model], env=env, timeout=PULL_SECONDS)
    return model


def server_failed(error):
    """True if ``error`` means the server stopped answering (no connection, a timeout or a server error)."""
    import openai
    if isinstance(error, openai.APIConnectionError):  # includes APITimeoutError
        return True
    return isinstance(error, openai.APIStatusError) and (error.status_code >= 500 or error.status_code == 404)


def _local(base_url):
    """Start a replay server (``upstream`` None) or a recording proxy; return its OpenAI-style URL."""
    from . import recordings
    address = urlparse(BASE_URL)
    host = address.hostname or "127.0.0.1"
    try:
        port = address.port if base_url is None else 0
        server = recordings.serve(host, port, upstream=base_url)
    except OSError:  # the address is taken (a server that is not answering): any free port will do
        server = recordings.serve(host, 0, upstream=base_url)
    return f"http://{host}:{server.server_address[1]}/v1"


def start():
    """Make sure a local model is being served at ``URL``, ready for an OpenAI client.

    If the server cannot start, print one line and serve the recordings at ``URL`` instead.
    """
    global READY, URL
    try:
        if os.environ.get("ADH_LLM_OFFLINE") == "1":
            raise Unavailable("offline by request")
        _install()
        _serve()
        READY = True
        pull(MODEL)
    except (Unavailable, OSError, subprocess.TimeoutExpired) as error:
        READY = False
        URL = _local(None)
        print("The model server could not start here, so cells that only show results show the run "
              "recorded on a GPU, and the checks that need the model will wait. "
              f"Run this cell again to retry. ({str(error)[:120]})")
        return URL
    READY = True
    URL = f"{BASE_URL}/v1"
    from . import recordings
    if recordings.recording():
        URL = _local(BASE_URL)
    if ON_GPU:
        print(f"Ready: {MODEL} is running on this machine's GPU.")
    else:
        print(f"Ready: no GPU here, so this is the smaller {MODEL}, on the CPU. It takes a few seconds per note "
              "and makes more mistakes. Steps that read many notes read some live and show a run recorded on a "
              "GPU for the rest.")
    return URL
