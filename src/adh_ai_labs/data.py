"""Loading the ECG data shared with the Multimodal Deep Learning lab.

Students put Ed Phillips's ``adh_multimodal_lab_data.zip`` (from Canvas) in Google
Drive at ``MyDrive/adh_multimodal_lab/``; the same file serves both Monday labs.
Outside Colab, ADH_LAB_ZIP points at a local copy. The ECGs and machine reports
are real (MIMIC-IV-ECG, ODbL); the EHR and outcomes are simulated.
"""

import os
import zipfile
from pathlib import Path

DRIVE_ZIP = Path("/content/drive/MyDrive/adh_multimodal_lab/adh_multimodal_lab_data.zip")
FOLDER = "adh_multimodal_lab_data"
FS = 250  # samples per second
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def find_zip():
    """Mount Drive in Colab if needed and return the path of the lab zip."""
    if os.environ.get("ADH_LAB_ZIP"):
        return Path(os.environ["ADH_LAB_ZIP"]).expanduser()
    try:
        from google.colab import drive  # noqa: F401
    except ImportError:
        return Path("adh_multimodal_lab_data.zip")
    if not DRIVE_ZIP.exists():
        drive.mount("/content/drive")
    return DRIVE_ZIP


def _existing_zip(zip_path=None):
    zip_path = Path(zip_path) if zip_path else find_zip()
    if not zip_path.exists():
        raise FileNotFoundError(
            f"No lab data at {zip_path}. Download adh_multimodal_lab_data.zip from Canvas and put "
            "it, still zipped, in a folder called adh_multimodal_lab at the top level of My Drive.")
    return zip_path


def unpack(zip_path=None, work_dir=None):
    """Unzip the lab data (once) and return the data folder."""
    zip_path = _existing_zip(zip_path)
    work_dir = Path(work_dir) if work_dir else (
        Path("/content") if Path("/content").is_dir() else zip_path.parent)
    data_dir = work_dir / FOLDER
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        if not all((work_dir / name).exists() for name in names):
            zf.extractall(work_dir)
    return data_dir


def load_ecgs(zip_path=None):
    """(metadata table, ECG array of shape (N, 12, 2500) as float32)."""
    import numpy as np
    import pandas as pd
    data_dir = unpack(zip_path)
    meta = pd.read_csv(data_dir / "metadata.csv")
    ecgs = np.load(data_dir / "ecgs.npy").astype(np.float32)
    return meta, ecgs


def load_metadata(zip_path=None):
    """The metadata table (one row per ECG, with its machine report), read without unzipping."""
    import pandas as pd
    with zipfile.ZipFile(_existing_zip(zip_path)) as zf, zf.open(f"{FOLDER}/metadata.csv") as f:
        return pd.read_csv(f)
