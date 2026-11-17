# AI for Efficient Healthcare Systems: labs

The Google Colab labs for the deep learning and large language model sessions of the
MSc in Applied Digital Health, University of Oxford, 2026.

| Lab | Session | Open |
|---|---|---|
| Introduction to deep learning: detecting atrial fibrillation from ECGs | Monday 16 November, 13:30 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BradSegal/OxfordADH2026AILabs/blob/main/notebooks/01_intro_to_deep_learning.ipynb) |
| LLM applications: how a language model reads and writes | Tuesday 17 November, 13:30 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BradSegal/OxfordADH2026AILabs/blob/main/notebooks/02_llm_applications.ipynb) |
| LLM integrations: a stroke-prevention audit with a language model | Tuesday 17 November, 15:00 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BradSegal/OxfordADH2026AILabs/blob/main/notebooks/03_llm_integrations.ipynb) |

## Before the session

1. **Download `adh_multimodal_lab_data.zip` from Canvas** (92 MB). It is the same file as
   the Multimodal Deep Learning lab, and is used by the first two labs (the third uses clinic
   notes that come with the course helpers).
2. **Put it in Google Drive** at `MyDrive/adh_multimodal_lab/adh_multimodal_lab_data.zip`:
   create an `adh_multimodal_lab` folder at the top level of My Drive, and leave the zip
   zipped. The notebook unzips it for you.

## Running a notebook

1. **Open it in Colab** with its button above. You need to be signed in to a Google account.
2. **Choose the runtime:** *Runtime > Change runtime type*.
   - Introduction to deep learning: a T4 GPU if you can; without one, training just takes a
     little longer.
   - LLM applications: no GPU needed; it is designed for the standard CPU runtime.
   - LLM integrations: a T4 GPU if you can. Without one, it uses a smaller model on the CPU and
     shows a run recorded on a GPU for the steps that read every note.
3. **Run the cells in order** (click a cell, then press Shift + Enter). The first cell
   installs the course helpers. When asked, connect the Google Drive that holds the zip.
4. **Keep your work:** *File > Save a copy in Drive*. Colab does not save your changes
   back to GitHub.

If something goes wrong, *Runtime > Restart session and run all* usually fixes it.

## Data and credits

Please keep the data within the course, and do not paste any of it into an AI chatbot.

- **MIMIC-IV-ECG:** Gow, B., Pollard, T., et al. (2023). MIMIC-IV-ECG: Diagnostic
  Electrocardiogram Matched Subset (version 1.0). PhysioNet. Open access under the Open
  Database License (ODbL): if you share anything built from it, keep this attribution and
  the licence. https://doi.org/10.13026/4nqg-sb35
- Goldberger, A., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. Circulation
  101(23), e215 to e220.
- **ECGFounder** (PKUDigitalHealth, NEJM AI 2025, MIT licence), whose embeddings appear in
  the final figure: https://github.com/PKUDigitalHealth/ECGFounder
- The ECGs and machine reports are real; the health records in the same file are
  simulated and are not used in these labs.
- **Clinic notes** (LLM integrations): forty-one synthetic notes written for this course; no
  real patient. CC BY 4.0.
- **MedMCQA** (an extension in LLM integrations): Pal, A., Umapathi, L. K. and Sankarasubbu, M.
  (2022). MedMCQA. PMLR 174, 248 to 260. Twenty questions, Apache License 2.0.
- **Models**, downloaded when a notebook runs, not stored here: Qwen2.5 0.5B Instruct, Qwen2.5
  1.5B Instruct and Qwen3 4B Instruct 2507 (Alibaba Cloud, Qwen team), all-MiniLM-L6-v2
  (sentence-transformers) and pubmedbert-base-embeddings (NeuML), all under the Apache License
  2.0. The LLM integrations lab runs its models with Ollama (MIT licence).

## Licence

The code is under the MIT licence ([LICENSE](LICENSE)); the written material is under
CC BY 4.0 ([LICENSE-content.md](LICENSE-content.md)).
