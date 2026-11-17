# Sources of the bundled assets

## medmcqa_20.json

Twenty multiple-choice questions from the validation split of MedMCQA, used in notebook 03 (LLM integrations). They are third-party material under the Apache License 2.0, not under this package's MIT licence; the full licence text is in `LICENSE-MedMCQA-Apache-2.0.txt` beside them. Neither upstream repository has a NOTICE file (checked 15 November 2026).

- Dataset: MedMCQA, `openlifescienceai/medmcqa` on Hugging Face (revision `91c6572c454088bf71b679ad90aa8dffcd0d5868`), https://huggingface.co/datasets/openlifescienceai/medmcqa.
- Licence: Apache License 2.0, as stated on the dataset card. (The authors' code repository, https://github.com/medmcqa/medmcqa, is under the MIT licence.)
- Citation: Pal A, Umapathi LK, Sankarasubbu M. MedMCQA: A Large-scale Multi-Subject Multi-Choice Dataset for Medical domain Question Answering. Proceedings of the Conference on Health, Inference, and Learning, PMLR 174:248-260, 2022. https://proceedings.mlr.press/v174/pal22a.html

Selection (15 November 2026): single-answer questions with no image, no "except" or "all of the above" wording, short enough to read quickly, mostly cardiovascular, and with an answer that is current in UK practice. Each answer key was checked by hand. Five questions have each answer letter, and the order is shuffled with a fixed seed.

Left out or replaced:

- routine antibiotic prophylaxis for dental extraction (the keyed answer is not UK practice);
- `960e3779-3e04-4c22-9cac-7e76c26c1e7e`, "Which drug is given to prevent acute mountain sickness?" (keyed acetazolamide): dexamethasone, also offered, is a recognised preventive too (CDC Yellow Book 2026; Wilderness Medical Society guideline 2024), so the question has two defensible answers. Replaced on 15 November 2026 by `e7f023ea-2147-47d3-9f2a-61fb60a900be` (antidote for fibrinolytic therapy, keyed epsilon aminocaproic acid), which keeps five questions per letter.

Edits: the text is as published except for these changes; only the stroke question's change adds words, and it adds context rather than changing the keyed answer.

| MedMCQA id | Subject | Edit |
|---|---|---|
| `9a901f04-8eb8-4e4f-a332-14cf2a3c70a3` | Pharmacology | tab characters replaced by spaces |
| `e7f023ea-2147-47d3-9f2a-61fb60a900be` | Pharmacology | none |
| `363aca43-6b66-4640-9cfe-93057f5c50f1` | Pharmacology | "hea rate" corrected to "heart rate" |
| `148f5ca1-0e61-4e4a-a19a-47df89cb1689` | Medicine | trailing full stops removed from the options |
| `edbdb125-7a27-4740-9e03-b379ae34773d` | Medicine | context added: "in a stroke patient" became "from symptom onset, in a patient with acute ischaemic stroke", the conditions under which the keyed 4.5 hours is the conventional answer (NICE NG128) |
| `92ee38b6-68d0-4626-8c03-053db30d77dc` | Medicine | a line break inside the question removed |
| `1f93c541-baaf-40f2-bf91-31be29732da2` | Pharmacology | none |
| `3fd17b81-42d6-41dc-915f-d4b612074dc5` | Pharmacology | none |
| `d3d896df-5856-43da-9baa-4f4ddaefde74` | Pharmacology | none |
| `70d87e90-4362-422a-9611-847021b23c33` | Biochemistry | none |
| `cbf6cefb-7428-4d61-9940-03967a1a56b8` | Medicine | "90deg" written as "90 degrees" |
| `3be9da5f-c0da-4e42-bbfe-a6a1373a7c5d` | Medicine | none |
| `c2cf6fda-d767-48e5-acbe-c663d376240b` | Medicine | none |
| `7a3a771f-2b56-4af8-b967-3c991b394bc6` | Pharmacology | none |
| `f447a6cc-d72e-4d0b-8330-ffc5c840394f` | Pharmacology | "hea rate" and "aerial" corrected to "heart rate" and "arterial" |
| `d454ee11-50e1-475c-b73f-e1c32f66d980` | Medicine | none |
| `ae07edee-fc24-489b-8c84-b89c77378ed8` | Medicine | none |
| `1d405e6b-5cd9-4ff7-ba6d-f7f748ad6050` | Pharmacology | none |
| `13e7b27f-6f0c-4366-955b-16836ed6f6cd` | Pathology | none |
| `4f22926c-1054-4056-8530-dd84d32600f9` | Medicine | none |

## Clinic notes (notebook 03)

`clinic_notes.json`: forty-one short clinic notes (GP consultations, clinic letters and discharge summary snippets) used in notebook 03 (LLM integrations) for a GP practice audit of stroke prevention in atrial fibrillation (NICE NG196). They were written for this course on 17 November 2026 and are entirely synthetic: no note describes or was derived from a real patient, and names, ages and dates are invented. Because no real patient data are involved, the notes may be shown to any language model, including a hosted one. (This does not change the rule for Monday's MIMIC-derived data, which must not be pasted into an AI chatbot.)

- Licence: Creative Commons Attribution 4.0 International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/.
- Labels: `af_status` and `in_audit` follow the audit definition given to students (a diagnosis of atrial fibrillation of any type, including patients now in sinus rhythm; not suspected AF, AF ruled out, atrial flutter only or a family history). The CHA2DS2-VASc inputs follow Lip GYH, Nieuwlaat R, Pisters R, Lane DA, Crijns HJGM. Refining clinical risk stratification for predicting stroke and thromboembolism in atrial fibrillation using a novel risk factor-based approach. Chest 2010;137(2):263-272, https://doi.org/10.1378/chest.09-1584. Each score is computed from those inputs, not typed by hand.

## Notebook 02 models

Notebook 02 (how a language model reads and writes) downloads three open models from Hugging Face at run time; none is bundled in this package. Licences checked on each model card and in the Hugging Face model API on 17 November 2026, at the revisions below.

| Model | Use in notebook 02 | Licence | Revision |
|---|---|---|---|
| `Qwen/Qwen2.5-0.5B-Instruct` (Alibaba Cloud, Qwen team) | tokeniser, next-token probabilities, generation, chat template | Apache License 2.0 (model card `license: apache-2.0`; LICENSE file in the repository) | `7ae557604adf67be50417f59c2c2f167def9a775` |
| `sentence-transformers/all-MiniLM-L6-v2` | general embedding model | Apache License 2.0 (model card) | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |
| `NeuML/pubmedbert-base-embeddings` | biomedical embedding model | Apache License 2.0 (model card); fine-tuned from `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`, which is under the MIT licence | `b79526d6ef3645e0df4530322e266f24c829f5ef` |

The clinical vocabulary in `language.CLINICAL_TERMS` is adapted from the course's own 2025 embeddings notebook (`BradSegal/ADH-LLM-Tutorials-2025`, notebook 07), trimmed and in British spelling.
