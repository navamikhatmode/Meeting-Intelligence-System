# Meeting-Intelligence-System
# 🧠 Meeting Intelligence System

**Inter IIT Tech Meet 15.0 · Bootcamp (#GoldforGuwahati) · Phase 2 – ML Problem Statement**
*IIT Guwahati Tech Board*

An AI-powered meeting assistant that turns a recorded meeting into an accurate transcript and a usable written record — summary, minutes, key decisions and action items — through a coordinated **multi-model pipeline**, wrapped in an interactive web app.

> **Core principle:** the recording is the only source of truth. If an owner or deadline was not stated, the system says **"Unspecified"** instead of inventing one. Proposals are never promoted to decisions.

---

## 📑 Table of Contents

1. [Problem Statement](#-problem-statement)
2. [Solution Overview](#-solution-overview)
3. [Models Used & Their Roles](#-models-used--their-roles)
4. [Architecture & Data Flow](#-architecture--data-flow)
5. [Features](#-features)
6. [Repository Structure](#-repository-structure)
7. [Setup & Run Instructions](#-setup--run-instructions)
8. [Using the Web App](#-using-the-web-app)
9. [Outputs](#-outputs)
10. [Anti-Hallucination Design](#-anti-hallucination-design)
11. [Error Handling](#-error-handling)
12. [Evaluation Rubric Mapping](#-evaluation-rubric-mapping)
13. [Limitations & Future Work](#-limitations--future-work)

---

## 🎯 Problem Statement

Build an AI-powered meeting assistant that turns a recorded meeting into an accurate transcript and a usable written record. The system must:

- **Transcribe** the audio with a speech-to-text model.
- **Refine** domain-specific terminology in the transcript using a language model, while preserving the speaker's meaning.
- **Generate** structured meeting minutes, key decisions and actionable tasks using a *separate* language model.
- **Orchestrate** all models into one reliable end-to-end workflow with an interactive interface and downloadable results.

The application must present the **raw transcript**, **refined transcript** and **generated record** in an interactive interface. Outputs must reflect what was said in the recording; where a task owner or deadline was not stated, it must be left unspecified rather than invented.

### Required model roles

| Model type | Required role |
|---|---|
| Speech-to-text model | Transcribe spoken content from the uploaded meeting recording |
| Language model for transcript refinement | Correct likely transcription errors, especially domain-specific terms, using the meeting context |
| Language model for meeting documentation | Generate structured minutes, key decisions and action items from the refined transcript |

The two language-model roles are implemented as **distinct processing stages**.

### Key functional requirements

- Accept an English-language meeting audio file and process it from upload to final output in **one run**.
- Retain both the **raw** and **refined** transcripts and produce a structured final record.
- Handle unsupported, empty or unreadable files with a **clear error message**.
- Generate outputs for new recordings **through the pipeline** — nothing is prewritten or hardcoded.
- Each task states the work to be done and includes an owner or deadline **only when the recording provides one**.
- Do not present a proposal as an agreed decision or an unstated assignment as a confirmed task.
- Show clear processing / failure status.

---

## 💡 Solution Overview

```
 Audio file ──► Stage 1: Speech-to-Text ──► Raw transcript
                    (Whisper)                    │
                                                 ▼
                              Stage 2: Domain-aware refinement ──► Refined transcript
                                   (Qwen2.5-7B-Instruct, 4-bit)          │
                                                                         ▼
                                           Stage 3: Documentation ──► Summary · Minutes ·
                                           (Qwen2.5-3B-Instruct)       Decisions · Action items
                                                                         │
                                                                         ▼
                                                    JSON · TXT · HTML report  +  Gradio UI
```

The project is delivered as a Jupyter notebook (developed on **Kaggle**) that builds the pipeline and launches a **Gradio** web app which re-uses the models already loaded in GPU memory.

---

## 🤖 Models Used & Their Roles

| Stage | Model | Role | Precision / Placement |
|---|---|---|---|
| 1. Speech-to-text | [`openai/whisper-small`](https://huggingface.co/openai/whisper-small) (Hugging Face `transformers` ASR pipeline) | Converts 16 kHz mono audio into the **raw transcript** (30 s chunking, batched) | GPU 0 |
| 2. Transcript refinement (LLM #1) | [`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | Corrects plausible recognition errors in technical terms, acronyms, software/model names, units and names — **without** summarising or changing meaning | 4-bit NF4 (bitsandbytes), GPU 0 |
| 3. Documentation (LLM #2) | [`Qwen/Qwen2.5-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) | Produces a structured JSON record: summary, minutes, decisions, action items | fp16, GPU 1 (falls back to GPU 0 on single-GPU sessions) |

### How outputs move between stages

1. **Audio → Whisper:** `ffmpeg` converts any supported input to 16 kHz mono WAV; Whisper returns the raw transcript text.
2. **Raw transcript → Refiner:** the transcript is split on sentence boundaries into chunks of ~250 words so long meetings are never truncated by the generation limit; each chunk is refined with greedy decoding (`do_sample=False`) and the parts are re-joined.
3. **Refined transcript → Documentation model:** the full refined transcript is the *only* input; the model must return strict JSON.
4. **JSON → Validator → Renderers:** the output is parsed (tolerating code fences / stray text), normalised so every section is a clean list, and rendered to the UI, JSON, TXT and HTML.

---

## ✨ Features

- 🎙️ **Upload or record** a meeting (MP3, WAV, M4A, FLAC, OGG, AAC, OPUS, MP4, WEBM, MKV …).
- 📊 **Live 4-step progress tracker:** Prepare audio → Transcribe → Refine → Document.
- 🔍 **Side-by-side comparison** of raw and refined transcripts.
- 🗂️ **Tabbed results:** Summary, Minutes, Decisions, Action Items, Transcripts.
- 📥 **Downloads:** structured **JSON**, human-readable **TXT**, and a styled standalone **HTML report** (printable to PDF).
- 📈 **Metrics tiles:** audio length, word count, number of decisions, number of action items.
- 🛡️ **Hallucination-guarded extraction** with explicit "Unspecified" handling.
- ⚠️ **Graceful error handling** for bad files, silent audio and GPU out-of-memory.

---

## 📁 Repository Structure

> Adjust to match your repository layout.

```
.
├── Meeting_Intelligence_System.ipynb   # Full pipeline + Gradio app (Kaggle notebook)
├── README.md                           # This file
├── requirements.txt                    # Python dependencies (see below)
├── prompts/                            # (optional) prompts exported from the notebook
│   ├── refinement_prompt.txt
│   └── documentation_prompt.txt
└── samples/
    ├── sample_meeting.<wav|mp3|m4a>    # Shareable demo recording
    ├── raw_transcript.txt
    ├── refined_transcript.txt
    ├── meeting_record.json
    ├── meeting_record.txt
    └── meeting_report.html
```

The prompts / model instructions live in the notebook:
- **Refinement prompt** → `refine_transcript()`
- **Documentation prompt** → `DOCUMENTATION_PROMPT`

---

## ⚙️ Setup & Run Instructions

### Option A — Kaggle (recommended, the environment this was built in)

1. Create a new Kaggle notebook and **import** `Meeting_Intelligence_System.ipynb`.
2. In **Session options**:
   - **Accelerator →** `GPU T4 x2` (best — the two LLMs get separate GPUs). `GPU P100` also works.
   - **Internet →** `On` (to download model weights and create the public Gradio link).
3. **Add your meeting audio:** *Add Input → Upload → New Dataset* and drop in your `.m4a/.mp3/.wav` file. The notebook auto-detects audio under `/kaggle/input/`. (Or hard-code `AUDIO_PATH` in the audio cell.)
4. Click **Run All** (or run cell by cell):
   - **Phase 1** – raw transcript (Whisper)
   - **Phase 2** – refined transcript (Qwen2.5-7B)
   - **Phase 3** – summary, minutes, decisions, action items (Qwen2.5-3B)
   - **Phase 4** – Gradio web app
5. The last cell prints a public link like `https://xxxx.gradio.live` — open it in any browser.

> ⚠️ Keep the notebook session **open** while using the app. "Save & Run All (Commit)" does not keep a web server running — use the interactive session. The public link is valid for ~72 hours while the session is alive.

### Option B — Local machine / other GPU environment

**Prerequisites**
- Python 3.10+
- An NVIDIA GPU with CUDA (≈ 16 GB VRAM total is comfortable; the 7B model runs in 4-bit)
- `ffmpeg` installed and on your `PATH`

**Install**

```bash
git clone <your-repo-url>
cd <your-repo-name>

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -U torch transformers accelerate bitsandbytes sentencepiece soundfile "gradio>=5,<6"
# ffmpeg: sudo apt-get install ffmpeg   (Linux)  |  brew install ffmpeg  (macOS)
```

**Run**

```bash
jupyter notebook Meeting_Intelligence_System.ipynb
```

Run the cells top to bottom. Note that the notebook contains a Kaggle-specific audio path; when running locally, set `AUDIO_PATH` to your own file (and `WORK_DIR` falls back to the current directory automatically). For the web app cell, set `share=False` if you don't need a public link.

### Dependencies

| Package | Purpose |
|---|---|
| `torch` | Model inference (CUDA) |
| `transformers`, `accelerate` | Whisper ASR pipeline & Qwen models |
| `bitsandbytes` | 4-bit quantisation of the 7B refiner |
| `sentencepiece` | Tokeniser support |
| `soundfile` | Reading WAV audio / duration |
| `gradio>=5,<6` | Interactive web interface |
| `ffmpeg` (system) | Audio conversion to 16 kHz mono WAV |

---

## 🖥️ Using the Web App

1. **Upload** an English meeting recording (or record via microphone).
2. Click **🚀 Process meeting**.
3. Watch the progress tracker move through *Prepare audio → Transcribe → Refine → Document*.
4. Inspect results in the tabs: **Summary · Minutes · Decisions · Action items · Transcripts** (raw and refined side by side).
5. **Download** the HTML report, JSON and TXT files.
6. Click **Clear** to reset and process another recording.

---

## 📤 Outputs

For each processed recording the app displays the following; the downloadable files are generated per run.

| Output | Contents | Where |
|---|---|---|
| **Raw transcript** | Speech-to-text result from Whisper | Transcripts tab · HTML report |
| **Refined transcript** | Transcript after domain-aware correction | Transcripts tab · HTML report |
| **Meeting minutes** | Concise summary and organised discussion points | UI · JSON · TXT · HTML |
| **Key decisions** | Only explicitly agreed/confirmed decisions (empty list if none) | UI · JSON · TXT · HTML |
| **Action items** | Task, owner and deadline; missing values shown as `Unspecified` | UI · JSON · TXT · HTML |

The meeting record is available in a **human-readable** format (`.txt`, `.html`) and a **machine-readable** format (`.json`). Both convey the same decisions and tasks.

### JSON schema

```json
{
  "summary": "Concise summary of what was discussed.",
  "minutes": [
    { "topic": "Topic title", "discussion": "What was discussed." }
  ],
  "decisions": [
    "Explicitly agreed decision."
  ],
  "action_items": [
    {
      "task": "Work to be done",
      "owner": "Name or Unspecified",
      "deadline": "Date/time or Unspecified"
    }
  ]
}
```

### Example (plain-text record)

```
MEETING RECORD
============================================================

MEETING SUMMARY
------------------------------------------------------------
...

KEY DECISIONS
------------------------------------------------------------
No explicit decisions identified.

ACTION ITEMS
------------------------------------------------------------
1. <task>
   Owner: Unspecified
   Deadline: Unspecified
```

---

## 🛡️ Anti-Hallucination Design

Accuracy of *commitments* matters more than fluency, so the pipeline is built defensively:

- **Strict refinement prompt:** the refiner may correct *only* likely ASR errors; it must preserve names, numbers, dates, units, negations, uncertainty, questions, disagreements and proposals, and must keep the original word when unsure.
- **Source-of-truth documentation prompt:** the transcript is the only allowed source; "if unsupported, do not include it".
- **Explicit decision test:** only language such as *"we decided / agreed / finalized / confirmed"* counts. *"We should… / maybe… / I suggest…"* are treated as proposals, never decisions.
- **Explicit action-item test:** a task requires clear assignment or commitment; suggestions, hypotheticals and questions are excluded.
- **Owner / deadline rules:** never inferred from who spoke, job title or urgency — otherwise `"Unspecified"`.
- **No-decision / no-action meetings are valid:** the model returns `[]` instead of manufacturing outcomes.
- **Deterministic decoding:** `do_sample=False` for both LLM stages.
- **Post-processing validator:** normalises malformed model output (e.g. the string `"None"`), fills missing owner/deadline with `Unspecified`, and tolerates code fences / stray text around the JSON.

---

## 🚨 Error Handling

| Situation | Behaviour |
|---|---|
| No file uploaded | "Please upload or record a meeting first." |
| Unsupported / corrupt / unreadable audio | "Could not read this audio file. Please try a different format (mp3, wav, m4a)." |
| Empty or silent recording (no speech) | "No speech was detected in this recording." |
| GPU out of memory | Clear message suggesting a shorter recording or a `GPU T4 x2` session |
| Model returns non-JSON output | Graceful fallback; the raw model output is shown in the summary with a warning |
| Any other failure | `Processing failed: <ErrorType>: <message>` shown in the interface |

---

## 📊 Evaluation Rubric Mapping

| Criterion | Pts | How this project addresses it |
|---|---|---|
| Speech transcription | 20 | Whisper-small with 30 s chunking + batching; 16 kHz mono preprocessing via ffmpeg |
| Transcript refinement | 20 | Dedicated 7B LLM stage with strict meaning-preservation rules; chunked for long recordings; raw vs refined shown side by side |
| Minutes and decisions | 25 | Separate documentation LLM; explicit-agreement test; proposals never become decisions |
| Action items | 15 | Task / owner / deadline schema; `Unspecified` for anything not stated |
| End-to-end application | 15 | One-click Gradio workflow: upload → process → inspect → download; progress & error states |
| Submission quality | 5 | This README, prompts in the repo, model roles documented, sample outputs, demo |

---

## ⚠️ Limitations & Future Work

- Designed for **English-language** recordings.
- Whisper-small trades some accuracy for speed; a larger Whisper variant can be swapped in via the `model=` argument of the ASR pipeline.
- No **speaker diarization** — owners are only captured when named in the speech itself.
- Very long meetings are handled by chunked refinement; the documentation stage receives the full refined transcript, so extremely long recordings may need chunked documentation with a merge step.
- The model-context window and the 3B documentation model limit extraction quality on very dense meetings.
- Possible extensions: diarization, timestamps in minutes, PDF/DOCX export, multilingual support.

---
