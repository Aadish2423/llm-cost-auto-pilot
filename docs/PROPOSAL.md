# Autopilot Edge
### A hybrid AI router for Snapdragon PCs: private by default, cloud only when it's worth it

**Snapdragon® AI Lab Build & Present Challenge proposal**

---

## One-line pitch

Autopilot Edge sits between your apps and AI models. For every prompt it decides on the
laptop whether the prompt can be answered **on the Snapdragon NPU**, where it's free, private
and works offline, or whether it's worth paying for a cloud model. **Prompts that contain
personal data never leave the device.**

## The problem

Developers and companies send almost every AI request to cloud models, even trivial ones:

- **Cost.** A cloud model is billed per token even for "convert this date" or "translate
  this line", which a small local model handles just as well.
- **Privacy.** Prompts routinely carry Aadhaar and PAN numbers, bank details, phone numbers,
  customer names and even API keys. All of it goes to a third-party server. India's
  Digital Personal Data Protection Act, 2023 makes that a compliance problem, not just a
  hygiene problem.
- **Availability.** When the network or the provider is down, the application stops.

Snapdragon X PCs have a dedicated AI engine (the Hexagon NPU), but software rarely decides
**per request** when to use it and when to use the cloud.

## The solution

Autopilot Edge extends my existing project, *LLM Cost Auto Pilot* (a cost-optimising LLM
router with a pre-call cost predictor, explainable routing, quality verification, a REST
API and a dashboard), into a hybrid on-device and cloud router.

For every prompt:

1. **On-device privacy guard (AI + rules), about 9 ms per prompt:**
   - Checksum-validated detectors for Aadhaar (Verhoeff), PAN, GSTIN, IFSC, UPI IDs,
     payment cards (Luhn), bank and passport numbers, phones, emails, API keys and passwords.
   - An **on-device BERT named-entity model** that finds people, organisations and places.
2. **Complexity classifier.** Each prompt gets a complexity tier (1–3) and the minimum model
   quality that tier needs.
3. **Routing decision:**
   - Sensitive prompt → **on-device only**. If no local model is available, the request is
     **blocked rather than sent to the cloud** (it fails closed).
   - Simple prompt → **on-device** (₹0, private, offline).
   - Hard prompt with no personal data → the **cheapest cloud model that meets the quality
     bar**.
4. **Explained and logged.** Every decision states its reason, and sensitive prompts and
   their answers are stored **redacted**.
5. **Resilience.**
   - If the cloud fails (outage, rate limit, offline), the prompt is answered on-device.
   - If the on-device call fails on a non-sensitive prompt, it retries in the cloud.

### Example decisions (live run)

| Prompt | Decision | Why |
|---|---|---|
| "What is the capital of Japan?" | 💻 on-device | Tier 1: the local model meets the quality bar |
| "Fill this KYC form… Aadhaar 4995 1234 5670, PAN ABCPE1234F…" | 💻 on-device (locked) | Contains AADHAAR, PAN, PERSON |
| "Why does my code return 401? …key sk-proj-…" | 💻 on-device (locked) | Contains API_KEY |
| "Summarize… and classify its sentiment" | ☁️ cloud (gpt-oss-20b) | Tier 2 needs medium quality |
| "Design an event-driven payments architecture…" | ☁️ cloud (gpt-oss-120b) | Tier 3 needs high quality |

---

## AI models

| Model | Source | Role | Runs on |
|---|---|---|---|
| **BERT-base NER** (dslim/bert-base-NER, 110M params, INT8 ONNX) | Open source, Hugging Face (MIT) | Privacy guard: names, organisations, places | Snapdragon NPU via ONNX Runtime **QNN Execution Provider**; CPU fallback |
| **Phi-3.5-mini** via Microsoft **Foundry Local** | Open source (MIT) | On-device language model for simple and sensitive prompts | Snapdragon NPU (Foundry Local serves the QNN build) |
| **Llama 3.2 3B** via Ollama | Open source (Llama licence) | On-device language model on any PC | CPU/GPU |
| gpt-oss-20b / gpt-oss-120b (Groq), Gemini | Cloud | Tier 2 / Tier 3 prompts without personal data | Cloud |

On-device runtimes are listed in priority order in `config/hybrid_config.yaml`. On a
Snapdragon PC the NPU runtime is picked first, and other models can be added to the same
list.

---

## 1. Technical implementation

| Component | What was built |
|---|---|
| Privacy guard | `app/privacy/`: 12 rule detectors (checksum- or context-validated where the format allows) plus an ONNX BERT NER. It scans long prompts in 510-token windows (no truncation), snaps entities to whole words, and ignores ID keywords the model mistakes for names. Configurable lock threshold. |
| NPU runtime | `app/edge/accelerators.py`: ONNX Runtime sessions that pick the **QNN execution provider (Hexagon NPU, burst mode)**, then DirectML, then CPU. Same code on every PC. |
| On-device LLM | `foundry_local_provider.py`: Microsoft Foundry Local (NPU build on Snapdragon X), with an Ollama fallback. Runtimes are health-probed with a 30 s cache. |
| Hybrid router | `app/router/hybrid.py`: privacy lock → quality-bar check → cloud, with fail-closed blocking and two-way fallbacks. It reuses the project's existing classifier and quality bars rather than redefining them. |
| Audit log | SQLite schema migrated in place: placement, reason, privacy categories (never the values), fallback, and redaction of sensitive prompts **and** responses. |
| API | `POST /v1/completions` (hybrid by default), `POST /v1/privacy/scan`, `GET /v1/on-device`, `GET /v1/stats`. |
| Dashboard | Live "Try it" demo (highlighted findings → decision + reason → answer → cost vs cloud), on-device %, privacy locks, Snapdragon status tab. |

**Measured results** (Intel i5-12450H development laptop, no NPU):

- **Privacy guard:** 28/28 hand-labelled prompts correct (17 that must lock, 11 that
  mustn't), with zero false locks and zero misses. Median scan time is **9 ms** on CPU.
  This is a regression check written alongside the rules, not an independent benchmark.
- **Live end-to-end run** (`scripts/hybrid_demo.py`, 12 mixed prompts): all 12 answered.
  **75% were answered on-device**, **4 of 4 prompts with personal data** were kept on-device,
  and total cost was **48.5% lower** than sending everything to the Tier-3 cloud model.
- **Snapdragon X Elite NPU:** *[to be measured on Snapdragon hardware / Qualcomm AI Hub]*.

## 2. Application use case and innovation

- **Privacy is a routing decision, not a setting.** The same app gets cloud-quality answers
  when the prompt is safe to share and stays fully on-device when it isn't, automatically,
  per prompt.
- **Fail closed.** A sensitive prompt with no local model available is blocked, never
  quietly sent to the cloud.
- **Built for India.** The guard checks Aadhaar, PAN, GSTIN, IFSC and UPI IDs with real
  checksum validation, so ordinary numbers don't trigger false alarms.
- **Uses the NPU the way Qualcomm intends:** always-on, low-power inference (a privacy scan on
  every prompt) plus an on-device language model, with the cloud kept for what only the
  cloud can do.
- **Explainable.** Every decision comes with a human-readable reason, which is essential for
  compliance teams.

## 3. Deployment and accessibility

- A **drop-in HTTP API**, so any existing app can route through Autopilot Edge. It also has a
  Streamlit dashboard and a Docker setup.
- **One codebase for x86 and Snapdragon.** On Snapdragon, installing `onnxruntime-qnn` and
  Foundry Local moves both models to the NPU with no code changes.
- It **works offline**: simple and sensitive prompts need no network, and a cloud outage
  falls back to on-device answers.
- Setup: `pip install -r requirements.txt` → `python scripts/download_models.py` →
  `streamlit run dashboard/app.py`.

## 4. Roadmap (Build phase)

| Phase | Deliverable |
|---|---|
| 1 (done) | Hybrid router, on-device privacy guard, Foundry Local provider, redacted audit log, API, dashboard |
| 2 | Run and benchmark on a Snapdragon X PC: NPU latency, power draw and share of prompts handled on-device |
| 3 | Compile the privacy NER through Qualcomm AI Hub (QNN, INT8/FP16) and profile it on Snapdragon X Elite |
| 4 | Replace the heuristic complexity classifier and TF-IDF routing memory with an on-device embedding model on the NPU |
| 5 | Multilingual PII (Hindi and other Indic names and addresses) and an organisation-level privacy policy editor |
| 6 | A local OpenAI-compatible proxy and tray app, so any desktop AI app gets hybrid routing with zero code changes |

## What changed from the original project

Before this challenge, *LLM Cost Auto Pilot* sent every prompt to a model chosen only by
cost and complexity, and it contained **no on-device AI models**. The new work is:
- the **on-device BERT privacy model** and detectors
- the **NPU-first ONNX Runtime layer**
- the **Foundry Local (Snapdragon NPU) language-model provider**
- the **hybrid routing engine** with fail-closed privacy locking and offline fallback
- redacted audit logging, the new API endpoints and the live demo dashboard

---

*Built on a development laptop without an NPU. Every Snapdragon-specific path uses the
official runtimes (ONNX Runtime QNN EP, Microsoft Foundry Local) but has not yet been run on
Snapdragon hardware; phase 2 of the roadmap covers that. All identity numbers in this
document are synthetic.*
