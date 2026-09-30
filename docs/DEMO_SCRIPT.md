# Autopilot Edge: demo video script (about 3 minutes)

## Before you hit record (10 min)

1. Make sure Ollama is running: the llama icon is in the system tray.
2. Start the dashboard from the project folder:
   ```
   .venv\Scripts\streamlit run dashboard/app.py
   ```
3. Open http://localhost:8501 in your browser and zoom to **110–125%** so the text is readable
   on video.
4. **Warm up:** route the "Simple question" example once *before* recording. The first
   on-device answer is slow while the model loads.
5. Close notifications and other windows. Start recording with **Win + Alt + R**; press it
   again to stop. The video is saved in `Videos\Captures`.
6. On-device answers take 5–15 s on a laptop without an NPU. Keep talking while they load
   (the lines are below), or cut the wait out afterwards.

---

## Script

### 0:00–0:20 · The problem (on screen: dashboard title)

> "Every AI app today sends every prompt to the cloud. That means paying per token even for
> trivial tasks, and it means your Aadhaar number, bank details or customer names end up on
> someone else's server. Snapdragon PCs have an NPU built for AI, but apps don't decide
> when to use it."

### 0:20–0:40 · What Autopilot Edge is

> "Autopilot Edge is a hybrid AI router. For every prompt it runs a privacy check on the
> laptop, then decides: answer on-device for free, or pay for a cloud model only when the
> task really needs it. Anything with personal data never leaves the device."

### 0:40–1:05 · Demo 1: simple prompt stays on-device

**Do:** Try it live → example **"Simple question → on-device"** → **Route it**.

> "A simple question. The privacy scan is clean, and the complexity classifier says Tier 1,
> so the small on-device model is good enough. Cost: zero. On a Snapdragon PC this runs on
> the NPU through Microsoft Foundry Local."

**Point at:** the 💻 ON-DEVICE badge, the "Why" line, and "Cost $0.000000".

### 1:05–1:50 · Demo 2: personal data is locked to the device (the key moment)

**Do:** example **"Aadhaar + PAN → locked on-device"** → **Route it**.

> "Now a KYC form with an Aadhaar number, a PAN and a name. The privacy guard catches all
> three in milliseconds. The Aadhaar is validated with its real checksum, and the name is
> found by a BERT model running on the device. So even though it's just a form-filling
> task, the router locks it to the laptop. If no local model were available, it would
> block the request rather than send it to the cloud."

**Point at:** the red and orange highlights, "SENSITIVE", and "privacy lock: contains
AADHAAR, PAN, PERSON".

> "It's even stored redacted in the logs, so personal data doesn't end up in the database
> either."

**Point at:** "Logged as request #… (stored redacted)".

### 1:50–2:20 · Demo 3: a hard prompt goes to the cloud

**Do:** example **"Hard reasoning → cloud (high)"** → **Route it**.

> "A complex architecture question with no personal data. This one really needs a big
> model, so the router sends it to the cheapest cloud model that meets the high quality
> bar, gpt-oss-120b on Groq, and explains why."

**Point at:** the ☁️ CLOUD badge, the "Why" line, and the cost next to the baseline.

### 2:20–2:40 · The results

**Do:** open the **📊 Savings & privacy** tab.

> "Across a realistic mix of prompts, 75% were answered on-device, every prompt with
> personal data stayed local, and total cost was about half of sending everything to the
> top cloud model."

**Point at:** Answered on-device, Privacy-locked prompts, and the headline percentage.

### 2:40–3:00 · Snapdragon and close

**Do:** open the **⚡ Snapdragon on-device** tab and expand "Moving everything onto the
Snapdragon NPU".

> "Everything on the device is built for the Snapdragon NPU. The privacy model uses ONNX
> Runtime's QNN execution provider, and the language model uses Foundry Local's NPU build.
> On a Snapdragon PC it's the same code, just a different runtime package. Autopilot Edge
> gives you private by default, and cloud only when it's worth it. Thank you."

---

## If something goes wrong while recording

| Problem | Fix |
|---|---|
| On-device answer takes very long | Keep talking and cut the pause later, or re-record that section |
| Cloud prompt shows an error | The router answers on-device automatically ("Fallback: cloud failed…"); present it as the offline-resilience feature |
| Dashboard shows "No logged requests" | Run `python scripts/hybrid_demo.py` once (about 2 min) |
| Local model refuses a prompt | Use the built-in examples; they're worded so llama3.2 answers them |
