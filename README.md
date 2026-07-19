# LLM Cost Auto Pilot

An intelligent routing layer that sits in front of multiple LLM providers,
predicts the cost of a request **before** sending it, routes to the
cheapest model capable of handling it at acceptable quality, explains every
routing decision, and continuously verifies that its routing was correct —
optimizing across cost, latency, quality, and data-locality, not cost alone.

This is a portfolio/learning project built in phases. Status below reflects
what's actually implemented right now, not the end-state design.

## Status: Phase 1 — Unified Model Interface ✅

What exists today:
- A model registry (`config/model_registry.yaml`) describing every model the
  router can pick from: provider, model ID, cost per input/output token,
  average latency, and a quality tier (high/medium/low).
- A single `send_request(prompt, model_config)` function
  (`app/models/dispatcher.py`) that hides provider-specific SDK calls behind
  one interface and always returns a standardized `LLMResponse`: output
  text, input/output tokens, latency, cost, model ID, and an `error` field
  if the call failed.
- Provider adapters for Gemini, Anthropic, Ollama (local), and a shared
  `OpenAICompatibleProvider` used for OpenAI, Groq, and Together AI (all
  three speak the same Chat Completions request/response shape, so one
  parameterized class covers them instead of three near-duplicate files).
- A test harness (`scripts/test_providers.py`) that sends the same 10
  prompts to every registered model and reports success rate, cost, and
  latency — skipping providers you don't have keys for instead of crashing.

**Live/tested:** Gemini (you have a key for this).
**Live, free tier, needs your own key:** Groq, Together AI — both offer a
free/no-payment-required tier, unlike OpenAI and Anthropic, so these are
the easiest way to get a second and third live data point without paying.
Sign up at console.groq.com and api.together.ai.
**Coded but dormant (needs a paid key):** OpenAI, Anthropic — adapters are
written against the real SDKs and follow the exact same interface as the
working providers, so they should work as soon as a key is added to `.env`.
**Local:** Ollama — adapter is written, but Ollama itself isn't installed
yet on this machine. See setup below.

> ⚠️ Never paste a real API key into any `.py` or `.yaml` file. Keys only
> ever go in `.env` (gitignored). If you accidentally paste one into
> tracked source, treat it as compromised and regenerate it.

## Setup

1. Create and activate a virtual environment (already created as `.venv`
   in this repo if you're continuing from the initial setup):
   ```
   python -m venv .venv
   .venv\Scripts\activate
   ```
   **Always run this project through `.venv`, not a different/global
   Python install** — a different environment won't have the exact
   packages `requirements.txt` pins, and may have unrelated environment
   variables (e.g. a stray `OPENAI_API_KEY` or `GROQ_API_KEY` from another
   project) that produce confusing 401 errors instead of clean "skipped"
   messages. VS Code is already configured (`.vscode/settings.json`) to
   default to `.venv` for this workspace — reload the window if it doesn't
   pick it up automatically.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and fill in the keys you have:
   ```
   copy .env.example .env
   ```
   Then open `.env` (not any `.py` file) and paste your key values in.
   At minimum, set `GEMINI_API_KEY`. Leave the others blank until you have
   them — those providers are simply skipped by the test harness.
4. (Optional, for the local tier) Install [Ollama](https://ollama.com/download),
   run `ollama serve`, then pull a small model, e.g. `ollama pull llama3.2`.

## Try it

```
python scripts/test_providers.py
```

This sends 10 fixed test prompts (spanning simple extraction to multi-step
reasoning) to every available model, prints a cost/latency/success summary
table, and saves the raw results to `data/phase1_test_results.json`.

## Repo structure

```
config/
  model_registry.yaml   pricing, latency, and quality tier per model
app/
  models/
    registry.py          ModelConfig dataclass + YAML loader + cost math
    response.py           standardized LLMResponse
    dispatcher.py          send_request() — the one function everything else calls
    providers/
      base.py               BaseProvider interface + ProviderError
      gemini_provider.py
      anthropic_provider.py
      ollama_provider.py
      openai_compatible.py  shared adapter for OpenAI, Groq, Together AI
scripts/
  test_providers.py      Phase 1 validation harness
data/
  phase1_test_results.json  (generated, gitignored is NOT set for this one file —
                              small enough to keep as a portfolio artifact)
```

## Known limitations / things to verify

- Pricing in `config/model_registry.yaml` is my best approximation as of
  early 2026 and **will drift** — cross-check against provider pricing
  pages before trusting any dollar figure the dashboard shows later. Groq
  and Together entries use their paid per-token rates, even though your
  own usage may fall under a free-tier allowance while testing.
- OpenAI and Anthropic adapters are untested against real traffic (no keys
  yet). They follow the identical pattern to the working Gemini adapter,
  so bugs there are more likely SDK-shape mismatches than logic errors.
- Ollama adapter is untested — Ollama isn't installed on this machine yet.
- If a provider that should be skipped instead fails with a `401` error,
  you likely have a stray API key for a *different* service set as a
  global environment variable on your machine under the same name (this
  happened during development with a leftover `OPENAI_API_KEY`). It's
  harmless — the call fails before anything is billed — but you can remove
  the stray variable from Windows environment variables if the noise
  bothers you.

## Roadmap

Build order:
1. Unified provider interface — **done**
2. Complexity classifier (heuristic v1 → scikit-learn v2) + routing map
3. Pre-call cost prediction
4. Explainable routing
5. Multi-objective optimization (cost/latency/quality/data-locality)
6. Async quality verification + auto-escalation
7. SQLite logging + Streamlit cost dashboard
8. **RAG** — added after deciding the core loop (2–7) needed to be solid first:
   - **Context compression**: chunk/embed/retrieve only the relevant pieces
     of a long document before it's sent to any model, instead of sending
     the whole thing. Directly cuts token cost on long-context requests —
     ties straight into the dashboard's savings metric.
   - **Routing memory** (built after compression): embed each incoming
     prompt and retrieve the k most similar past requests plus their
     *verified* outcomes (which model actually succeeded/failed per the
     Phase 6 verifier) as a second signal alongside the classifier. Needs
     real logged history to be useful, which is why it comes after
     compression, not before.
9. Simulation mode + live benchmarking
10. FastAPI service + Docker
11. Load test + portfolio write-up

Explicitly deferred past v1 (documented here so they're not forgotten,
not because they're bad ideas):
- Prompt optimizer (auto-rewrite prompts to cut tokens)
- User preference dials (speed>quality, budget-per-day)
- Agent-specific routing policies (coding/research/creative/OCR/math)
- Self-learning/RL router (online learning from routing feedback)
- Budget scheduler (automatic degradation as a monthly budget is consumed)
- Weekly classifier auto-retraining flywheel
