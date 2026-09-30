# LLM Cost Auto Pilot

An intelligent routing layer that sits in front of multiple LLM providers,
predicts the cost of a request **before** sending it, routes to the
cheapest model capable of handling it at acceptable quality, explains every
routing decision, and continuously verifies that its routing was correct —
optimizing across cost, latency, quality, and data-locality, not cost alone.

This is a portfolio/learning project built in phases. Status below reflects
what's actually implemented right now, not the end-state design.

## Autopilot Edge — the Snapdragon edition (Phase 12)

The router now runs **hybrid**: an on-device privacy guard scans every
prompt first, prompts containing personal data (Aadhaar, PAN, bank
details, phone numbers, names, API keys…) **never leave the machine**,
simple prompts are answered free by an on-device model, and only hard,
non-sensitive prompts go to the cheapest capable cloud model. On a
Snapdragon X PC both on-device models run on the Hexagon NPU. Details in
Phase 12 below; the challenge write-up is in `docs/PROPOSAL.md`.

```
streamlit run dashboard/app.py          # live demo: "Try it live" tab
python scripts/test_privacy_guard.py    # offline privacy-guard check
python scripts/hybrid_demo.py           # route + log a realistic prompt mix
```

## Status: All 12 Phases Complete ✅

### Phase 1 — Unified Model Interface
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
- **Hardening pass:** every provider's response-parsing code now lives
  inside the same try/except as its network call — previously a malformed
  or safety-filtered response (empty `choices`, missing `usage`, a non-text
  content block) could raise *outside* the error handling and crash the
  whole batch run instead of being logged as one failed request. The
  dispatcher also got a last-resort catch-all so an unanticipated bug in a
  provider still becomes data on the response, never a crash. All
  providers now cap output at the same 1024 tokens so the "same prompt to
  every model" benchmark is comparing like with like.

### Phase 2 — Complexity Classifier + Routing Map
- `app/classifier/features.py` — extracts signals from a prompt (keyword
  matches in three weighted categories, constraint count, word count,
  presence of a long quoted/context block). Word-boundary regex matching,
  not substring — an earlier version incorrectly matched "list" inside
  "checklist" and inside prompts that merely *mention* a list.
- `app/classifier/heuristic.py` — the v1 classifier: scores a prompt from
  the features above and buckets it into `ComplexityTier.TIER_1/2/3`. No
  training data required; this is what gets routing working today.
- `app/classifier/tiers.py` — the `ComplexityTier` enum shared by the
  classifier and the routing config.
- `config/routing_config.yaml` — maps each tier to a specific
  provider/model. Currently: Tier 1 → Groq `llama-3.1-8b-instant`, Tier
  2/3 → Gemini `gemini-2.5-flash` / `gemini-2.5-pro` (picked because those
  are the providers actually live-tested or free-tier so far — repoint
  freely once you have more keys).
- `app/router/routing_engine.py` — `route_request(prompt)`, the one
  function that ties classifier → routing map → Phase 1's `send_request()`
  together and returns a `RoutingResult` (response, tier, score, the
  specific signals that fired).
- `scripts/test_classifier.py` — offline sanity check (no API keys, no
  network calls) that runs the classifier against 10 example prompts
  spanning all three tiers and flags any tier that doesn't match a hand
  -set expectation. Run with `-v` to see which signals drove each score.
- `data/labeled_prompts.csv` — empty (`prompt,tier,notes` header only) and
  waiting for your 200+ hand-labeled examples. Not required for Phase 2 to
  work — the heuristic classifier needs no training data — but required
  before the Phase 2b scikit-learn upgrade can happen.
- **Hardening pass:** `load_routing_map()` now fails fast with a clear
  `ValueError` if `routing_config.yaml` is missing its `routing` key or
  missing an entry for any tier, instead of crashing later with a bare,
  confusing `KeyError` from deep inside `route_request()`.

### Phase 3 — Pre-call Cost Prediction
- `app/router/cost_predictor.py` — `predict_cost(prompt)` estimates input
  tokens (~4 chars/token rule of thumb) and output tokens (a fixed
  per-tier assumption keyed off the Phase 2 classifier — 80/250/500 tokens
  for Tier 1/2/3), then computes what the prompt would cost on *every*
  model in the registry, sorted cheapest first. It never calls a provider —
  pure math against `config/model_registry.yaml`.
- **"Recommended"** = cheapest model whose static `quality_tier` meets a
  minimum bar for the prompt's complexity tier (Tier 3 won't recommend a
  "low" quality model even if it's cheapest). This is a light preview of
  Phase 5's full multi-objective scoring, not the real thing yet.
- **Deliberately not doing what the original mockup showed:** no fake
  "Expected Quality: 97%" — there's no real quality measurement in the
  system until Phase 6's verification loop exists, so showing a precise,
  unmeasured percentage would be presenting a guess as fact. The
  `quality_tier` (high/medium/low) shown instead is honest about what we
  actually know right now.
- `scripts/predict_cost.py` — CLI demo: `python scripts/predict_cost.py
  "your prompt"` prints the full cost/quality table across all models,
  flagging both the recommended pick and whatever `routing_config.yaml`
  currently routes that tier to (they can disagree — see below).
- **Real finding from testing this:** for Tier 1 prompts, the recommended
  model is now Ollama (`$0`, meets the "low" bar) — cheaper than Groq,
  which is what `routing_config.yaml` currently routes Tier 1 to. For
  Tier 2, `gpt-4o-mini` (dormant, no key yet) undercuts the current
  Gemini Flash route by ~4x. Not fixed automatically — Phase 3 only
  *predicts and surfaces* this, it doesn't rewrite the routing config;
  that's a config change you can make once you decide you want it.
- **Hardening pass:** `scripts/predict_cost.py` called `max()` on the
  estimates list to size a table column with no empty-list guard — would
  crash with `ValueError: max() arg is an empty sequence` if the registry
  were ever empty. The line right above it already guarded this exact
  case for a different purpose; the guard just didn't extend far enough.
  Fixed with an early return.

### Phase 4 — Explainable Routing
- `app/router/explainer.py` — `explain(routing_result)` turns a completed
  routing decision into a `RoutingExplanation`: the tier and *why*
  (reuses Phase 2's `matched_signals` directly, no new logic), a
  confidence score, the chosen model's cost/latency, and the cheapest
  qualifying alternative with a cost/quality delta against it. Built
  entirely by combining Phase 2 and Phase 3's existing outputs — no new
  estimation logic.
- **Confidence, honestly:** derived from how far the classifier's raw
  score sits from the nearest tier boundary — a score right on a boundary
  (e.g. Tier 3 at score=3, the minimum) gets 50% ("could easily have
  landed in the neighboring tier"); a score deep inside a tier's range
  approaches 100%. This is **not a calibrated probability** — it's a
  transparent function of the same heuristic score, documented as such.
  Same honesty rule as Phase 3's refusal to show a fabricated quality
  percentage.
- `RoutingResult` (Phase 2) gained a `prompt` field so it's self-contained
  — the explainer needs the original prompt to re-run the cost predictor,
  and Phase 7's audit log will want it stored alongside the response too.
- `scripts/explain_routing.py` — CLI demo: routes a real prompt (this one
  *does* make a live provider call, unlike the Phase 2/3 demo scripts)
  and prints the full explanation — tier, confidence, matched signals,
  chosen model, cheapest alternative with cost/quality delta, and the
  actual response or error.
- Verified against all three tiers and both cost-delta directions (chosen
  model cheaper than the alternative, and chosen model more expensive) —
  confirmed via a direct Ollama call, since `routing_config.yaml`
  currently doesn't route anything to it (see Phase 3's finding above).
- **Hardening pass (found during the Phase 5/6 audit):** `explain()` was
  computing whether the chosen model meets its own tier's quality bar
  (`chosen_estimate.meets_quality_bar`) and then silently discarding that
  signal — for a phase whose entire purpose is surfacing routing risk,
  quietly dropping "this choice might not even clear the bar for its own
  tier" undermined the point. Added a `chosen_meets_quality_bar` field and
  a WARNING line in `explain_routing.py` when it's `False`. Verified by
  simulating a deliberately misconfigured routing (a Tier 3 prompt forced
  onto a "low" quality model) and confirming the flag correctly flips.

### Phase 5 — Multi-objective Optimization
- `app/router/multi_objective.py` — `select_model(prompt, objectives)`
  scores every candidate model on **cost + latency + quality**, each
  min-max normalized to 0–1 and combined via configurable weights
  (`RoutingObjectives`). Five named profiles ship as presets: `BALANCED`,
  `COST_FOCUSED`, `LATENCY_FOCUSED`, `QUALITY_FOCUSED`, and
  `ON_PREM_ONLY`.
- **Data-locality is a hard constraint, not a weight** — matches the
  original design's own framing ("Hospital → must stay on-prem → choose
  Ollama"). A request that must stay local doesn't get a spectrum; models
  that aren't `local: true` are filtered out *before* scoring, the same
  way a hard `max_latency_ms` ceiling filters out anything too slow.
  Weights only ever apply to whatever survives the hard filters.
- `scripts/select_model_demo.py` — CLI demo: runs the same prompt through
  all five profiles side by side. Never calls a provider — pure scoring
  against the registry, like Phase 3.
- **Real finding from testing this:** for the current registry, Groq's
  `llama-3.3-70b-versatile` is close to Pareto-dominant — cheapest,
  fastest, *and* highest quality tier simultaneously among eligible
  candidates for most prompts — so soft weight changes (balanced vs.
  cost-focused vs. quality-focused) rarely flip the pick. Only a **hard**
  constraint does: `ON_PREM_ONLY` forces Ollama, and a `max_latency_ms:
  300` cap forces Groq's smaller 8B model instead of the 70B. This is an
  honest reflection of the registry's current numbers, not a demo
  artifact — a genuinely close-to-dominant option existing is itself a
  real, useful thing to have found.
- **Hardening pass (caught before it could bite):** the demo script sized
  its "chosen model" and "profile" table columns with fixed-width guesses
  — the same bug class that hit `predict_cost.py` in Phase 3 (Together
  AI's model keys and the longer profile names both exceed those
  guesses). Fixed proactively with the same dynamic-width approach used
  there, before it ever produced misaligned output.

### Phase 6 — Async Quality Verification + Auto-escalation
- `app/router/verifier.py` — `verify_and_escalate(routing_result)` sends
  the same prompt to the strongest configured model (whatever Tier 3
  routes to) as a reference, scores text-similarity agreement against the
  originally-routed model's answer, and swaps in the reference answer
  when agreement falls below a threshold (default 0.5). Returns a
  `VerificationResult` with one of four statuses: `verified`, `escalated`,
  `already_top_tier` (nothing stronger exists to check against), or
  `comparison_failed` (the original and/or reference call itself failed —
  handled distinctly from genuine disagreement so the two are never
  conflated).
- **"Async," honestly:** the original design wants this as a
  fire-and-forget background job that never blocks the user-facing
  response. There's no job queue or background worker yet — that's
  genuinely Phase 10's job (FastAPI + Docker + a background worker
  process) — so `verify_and_escalate()` is **synchronous** for now: it
  makes the reference call and returns inline. When Phase 10 adds a real
  worker, this is the function that gets queued; the verification logic
  itself won't need to change.
- **Scoring, honestly:** uses `difflib` text similarity, not an
  LLM-as-judge, even though the original design names both ("Custom
  scoring + LLM-as-judge"). Text similarity is a weak proxy for semantic
  agreement — two answers can be worded completely differently but mean
  the same thing (scores low when it shouldn't), or be superficially
  similar but factually wrong (scores high when it shouldn't). An
  LLM-as-judge call would be a meaningfully better signal; it's a
  documented, open follow-up, not something built here — adding it now
  would mean shipping two half-verified pieces instead of one solid one.
- `scripts/verify_routing.py` — CLI demo: routes a prompt for real, then
  runs verification against it, printing the agreement score and whether
  escalation happened. Makes up to two live provider calls.
- Verified all four status paths: `already_top_tier` and
  `comparison_failed` confirmed via live runs (routing config currently
  has no working keys, which conveniently exercises the failure path for
  free); `verified` and `escalated` confirmed via deterministic tests
  with mocked responses (identical text → agreement 1.0, no escalation;
  divergent text → agreement 0.21, correctly escalates and swaps the
  final response to the reference model).
- **Hardening pass (fresh audit of Phases 5-6):** the quality-tier
  ordering (`low` < `medium` < `high`) was independently redefined in
  **three separate files** (`cost_predictor.py`, `explainer.py`,
  `multi_objective.py`) — worse, `QualityTier`'s actual declaration order
  in `registry.py` is `HIGH, MEDIUM, LOW`, the *opposite* of what all
  three assumed. Not a live bug yet, but exactly the kind of thing that
  silently drifts if a fourth tier is ever added and only one copy gets
  updated. Consolidated into a single `QUALITY_RANK` dict in
  `registry.py`; all three modules now import it. Verified behavior is
  byte-for-byte identical before/after (`select_model_demo.py`'s scores
  matched exactly).

### Phase 7 — SQLite Logging + Streamlit Dashboard
- `app/logging/db.py` — `log_request(routing_result, verification_result)`
  writes one row per request to `data/requests.db`: prompt (hash *and*
  full text — see deviation note below), response text, tier, routed
  model, tokens, cost, latency, and — if verification ran — the
  agreement score and escalation outcome. Also computes a
  **baseline_cost_usd**: what the request would have cost on whatever
  Tier 3 currently routes to, using the *actual* token counts from the
  real response, not an estimate. This is what the dashboard's headline
  savings number is built from.
- **Deliberate deviation from the original spec:** it lists "prompt
  hash" as the logged field (sensible for privacy at production scale).
  This is a personal project with no real user data at stake, and the
  dashboard's drill-down needs to show actual past requests to be useful
  — so both `prompt_hash` (kept, matches the spec) and `prompt_text` /
  `response_text` (added, for practical debugging/demo value) are
  stored.
- `scripts/run_and_log.py` — routes a prompt, verifies it, and logs the
  full result in one call. This is what actually populates the
  database; routing/verification alone (Phases 2-6) never persist
  anything on their own.
- `scripts/seed_demo_data.py` — dev utility that routes 15 varied
  prompts directly through Ollama (the registry's static routing map
  doesn't point anywhere live right now) and logs them, so the
  dashboard has real data to show. **Deliberately skips verification**:
  Tier 3's routing target is also Ollama's `llama3.2`, so "verifying"
  would compare the routed model against itself and always
  short-circuit to `already_top_tier` — not a real comparison. Faking a
  second model just to manufacture demo variety would be less honest
  than simply not running it; the verification mechanism itself is
  already proven correct by Phase 6's deterministic tests.
- `dashboard/app.py` — `streamlit run dashboard/app.py`. Shows the
  headline cost-reduction percentage, routing/tier distribution charts,
  escalation stats, a cost-over-time chart, and a drill-down into any
  individual logged request (prompt, actual response, full metadata).
  Read-only — computes nothing new, just visualizes what the logger
  already wrote.
- **Verified in an actual browser**, not just "it imports without
  error": ran it against the 15 seeded Ollama requests and confirmed
  real numbers rendered correctly — 100% cost reduction (every request
  routed to $0 Ollama vs. a computed Tier-3 baseline of $0.0292),
  correct tier/routing distribution, and the drill-down showing a real
  generated response with accurate token counts and latency. Also
  caught and fixed a real gap while building it: the schema never
  actually stored the model's response text (only the prompt) — a
  drill-down dashboard is far less useful if it can't show what the
  model said. Fixed before any real data was persisted against the
  broken schema.
- **Hardening pass:** `use_container_width=True` is deprecated in the
  installed Streamlit version (1.60.0) and past its stated removal
  date — replaced with `width="stretch"` before it could start failing.

### Phase 8 — RAG (Context Compression + Routing Memory)
- `app/rag/compression.py` — `compress_context(query, document)` chunks
  a long document (~200 words/chunk), scores each chunk's relevance to
  the query via TF-IDF + cosine similarity, and keeps only the top-k
  most relevant chunks, preserving original reading order. Verified with
  a synthetic multi-topic "API documentation" text (auth, rate limits,
  pricing, webhooks, error codes, changelog) — a query about 429 rate
  limit errors correctly retrieved *only* the Rate Limits and Error
  Codes sections (60.4% word-count reduction), correctly excluding
  Authentication, Webhooks, and Pricing. A checkable, honest result, not
  a cherry-picked one.
- `app/rag/routing_memory.py` — `get_routing_memory_signal(prompt)`
  retrieves the k most similar *past* logged requests (from Phase 7's
  database) via the same TF-IDF approach, plus their verified outcomes
  (tier, escalation rate) as an advisory second signal. Like Phase 3's
  cost prediction and Phase 5's multi-objective scoring, this is
  advisory only — it does not automatically feed into `route_request()`.
  Verified against the 15 seeded requests: a query about a payment
  ticket correctly ranked the one genuinely similar past prompt
  ("classify this support ticket... charged twice") at similarity 0.33,
  with everything else honestly scoring 0.00 (no shared vocabulary) —
  exactly the expected behavior for lexical similarity.
- **"Semantic," honestly:** both use TF-IDF (term frequency) + cosine
  similarity, not real embeddings. That's **lexical** similarity — it
  catches shared vocabulary, not shared meaning ("car" and "automobile"
  won't match each other even though they mean the same thing). Chosen
  over real embeddings deliberately: genuine semantic embeddings would
  mean either a heavy new dependency (`sentence-transformers` + `torch`,
  multi-gigabyte), another blocking Ollama model pull mid-project, or a
  live Gemini API dependency we can't yet confirm works. TF-IDF needed
  none of that and is a legitimate, if weaker, "custom scoring"
  technique — documented as a known limitation and upgrade path, not
  quietly oversold as more capable than it is.
- `scripts/compress_context_demo.py` and `scripts/routing_memory_demo.py`
  — CLI demos, both offline except routing memory's dependency on
  Phase 7's logged history (run `seed_demo_data.py` first if empty).

### Phase 9 — Simulation Mode + Live Benchmarking
- `app/simulation/simulator.py` — `simulate(prompts)` replays a batch of
  prompts against every "always use provider X's best model" baseline,
  the static Phase 2 router, and Phase 5's multi-objective balanced
  scorer, comparing total projected cost, latency, and quality mix.
  Uses registry math throughout (Phase 3's token estimates, Phase 5's
  scoring), not live calls — running N prompts against every strategy
  live would need a working key for *every* provider and cost real
  money per simulation run, defeating the point of a pre-flight
  simulation.
- **Real result from running this on 15 prompts:** the static router
  saves **70.8%** vs. always using the most expensive strategy
  (Claude Sonnet) — the actual headline number this whole project
  exists to produce, and it fell out of real math, not a tuned demo.
  It also showed Phase 5's multi-objective router beating Phase 2's
  static router by a wide margin ($0.0029 vs. $0.0152) on the same
  prompts — a genuine, unprompted validation that Phase 5's smarter
  scoring outperforms Phase 2's fixed tier map.
- `app/simulation/live_benchmark.py` — `benchmark_available_models()`
  tests every model that actually has a working key against a couple of
  real prompts and reports measured latency vs. what
  `model_registry.yaml` assumes. **On-demand, not scheduled** — there's
  no cron/scheduler infrastructure in this project, so "live
  benchmarking" here means "run this command for fresh numbers
  whenever you want them," not an automatic periodic refresh. Doesn't
  rewrite the registry automatically either — same "predicts and
  surfaces, doesn't auto-edit config" pattern as Phase 3/5.
- **Real finding from running this:** Ollama's measured latency was
  3308ms against the registry's static 800ms assumption — a genuine
  +2508ms drift, consistent with the partial GPU/CPU split found
  earlier in this project. A concrete example of exactly the kind of
  thing this phase exists to catch.
- **Hardening pass (shared-code consolidation):** `test_providers.py`
  had its own local `is_available()` helper checking whether a model's
  key is present; `live_benchmark.py` needed the identical check. Rather
  than duplicate it a second time, moved it into `registry.py` as
  `is_model_available()` — the same fix pattern as `QUALITY_RANK` above.
  `test_providers.py` re-verified working against the shared version
  before and after (Ollama 10/10 ok, everything else skipping/failing
  identically).
- `scripts/simulate_demo.py` (offline) and `scripts/live_benchmark_demo.py`
  (makes real calls to whatever has a working key) — CLI demos.
- **Hardening pass (fresh audit of Phases 7-9):** three small but real
  fixes. `db.py`'s baseline-cost lookup only caught `ValueError`/
  `FileNotFoundError`, missing `yaml.YAMLError` from a malformed
  routing config — broadened to a catch-all, matching `dispatcher.py`'s
  belt-and-suspenders pattern (low practical risk, since `route_request()`
  already reads the same file successfully earlier in the same call, but
  logging a request should never itself be able to crash). `compression.py`
  defined a `CHARS_PER_TOKEN_ESTIMATE` constant that was never actually
  used — the token estimate properties used a different, un-labeled ratio
  instead; removed the dead constant and clarified the comment.
  `simulator.py` found its own router result by string-matching a
  strategy name (`s.name.startswith("router (static tier map...")`) — a
  silent `StopIteration` crash waiting to happen if that name is ever
  edited; replaced with a direct object reference. All three verified
  with no behavior change (`simulate_demo.py`'s 70.8% figure identical
  before/after; `run_and_log.py` still logs correctly).

### Phase 10 — FastAPI Service + Docker
- `app/api/main.py` — a real HTTP API, run with `uvicorn app.api.main:app`
  from the project root:
  - `POST /v1/completions` — routes a prompt (Phase 2), optionally runs
    Phase 6 verification (`"verify": true`), optionally logs to Phase 7's
    database (`"log": true`, the default), and returns the response plus
    routing metadata. You don't choose the model; the router does.
  - `GET /v1/models` — every registered model's pricing/latency/quality
    tier plus whether it's actually callable right now (`is_model_available()`).
  - `GET /v1/stats` — Phase 7's `get_summary()` as JSON: cost, baseline,
    savings %, escalation rate.
  - `PUT /v1/routing-config` — repoints a tier to a different provider/model
    **without redeploying**, validating the tier name and that the model
    actually exists in the registry first (400 with a clear message
    otherwise).
- **All four endpoints verified live**, not just "it imports without
  error" — started the server, hit every endpoint with real HTTP
  requests (`GET`s via browser navigation, `POST`/`PUT`s via `fetch`),
  and confirmed: `/v1/models` returns all 12 registry entries with
  correct availability flags; `/v1/completions` with `verify: true`
  produced a real Ollama response, attempted verification, and logged
  row #19; both `PUT` validation error paths return clean `400`s with
  the exact error messages `get_model()` and the `ComplexityTier` enum
  already produce.
- **A real, kept change, not just a test:** used `PUT /v1/routing-config`
  to actually point Tier 1 at Ollama — the choice Phase 3, Phase 5, and
  Phase 9 each independently found was better, but that nothing had
  wired in yet. `routing_config.yaml`'s explanatory comments (stripped
  by the endpoint's `yaml.dump`, a documented PyYAML limitation) were
  restored by hand afterward with a note explaining why Tier 1 changed.
  Tiers 2/3 still point at Gemini, untested live pending a working key.
- `Dockerfile` / `docker-compose.yml` (`api` + `dashboard` services,
  sharing one image, both mounting `./data` and `./config`) —
  **written but not build-tested**: Docker isn't installed in this
  environment. Written carefully against standard, well-documented
  patterns and flagged honestly rather than claimed as verified, same
  as the untested OpenAI/Anthropic adapters back in Phase 1. No
  separate "worker" service — Phase 6's verification is synchronous
  and runs inline in the API process; a placeholder service that did
  nothing would be decoration, not honesty.

### Phase 11 — Load Test + Portfolio Write-up
- `scripts/load_test.py` — generates a batch of programmatically-varied
  prompts (16 topics × 12 templates spanning all three tiers) and pushes
  them through the full route+log pipeline, printing final aggregate
  cost/savings numbers.
- **Scope, honestly:** the original design calls for 500-1,000 prompts.
  This session ran **40** live — Ollama is the only consistently free
  live provider available, and sequential calls average several seconds
  each on this hardware, so 500-1,000 would take multiple hours. The
  script itself scales to any `--count`; this is a documented scope
  decision, not a shortcut taken quietly.
- **Real run result:** 17/40 succeeded (only Tier 1 requests — Tier 2/3
  route to Gemini, which has no working key here, so they failed
  cleanly exactly as designed). Combined with earlier development runs,
  `data/requests.db` now holds 62 real logged requests, $0.045686 saved,
  100% cost reduction — with the honest caveat that 100% only holds
  because the sole currently-working tier is also the free one. See
  `CASE_STUDY.md` for the fuller, more representative headline number
  (Phase 9's 70.8% simulation, which reflects real pricing across every
  registered provider, not just whichever one has a key today).
- `CASE_STUDY.md` — the portfolio-facing write-up: problem, approach,
  the feedback loop, RAG as a cost lever, and — deliberately included
  rather than omitted — a "what's honestly still missing" section
  naming the two gaps a careful reader would find anyway (quality-parity
  claims need a second working paid key; the better-performing Phase 5
  router isn't wired in as the default yet).

### Phase 12 — Autopilot Edge: Hybrid On-device / Cloud Routing for Snapdragon PCs
- `app/privacy/` — an **on-device privacy guard** that runs before any
  routing decision:
  - `detectors.py` — rules for Indian and general identifiers: Aadhaar
    (Verhoeff checksum), PAN, GSTIN, IFSC, UPI IDs, payment cards (Luhn
    checksum), bank account and passport numbers (only next to a context
    word), phone numbers, emails, API keys and passwords/OTPs. Checksums
    mean a random 12-digit order number doesn't trip the Aadhaar rule.
  - `ner.py` — **dslim/bert-base-NER** (MIT, INT8 ONNX, ~109 MB, fetched by
    `scripts/download_models.py`) finds people, organisations and places
    the rules can't. It scans long prompts in 510-token windows instead of
    truncating, snaps entities to whole words, and ignores ID keywords the
    English model mistakes for names ("Aadhaar", "IFSC").
  - `guard.py` — merges both into a `PrivacyReport`: findings, a redacted
    copy, and `sensitive` (any finding at or above `lock_on_severity` in
    `config/hybrid_config.yaml`; default `medium` = names/contact details
    and up; organisations and places alone don't lock).
- `app/edge/accelerators.py` — ONNX Runtime sessions that pick the
  **Snapdragon Hexagon NPU** (QNN execution provider) first, then DirectML,
  then CPU. Same code on every PC; only the installed `onnxruntime` package
  differs (`onnxruntime-qnn` on Snapdragon).
- `app/models/providers/foundry_local_provider.py` — on-device LLMs via
  **Microsoft Foundry Local**, which serves the NPU (QNN) build of models
  like Phi-3.5-mini on Snapdragon X behind an OpenAI-compatible localhost
  endpoint. Registered as `foundry_local:phi-3.5-mini` (local, $0).
- `app/router/hybrid.py` — `route_hybrid(prompt)`:
  1. privacy guard → sensitive prompts go on-device only; if no on-device
     runtime is reachable the request is **blocked, not sent to the cloud**
     (fail closed);
  2. otherwise on-device if the local model meets the tier's existing
     quality bar (`MIN_QUALITY_FOR_TIER` — reused, not redefined);
  3. otherwise the cloud model `routing_config.yaml` maps the tier to.
  Fallbacks: a failed on-device call on a non-sensitive prompt retries in
  the cloud; a failed cloud call (outage, 429, offline) is answered
  on-device. On-device runtimes are probed (cached 30 s) in the order
  `hybrid_config.yaml` lists them: Foundry Local (NPU) first, Ollama second.
- **Logging (Phase 7 schema, migrated in place):** new columns `placement`,
  `routing_reason`, `privacy_locked`, `privacy_categories` (finding kinds,
  never values), `on_device_runtime`, `fallback`, `redacted`. Privacy-locked
  prompts **and** the responses to them are stored redacted, so personal
  data doesn't end up in `requests.db` either. Pre-hybrid rows keep
  working (placement inferred from provider).
- **API:** `POST /v1/completions` now defaults to `"mode": "hybrid"`
  (`"static"` keeps the Phase 2 behaviour) and returns placement, reason,
  privacy categories and fallback; privacy-locked prompts are never sent to
  a cloud model for Phase 6 verification. New: `POST /v1/privacy/scan`
  (guard only, no model call) and `GET /v1/on-device` (runtimes +
  accelerators).
- **Dashboard:** new "Try it live" tab (privacy scan with highlighted
  findings → routing decision with its reason → answer → cost vs. Tier-3
  baseline), on-device % and privacy-lock metrics, and a Snapdragon tab with
  runtime status and a privacy-guard latency benchmark.
- **Registry/routing changes found while testing live (2026-09-30):**
  Groq retired its llama-3.x models for this key (404 `model_not_found`),
  so Groq entries now point at `openai/gpt-oss-120b` (high) and
  `openai/gpt-oss-20b` (medium). `gemini-flash-latest` returned 503
  repeatedly and `gemini-pro-latest` still has no free-tier quota, so
  Tier 2/3 now route to those Groq models — the only cloud provider
  answering live here.
- **Verified:** `scripts/test_privacy_guard.py` — 28 hand-labelled prompts
  (17 should lock, 11 shouldn't), 28/28 correct, median scan ~9 ms on an
  Intel i5-12450H CPU. Honest caveat: the cases were written alongside the
  rules, so this is a regression check, not an independent benchmark.
  `scripts/hybrid_demo.py` routed 12 live prompts end-to-end (results in
  `docs/PROPOSAL.md`).
- **Not verified here:** nothing has run on a real Snapdragon NPU yet —
  this machine is x86 with no NPU, so both on-device models ran on CPU
  (NER via ONNX Runtime, LLM via Ollama). The Foundry Local provider is
  written against its documented SDK but untested on hardware. Small local
  models also sometimes refuse prompts containing ID numbers even though
  they run locally; the demo prompts are phrased as form-filling / email
  tasks, which llama3.2 handles.

**Live/tested:** Gemini (you have a key for this). Ollama (`llama3.2`,
installed and pulled locally — verified end-to-end through
`send_request()`, no API key needed, $0 cost).
**Live, free tier, needs your own key:** Groq, Together AI — both offer a
free/no-payment-required tier, unlike OpenAI and Anthropic, so these are
the easiest way to get a second and third live cloud data point without
paying. Sign up at console.groq.com and api.together.ai.
**Coded but dormant (needs a paid key):** OpenAI, Anthropic — adapters are
written against the real SDKs and follow the exact same interface as the
working providers, so they should work as soon as a key is added to `.env`.

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
   then pull the model the registry references: `ollama pull llama3.2`.
   Ollama runs as a background Windows service once installed, so you
   don't need to manually run `ollama serve` yourself. Note: the `ollama`
   CLI is registered for PowerShell/cmd, not Git Bash — if you're in Git
   Bash and `ollama` isn't found, that's a PATH difference between shells,
   not a broken install; the project's Python code talks to it over HTTP
   (`localhost:11434`) regardless of which shell you're in.

## Try it

```
python scripts/test_providers.py
```

Sends 10 fixed test prompts (spanning simple extraction to multi-step
reasoning) to every available model, prints a cost/latency/success summary
table, and saves the raw results to `data/phase1_test_results.json`.

```
python scripts/test_classifier.py -v
```

Runs the heuristic classifier against 10 example prompts and prints the
tier it picked, the score, and (with `-v`) exactly which signals fired —
no API keys or network calls needed, this only exercises the classifier
logic.

```
python scripts/predict_cost.py "Summarize this document in 3 bullet points"
```

Prints the estimated cost of that prompt across every registered model,
cheapest first, with the recommended pick and the currently-configured
route both flagged. No API keys or network calls needed.

```
python scripts/explain_routing.py "Why did you pick this model?"
```

Routes the prompt for real (this one does make a live call) and prints
the full explanation: tier, confidence, matched signals, the model
chosen, the cheapest qualifying alternative with a cost/quality delta,
and the actual response or error.

```
python scripts/select_model_demo.py "Your prompt here"
```

Shows how the same prompt gets routed under five different objective
profiles (balanced, cost-focused, latency-focused, quality-focused,
on-prem-only) side by side, plus a strict-latency hard-constraint
example. No API keys or network calls needed.

```
python scripts/verify_routing.py "Your prompt here"
```

Routes the prompt for real, then verifies it against the strongest
configured model and reports whether it escalated. Makes up to two live
provider calls.

```
python scripts/run_and_log.py "Your prompt here"
python scripts/seed_demo_data.py
streamlit run dashboard/app.py
```

The first routes+verifies+logs one real request to `data/requests.db`.
The second populates it with 15 varied requests via Ollama in one go
(no keys needed). The third opens the cost dashboard in your browser —
run it after at least one of the first two.

```
python scripts/compress_context_demo.py "your query"
python scripts/routing_memory_demo.py "your prompt"
```

Context compression (offline, no keys) shows a long synthetic document
getting cut down to just the chunks relevant to your query. Routing
memory (needs logged history — run `seed_demo_data.py` first) shows the
most similar past requests to a new prompt.

```
python scripts/simulate_demo.py
python scripts/live_benchmark_demo.py
```

Simulation mode (offline, no keys) replays 15 prompts across every
routing strategy and reports the cost-savings comparison. Live
benchmarking makes real calls to whatever providers currently have a
working key and reports measured vs. assumed latency.

```
uvicorn app.api.main:app --reload
```

Starts the API on http://localhost:8000 (interactive docs at `/docs`).
Run from the project root. Try:
```
curl http://localhost:8000/v1/models
curl -X POST http://localhost:8000/v1/completions -H "Content-Type: application/json" -d "{\"prompt\": \"What is the capital of Spain?\", \"verify\": true}"
```

```
python scripts/load_test.py --count 50
```

Runs a batch of programmatically-varied prompts through the full
route+log pipeline and prints the final cost-savings numbers — the
actual portfolio deliverable. See the Phase 11 section above for the
real numbers from this session's run.

## Repo structure

```
config/
  model_registry.yaml   pricing, latency, and quality tier per model
  routing_config.yaml   tier -> provider/model map
app/
  models/
    registry.py          ModelConfig, QUALITY_RANK, is_model_available(), YAML loader + cost math
    response.py           standardized LLMResponse
    dispatcher.py          send_request() — the one function everything else calls
    providers/
      base.py               BaseProvider interface + ProviderError
      gemini_provider.py
      anthropic_provider.py
      ollama_provider.py
      openai_compatible.py  shared adapter for OpenAI, Groq, Together AI
  classifier/
    tiers.py              ComplexityTier enum
    features.py            prompt -> PromptFeatures
    heuristic.py            PromptFeatures -> ComplexityTier (v1, rule-based)
  router/
    routing_engine.py     route_request() — classifier + routing map + dispatcher
    cost_predictor.py      predict_cost() — pre-call cost/quality estimate, all models
    explainer.py            explain() — tier/confidence/signals/alternative writeup
    multi_objective.py      select_model() — weighted cost/latency/quality scoring
    verifier.py              verify_and_escalate() — reference-check + auto-escalation
  logging/
    db.py                  SQLite schema, log_request(), get_summary() and friends
  rag/
    compression.py          compress_context() — TF-IDF chunk retrieval
    routing_memory.py       get_routing_memory_signal() — similar-past-request lookup
  simulation/
    simulator.py            simulate() — strategy comparison over a prompt batch
    live_benchmark.py       benchmark_available_models() — real-call latency snapshot
  api/
    main.py                 FastAPI app: /v1/completions, /v1/models, /v1/stats, /v1/routing-config
    schemas.py               Pydantic request/response models
dashboard/
  app.py                  Streamlit cost dashboard (streamlit run dashboard/app.py)
scripts/
  test_providers.py      Phase 1 validation harness
  test_classifier.py     Phase 2 offline classifier sanity check
  predict_cost.py        Phase 3 cost-prediction CLI demo
  explain_routing.py     Phase 4 explanation CLI demo (makes a real provider call)
  select_model_demo.py   Phase 5 multi-objective CLI demo
  verify_routing.py      Phase 6 verification CLI demo (makes real provider calls)
  run_and_log.py         Phase 7 route + verify + log pipeline
  seed_demo_data.py      Phase 7 dev utility — populates data/requests.db via Ollama
  compress_context_demo.py  Phase 8a context compression CLI demo
  routing_memory_demo.py    Phase 8b routing memory CLI demo (needs logged history)
  simulate_demo.py       Phase 9a simulation CLI demo
  live_benchmark_demo.py Phase 9b live benchmark CLI demo (makes real provider calls)
  load_test.py            Phase 11 load test — final cost-savings numbers
data/
  phase1_test_results.json  (generated; kept as a portfolio artifact)
  labeled_prompts.csv        (empty header, waiting for your hand-labels)
  requests.db                 (generated, gitignored — Phase 7's audit trail)
Dockerfile / docker-compose.yml / .dockerignore   Phase 10 (untested — no Docker here)
CASE_STUDY.md            Phase 11 portfolio write-up
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
- Ollama's **first** request after the model has been idle is slow —
  observed ~155 seconds for `llama3.2` while Ollama loads the 2GB model
  into memory. Subsequent requests are fast (normal LLM latency) as long
  as the model stays loaded, but Ollama unloads idle models after a few
  minutes by default, so the next request after a gap pays that cost
  again. The `avg_latency_ms: 800` in `config/model_registry.yaml`
  reflects warm-model latency, not this cold-start spike — worth knowing
  before Phase 3's cost/latency predictions start factoring it in.
- If a provider that should be skipped instead fails with a `401` error,
  you likely have a stray API key for a *different* service set as a
  global environment variable on your machine under the same name (this
  happened during development with a leftover `OPENAI_API_KEY`). It's
  harmless — the call fails before anything is billed — but you can remove
  the stray variable from Windows environment variables if the noise
  bothers you.
- The heuristic classifier matches whole words (`\bcompare\b`), not stems —
  "compare" matches but "compares"/"comparing" won't, and "tradeoff"
  matches but "tradeoffs" won't. This under-counts complex-keyword signals
  on inflected prompts. Acceptable for a rule-based v1 (the scikit-learn
  v2 classifier won't have this limitation), but worth knowing if a tier
  assignment looks off.
- Routing map (`config/routing_config.yaml`) now points Tier 1 at Ollama
  (updated via the Phase 10 API, live/working) and Tiers 2/3 at Gemini
  (still no working key here). Tier 2/3 calls will fail with a clean
  error until `GEMINI_API_KEY` is added to `.env` — `route_request()`
  still returns a full `RoutingResult` in that case, just with
  `response.error` set, so this is expected, not a bug, until a key is
  added.
- `predict_cost.py`'s output-token estimate is a flat number per
  complexity tier (80/250/500), not per-model or per-prompt — a terse
  Tier 2 prompt and a verbose one get the same output estimate today.
  Expect the dollar figures to be rough, not exact, until Phase 7's
  logging lets this be replaced with real historical averages.
- `predict_cost.py` and `route_request()` currently disagree for Tier 1
  (Ollama is now cheaper than Groq, per the "real finding" note above)
  and are unlikely to agree in general — prediction doesn't feed back
  into the routing config automatically. If you want the router to
  actually route to what's predicted cheapest, that's a manual edit to
  `config/routing_config.yaml` for now.
- `explain()`'s confidence score is a transparent function of the
  classifier's raw score, not a calibrated probability — don't read
  "74%" as "correct 74% of the time." It only tells you how close the
  score sat to a tier boundary.
- `select_model()` (Phase 5) and `route_request()` (Phase 2) are
  independent — multi-objective scoring doesn't feed back into
  `routing_config.yaml` any more than Phase 3's predictions do. If you
  want the static router to actually route by weighted score instead of
  a fixed tier→model map, that's a design change for a later phase, not
  something Phase 5 does automatically.
- `select_model()`'s cost/latency/quality weighting only distinguishes
  candidates when the registry actually has spread on those axes for a
  given tier — see Phase 5's "real finding" above. Don't expect every
  profile to pick a different model; expect hard constraints
  (`require_local`, `max_latency_ms`) to matter more than soft weights
  for this particular registry.
- `verify_and_escalate()` (Phase 6) always checks against whatever Tier 3
  routes to — if `routing_config.yaml`'s Tier 3 entry doesn't have a
  working key, verification will always come back `comparison_failed`
  (or `already_top_tier` for anything already routed to Tier 3's model)
  until that key exists. Not a bug — there's nothing to compare against.
- `verify_and_escalate()` doubles the number of live calls for anything
  not already on the top tier (one for the route, one for the reference
  check) — real cost to be aware of if you run it at volume, since it's
  fully synchronous right now (see the "async, honestly" note above).
- `data/requests.db` has 62 real logged requests as of Phase 11's load
  test (15 seeded + individual test/demo runs + a 40-prompt load test,
  17 of which succeeded — the rest correctly failed on Tier 2/3's
  missing Gemini key). The dashboard's numbers are only as representative
  as what's actually been logged — enough to prove the mechanism works
  end-to-end, not enough volume to draw statistically meaningful
  conclusions from. `CASE_STUDY.md` is explicit about this caveat rather
  than presenting the 100%-savings figure without context.
- Phase 8's TF-IDF similarity is lexical, not semantic (see the "Phase 8"
  section above) — it will miss genuinely related prompts that don't
  share vocabulary, and can be fooled by shared words in unrelated
  contexts. Routing memory's signal is advisory only; nothing in the
  router currently acts on it automatically.
- Phase 9's `simulate()` costs are projections (Phase 3/5's estimation
  logic), not measurements — treat the 70.8% savings figure as "what the
  math says," reproducible and inspectable, but not the same class of
  claim as `live_benchmark.py`'s measured latency numbers.
- `live_benchmark_demo.py` only benchmarks models with a working key —
  right now that's Ollama alone, so the "measured vs. assumed" comparison
  is only meaningful for one model until more keys are added.
- `PUT /v1/routing-config` overwrites `config/routing_config.yaml` via
  `yaml.dump`, which does not preserve comments — every call strips the
  file's explanatory notes. This already happened once during Phase 10
  testing; the comments were restored by hand afterward. If you call
  this endpoint yourself, expect to lose the comments again.
- `Dockerfile` / `docker-compose.yml` are written but **not
  build-tested** — Docker isn't installed in this environment. They
  follow standard, well-documented patterns, but treat them the same
  way as the untested OpenAI/Anthropic adapters: probably fine,
  unverified, worth a real `docker compose up` before trusting them for
  anything important.
- `load_test.py`'s 40-prompt run is a scope-reduced stand-in for the
  original 500-1,000 prompt target (see Phase 11 section above for why)
  — the script scales, the demonstrated run doesn't reach that volume.
- The 100% savings figure from the load test is real but narrow: it
  holds only because the sole currently-working live tier (Tier 1 →
  Ollama) is also the free one. Treat Phase 9's 70.8% simulation figure
  as the more representative claim — it reflects real pricing across
  every registered provider, not just whichever one has a key today.

## Roadmap

Build order:
1. Unified provider interface — **done**
2. Complexity classifier (heuristic v1) + routing map — **done** (v1 is
   rule-based and live; the scikit-learn v2 upgrade is still blocked on
   your 200+ hand-labeled `data/labeled_prompts.csv` rows)
3. Pre-call cost prediction — **done**
4. Explainable routing — **done**
5. Multi-objective optimization (cost/latency/quality/data-locality) — **done**
6. Async quality verification + auto-escalation — **done** (synchronous
   for now — Phase 10 shipped the API but deliberately not a background
   worker either; see Phase 10's section above for why a placeholder
   worker would've been decoration, not a real fix)
7. SQLite logging + Streamlit cost dashboard — **done**
8. **RAG** — **done** (TF-IDF, not real embeddings — see the Phase 8
   section above for why):
   - **Context compression**: chunk/retrieve only the relevant pieces of
     a long document before it's sent to any model, instead of sending
     the whole thing. Verified with a checkable synthetic example
     (60.4% reduction, correctly kept only the relevant sections).
   - **Routing memory**: retrieve the k most similar past requests plus
     their verified outcomes as an advisory second signal. Needs real
     logged history to be useful, which is why it's built on top of
     Phase 7's database.
9. Simulation mode + live benchmarking — **done** (simulation found the
   router saves 70.8% vs. the most expensive strategy on 15 real
   prompts; live benchmarking found Ollama's measured latency drifts
   +2508ms from the registry's static assumption)
10. FastAPI service + Docker — **done** (API fully verified live against
    real HTTP requests; Docker files written but not build-tested — no
    Docker in this environment)
11. Load test + portfolio write-up — **done** (40 real prompts, scope
    reduced from the original 500-1,000 — see Phase 11 section above;
    `CASE_STUDY.md` is the portfolio write-up)

**All 11 phases of the original build plan are now implemented.**
Remaining work is depth, not breadth: a real second live provider key,
the scikit-learn v2 classifier upgrade (blocked on hand-labeling), and
the items below that were always explicitly out of scope for v1.

Explicitly deferred past v1 (documented here so they're not forgotten,
not because they're bad ideas):
- Prompt optimizer (auto-rewrite prompts to cut tokens)
- User preference dials (speed>quality, budget-per-day)
- Agent-specific routing policies (coding/research/creative/OCR/math)
- Self-learning/RL router (online learning from routing feedback)
- Budget scheduler (automatic degradation as a monthly budget is consumed)
- Weekly classifier auto-retraining flywheel
