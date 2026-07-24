# LLM Cost Auto Pilot — Case Study

## The number

**Projected savings: 70.8%** vs. always routing every request to the most
expensive available model — measured by replaying 15 real prompts through
every routing strategy the system supports (Phase 9's simulation mode),
using each model's real published pricing.

A second, narrower number from actually running the system end-to-end:
**100% cost reduction, $0.045686 saved, across 62 real logged requests**
(Phase 11's load test plus earlier development runs). That number needs an
honest caveat, and this case study leads with the caveat rather than
burying it: it's 100% right now only because the one tier with a
currently-working live API key (Tier 1, routed to a local Ollama model) is
also the free one. The 70.8% simulation figure is the more representative
claim — it reflects real cost math across every provider this project
supports (OpenAI, Anthropic, Gemini, Groq, Together AI, Ollama), not just
whichever one happens to have a key configured in this environment today.
Both numbers are real, reproducible, and come from code in this repo, not
from a slide.

## The problem

Every team running LLMs in production sends some meaningful fraction of
requests to a model that's more expensive than the task needs. Asking a
frontier model to extract a date from an invoice, when a model 50x
cheaper would get it right too, is the default behavior of most
integrations — nobody built the layer that stops to ask "does this
actually need the expensive model?" before spending money.

## The approach

A routing layer sits between the caller and every LLM provider. For each
request, it:

1. **Classifies complexity** (Phase 2) — a rule-based scorer buckets the
   prompt into Tier 1 (simple: extraction, reformatting, basic Q&A),
   Tier 2 (moderate: summarization, classification), or Tier 3 (complex:
   multi-step reasoning, creative generation), from keyword signals,
   constraint counts, and length. No training data required to start.
2. **Predicts cost before sending anything** (Phase 3) — estimates what
   the request would cost on *every* registered model, and recommends the
   cheapest one that clears a quality bar for that tier.
3. **Routes and explains the decision** (Phase 2 + 4) — picks a model via
   a configurable tier→model map, and can produce a full writeup of *why*:
   the matched signals, a transparency-first confidence score (explicitly
   not a calibrated probability — documented as such rather than
   overclaiming precision it doesn't have), and what the next-cheapest
   qualifying alternative would have cost instead.
4. **Optimizes across more than cost** (Phase 5) — cost, latency, and
   quality are weighted and combined into one score; data-locality is a
   *hard constraint*, not a weight (a request that must stay on-prem
   doesn't get a spectrum — it's filtered to local-only models before
   scoring even starts, the same way a strict latency ceiling is).
5. **Checks its own work** (Phase 6) — after routing, the same prompt can
   be sent to the strongest configured model as a reference, the two
   answers scored for agreement, and the response transparently
   upgraded if they diverge — an explicit, logged escalation, not a
   silent failure.

## The feedback loop

Every request — routed, verified, escalated or not — gets one row in a
SQLite audit trail (Phase 7), including what the request would have cost
on the strongest model, computed from the *actual* token counts of the
real response. A Streamlit dashboard reads straight from that table:
total cost vs. baseline, routing distribution, escalation rate, and a
drill-down into any individual request's actual prompt and response.

That same logged history feeds Phase 8's routing memory: a new prompt can
be compared against the k most similar *past* requests and their verified
outcomes as a second, advisory signal — not yet wired into the routing
decision automatically, but the retrieval mechanism is built and verified
(a query about a payment dispute correctly found the one genuinely
similar past support-ticket prompt among 15 logged requests, correctly
scoring everything else at zero similarity).

## Where the money actually goes: RAG as a cost lever, not a chatbot feature

The other half of Phase 8 treats retrieval as a *cost-cutting* technique,
not a knowledge-injection one: a long document gets chunked, and only the
chunks relevant to the actual question are sent, instead of the whole
thing. Verified on a synthetic multi-topic "API documentation" text
(authentication, rate limits, pricing, webhooks, error codes, changelog)
— a query about handling a 429 rate-limit error correctly retrieved only
the Rate Limits and Error Codes sections, correctly excluding
Authentication, Webhooks, and Pricing, for a 60.4% word-count reduction.
A checkable result on a document built specifically so "did it get this
right" has an unambiguous answer, not a cherry-picked demo.

## What's honestly still missing

This case study is deliberately not overselling a system with gaps. Two
things worth naming directly:

- **Quality-parity claims need a second working paid key.** The
  verification loop (Phase 6) is built and its logic is proven correct
  by deterministic tests, but a live "X% cost reduction at Y% quality
  parity" claim needs two independently capable models actually
  disagreeing sometimes — right now only one provider (Ollama, local and
  free) has a consistently working key in this environment, so there's
  nothing weaker to check against something stronger yet.
- **The routing map is static; the smarter scorer (Phase 5) isn't wired
  in as the default.** Simulation mode already shows Phase 5's
  multi-objective router beating the static Phase 2 router by a wide
  margin on the same prompts ($0.0029 vs. $0.0152 across 15 prompts) —
  that's a concrete, unprompted argument for making it the default, not
  yet acted on.

Both are one config change and one API key away, not a redesign — see the
README's "Known limitations" sections for the full, itemized list this
case study is summarizing.

## Try it yourself

```
git clone <this repo>
cd "LLM Cost Auto Pilot"
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env   # add your own keys — see README
python scripts/seed_demo_data.py
streamlit run dashboard/app.py
```

Full setup, every phase's design rationale, and the complete list of
known limitations are in [README.md](README.md).
