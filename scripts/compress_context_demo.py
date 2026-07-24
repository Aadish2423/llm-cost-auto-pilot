"""Phase 8a demo: compress a long multi-topic document down to just the
chunks relevant to a specific query. Never calls a provider - pure
TF-IDF retrieval, like the other offline demo scripts.

The demo document deliberately covers several unrelated topics
(authentication, rate limits, pricing, webhooks, changelog) so a query
about one topic has a clear, checkable right answer for which chunks
should survive.

Usage:
    python scripts/compress_context_demo.py "your query"
    python scripts/compress_context_demo.py             (demo query)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.rag.compression import compress_context  # noqa: E402

DEMO_DOCUMENT = """
Authentication. All requests to the API must include an API key in the
Authorization header, formatted as "Bearer <key>". Keys are generated
from the dashboard under Settings > API Keys. Keys never expire but can
be revoked at any time, which takes effect within sixty seconds. There
is no support for session cookies or OAuth for server-to-server calls;
OAuth is only used for the optional user-facing web login flow, which is
a separate system from API key authentication and issues short-lived
tokens instead.

Rate limits. Free tier accounts are limited to 60 requests per minute
and 5,000 requests per day. Paid tier accounts are limited to 600
requests per minute and 200,000 requests per day. Exceeding the limit
returns HTTP 429 with a Retry-After header indicating how many seconds
to wait. Repeated 429 responses without backoff may result in a
temporary IP-level block lasting up to fifteen minutes. Rate limit
headers (X-RateLimit-Remaining, X-RateLimit-Reset) are included on every
response so clients can self-throttle before hitting the limit.

Pricing. The free tier includes 5,000 requests per day at no cost. The
paid "Growth" tier is $49 per month and includes 200,000 requests per
day, with overage billed at $0.001 per request beyond that. The
"Enterprise" tier is custom-priced and includes a dedicated rate limit,
a service-level agreement, and priority support. Downgrading from a paid
tier takes effect at the start of the next billing cycle, not
immediately. Refunds are not issued for partial billing periods.

Webhooks. You can register a webhook URL to receive event notifications
for account activity: request.completed, request.failed, and
key.revoked. Webhook payloads are signed with HMAC-SHA256 using a secret
shown once at creation time; verify the X-Signature header before
trusting the payload. Failed webhook deliveries are retried up to five
times with exponential backoff, then dropped. There is currently no way
to replay a dropped webhook event after the retry window expires.

Error codes. 400 means the request body was malformed or missing a
required field. 401 means the API key is missing, invalid, or revoked.
403 means the key is valid but lacks permission for that endpoint. 404
means the resource does not exist or does not belong to this account.
429 means the rate limit was exceeded (see Rate limits above). 500
means an internal error occurred; these are logged automatically and
usually resolved without any action needed from the caller.

Changelog. Version 3.2 added webhook signature verification. Version
3.1 raised the Growth tier's daily limit from 100,000 to 200,000
requests. Version 3.0 deprecated the legacy /v1/ endpoints in favor of
/v2/; the legacy endpoints will be removed entirely six months after
this release. Version 2.5 introduced per-key rate limit overrides for
Enterprise accounts.
""".strip()

DEMO_QUERY = "How do I handle a 429 rate limit error and what are the actual rate limits?"


def main() -> None:
    query = " ".join(sys.argv[1:]) or DEMO_QUERY
    result = compress_context(query, DEMO_DOCUMENT, top_k=2, chunk_size_words=90)

    print(f'Query: "{query}"')
    print()
    print(f"Original: {result.original_word_count} words (~{result.original_est_tokens} tokens)")
    print(f"Compressed: {result.compressed_word_count} words (~{result.compressed_est_tokens} tokens)")
    print(f"Reduction: {result.reduction_pct:.1f}%")
    print(f"Chunks kept: {result.chunks_kept} / {result.chunks_total} (indices {result.selected_chunk_indices})")
    print()
    print("=== Compressed context that would actually get sent ===")
    print(result.compressed_text)


if __name__ == "__main__":
    main()
