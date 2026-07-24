"""Phase 9b demo: benchmark every model that actually has a working key
with a couple of real prompts, reporting measured latency vs. what the
registry assumes. A snapshot you run on demand - see live_benchmark.py's
docstring for why this isn't a scheduled/automatic refresh.

Usage:
    python scripts/live_benchmark_demo.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.simulation.live_benchmark import benchmark_available_models  # noqa: E402


def main() -> None:
    print("Benchmarking every model with a working key (this makes real API calls)...\n")
    results = benchmark_available_models()

    headers = ["model", "status", "measured latency", "registry assumes", "drift"]
    rows = []
    for r in results:
        if r.skipped_reason:
            rows.append([r.model_config.key, f"skipped ({r.skipped_reason})", "-", "-", "-"])
            continue
        status = f"{r.succeeded}/{r.attempted} ok"
        measured = f"{r.avg_latency_ms:.0f} ms" if r.avg_latency_ms is not None else "n/a (all failed)"
        drift = f"{r.latency_drift_ms:+.0f} ms" if r.latency_drift_ms is not None else "-"
        rows.append([r.model_config.key, status, measured, f"{r.registry_avg_latency_ms:.0f} ms", drift])

    widths = [max(len(headers[i]), max(len(row[i]) for row in rows)) for i in range(len(headers))]
    print(" | ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(" | ".join(str(c).ljust(w) for c, w in zip(row, widths)))

    print()
    print("'drift' = measured minus config/model_registry.yaml's avg_latency_ms assumption.")
    print("Nothing here rewrites the registry automatically - large drift is a signal to")
    print("go update it by hand, same as Phase 3/5's predictions never auto-edit config.")


if __name__ == "__main__":
    main()
