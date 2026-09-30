"""Offline check of the on-device privacy guard against hand-labelled prompts.

Every case says whether the prompt should be locked to the device and which
finding kinds must be present. All identifiers below are synthetic (they
pass the Verhoeff / Luhn checksums but belong to nobody).

Usage:
    python scripts/test_privacy_guard.py        # summary
    python scripts/test_privacy_guard.py -v     # every case
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.privacy.guard import scan  # noqa: E402

# (prompt, should_lock, kinds that must be found)
CASES = [
    # ── should stay on-device ───────────────────────────────────────────────
    ("My Aadhaar is 4995 1234 5670, update my address", True, {"aadhaar"}),
    ("Aadhaar: 234567890124", True, {"aadhaar"}),
    ("File ITR for PAN ABCPE1234F", True, {"pan"}),
    ("Pay with card 4111 1111 1111 1111 exp 12/29", True, {"card_number"}),
    ("Transfer to account number 50100234567890 IFSC HDFC0001234", True, {"bank_account", "ifsc"}),
    ("Send 500 to rahul.v@okhdfcbank on UPI", True, {"upi_id"}),
    ("Our GSTIN is 27ABCPE1234F1Z5, draft an invoice", True, {"gstin"}),
    ("Call me on +91 98765 43210 tomorrow", True, {"phone"}),
    ("My number is 9876543210", True, {"phone"}),
    ("Reply to priya.sharma@gmail.com with the report", True, {"email"}),
    ("Draft a leave letter for Rahul Verma to his manager", True, {"person"}),
    ("Remind Ananya Iyer that rent is due Friday", True, {"person"}),
    ("The db password is Hunter2!, why can't I log in?", True, {"password"}),
    ("My OTP: 482913 isn't working", True, {"password"}),
    ("Why does sk-proj-4f9aQ2mZx81LwT7vB0cR5yN3 return 401?", True, {"api_key"}),
    ("export GROQ_API_KEY=gsk_abcdefghijklmnopqrstuvwxyz0123", True, {"api_key"}),
    ("My passport number is K1234567, renew it", True, {"passport"}),
    # ── fine to send to the cloud ───────────────────────────────────────────
    ("What is the capital of France?", False, set()),
    ("Summarize the history of Microsoft in Seattle", False, set()),
    ("Explain event sourcing vs CRUD for a payments platform", False, set()),
    ("Order 123412341234 shipped; revenue was 45000000 in 2026", False, set()),
    ("Convert 2026-09-30 to DD/MM/YYYY", False, set()),
    ("Write a Python function that validates a PAN format", False, set()),
    ("How does UPI settlement work between banks?", False, set()),
    ("Compare Infosys and TCS revenue growth", False, set()),
    ("Translate 'good morning' into Hindi", False, set()),
    ("The invoice total is 1,250.00 due in 30 days", False, set()),
    ("What is 1234567890 divided by 7?", False, set()),
]


def main() -> None:
    verbose = "-v" in sys.argv
    tp = tn = fp = fn = 0
    kind_misses = []
    times = []

    for prompt, should_lock, kinds in CASES:
        report = scan(prompt)
        times.append(report.scan_ms)
        missing = kinds - set(report.categories)
        if should_lock and report.sensitive:
            tp += 1
        elif should_lock:
            fn += 1
        elif report.sensitive:
            fp += 1
        else:
            tn += 1
        if missing:
            kind_misses.append((prompt, missing))
        ok = report.sensitive == should_lock and not missing
        if verbose or not ok:
            mark = "ok  " if ok else "FAIL"
            print(f"[{mark}] lock={report.sensitive!s:5} expected={should_lock!s:5} "
                  f"{report.categories} :: {report.redacted}")

    total = len(CASES)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    times = sorted(times[1:])  # first scan includes model load
    print(f"\n{total} cases | lock accuracy {(tp + tn) / total:.1%} | precision {precision:.1%} | "
          f"recall {recall:.1%} | FP {fp} FN {fn} | kind misses {len(kind_misses)}")
    print(f"scan latency: median {times[len(times) // 2]:.1f} ms, max {times[-1]:.1f} ms "
          f"(on {scan('x').ner_accelerator or 'rules only'})")
    sys.exit(0 if fp == fn == 0 and not kind_misses else 1)


if __name__ == "__main__":
    main()
