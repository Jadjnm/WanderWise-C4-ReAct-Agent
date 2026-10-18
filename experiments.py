"""Logged experiment harness for Assignment C4 (personas, prompting, LLM parameters).

Every run is appended to logs/c4_experiments.jsonl (full config, trace, raw LLM output,
metrics) and each invocation writes a Markdown comparison table to
logs/c4_experiments_summary_<run_id>.md that you can paste into the notebook/report.

Usage:
    python experiments.py --suite personas
    python experiments.py --suite prompting --persona strict_auditor
    python experiments.py --suite params --persona friendly_advisor --sleep 6
    python experiments.py --suite all --repeats 2
    python experiments.py --list
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from typing import Any

import react_core as core

EXPERIMENT_LOG = core.BASE_DIR / "logs" / "c4_experiments.jsonl"
# Tool side effects from experiments go here so the demo logs stay clean
SANDBOX_TOOL_LOG_DIR = core.BASE_DIR / "logs" / "experiment_tool_logs"

# ------------------------------------------------------------------------------
# Test cases: each has the tools a correct agent must call, facts the answer must
# contain, and patterns that indicate a hallucination.
# ------------------------------------------------------------------------------
_PACKAGE_PRICES = "|".join(f"{p['base_price_pp']:,}" for p in core.CATALOG.values())


def _wrong_total(correct: str, subtotal: str) -> str:
    """A "total" followed by an amount that is none of the legitimate figures (the verified total, the
    base subtotal, an add-on price, a package price, or a per-person amount). Catches answers that quote
    the right figure in a breakdown but end on a wrong "Grand Total"."""
    allowed = "|".join(re.escape(x) for x in (correct, subtotal, f"{core.ROOM_UPGRADE_PP * 2:,}",
                                               f"{core.PRIVATE_GUIDE_FLAT:,}")) + "|" + _PACKAGE_PRICES
    return (r"(?<!base )(?<!base rate )\btotal(?! per)[^$\n]{0,40}\$\s?(?!(?:" + allowed + r")\b)\d[\d,]*"
            r"(?![\d,]*\s*(base\s+)?(per person|pp\b|each|/person))")


# A dollar amount in the same sentence as airfare/flights, excluding the catalog's own package prices.
AIRFARE_QUOTE = (r"\b(airfares?|flights?|fares?)\b[^.!?\n]{0,160}\$\s?(?!(?:" + _PACKAGE_PRICES + r")\b)\d")

TEST_CASES: list[dict[str, Any]] = [
    {
        "id": "budget_bali_upgrade",
        "query": "How much would a 7-day trip to Bali cost for 2 people with a deluxe room upgrade?",
        "expected_tools": ["calculate_trip_budget"],
        "must_include": ["4,200"],
        "must_not_match": [_wrong_total("4,200", subtotal="3,700")],
    },
    {
        "id": "budget_swiss_guide",
        "query": "Calculate the total budget for 2 people on Swiss Alpine Discovery with a private guide.",
        "expected_tools": ["calculate_trip_budget"],
        "must_include": ["4,700"],
        "must_not_match": [_wrong_total("4,700", subtotal="4,300")],
    },
    {
        "id": "policy_cancellation_unknown",
        "query": "What happens if I need to cancel my Santorini trip because of a storm? Do I get money back?",
        "expected_tools": ["record_unanswered_inquiry"],
        "must_include": [],
        "must_not_match": [r"full refund", r"\d+\s*%\s*(refund|back)", r"refund(ed)? within \d+"],
    },
    {
        "id": "policy_flights",
        "query": "Do you book flights to Rome, and roughly how much is the airfare?",
        "expected_tools": ["lookup_policy_rule"],
        "must_include": ["flight"],
        "must_not_match": [AIRFARE_QUOTE],
    },
    {
        "id": "search_japan_group",
        "query": "Tell me about the highlights of the Japan package and whether group tours are planned.",
        "expected_tools": ["search_travel_packages"],
        "must_include": ["kyoto"],
        "must_not_match": [r"group tours? (are|is) (planned|scheduled|available)"],
    },
    {
        "id": "lead_capture",
        "query": "I'd like a consultation for a Rome honeymoon. I'm Dana Saleh, dana@example.com, 2 travelers in June.",
        "expected_tools": ["record_customer_lead"],
        "must_include": [],
        "must_not_match": [r"\b(booked|reservation (is )?confirmed)\b"],
    },
]

# ------------------------------------------------------------------------------
# Experiment conditions. Each condition is a label plus run_react_agent kwargs.
# ------------------------------------------------------------------------------
BASELINE = {"use_cot": True, "few_shot": True, "instruction_placement": "system"}


def persona_conditions(_persona: str) -> list[tuple[str, dict[str, Any]]]:
    return [(f"persona={key}", {**BASELINE, "persona_key": key}) for key in core.PERSONA_PROFILES]


def prompting_conditions(persona: str) -> list[tuple[str, dict[str, Any]]]:
    """One-factor-at-a-time ablation from the baseline (keeps API usage manageable)."""
    base = {**BASELINE, "persona_key": persona}
    return [
        ("baseline (CoT, few-shot, system)", base),
        ("no CoT", {**base, "use_cot": False}),
        ("zero-shot (no examples)", {**base, "few_shot": False}),
        ("instructions in user message", {**base, "instruction_placement": "user"}),
        ("zero-shot + no CoT", {**base, "use_cot": False, "few_shot": False}),
    ]


def params_conditions(persona: str) -> list[tuple[str, dict[str, Any]]]:
    base = {**BASELINE, "persona_key": persona}
    variants: list[tuple[str, dict[str, Any]]] = [
        ("persona defaults", {}),
        ("temperature=0.0", {"temperature": 0.0}),
        ("temperature=1.2", {"temperature": 1.2}),
        ("top_p=0.3", {"top_p": 0.3}),
        ("top_p=1.0", {"top_p": 1.0}),
        ("top_k=1", {"top_k": 1}),
        ("top_k=100", {"top_k": 100}),
        ("max_tokens=120", {"max_tokens": 120}),
        ("frequency_penalty=1.0", {"frequency_penalty": 1.0}),
        ("presence_penalty=1.0", {"presence_penalty": 1.0}),
    ]
    # Model ids are provider-specific: BEDROCK_ALT_MODEL on Bedrock, EXPERIMENT_ALT_MODEL elsewhere.
    alt_var = "BEDROCK_ALT_MODEL" if core.get_active_provider() == "bedrock" else "EXPERIMENT_ALT_MODEL"
    alt_model = core.os.getenv(alt_var, "").strip()
    if alt_model:
        variants.append((f"model={alt_model}", {"model": alt_model}))
    return [(label, {**base, "llm_overrides": ov}) for label, ov in variants]


def model_list() -> list[str]:
    """Models for the models suite: comma-separated EXPERIMENT_MODELS (set by --models)."""
    return [m.strip() for m in core.os.getenv("EXPERIMENT_MODELS", "").split(",") if m.strip()]


def models_conditions(_persona: str) -> list[tuple[str, dict[str, Any]]]:
    """Every persona on every listed model (same prompt setup), so you can see whether the persona
    ranking holds across models. The model is kept per run, so each model gets its own summary rows."""
    models = model_list()
    if not models:
        raise ValueError("The models suite needs --models model_a,model_b (ids for the active provider).")
    return [(f"persona={key}", {**BASELINE, "persona_key": key, "llm_overrides": {"model": model}})
            for model in models for key in core.PERSONA_PROFILES]


def guardrails_conditions(_persona: str) -> list[tuple[str, dict[str, Any]]]:
    """Every persona with code guardrails off vs on (the guard checks the finished answer only)."""
    return [(f"persona={key} | guardrails={'on' if on else 'off'}", {**BASELINE, "persona_key": key, "guardrails": on})
            for key in core.PERSONA_PROFILES for on in (False, True)]


def instructions_conditions(_persona: str) -> list[tuple[str, dict[str, Any]]]:
    """Every persona with full instructions vs format-only (operational rules and worked examples removed)."""
    return [
        (f"persona={key} | instructions={mode}", {**BASELINE, "persona_key": key, "instructions_mode": mode})
        for key in core.PERSONA_PROFILES
        for mode in ("full", "format_only")
    ]


def loop_guards_conditions(_persona: str) -> list[tuple[str, dict[str, Any]]]:
    """Strict auditor with loop guards off vs on. Defaults to xing4.0 where loops were common."""
    model = core.os.getenv("LOOP_GUARD_MODEL", "xingchen-agi/xing4.0-29b-a4b:free").strip()
    return [
        (f"loop_guards={'on' if on else 'off'}", {
            **BASELINE,
            "persona_key": "strict_auditor",
            "loop_guards": on,
            "llm_overrides": {"model": model} if model else {},
        })
        for on in (False, True)
    ]


SUITES = {
    "personas": persona_conditions,
    "prompting": prompting_conditions,
    "params": params_conditions,
    "models": models_conditions,
    "guardrails": guardrails_conditions,
    "instructions": instructions_conditions,
    "loop_guards": loop_guards_conditions,
}
CORE_SUITES = ("personas", "prompting", "params")  # what --suite all runs


# ------------------------------------------------------------------------------
# Scoring
# ------------------------------------------------------------------------------
_NEGATION_RE = re.compile(r"\b(no|not|never|cannot|can't|won't|don't|isn't|aren't|without)\b|n't\b", re.I)


def _asserted_matches(pattern: str, answer: str) -> bool:
    """True if the pattern matches somewhere it is asserted, not negated: the denial
    "No group tour is planned" must not count as claiming "group tour is planned"."""
    # Overlapping search: a negated mention early in a sentence must not hide a later asserted one.
    for m in re.finditer(f"(?=({pattern}))", answer, flags=re.I):
        start = m.start()
        clause_start = 0
        for sep in (".", "!", "?", "\n", ";", "—", " but "):
            i = answer.rfind(sep, 0, start)
            if i >= 0:
                clause_start = max(clause_start, i + len(sep))
        if not _NEGATION_RE.search(answer[max(clause_start, start - 40):start]):
            return True
    return False


def score_run(case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    answer = result["final_answer"].lower()
    tools_ok = all(t in result["tools_used"] for t in case["expected_tools"])
    facts_ok = all(f.lower() in answer for f in case["must_include"])
    hallucinated = [p for p in case["must_not_match"] if _asserted_matches(p, answer)]
    ok = result["stopped_reason"] in ("final_answer", "forced_answer")
    passed = ok and tools_ok and facts_ok and not hallucinated
    return {
        "passed": passed,
        "completed": ok,
        "tools_ok": tools_ok,
        "facts_ok": facts_ok,
        "hallucination_hits": hallucinated,
        "answer_words": len(result["final_answer"].split()),
    }


# ------------------------------------------------------------------------------
# Running
# ------------------------------------------------------------------------------
class ExperimentAborted(RuntimeError):
    """Raised after repeated LLM errors (e.g. exhausted quota); carries the runs completed so far."""

    def __init__(self, records: list[dict[str, Any]]):
        super().__init__("experiment aborted after repeated LLM errors")
        self.records = records


def run_conditions(
    suite: str,
    conditions: list[tuple[str, dict[str, Any]]],
    cases: list[dict[str, Any]],
    repeats: int = 1,
    sleep_s: float = 0.0,
    run_id: str | None = None,
    verbose: bool = True,
    max_consecutive_errors: int = 3,
    skip: set[tuple[str, str, str, str, int]] | None = None,
) -> list[dict[str, Any]]:
    """skip: run keys (model, suite, condition, case_id, repeat) already logged successfully (--resume)."""
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    EXPERIMENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    original_log_dir = core.LOG_DIR
    core.LOG_DIR = SANDBOX_TOOL_LOG_DIR
    records = []
    consecutive_errors = 0
    try:
        total = len(conditions) * len(cases) * repeats
        n = 0
        for label, kwargs in conditions:
            model = (kwargs.get("llm_overrides") or {}).get("model") or core.get_active_model()
            for case in cases:
                for rep in range(repeats):
                    n += 1
                    if skip and (model, suite, label, case["id"], rep) in skip:
                        continue
                    result = core.run_react_agent(case["query"], **kwargs)
                    record = {
                        "run_id": run_id,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "suite": suite,
                        "condition": label,
                        "case_id": case["id"],
                        "repeat": rep,
                        **result,
                        "metrics": score_run(case, result),
                    }
                    records.append(record)
                    with EXPERIMENT_LOG.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    if verbose:
                        m = record["metrics"]
                        print(f"[{n}/{total}] {suite} | {label} | {case['id']} -> "
                              f"{'PASS' if m['passed'] else 'FAIL'} tools={result['tools_used']} "
                              f"steps={result['iterations']} {result['latency_s']}s"
                              + (f" ERROR: {result['error'][:80]}" if result["error"] else ""))
                    consecutive_errors = consecutive_errors + 1 if result["error"] else 0
                    if consecutive_errors >= max_consecutive_errors:
                        print(f"\nStopping: {consecutive_errors} LLM errors in a row (last: {result['error'][:160]})")
                        raise ExperimentAborted(records)
                    if sleep_s:
                        time.sleep(sleep_s)
    finally:
        core.LOG_DIR = original_log_dir
    return records


def summarize(records: list[dict[str, Any]]) -> str:
    """Markdown comparison table: one row per (model, suite, condition), so runs from
    different providers/models logged in the same file are never blended together."""
    if not records:
        return "_No records._"
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for r in records:
        groups.setdefault((f"{r['provider']}/{r['model']}", r["suite"], r["condition"]), []).append(r)

    def pct(rows: list[dict[str, Any]], key: str) -> str:
        return f"{100 * sum(bool(r['metrics'][key]) for r in rows) / len(rows):.0f}%"

    lines = []
    if any(r["simulated"] for r in records):
        lines.append("> **WARNING: these runs used the offline simulator, not a real LLM. "
                     "Do not use them as evidence in the reflection.**\n")
    models = sorted({key[0] for key in groups})
    run_ids = sorted({r["run_id"] for r in records})
    lines.append(f"Models: {', '.join(f'`{m}`' for m in models)} | run_ids: {', '.join(f'`{x}`' for x in run_ids)}\n")
    lines.append("| Model | Suite | Condition | Runs | Pass | Correct tools | Facts present | Hallucinations | "
                 "Errors | Avg steps | Format errors | Avg words | Avg latency (s) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for (model, suite, cond), rows in groups.items():
        lines.append(
            f"| {model.split('/', 1)[1]} | {suite} | {cond} | {len(rows)} | {pct(rows, 'passed')} | {pct(rows, 'tools_ok')} | "
            f"{pct(rows, 'facts_ok')} | {sum(bool(r['metrics']['hallucination_hits']) for r in rows)} | "
            f"{sum(bool(r['error']) for r in rows)} | "
            f"{sum(r['iterations'] for r in rows) / len(rows):.1f} | "
            f"{sum(r['format_errors'] for r in rows)} | "
            f"{sum(r['metrics']['answer_words'] for r in rows) / len(rows):.0f} | "
            f"{sum(r['latency_s'] for r in rows) / len(rows):.1f} |"
        )
    return "\n".join(lines)


def load_records(run_id: str | None = None, model: str | None = None) -> list[dict[str, Any]]:
    """Read logged runs back (optionally one run_id and/or one model) for analysis in the notebook."""
    if not EXPERIMENT_LOG.exists():
        return []
    rows = [json.loads(line) for line in EXPERIMENT_LOG.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows
            if (run_id is None or r["run_id"] == run_id) and (model is None or r["model"] == model)]


def _run_key(r: dict[str, Any]) -> tuple[str, str, str, str, int]:
    return (r["model"], r["suite"], r["condition"], r["case_id"], r.get("repeat", 0))


def rescore(records: list[dict[str, Any]], latest_only: bool = True) -> list[dict[str, Any]]:
    """Recompute metrics for logged runs with the current scorer (the raw log is left untouched).
    latest_only keeps just the newest record per run key, so a run retried with --resume
    (e.g. after an error) replaces the failed attempt instead of being counted twice."""
    if latest_only:
        latest: dict[tuple[str, str, str, str, int], dict[str, Any]] = {}
        for r in records:  # log order is chronological, so later records win
            latest[_run_key(r)] = r
        records = list(latest.values())
    cases = {c["id"]: c for c in TEST_CASES}
    return [{**r, "metrics": score_run(cases[r["case_id"]], r)} for r in records if r["case_id"] in cases]


def run_suite(
    suite: str,
    persona: str = "friendly_advisor",
    case_ids: list[str] | None = None,
    repeats: int = 1,
    sleep_s: float = 0.0,
    verbose: bool = True,
    resume: bool = False,
) -> list[dict[str, Any]]:
    """Notebook-friendly entry point. suite: personas | prompting | params | all.
    resume=True skips runs already logged without an error (to continue after a stop)."""
    if suite == "loop_guards" and case_ids is None:
        case_ids = ["policy_flights", "policy_cancellation_unknown"]
    cases = [c for c in TEST_CASES if not case_ids or c["id"] in case_ids]
    if not cases:
        raise ValueError(f"No test cases match {case_ids}. Available: {[c['id'] for c in TEST_CASES]}")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    skip = {_run_key(r) for r in load_records() if not r["error"]} if resume else None
    names = list(CORE_SUITES) if suite == "all" else [suite]
    records = []
    for name in names:
        if name not in SUITES:
            raise ValueError(f"Unknown suite '{name}'. Choose from: {', '.join(SUITES)}, all")
        try:
            records += run_conditions(name, SUITES[name](persona), cases, repeats, sleep_s, run_id, verbose, skip=skip)
        except ExperimentAborted as aborted:
            records += aborted.records
            print("Remaining runs skipped. Partial results are logged; re-run the same command with --resume to continue.")
            break
    return records


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Run logged C4 ReAct experiments.")
    parser.add_argument("--suite", default="personas", choices=[*SUITES, "all"],
                        help="all = personas + prompting + params; models needs --models.")
    parser.add_argument("--models", help="Comma-separated model ids for --suite models (active provider's ids).")
    parser.add_argument("--persona", default="friendly_advisor", choices=list(core.PERSONA_PROFILES),
                        help="Persona held fixed for the prompting/params suites.")
    parser.add_argument("--cases", nargs="*", help="Subset of test case ids (default: all).")
    parser.add_argument("--repeats", type=int, default=1, help="Repeat each run to measure sampling variance.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds between runs (free-tier rate limits).")
    parser.add_argument("--list", action="store_true", help="List test cases and conditions, then exit.")
    parser.add_argument("--rescore", nargs="?", const="all", metavar="RUN_ID",
                        help="Re-score logged runs with the current scorer (one run_id, or all) and write a summary.")
    parser.add_argument("--model", help="Model id override for --suite loop_guards or filter for --rescore.")
    parser.add_argument("--resume", action="store_true",
                        help="Skip runs already logged without an error (continue after a stop).")
    parser.add_argument("--log", help="Experiment log file (default logs/c4_experiments.jsonl). Give parallel "
                                      "runs separate files so their records never interleave.")
    args = parser.parse_args()

    global EXPERIMENT_LOG
    if args.models:
        core.os.environ["EXPERIMENT_MODELS"] = args.models
    if args.model and not args.rescore:
        core.os.environ["LOOP_GUARD_MODEL"] = args.model
    if args.log:
        EXPERIMENT_LOG = (core.BASE_DIR / args.log).resolve()
    log_tag = "" if EXPERIMENT_LOG.name == "c4_experiments.jsonl" else f"_{EXPERIMENT_LOG.stem}"

    if args.rescore:
        records = rescore(load_records(None if args.rescore == "all" else args.rescore, args.model))
        if not records:
            print("No logged runs found.")
            return
        summary = summarize(records)
        suffix = args.rescore + (f"_{re.sub(r'[^A-Za-z0-9.-]+', '-', args.model)}" if args.model else "")
        out = core.BASE_DIR / "logs" / f"c4_experiments_summary_rescored_{suffix}{log_tag}.md"
        out.write_text(summary + "\n", encoding="utf-8")
        print(summary + f"\n\nRe-scored {len(records)} runs. Summary written to {out}")
        return

    if args.list:
        for c in TEST_CASES:
            print(f"{c['id']:<30} {c['query']}")
        for name, fn in SUITES.items():
            print(f"\n[{name}]")
            if name == "models" and not model_list():
                print("  (pass --models a,b to list its conditions)")
                continue
            for label, kwargs in fn(args.persona):
                model = (kwargs.get("llm_overrides") or {}).get("model")
                print(f"  {label}" + (f"  @ {model}" if name == "models" else ""))
        return

    provider = core.get_active_provider()
    print(f"Provider: {provider} | model: {core.get_active_model(provider)}")
    if provider == "simulated":
        print("WARNING: no API key configured (or REACT_SIMULATE=1). Results come from the offline simulator.")

    records = run_suite(args.suite, args.persona, args.cases, args.repeats, args.sleep, resume=args.resume)
    if not records:
        print("No runs completed.")
        return
    summary = summarize(records)
    out = core.BASE_DIR / "logs" / f"c4_experiments_summary_{records[0]['run_id']}{log_tag}.md"
    out.write_text(summary + "\n", encoding="utf-8")
    print("\n" + summary)
    print(f"\nFull runs appended to {EXPERIMENT_LOG}\nSummary written to {out}")


if __name__ == "__main__":
    main()
