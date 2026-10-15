#!/usr/bin/env python3
"""Runs run.py (in either --mode) or the separate experimental ML
comparison (scripts/run_ml_evaluation.py) three times, in three separate
fresh processes, against the same untouched input files, and verifies the
resulting CSVs are byte-for-byte identical.

Three distinct things can be checked here -- don't confuse them:

  --mode ml_driven   (default) run.py's new default: the calibrated
                      LOF-driven PRODUCTION pipeline (ml/official_pipeline.py)
                      that now writes submission/output.csv.
  --mode rule_based   run.py --mode rule_based: the original deterministic
                      +15pp-threshold pipeline, fully preserved, selectable
                      explicitly, output saved as a benchmark artifact.
  --experimental-ml   scripts/run_ml_evaluation.py: the separate, earlier,
                      side-by-side Isolation-Forest-vs-LOF-vs-baseline
                      COMPARISON (never wired into any official output) --
                      unrelated to which pipeline is production-primary.

Usage:
    python3 scripts/check_reproducibility.py                  # ml_driven (default)
    python3 scripts/check_reproducibility.py --mode rule_based
    python3 scripts/check_reproducibility.py --experimental-ml

Writes a report to docs/ml_driven_reproducibility_report.md,
docs/reproducibility_report.md, or
docs/experimental_ml_comparison_reproducibility_report.md
respectively, and exits non-zero if any run differs.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
N_RUNS = 3


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_run_py(mode: str, report_name: str, report_title: str) -> int:
    run_py = ROOT / "run.py"
    run_dirs = []
    for i in range(1, N_RUNS + 1):
        out_dir = ROOT / "output" / f"_repro_{mode}_run{i}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / "output.csv"
        ledger_csv = out_dir / "evidence_ledger.csv"
        result = subprocess.run(
            [sys.executable, str(run_py), "--mode", mode, "--output", str(out_csv), "--ledger", str(ledger_csv)],
            cwd=ROOT, capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"run {i} FAILED:\n{result.stderr}", file=sys.stderr)
            return 1
        run_dirs.append((out_csv, ledger_csv))

    output_hashes = [sha256(o) for o, _ in run_dirs]
    ledger_hashes = [sha256(l) for _, l in run_dirs]
    output_identical = len(set(output_hashes)) == 1
    ledger_identical = len(set(ledger_hashes)) == 1
    overall_pass = output_identical and ledger_identical

    lines = [
        f"# {report_title}",
        "",
        f"Command: `python3 {run_py.relative_to(ROOT)} --mode {mode}` run {N_RUNS} times in separate "
        "processes, same untouched inputs.",
        "",
        "## output.csv SHA-256 per run",
        "",
    ]
    for i, h in enumerate(output_hashes, start=1):
        lines.append(f"- run {i}: `{h}`")
    lines += ["", f"**Identical across all {N_RUNS} runs: {'YES' if output_identical else 'NO'}**", ""]

    lines += ["## evidence_ledger.csv SHA-256 per run", ""]
    for i, h in enumerate(ledger_hashes, start=1):
        lines.append(f"- run {i}: `{h}`")
    lines += ["", f"**Identical across all {N_RUNS} runs: {'YES' if ledger_identical else 'NO'}**", ""]

    if not output_identical:
        diff = subprocess.run(["diff", str(run_dirs[0][0]), str(run_dirs[1][0])], capture_output=True, text=True)
        lines += ["## output.csv diff (run 1 vs run 2)", "", "```", diff.stdout or "(no stdout)", "```", ""]

    randomness_note = (
        "Both the rule-based and ML-driven pipelines are fully deterministic: the rule-based path uses "
        "no randomness at all, and the ML-driven path fixes RANDOM_SEED=42 (ml/config.py) for every "
        "StandardScaler/LOF fit, with feature rows always explicitly sorted before any split -- no LLM "
        "calls, no unseeded sampling, in either path."
        if mode == "ml" else
        "No randomness is used anywhere in the rule-based pipeline (no LLM calls, no sampling, no "
        "wall-clock-dependent values in the computed fields) so exact equality is expected and enforced "
        "by this check."
    )
    lines += [
        "## Overall result",
        "",
        f"**{'PASS' if overall_pass else 'FAIL'}**: {N_RUNS} untouched runs on the same input "
        f"produced {'identical' if overall_pass else 'DIFFERING'} output.csv and evidence_ledger.csv "
        "(byte-for-byte, via SHA-256).",
        "",
        randomness_note,
    ]

    report_path = ROOT / "docs" / report_name
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {report_path}")
    print("PASS" if overall_pass else "FAIL")

    for out_csv, ledger_csv in run_dirs:
        out_csv.unlink()
        ledger_csv.unlink()
        out_csv.parent.rmdir()

    return 0 if overall_pass else 1


def check_experimental_ml_comparison() -> int:
    ml_script = ROOT / "scripts" / "run_ml_evaluation.py"
    run_dirs = []
    for i in range(1, N_RUNS + 1):
        out_dir = ROOT / "output" / f"_repro_experimental_ml_run{i}"
        out_dir.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            [sys.executable, str(ml_script), "--out-dir", str(out_dir)],
            cwd=ROOT, capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"ML run {i} FAILED:\n{result.stderr}", file=sys.stderr)
            return 1
        run_dirs.append(out_dir)

    pred_hashes = [sha256(d / "experimental_ml_predictions.csv") for d in run_dirs]
    cmp_hashes = [sha256(d / "experimental_ml_model_comparison.csv") for d in run_dirs]
    pred_identical = len(set(pred_hashes)) == 1
    cmp_identical = len(set(cmp_hashes)) == 1
    overall_pass = pred_identical and cmp_identical

    lines = [
        "# Experimental ML Comparison Reproducibility Report",
        "",
        "Checks scripts/run_ml_evaluation.py -- the separate, exploratory side-by-side comparison of "
        "Isolation Forest / LOF / the rule-based baseline (submission/ml_evaluation_report.md). This is "
        "NOT the production ML-driven pipeline (see docs/ml_driven_reproducibility_report.md for that).",
        "",
        f"Command: `python3 {ml_script.relative_to(ROOT)} --out-dir <tmp>` run {N_RUNS} times in "
        "separate processes, same untouched inputs, same RANDOM_SEED.",
        "",
        "## experimental_ml_predictions.csv SHA-256 per run",
        "",
    ]
    for i, h in enumerate(pred_hashes, start=1):
        lines.append(f"- run {i}: `{h}`")
    lines += ["", f"**Identical across all {N_RUNS} runs: {'YES' if pred_identical else 'NO'}**", ""]

    lines += ["## experimental_ml_model_comparison.csv SHA-256 per run", ""]
    for i, h in enumerate(cmp_hashes, start=1):
        lines.append(f"- run {i}: `{h}`")
    lines += ["", f"**Identical across all {N_RUNS} runs: {'YES' if cmp_identical else 'NO'}**", ""]

    if not pred_identical:
        diff = subprocess.run(
            ["diff", str(run_dirs[0] / "experimental_ml_predictions.csv"), str(run_dirs[1] / "experimental_ml_predictions.csv")],
            capture_output=True, text=True,
        )
        lines += ["## experimental_ml_predictions.csv diff (run 1 vs run 2)", "", "```", diff.stdout or "(no stdout)", "```", ""]

    lines += [
        "## Overall result",
        "",
        f"**{'PASS' if overall_pass else 'FAIL'}**: {N_RUNS} untouched runs on the same input and "
        "RANDOM_SEED produced "
        f"{'identical' if overall_pass else 'DIFFERING'} experimental_ml_predictions.csv and experimental_ml_model_comparison.csv "
        "(byte-for-byte, via SHA-256).",
        "",
        "`ml_evaluation_report.md`'s prose is intentionally excluded from this byte-identity check: it "
        "contains a measured wall-clock runtime line that varies run to run by construction -- every "
        "*number* in that report is re-derived from the same underlying `experimental_ml_predictions.csv` checked "
        "here, so this check extends to its content transitively.",
    ]

    report_path = ROOT / "docs" / "experimental_ml_comparison_reproducibility_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {report_path}")
    print("PASS" if overall_pass else "FAIL")

    for d in run_dirs:
        shutil.rmtree(d)

    return 0 if overall_pass else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="3-process reproducibility check")
    parser.add_argument(
        "--mode", choices=["ml_driven", "rule_based"], default="ml_driven",
        help="which run.py pipeline to check (default: ml_driven, matching run.py's own default)",
    )
    parser.add_argument(
        "--experimental-ml", action="store_true",
        help="check the separate experimental Isolation-Forest-vs-LOF-vs-baseline comparison instead "
             "(scripts/run_ml_evaluation.py) -- not the production pipeline",
    )
    args = parser.parse_args()

    if args.experimental_ml:
        return check_experimental_ml_comparison()
    if args.mode == "rule_based":
        return check_run_py("rule_based", "reproducibility_report.md", "Reproducibility Report (rule-based)")
    return check_run_py("ml", "ml_driven_reproducibility_report.md", "Reproducibility Report (ML-driven, production default)")


if __name__ == "__main__":
    sys.exit(main())
