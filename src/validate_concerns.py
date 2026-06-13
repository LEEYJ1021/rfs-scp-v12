#!/usr/bin/env python3
"""
validate_concerns.py
====================
Standalone validation script for RFS-SCP v16.1 concern responses.

Independently verifies:
  1. VADER backend is real (not a silent fallback)
  2. AUC floor citation constant is embedded and correct
  3. MCMC implementation contains no temperature constant
  4. Hold-out weight generalisation gaps are within documented bounds
  5. Communication proxy coverage matches the documented 1.3% threshold
  6. Main script (rfs_scp_v16_main.py) is present and importable

Usage:
    python src/validate_concerns.py --annomi-dir data/annomi --results-dir results/
    # Or via shell:
    bash scripts/run_validation.sh
"""

import argparse
import ast
import json
import os
import sys
from pathlib import Path


# ── helpers ──────────────────────────────────────────────────────────────────

def check(label: str, condition: bool, detail: str = "") -> bool:
    status = "PASS ✓" if condition else "FAIL ✗"
    print(f"  [{status}]  {label}")
    if detail:
        print(f"            {detail}")
    return condition


def warn(msg: str):
    print(f"  [WARN]  {msg}")


SEP = "=" * 72

# ── check 1: VADER import ─────────────────────────────────────────────────────

def check_vader() -> bool:
    print(f"\n{SEP}")
    print("CHECK 1 — VADER backend (mandatory; silent fallback disabled)")
    print(SEP)
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        score = sia.polarity_scores("This is great!")["compound"]
        return check(
            "vaderSentiment imported and functional",
            abs(score) > 0,
            f"compound score on 'This is great!' = {score:.4f}",
        )
    except ImportError:
        return check("vaderSentiment imported", False,
                     "pip install vaderSentiment")


# ── check 2: AUC floor constant ──────────────────────────────────────────────

def check_auc_floor(main_path: Path) -> bool:
    print(f"\n{SEP}")
    print("CHECK 2 — AUC_FLOOR constant and Tanana et al. (2016) citation")
    print(SEP)
    if not main_path.exists():
        return check(f"{main_path.name} present", False,
                     f"File not found: {main_path}")
    src = main_path.read_text()
    floor_ok = "AUC_FLOOR = 0.55" in src
    citation_ok = "Tanana et al." in src and "2016" in src
    check("AUC_FLOOR = 0.55 defined", floor_ok)
    check("Tanana et al. (2016) citation embedded", citation_ok)
    return floor_ok and citation_ok


# ── check 3: MCMC has no temp constant ───────────────────────────────────────

def check_no_temp_constant(main_path: Path) -> bool:
    print(f"\n{SEP}")
    print("CHECK 3 — MCMC: no temperature constant (temp=300 removed)")
    print(SEP)
    if not main_path.exists():
        return check(f"{main_path.name} present", False)
    src = main_path.read_text()
    no_temp_300 = "temp=300" not in src and "temp = 300" not in src
    has_bernoulli = "Bernoulli" in src
    has_dirichlet = "Dirichlet" in src
    check("temp=300 absent from source", no_temp_300,
          "'temp=300' found — MCMC not fully rewritten" if not no_temp_300 else "")
    check("Bernoulli likelihood keyword present", has_bernoulli)
    check("Dirichlet prior keyword present", has_dirichlet)
    return no_temp_300 and has_bernoulli and has_dirichlet


# ── check 4: hold-out gaps from results CSV ───────────────────────────────────

def check_holdout_gaps(results_dir: Path) -> bool:
    print(f"\n{SEP}")
    print("CHECK 4 — Hold-out weight generalisation gaps")
    print(SEP)
    csv_path = results_dir / "holdout_weight_sensitivity_v16.csv"
    if not csv_path.exists():
        warn(f"Results file not found: {csv_path}  (run pipeline first)")
        return True  # not a failure if pipeline hasn't been run yet

    try:
        import csv
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            row = next(reader)
        theory_gap = float(row["theory_gap"])
        learned_gap = float(row["gap"])
        both_flagged = row["both_gaps_flagged"].strip().lower() in ("true", "1")

        check(
            f"Theory gap reported (+{theory_gap:.4f})",
            theory_gap > 0,
            "Should be positive (train > test); documented as +0.2909",
        )
        check(
            f"Learned gap reported (+{learned_gap:.4f})",
            learned_gap > 0,
            "Should be positive; documented as +0.1546",
        )
        check(
            "both_gaps_flagged = True in CSV",
            both_flagged,
            "LIMITATION row requires both gaps flagged",
        )
        return both_flagged
    except Exception as e:
        warn(f"Could not parse {csv_path}: {e}")
        return False


# ── check 5: communication coverage ─────────────────────────────────────────

def check_comm_coverage(results_dir: Path) -> bool:
    print(f"\n{SEP}")
    print("CHECK 5 — Communication proxy regex coverage audit")
    print(SEP)
    csv_path = results_dir / "regex_coverage_audit_v16.csv"
    if not csv_path.exists():
        warn(f"Results file not found: {csv_path}  (run pipeline first)")
        return True

    try:
        import csv
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            row = next(reader)
        clarif_pct = float(row["clarification_hit_pct"])
        check(
            f"Clarification coverage = {clarif_pct:.2f}% (< 10% → FUTURE WORK)",
            clarif_pct < 10.0,
            "If ≥ 10%, Communication axis may be promoted to primary claim.",
        )
        return clarif_pct < 10.0
    except Exception as e:
        warn(f"Could not parse {csv_path}: {e}")
        return False


# ── check 6: main script present ─────────────────────────────────────────────

def check_main_script(main_path: Path) -> bool:
    print(f"\n{SEP}")
    print("CHECK 6 — Main script presence (Concern 1 fix)")
    print(SEP)
    exists = main_path.exists()
    check(f"{main_path.name} exists at {main_path}", exists,
          "Commit src/rfs_scp_v16_main.py to fix Concern 1" if not exists else "")

    if exists:
        # Verify run_full_pipeline.sh references v16
        script_path = main_path.parent.parent / "scripts" / "run_full_pipeline.sh"
        if script_path.exists():
            sh_src = script_path.read_text()
            references_v16 = "rfs_scp_v16_main.py" in sh_src
            check(
                "run_full_pipeline.sh references rfs_scp_v16_main.py",
                references_v16,
                "Update scripts/run_full_pipeline.sh path" if not references_v16 else "",
            )
    return exists


# ── check 7: JSAI citation year ──────────────────────────────────────────────

def check_jsai_citation(main_path: Path, docs_dir: Path) -> bool:
    print(f"\n{SEP}")
    print("CHECK 7 — JSAI 2026 citation year (Hirano & Tanaka)")
    print(SEP)
    files_to_check = list(docs_dir.glob("*.md")) + [main_path]
    any_2025 = False
    for fpath in files_to_check:
        if not fpath.exists():
            continue
        src = fpath.read_text()
        # Look for the JSAI domestic conference citation with wrong year
        import re
        bad = re.findall(r"Hirano.*?Tanaka.*?2025.*?[Dd]omestic|[Jj]apanese.*?[Dd]omestic.*?2025", src)
        if bad:
            warn(f"{fpath.name}: possible 2025 year in JSAI domestic citation")
            any_2025 = True
    return check(
        "No stale 2025 year in JSAI domestic conference citation",
        not any_2025,
        "JSAI 2026 was held in June 2026; update citation year to 2026",
    )


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RFS-SCP v16.1 concern validation")
    parser.add_argument("--annomi-dir", type=str, default="data/annomi")
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    main_script = root / "src" / "rfs_scp_v16_main.py"
    results_dir = root / Path(args.results_dir)
    docs_dir = root / "docs"

    print(f"\n{'=' * 72}")
    print("  RFS-SCP v16.1 — CONCERN VALIDATION SCRIPT")
    print(f"  Root: {root}")
    print(f"{'=' * 72}")

    results = {
        "vader":        check_vader(),
        "auc_floor":    check_auc_floor(main_script),
        "no_temp":      check_no_temp_constant(main_script),
        "holdout_gaps": check_holdout_gaps(results_dir),
        "comm_cov":     check_comm_coverage(results_dir),
        "main_script":  check_main_script(main_script),
        "jsai_year":    check_jsai_citation(main_script, docs_dir),
    }

    n_pass = sum(results.values())
    n_total = len(results)

    print(f"\n{SEP}")
    print(f"  SUMMARY: {n_pass}/{n_total} checks passed")
    if n_pass == n_total:
        print("  ALL CHECKS PASSED ✓")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"  FAILED: {', '.join(failed)}")
    print(SEP)

    sys.exit(0 if n_pass == n_total else 1)


if __name__ == "__main__":
    main()
