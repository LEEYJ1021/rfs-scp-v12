#!/usr/bin/env python3
"""
annomi_eda.py — Exploratory Data Analysis for AnnoMI corpus

Run standalone:
    python src/eda/annomi_eda.py --annomi-dir data/annomi

Outputs written to rfs_v16_outputs/eda/.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--annomi-dir", type=str, default="data/annomi")
args, _ = parser.parse_known_args()

ANNOMI_DIR = Path(args.annomi_dir)
OUT_DIR = Path(os.environ.get("RFS_OUT_v16", "rfs_v16_outputs")) / "eda"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ANNOMI_FULL   = ANNOMI_DIR / "AnnoMI-full.csv"
ANNOMI_SIMPLE = ANNOMI_DIR / "AnnoMI-simple.csv"

print("[EDA] Loading AnnoMI …")
if ANNOMI_FULL.exists():
    df = pd.read_csv(ANNOMI_FULL)
    src = "full"
elif ANNOMI_SIMPLE.exists():
    df = pd.read_csv(ANNOMI_SIMPLE)
    src = "simple"
else:
    raise FileNotFoundError(f"AnnoMI CSV not found in {ANNOMI_DIR}")

print(f"  Source: AnnoMI-{src}  |  {len(df):,} utterances  |  "
      f"{df.transcript_id.nunique()} sessions")

# ── Session-level summary ─────────────────────────────────────────────────
sess_summary = df.groupby("transcript_id").agg(
    mi_quality=("mi_quality", "first"),
    topic=("topic", "first"),
    n_turns=("utterance_id", "count"),
    n_therapist=("interlocutor", lambda x: (x == "therapist").sum()),
    n_client=("interlocutor", lambda x: (x == "client").sum()),
).reset_index()

print(f"\n  MI quality distribution:")
print(sess_summary.mi_quality.value_counts().to_string())
print(f"\n  Turns per session (mean ± std): "
      f"{sess_summary.n_turns.mean():.1f} ± {sess_summary.n_turns.std():.1f}")
print(f"  Topic clusters: {sess_summary.topic.nunique()}")

# ── Figure: class balance + turn distribution ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
fig.suptitle("AnnoMI EDA — Session Overview", fontweight="bold")

vc = sess_summary.mi_quality.value_counts()
axes[0].bar(vc.index, vc.values,
            color=["#1E8449", "#C0392B"], alpha=0.85, edgecolor="white")
for i, v in enumerate(vc.values):
    axes[0].text(i, v + 0.5, str(v), ha="center", fontweight="bold")
axes[0].set_ylabel("Number of sessions")
axes[0].set_title("(A) MI quality label distribution")

axes[1].hist(sess_summary[sess_summary.mi_quality == "high"].n_turns,
             bins=20, alpha=0.65, color="#1E8449", label="High-MI", edgecolor="white")
axes[1].hist(sess_summary[sess_summary.mi_quality == "low"].n_turns,
             bins=20, alpha=0.65, color="#C0392B", label="Low-MI",  edgecolor="white")
axes[1].set_xlabel("Number of turns per session")
axes[1].set_ylabel("Count")
axes[1].set_title("(B) Turn count distribution by MI quality")
axes[1].legend()

fig.savefig(OUT_DIR / "eda_overview.png", dpi=150, bbox_inches="tight",
            facecolor="white")
plt.close()
print(f"\n  Figure saved: {OUT_DIR / 'eda_overview.png'}")

sess_summary.to_csv(OUT_DIR / "session_summary_eda.csv", index=False)
print(f"  CSV saved:   {OUT_DIR / 'session_summary_eda.csv'}")
print("[EDA] Done.")
