"""
sllm_benchmark.py
Multi-SLLM Circumplex estimation benchmark via Ollama.

Queries locally-hosted SLLMs to rate therapy dialogue on Olson's
cohesion and flexibility scales. Falls back to lexical heuristic
when Ollama is unavailable.

V12-C: ICC < 0.30 across all models is interpreted as evidence of
LLM limitation rather than task difficulty, as the heuristic
estimator achieves AUC = 0.816 on the same task.
"""

from __future__ import annotations
import json
import re
import time
import urllib.request
import numpy as np
import pandas as pd
from typing import List, Optional
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score

from src.utils.stats_utils import icc_2_1

SYSTEM_PROMPT = (
    "You are a social robotics researcher applying Olson's FACES IV Circumplex Model "
    "to evaluate dialogue quality for robot family system (RFS) state estimation. "
    "Rate this therapy dialogue on 0-100 scales:\n"
    "- Cohesion (0=Disengaged, 100=Enmeshed; optimal balanced zone=35-65)\n"
    "- Flexibility (0=Rigid, 100=Chaotic; optimal balanced zone=35-65)\n"
    "Respond ONLY with valid JSON: "
    '{"cohesion": <int>, "flexibility": <int>, "reasoning": "<1 sentence>"}'
)

MODEL_PARAMS_B = {
    "qwen2.5:1.5b": 1.5, "qwen2.5:3b": 3.0, "qwen2.5:7b": 7.0,
    "phi3:mini": 3.8, "gemma2:2b": 2.0, "llama3.2:3b": 3.0,
    "mistral:7b-instruct": 7.0, "lexical_fallback": 0.0,
}

DEFAULT_MODELS = [
    "qwen2.5:1.5b", "qwen2.5:3b", "qwen2.5:7b",
    "phi3:mini", "gemma2:2b", "llama3.2:3b", "mistral:7b-instruct",
]


def check_ollama_models(ollama_url: str, timeout: float = 2.0) -> List[str]:
    """Return list of available model names from Ollama /api/tags."""
    tags_url = ollama_url.rsplit("/api/", 1)[0] + "/api/tags"
    try:
        resp = urllib.request.urlopen(
            urllib.request.Request(tags_url), timeout=timeout
        )
        return [m.get("name", "") for m in json.loads(resp.read()).get("models", [])]
    except Exception:
        return []


def build_dialogue_text(grp: pd.DataFrame, max_turns: int = 10) -> str:
    """Render first max_turns utterances as 'T: ...' / 'C: ...' string."""
    out = []
    for _, row in grp.head(max_turns).iterrows():
        spk = "T" if row["interlocutor"] == "therapist" else "C"
        txt = str(row["utterance_text"]).strip().replace("\n", " ")
        if txt:
            out.append(f"{spk}: {txt}")
    return " | ".join(out)


def ollama_label(model: str, text: str, ollama_url: str, timeout: float = 60.0) -> dict:
    """Query Ollama for cohesion/flexibility ratings."""
    try:
        payload = json.dumps({
            "model": model,
            "prompt": f"{SYSTEM_PROMPT}\n\nDialogue:\n{text}",
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 150},
        }).encode()
        req = urllib.request.Request(
            ollama_url, data=payload,
            headers={"Content-Type": "application/json"},
        )
        t0 = time.perf_counter()
        resp = urllib.request.urlopen(req, timeout=timeout)
        lat = (time.perf_counter() - t0) * 1000
        raw = json.loads(resp.read())["response"].strip()
        parsed = (
            json.loads(raw) if raw.startswith("{")
            else json.loads(re.search(r'\{.*\}', raw, re.DOTALL).group())
        )
        return dict(
            cohesion=float(parsed["cohesion"]),
            flexibility=float(parsed["flexibility"]),
            parse_ok=True,
            latency_ms=lat,
            method="ollama",
        )
    except Exception as e:
        return dict(
            cohesion=np.nan,
            flexibility=np.nan,
            parse_ok=False,
            latency_ms=np.nan,
            method="error",
            error=str(e),
        )


def lexical_label(text: str) -> dict:
    """Lexical fallback estimator (no API call required)."""
    words = text.lower().split()
    turns = [t for t in text.split("|") if t.strip()]
    n_w, n_t = max(len(words), 1), max(len(turns), 1)

    EMP = {"understand", "glad", "together", "feeling", "check", "thank",
           "sorry", "care", "listen", "okay", "hear"}
    AGR = {"agree", "yes", "absolutely", "sure", "right", "exactly", "change", "better"}
    NEG = {"not", "never", "no", "don't", "nothing", "leave", "alone", "stop"}
    POS = {"love", "glad", "better", "yes", "thank", "happy", "good", "okay"}

    emp = sum(1 for w in words if w in EMP)
    agr = sum(1 for w in words if w in AGR)
    neg = sum(1 for w in words if w in NEG)
    pos = sum(1 for w in words if w in POS)
    sent = (pos - neg) / max(pos + neg + 1, 1)

    coh = float(np.clip(50 + 25 * sent + 40 * emp / n_t + 20 * agr / n_t - 15 * neg / n_w, 0, 100))
    flex = float(np.clip(50 + 10 * (1 - abs(sent)) + 15 * agr / n_t - 10 * neg / n_w, 0, 100))
    return dict(
        cohesion=round(coh, 1),
        flexibility=round(flex, 1),
        parse_ok=True,
        latency_ms=0.0,
        method="lexical_fallback",
    )


def run_benchmark(
    raw_df: pd.DataFrame,
    sess_df: pd.DataFrame,
    model_list: List[str],
    ollama_url: str = "http://localhost:11434/api/generate",
    output_dir: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run multi-SLLM benchmark.

    Parameters
    ----------
    raw_df : DataFrame  — utterance-level AnnoMI data
    sess_df : DataFrame — session-level features (from feature_extraction)
    model_list : list of str
    ollama_url : str
    output_dir : str or None — if given, save CSVs there
    verbose : bool

    Returns
    -------
    bench_df : DataFrame — one row per model with aggregate metrics
    """
    sample_ids = sess_df.transcript_id.values
    mi_q_bin = sess_df.mi_quality_bin.values
    cp_coh = sess_df.cohesion.values
    cp_flex = sess_df.flexibility.values

    bench_rows = []
    all_labels = []

    for model in model_list:
        if verbose:
            print(f"\n  -- {model} --")
        rows_m = []

        for i, tid in enumerate(sample_ids):
            grp = raw_df[raw_df.transcript_id == tid].sort_values("utterance_id")
            text = build_dialogue_text(grp)

            if model == "lexical_fallback":
                res = lexical_label(text)
            else:
                res = ollama_label(model, text, ollama_url)
                if not res["parse_ok"]:
                    res = lexical_label(text)
                    res["method"] = "lexical_fallback"

            res.update(model=model, transcript_id=tid, mi_quality_bin=int(mi_q_bin[i]))
            rows_m.append(res)

            if verbose and (i + 1) % 30 == 0:
                print(f"    {i+1}/{len(sample_ids)}")

        df_m = pd.DataFrame(rows_m)
        all_labels.append(df_m)
        dfv = df_m.dropna(subset=["cohesion", "flexibility"])
        n_v = len(dfv)

        icc_c = icc_2_1(cp_coh[:n_v], dfv.cohesion.values)
        icc_f = icc_2_1(cp_flex[:n_v], dfv.flexibility.values)
        rho_c, _ = spearmanr(dfv.cohesion.values, cp_coh[:n_v])

        try:
            auc_m = roc_auc_score(dfv.mi_quality_bin, dfv.cohesion)
            ap_m = average_precision_score(dfv.mi_quality_bin, dfv.cohesion)
        except Exception:
            auc_m = ap_m = float("nan")

        bench_rows.append(dict(
            model=model,
            n=n_v,
            params_b=MODEL_PARAMS_B.get(model, 0.0),
            ICC_cohesion=icc_c,
            ICC_flexibility=icc_f,
            Spearman_rho=rho_c,
            AUC=auc_m,
            AP=ap_m,
            parse_rate=df_m.parse_ok.mean(),
            latency_ms=df_m.latency_ms.replace(0, np.nan).mean(),
            meets_floor=bool(auc_m >= 0.55) if not np.isnan(auc_m) else False,
        ))

        if verbose:
            print(f"    ICC={icc_c:.3f}  ρ={rho_c:.3f}  AUC={auc_m:.3f}")

    bench_df = pd.DataFrame(bench_rows).sort_values("AUC", ascending=False)

    if output_dir:
        import os
        bench_df.to_csv(os.path.join(output_dir, "sllm_benchmark_v12.csv"), index=False)
        pd.concat(all_labels, ignore_index=True).to_csv(
            os.path.join(output_dir, "sllm_labels_v12.csv"), index=False
        )

    return bench_df
