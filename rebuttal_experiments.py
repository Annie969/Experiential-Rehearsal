"""Ablation 4 experiments: prefix variants.

Sidecar to run_experiment.py. Each variant reuses an existing game
trajectory stored in logs/experiment_*.json (no game replay) and
produces logs/rebuttal_<variant>_*.json so the main-pipeline outputs
are untouched.

Variants reported in Sec. 4.6 and Appendix F of the paper:
  e1a: Fixed generic prompt ("You tend to over-report confidence ...").
  e1b: Adaptive generic prompt (direction read from game trajectory).
  e2 : Cross-model prefix (LLaMA test with Gemma prefix or vice versa).
  e3a: Shuffled correctness (permute Correct/Incorrect labels in prefix).
  e3b: Random scores (permute per-round log-rule scores).
  e4 : Model-authored 5-sentence compressed prefix.
"""
# THREAD-CAP: some multi-user login hosts cap per-user pids low enough
# that HuggingFace datasets' thread_map crashes; only applies when
# --backend api runs this script on a shared host.
import sys as _sys_cap
if "api" in _sys_cap.argv:
    import os as _os_cap
    for _k, _v in {"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1",
                   "NUMEXPR_NUM_THREADS":"1","RAYON_NUM_THREADS":"1","TOKENIZERS_PARALLELISM":"false",
                   "HF_DATASETS_DISABLE_PROGRESS_BAR":"1"}.items():
        _os_cap.environ.setdefault(_k, _v)
    import concurrent.futures as _cf
    _orig_tpe = _cf.ThreadPoolExecutor
    class _CappedTPE(_orig_tpe):
        def __init__(self, max_workers=None, *a, **kw):
            m = 2 if (max_workers is None or max_workers > 2) else max_workers
            super().__init__(max_workers=m, *a, **kw)
    _cf.ThreadPoolExecutor = _CappedTPE
# END THREAD-CAP

import argparse, json, os, random, re, time, glob
from datetime import datetime
from pathlib import Path

import run_experiment as rx

PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# Generic prompt templates (E1a, E1b)
#   Kept OUTSIDE POSTGAME_PREAMBLE to avoid the "past game" framing —
#   these are truly generic reminders, not game summaries.
# ─────────────────────────────────────────────
E1A_FIXED = (
    "[Calibration Reminder]\n"
    "Note: You tend to over-report your confidence when uncertain about your answer. "
    "When giving a confidence score, adjust it downward if you have any doubt about "
    "your correctness.\n"
    "---------------------\n\n"
    "[Current Task]\n"
    "Answer the following question carefully.\n"
)

E1B_TEMPLATES = {
    "overconfident": (
        "Note: You tend to over-report your confidence when uncertain. "
        "Adjust your confidence downward if you have any doubt about your correctness."
    ),
    "underconfident": (
        "Note: You tend to under-report your confidence even when your answer is correct. "
        "Adjust your confidence upward when you are confident."
    ),
    "well_calibrated": (
        "Note: Your reported confidence tends to align with your accuracy. "
        "Continue to report confidence honestly."
    ),
}
E1B_WRAPPER = (
    "[Calibration Reminder]\n"
    "{note}\n"
    "---------------------\n\n"
    "[Current Task]\n"
    "Answer the following question carefully.\n"
)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def load_existing_game_summary(model_short, dataset, seed):
    """Find newest FINAL JSON for (model, dataset, seed); return (summary, path)."""
    ds_tag = "" if dataset == "mmlu_pro" else f"_{dataset}"
    pat = str(LOG_DIR / f"experiment_{model_short}{ds_tag}_seed{seed}_*.json")
    files = [f for f in sorted(glob.glob(pat))
             if "INTERMEDIATE" not in os.path.basename(f)
             and "BACKUP" not in os.path.basename(f)
             and re.search(r"_M\d+", os.path.basename(f)) is None
             and "_symm" not in os.path.basename(f)]
    if not files:
        raise FileNotFoundError(f"No FINAL JSON for {model_short}/{dataset}/seed{seed}")
    path = files[-1]  # newest
    with open(path) as f:
        d = json.load(f)
    summary = d.get("game_summary")
    if not summary:
        raise ValueError(f"JSON {path} has no game_summary")
    return summary, path


def detect_direction(game_summary):
    """Extract calibration direction from game summary's final status."""
    tail = game_summary[-500:]  # look at end
    if "You are currently overconfident" in tail:
        return "overconfident"
    if "You are currently underconfident" in tail:
        return "underconfident"
    if "well-calibrated" in tail or "well calibrated" in tail:
        return "well_calibrated"
    return "overconfident"  # default (all our test models are)


def shuffle_correctness(game_summary, shuffle_seed):
    """Randomly permute Correct/Incorrect labels in Feedback lines."""
    rng = random.Random(shuffle_seed)
    pattern = re.compile(r'Feedback: (Correct|Incorrect)')
    tags = [m.group(1) for m in pattern.finditer(game_summary)]
    if not tags:
        return game_summary
    rng.shuffle(tags)
    idx = [0]
    def _repl(m):
        val = tags[idx[0]]
        idx[0] += 1
        return f'Feedback: {val}'
    return pattern.sub(_repl, game_summary)


def shuffle_scores(game_summary, shuffle_seed):
    """Randomly permute numeric per-round scores (score=+NN or score=-NN).
    Leaves correctness marks, running totals, accuracy, confidence, and cal tag intact —
    isolates whether the log-rule score signal itself carries calibration information
    beyond the ✓/✗ labels the model already sees."""
    import random as _rnd, re as _re
    rng = _rnd.Random(shuffle_seed)
    pattern = _re.compile(r"score=([+-]?\d+)")
    scores = [m.group(1) for m in pattern.finditer(game_summary)]
    if not scores:
        return game_summary
    rng.shuffle(scores)
    idx = [0]
    def _repl(m):
        val = scores[idx[0]]
        idx[0] += 1
        return "score=" + val
    return pattern.sub(_repl, game_summary)


def run_e3b(llm, samples, game_summary, shuffle_seed=42):
    shuffled = shuffle_scores(game_summary, shuffle_seed)
    print(f"[E3b Shuffled Scores] Original first 200 chars:\n  {game_summary[:200]}", flush=True)
    print(f"[E3b] Shuffled first 200 chars:\n  {shuffled[:200]}", flush=True)
    return rx.run_postgame_kojima(llm, samples, shuffled)


def compress_to_5_sentences(llm, game_summary):
    """Ask the same LLM to compress game_summary to 5 sentences."""
    prompt = (
        "Below is a record of a calibration game you played. In exactly 5 sentences, "
        "summarize your calibration behavior, focusing on: (1) overall accuracy vs "
        "reported confidence gap, (2) high-confidence errors, (3) low-confidence correct "
        "answers if any, (4) trend across rounds, (5) final calibration status "
        "(overconfident / underconfident / well-calibrated).\n\n"
        "--- GAME RECORD ---\n"
        f"{game_summary}\n"
        "--- END ---\n\n"
        "Provide exactly 5 sentences. Write plain prose without bullets, numbering, or headings."
    )
    msgs = [{"role": "user", "content": prompt}]
    results = llm.batch_chat([msgs], max_tokens=400)
    return results[0].strip()


# ─────────────────────────────────────────────
# Experiment functions
# ─────────────────────────────────────────────
def run_e1a(llm, samples):
    def fmt(s):
        return E1A_FIXED + (
            rx.format_question_kojima_open(s["question"], s["open_type"])
            if "open_type" in s
            else rx.format_question_kojima(s["question"], s["options"])
        )
    print("[E1a Fixed Generic] Running batch inference...", flush=True)
    return rx._batch_with_retry(llm, samples, format_fn=fmt, max_tokens=500)


def run_e1b(llm, samples, game_summary):
    direction = detect_direction(game_summary)
    print(f"[E1b Adaptive] Direction: {direction}", flush=True)
    preamble = E1B_WRAPPER.format(note=E1B_TEMPLATES[direction])
    def fmt(s):
        return preamble + (
            rx.format_question_kojima_open(s["question"], s["open_type"])
            if "open_type" in s
            else rx.format_question_kojima(s["question"], s["options"])
        )
    return rx._batch_with_retry(llm, samples, format_fn=fmt, max_tokens=500)


def run_e2(llm, samples, cross_summary):
    print("[E2 Cross-Model Prefix] Running with other-model prefix...", flush=True)
    return rx.run_postgame_kojima(llm, samples, cross_summary)


def run_e3a(llm, samples, game_summary, shuffle_seed=42):
    shuffled = shuffle_correctness(game_summary, shuffle_seed)
    print(f"[E3a Shuffled Correctness] Original first 200 chars:\n  {game_summary[:200]}", flush=True)
    print(f"[E3a] Shuffled first 200 chars:\n  {shuffled[:200]}", flush=True)
    return rx.run_postgame_kojima(llm, samples, shuffled)


def run_e4(llm, samples, game_summary):
    print(f"[E4 Compressed] Summarizing {len(game_summary)}-char prefix...", flush=True)
    compressed = compress_to_5_sentences(llm, game_summary)
    print(f"[E4] Compressed prefix ({len(compressed)} chars):\n{compressed}\n", flush=True)
    return rx.run_postgame_kojima(llm, samples, compressed)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", required=True, choices=["e1a", "e1b", "e2", "e3a", "e3b", "e4"])
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True, choices=["mmlu_pro", "triviaqa", "gsm8k"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--n_test", type=int, default=500)
    p.add_argument("--backend", choices=["local", "api"], default="local")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--dp", type=int, default=1)
    p.add_argument("--max_model_len", type=int, default=32768)
    p.add_argument("--quantization", default=None)
    p.add_argument("--disable_guided", action="store_true")
    p.add_argument("--cross_model_short", default=None,
                   help="For e2: short name of other model whose summary to use")
    p.add_argument("--shuffle_seed", type=int, default=42,
                   help="For e3a: seed for correctness label permutation")
    args = p.parse_args()

    model_short = args.model.split("/")[-1]
    print(f"=== Rebuttal {args.experiment.upper()} | {model_short} | {args.dataset} | seed{args.seed} ===",
          flush=True)

    # Load required game_summary
    src_summary = None; src_path = None
    if args.experiment in ("e1b", "e3a", "e3b", "e4"):
        src_summary, src_path = load_existing_game_summary(model_short, args.dataset, args.seed)
        print(f"[Loaded] game_summary from {os.path.basename(src_path)} ({len(src_summary)} chars)",
              flush=True)

    cross_summary = None
    if args.experiment == "e2":
        assert args.cross_model_short, "e2 requires --cross_model_short"
        cross_summary, src_path = load_existing_game_summary(args.cross_model_short, args.dataset, args.seed)
        print(f"[Loaded cross] game_summary from {os.path.basename(src_path)} ({len(cross_summary)} chars)",
              flush=True)

    # Load test samples (matches main pipeline's subsampling exactly via same seed)
    test_samples, test_K = rx.load_test_data(args.dataset, args.seed, args.n_test)
    is_open = test_K is None
    if not is_open:
        rx.K_CHOICES = test_K
        rx.MIN_CONF = 1.0 / test_K
    else:
        rx.K_CHOICES = 4
        rx.MIN_CONF = 0.10
    print(f"[Loaded] {len(test_samples)} test samples (K={test_K})", flush=True)

    # Instantiate LLM
    if args.backend == "local":
        llm = rx.LocalLLM(args.model, max_model_len=args.max_model_len,
                          tp_size=args.tp, quantization=args.quantization,
                          disable_guided=args.disable_guided, dp_size=args.dp)
    else:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("Set OPENAI_API_KEY")
        llm = rx.APILLM(args.model, base_url="https://api.openai.com/v1", api_key=api_key, max_concurrency=5)

    # Dispatch
    t0 = time.time()
    if args.experiment == "e1a":
        corr, conf, parsed, resp = run_e1a(llm, test_samples)
    elif args.experiment == "e1b":
        corr, conf, parsed, resp = run_e1b(llm, test_samples, src_summary)
    elif args.experiment == "e2":
        corr, conf, parsed, resp = run_e2(llm, test_samples, cross_summary)
    elif args.experiment == "e3a":
        corr, conf, parsed, resp = run_e3a(llm, test_samples, src_summary, args.shuffle_seed)
    elif args.experiment == "e3b":
        corr, conf, parsed, resp = run_e3b(llm, test_samples, src_summary, args.shuffle_seed)
    elif args.experiment == "e4":
        corr, conf, parsed, resp = run_e4(llm, test_samples, src_summary)

    m = rx.compute_metrics(corr, conf, parsed, responses=resp, samples=test_samples)
    m["time"] = time.time() - t0
    m["experiment"] = args.experiment
    m["source_json"] = str(src_path) if src_path else None
    m["cross_model_short"] = args.cross_model_short

    ds_tag = f"_{args.dataset}" if args.dataset != "mmlu_pro" else ""
    out = LOG_DIR / f"rebuttal_{args.experiment}_{model_short}{ds_tag}_seed{args.seed}_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(out, "w") as f:
        json.dump({
            "experiment": args.experiment,
            "model": args.model,
            "config": {"n_test": args.n_test, "seed": args.seed,
                       "dataset": args.dataset, "shuffle_seed": args.shuffle_seed,
                       "cross_model_short": args.cross_model_short},
            "results": {args.experiment: m},
        }, f, indent=2, default=str)

    print(f"\n[Saved] {out}", flush=True)
    auroc_str = "N/A" if m.get("auroc") is None else f"{m['auroc']:.3f}"
    print(f"[Summary] {args.experiment} | {model_short} | {args.dataset} | s{args.seed}: "
          f"Acc={m['accuracy']:.3f} ECE={m['ece']:.3f} Brier={m['brier']:.3f} "
          f"AUROC={auroc_str} (n={m['n']}, time={m['time']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
