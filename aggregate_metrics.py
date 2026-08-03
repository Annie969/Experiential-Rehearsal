"""Reproduce the paper's table numbers (Accuracy / ECE / Brier / AUROC) from
raw per-sample JSON outputs of `run_experiment.py`.

Aggregation rule (matches the paper):
- Iterate over the per-sample list at `data["results"][<method>]["per_sample"]`.
- Unparsed items (`parsed == False`) are treated as (parsed_correct=0, parsed_conf=0)
  when computing all four metrics. This "include-unparsed with zeroed conf"
  convention was used consistently across all cells in the paper.
- ECE uses 10 equal-width bins on [0, 1].
- AUROC is computed by pairwise comparison of correct vs wrong confidences
  (ties contribute 0.5).

Usage:
    python aggregate_metrics.py --json_dir ./logs \\
        --model Llama-3.1-8B-Instruct --dataset mmlu_pro \\
        --seeds 42 43 44 45 46 --method postgame_kojima

Method names correspond to the keys in `results` written by run_experiment.py:
  postgame_kojima : REHEARSE (Ours) - trajectory prefix + Kojima CoT trigger
  uncalibrated    : single-turn baseline
  selfcal         : Self-Cal (Kadavath 2022)
  topk            : Top-K (Tian et al., 2023)
  far             : FaR (Zhao et al., 2024)
  xiong_cot       : Xiong CoT baseline (Xiong et al., 2024)
  kojima_cot      : plain Kojima CoT (no prefix)

Prints per-seed metrics and mean +/- std across seeds.
"""
import argparse
import glob
import json
import math
import os
from statistics import mean


def compute_metrics(per_sample):
    """Compute (acc, ece, brier, auroc) from the paper's aggregation rule.

    Unparsed entries (`parsed=False`) contribute (correct=0, conf=0).
    """
    y, c = [], []
    for r in per_sample:
        if r.get("parsed") is False:
            y.append(0)
            c.append(0.0)
        else:
            y.append(int(r.get("parsed_correct", 0)))
            c.append(float(r.get("parsed_conf", 0.0)))

    n = len(y)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")

    acc = sum(y) / n
    brier = sum((c[i] - y[i]) ** 2 for i in range(n)) / n

    n_bins = 10
    bin_totals = [0] * n_bins
    bin_correct = [0] * n_bins
    bin_conf_sum = [0.0] * n_bins
    for i in range(n):
        b = min(int(c[i] * n_bins), n_bins - 1)
        bin_totals[b] += 1
        bin_correct[b] += y[i]
        bin_conf_sum[b] += c[i]
    ece = 0.0
    for b in range(n_bins):
        if bin_totals[b] == 0:
            continue
        avg_conf = bin_conf_sum[b] / bin_totals[b]
        acc_b = bin_correct[b] / bin_totals[b]
        ece += (bin_totals[b] / n) * abs(avg_conf - acc_b)

    pos = [c[i] for i in range(n) if y[i] == 1]
    neg = [c[i] for i in range(n) if y[i] == 0]
    if not pos or not neg:
        auroc = float("nan")
    else:
        gt = eq = 0
        for cp in pos:
            for cn in neg:
                if cp > cn:
                    gt += 1
                elif cp == cn:
                    eq += 1
        auroc = (gt + 0.5 * eq) / (len(pos) * len(neg))

    return acc, ece, brier, auroc


def std(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def load_seed_json(json_dir, model, dataset, seed):
    """Locate the FINAL (non-intermediate) JSON for one (model, dataset, seed)."""
    ds_tag = "" if dataset == "mmlu_pro" else f"_{dataset}"
    pat = os.path.join(
        json_dir, f"experiment_{model}{ds_tag}_seed{seed}_*.json"
    )
    candidates = [
        p for p in glob.glob(pat)
        if "INTERMEDIATE" not in os.path.basename(p)
        and "symm" not in os.path.basename(p)
        and "_M" not in os.path.basename(p).replace(model, "")
    ]
    if not candidates:
        raise FileNotFoundError(f"no FINAL JSON for {model}/{dataset}/seed{seed} under {json_dir}")
    return sorted(candidates)[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json_dir", required=True, help="directory of run_experiment.py JSONs")
    ap.add_argument("--model", required=True, help="e.g. Llama-3.1-8B-Instruct")
    ap.add_argument("--dataset", required=True, choices=["mmlu_pro", "triviaqa", "gsm8k"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--method", default="postgame_kojima",
                    help="key from results dict (default: postgame_kojima = REHEARSE Ours)")
    args = ap.parse_args()

    per_seed = []
    for s in args.seeds:
        path = load_seed_json(args.json_dir, args.model, args.dataset, s)
        with open(path) as f:
            data = json.load(f)
        results = data.get("results", {})
        if args.method not in results:
            raise KeyError(f"method '{args.method}' not in {os.path.basename(path)}. "
                           f"Available: {sorted(results.keys())}")
        per_sample = results[args.method].get("per_sample", [])
        per_seed.append(compute_metrics(per_sample))
        print(f"  seed {s}: {os.path.basename(path)}  -> "
              f"Acc={per_seed[-1][0]:.3f}  ECE={per_seed[-1][1]:.3f}  "
              f"Brier={per_seed[-1][2]:.3f}  AUROC={per_seed[-1][3]:.3f}")

    print(f"\n[{args.method}]  {len(per_seed)}-seed aggregate:")
    for i, name in enumerate(["Acc", "ECE", "Brier", "AUROC"]):
        vs = [row[i] for row in per_seed]
        print(f"  {name:5s}: {mean(vs):.3f} +/- {std(vs):.3f}")


if __name__ == "__main__":
    main()
