"""CoT-matched baseline sweep (Ablation 4 and Appendix G in the paper).

Runs Top-K, Self-Cal, and FaR with a kojima-style CoT reasoning scaffold
added to the test-time prompt, so each baseline uses the same CoT trigger
that REHEARSE uses via run_postgame_kojima. This isolates the prefix
contribution from a possible CoT-asymmetry effect.

Design:
  - Prepend a kojima CoT instruction ("Let's think step by step.") to the
    primary inference prompt of each baseline.
  - Disable guided regex on that step so the model can emit reasoning
    tokens before the required answer format.
  - Increase max_tokens accordingly.
  - Reuse each baseline's parse function (parse_topk / parse_answer_conf /
    parse_far) unchanged.

Results saved as rebuttal_cotb_<method>_<model>_<ds>_seed<s>_*.json.
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

import argparse, json, os, time, glob
from datetime import datetime
from pathlib import Path

import run_experiment as rx

PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

COT_INSTRUCTION = "Let's think step by step, then follow the required format below.\n\n"


def _wrap_cot(orig_format_fn):
    """Return a format_fn that prepends the CoT instruction to the base prompt."""
    def wrapped(*args, **kwargs):
        return COT_INSTRUCTION + orig_format_fn(*args, **kwargs)
    return wrapped


def run_topk_cot(llm, samples):
    """Top-K with CoT scaffold. Uses batched inference with parse_topk."""
    print("[TopK+CoT] Running batch inference (guided_regex disabled)...", flush=True)
    any_open = any(rx.is_open_sample(s) for s in samples)
    if any_open:
        n = len(samples)
        correct = [0]*n; conf = [rx.MIN_CONF]*n; parsed = [False]*n; raw = [""]*n
        msgs = [[{"role":"system","content":rx.SYS_MSG},
                 {"role":"user","content": COT_INSTRUCTION + rx.format_question_topk_open(
                     s["question"], s["open_type"])}] for s in samples]
        responses = llm.batch_chat(msgs, max_tokens=1000)
        for i, (s, resp) in enumerate(zip(samples, responses)):
            raw[i] = resp
            corr, c = rx.parse_topk_open(resp, s)
            if c is not None:
                correct[i]=corr; conf[i]=c; parsed[i]=True
        print(f"  Parsed: {sum(parsed)}/{n}", flush=True)
        return correct, conf, parsed, raw
    # MC path — bypass guided regex so reasoning tokens can appear
    orig = llm.disable_guided
    llm.disable_guided = True
    try:
        return rx._batch_with_retry(llm, samples,
            format_fn=lambda s: COT_INSTRUCTION + rx.format_question_topk(s["question"], s["options"]),
            max_tokens=1000,
            guided_regex=None,
            parse_fn=rx.parse_topk)
    finally:
        llm.disable_guided = orig


def run_selfcal_cot(llm, samples):
    """Self-Cal with CoT scaffold added to step-1 (answer generation).
    Step-2 (verify) left unchanged — it's a short yes/no probe."""
    n = len(samples)
    correct = [0]*n; conf = [rx.MIN_CONF]*n; parsed = [False]*n; raw = [""]*n

    # Step 1 — CoT reasoning + answer + confidence; NO guided regex
    def _step1_fmt(s):
        base = (rx.format_question_open(s["question"], s["open_type"])
                if "open_type" in s else rx.format_question(s["question"], s["options"]))
        return COT_INSTRUCTION + base
    msgs1 = [[{"role":"system","content":rx.SYS_MSG},
              {"role":"user","content": _step1_fmt(s)}] for s in samples]
    print("[SelfCal+CoT] Step 1 (CoT reasoning)...", flush=True)
    resp1 = llm.batch_chat(msgs1, max_tokens=1000)  # no guided_regex

    answers = []
    open_answers = {}
    for i, (s, r) in enumerate(zip(samples, resp1)):
        raw[i] = f"[STEP1+CoT]\n{r}"
        if rx.is_open_sample(s):
            corr, _ = rx.parse_answer_open(r, s)
            import re as _re
            m = _re.search(r"Answer:\s*(.+?)\.?\s*Confidence:", r, _re.DOTALL | _re.IGNORECASE)
            ans_text = m.group(1).strip() if m else r.split("Confidence:")[0].strip()
            open_answers[i] = (corr, ans_text)
            answers.append(0)  # placeholder
        else:
            ans, _ = rx.parse_answer_conf(r)
            answers.append(ans)

    # Step 2 — verify (unchanged from run_selfcal)
    msgs2 = []; valid = []
    LETTERS = rx.LETTERS
    for i, (s, ans) in enumerate(zip(samples, answers)):
        if ans is None: continue
        if rx.is_open_sample(s):
            _, ans_text = open_answers[i]
            verify = (f"Question: {s['question']}\n\n"
                      f"Proposed answer: {ans_text}\n\n" + rx.SELF_CAL_VERIFY)
        else:
            choices_str = "\n".join(f"{LETTERS[j]}. {c}" for j, c in enumerate(s["options"]))
            verify = (f"Question: {s['question']}\nChoices:\n{choices_str}\n\n"
                      f"Proposed answer: {LETTERS[ans]}\n\n" + rx.SELF_CAL_VERIFY)
        msgs2.append([{"role":"system","content":rx.SYS_MSG},
                      {"role":"user","content":verify}])
        valid.append(i)
    print(f"[SelfCal+CoT] Step 2 verify ({len(msgs2)} valid)...", flush=True)
    resp2 = llm.batch_chat(msgs2, guided_regex=rx.YESNO_CONF_REGEX)

    for idx, r in zip(valid, resp2):
        raw[idx] = raw[idx] + f"\n[STEP2]\n{r}"
        is_yes, c = rx.parse_yes_no_conf(r)
        if is_yes is None or c is None: continue
        final = max(rx.MIN_CONF, min(0.99, c if is_yes else 1.0 - c))
        if rx.is_open_sample(samples[idx]):
            correct[idx] = open_answers[idx][0] if open_answers[idx][0] is not None else 0
        else:
            correct[idx] = int(answers[idx] == samples[idx]["answer_index"])
        conf[idx] = final; parsed[idx] = True
    print(f"  Parsed: {sum(parsed)}/{n}", flush=True)
    return correct, conf, parsed, raw


def run_far_cot(llm, samples):
    """FaR with CoT scaffold prepended. FaR already has fact/reflect structure
    but we prepend the uniform CoT trigger for consistency with the other two."""
    print("[FaR+CoT] Running batch inference (guided_regex disabled)...", flush=True)
    orig = llm.disable_guided
    llm.disable_guided = True
    try:
        return rx._batch_with_retry(llm, samples,
            format_fn=lambda s: (COT_INSTRUCTION + rx.format_question_far_open(s["question"], s["open_type"])
                                 if "open_type" in s else
                                 COT_INSTRUCTION + rx.format_question_far(s["question"], s["options"])),
            max_tokens=1200,
            guided_regex=None)  # free-form response; parse_answer_conf on tail
    finally:
        llm.disable_guided = orig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=["topk_cot", "selfcal_cot", "far_cot"])
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True, choices=["mmlu_pro", "triviaqa", "gsm8k"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--n_test", type=int, default=500)
    p.add_argument("--backend", choices=["local","api"], default="local")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--dp", type=int, default=1)
    p.add_argument("--max_model_len", type=int, default=32768)
    p.add_argument("--quantization", default=None)
    p.add_argument("--disable_guided", action="store_true")
    p.add_argument("--shuffle_seed", type=int, default=42)
    args = p.parse_args()

    model_short = args.model.split("/")[-1]
    print(f"=== Rebuttal CoTB {args.method.upper()} | {model_short} | {args.dataset} | seed{args.seed} ===",
          flush=True)

    test_samples, test_K = rx.load_test_data(args.dataset, args.seed, args.n_test)
    is_open = test_K is None
    if not is_open:
        rx.K_CHOICES = test_K; rx.MIN_CONF = 1.0 / test_K
    else:
        rx.K_CHOICES = 4; rx.MIN_CONF = 0.10
    print(f"[Loaded] {len(test_samples)} test samples (K={test_K})", flush=True)

    if args.backend == "local":
        llm = rx.LocalLLM(args.model, max_model_len=args.max_model_len,
                          tp_size=args.tp, quantization=args.quantization,
                          disable_guided=args.disable_guided, dp_size=args.dp)
    else:
        api_key = os.environ.get("OPENAI_API_KEY","")
        if not api_key: raise RuntimeError("Set OPENAI_API_KEY")
        llm = rx.APILLM(args.model, base_url="https://api.openai.com/v1", api_key=api_key, max_concurrency=5)

    t0 = time.time()
    if args.method == "topk_cot":
        corr, conf, parsed, resp = run_topk_cot(llm, test_samples)
    elif args.method == "selfcal_cot":
        corr, conf, parsed, resp = run_selfcal_cot(llm, test_samples)
    elif args.method == "far_cot":
        corr, conf, parsed, resp = run_far_cot(llm, test_samples)

    m = rx.compute_metrics(corr, conf, parsed, responses=resp, samples=test_samples)
    m["time"] = time.time() - t0
    m["method"] = args.method

    ds_tag = f"_{args.dataset}" if args.dataset != "mmlu_pro" else ""
    out = LOG_DIR / f"rebuttal_cotb_{args.method}_{model_short}{ds_tag}_seed{args.seed}_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(out, "w") as f:
        json.dump({
            "experiment": args.method,
            "model": args.model,
            "config": {"n_test": args.n_test, "seed": args.seed,
                       "dataset": args.dataset, "shuffle_seed": args.shuffle_seed},
            "results": {args.method: m},
        }, f, indent=2, default=str)

    print(f"\n[Saved] {out}", flush=True)
    auroc_str = "N/A" if m.get("auroc") is None else f"{m['auroc']:.3f}"
    print(f"\n[Summary] {args.method} | {model_short} | {args.dataset} | s{args.seed}: "
          f"Acc={m['accuracy']:.3f} ECE={m['ece']:.3f} Brier={m['brier']:.3f} "
          f"AUROC={auroc_str} (n={m['n']}, time={m['time']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
