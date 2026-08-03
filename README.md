# REHEARSE: Experiential Rehearsal for Verbal Confidence Calibration

Anonymized code release for the ACL/ARR submission
*"REHEARSE: Experiential Rehearsal for Verbal Confidence Calibration in Large Language Models"*.

REHEARSE is a training-free calibration method: the model plays a credence-calibration
game under a strictly proper log scoring rule, the game history is condensed into a
compact natural-language *trajectory prefix*, and at inference the prefix is prepended
to each question along with the CoT trigger `"Let's think step by step."`.

## Install

```bash
pip install -r requirements.txt

# For local vLLM serving (open-weight backbones as reported in the paper):
pip install "vllm>=0.5" "xgrammar>=0.1" "transformers>=4.44"

# API access:
export OPENAI_API_KEY=<key>       # for o3-mini (or any OpenAI-compatible endpoint)
```

## Reproducing the paper

`run_experiment.py` is the single, self-contained pipeline that produced every
number in the paper. It runs the credence game, builds the trajectory prefix,
and evaluates REHEARSE together with all inference-time baselines
(Uncalibrated, Top-K, Self-Cal, FaR, plus additional CoT and reflective
variants) in one process. Open-weight models are served with local vLLM on
A100-80GB (bf16); o3-mini is called through the OpenAI-compatible API.

```bash
# LLaMA-3.1-8B on MMLU-Pro, seed 42, local vLLM
python run_experiment.py --backend local \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --dataset mmlu_pro --seed 42 --n_test 500 --n_game 50

# o3-mini on GSM8K, seed 42, API
python run_experiment.py --backend api \
    --model o3-mini \
    --dataset gsm8k --seed 42 --n_test 500 --n_game 50
```

Each run writes a JSON to
`./logs/experiment_<model>_<dataset>_seed<S>_<timestamp>.json`
containing per-method aggregate metrics *and* the per-sample outputs
(`parsed_correct`, `parsed_conf`, `parsed`) for post-hoc auditing.

**Ablation sweeps** are controlled from the same script:
- `--n_game {10,25,50,100}` — game round count (Ablation 2)
- `--scoring_rule {log,symmetric}` — scoring rule (Ablation 3)

## Reproducing table numbers

```bash
python aggregate_metrics.py --json_dir ./logs \
    --model Llama-3.1-8B-Instruct --dataset mmlu_pro \
    --seeds 42 43 44 45 46 --method postgame_kojima
```

Prints per-seed metrics and mean +/- std. The aggregation rule matches the
paper exactly: unparsed items are counted as
`(parsed_correct=0, parsed_conf=0)`, ECE uses 10 equal-width bins, AUROC is
the pairwise rank-based estimator. Use a different `--method` to reproduce
any baseline row:

| `--method`         | Paper column        |
| ------------------ | ------------------- |
| `postgame_kojima`  | REHEARSE (Ours)     |
| `uncalibrated`     | Uncalibrated        |
| `topk`             | Top-K               |
| `selfcal`          | Self-Cal            |
| `far`              | FaR                 |
| `xiong_cot`        | Xiong CoT (App.)    |
| `kojima_cot`       | Kojima CoT (App.)   |

## Reproducing paper figures

```bash
export GAME_HISTORY_DIR=./game_history    # dir of *.txt game logs
export FIG_OUT_DIR=./figures
python plot_paper.py     # game-score trajectory curves
python plot_pareto.py    # ECE-vs-Accuracy Pareto plot
```

## Repo layout

```
run_experiment.py     # canonical paper pipeline (game + prefix + REHEARSE + all baselines)
aggregate_metrics.py  # reproduce paper's Acc/ECE/Brier/AUROC from run_experiment.py JSONs
plot_paper.py         # game-trajectory figure
plot_pareto.py        # ECE-Accuracy Pareto figure
requirements.txt
LICENSE
```

## Data

- **TruthfulQA MC1** (game only), loaded via HuggingFace `datasets`.
- **MMLU-Pro** (`TIGER-Lab/MMLU-Pro`), 500-item subsample per seed.
- **TriviaQA** (`mandarjoshi/trivia_qa`, `rc.wikipedia.nocontext`), 500-item subsample per seed.
- **GSM8K** (`openai/gsm8k`), 500-item subsample per seed.

Seeds 42-46 control both the subsample and the game item order.

## License

MIT (see `LICENSE`).
