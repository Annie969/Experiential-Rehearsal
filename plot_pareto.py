import os
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = os.environ.get('FIG_OUT_DIR', './figures')
os.makedirs(OUT_DIR, exist_ok=True)

# Data from Table 1: (model, dataset, method, ECE, Accuracy)
raw = [
    ("LLaMA-8B", "MMLU-Pro", "Uncalibrated", 0.6171, 0.2972),
    ("LLaMA-8B", "MMLU-Pro", "Self-Cal", 0.5993, 0.3162),
    ("LLaMA-8B", "MMLU-Pro", "Game-Sym", 0.5635, 0.2592),
    ("LLaMA-8B", "MMLU-Pro", "Game-Exp", 0.5391, 0.2815),
    ("LLaMA-70B", "MMLU-Pro", "Uncalibrated", 0.5985, 0.3870),
    ("LLaMA-70B", "MMLU-Pro", "Self-Cal", 0.5771, 0.4124),
    ("LLaMA-70B", "MMLU-Pro", "Game-Sym", 0.5382, 0.3611),
    ("LLaMA-70B", "MMLU-Pro", "Game-Exp", 0.5324, 0.3597),
    ("LLaMA-8B", "TriviaQA", "Uncalibrated", 0.2581, 0.6654),
    ("LLaMA-8B", "TriviaQA", "Self-Cal", 0.2512, 0.6899),
    ("LLaMA-8B", "TriviaQA", "Game-Sym", 0.2066, 0.6478),
    ("LLaMA-8B", "TriviaQA", "Game-Exp", 0.2039, 0.6489),
    ("LLaMA-70B", "TriviaQA", "Uncalibrated", 0.1762, 0.8028),
    ("LLaMA-70B", "TriviaQA", "Self-Cal", 0.1765, 0.8105),
    ("LLaMA-70B", "TriviaQA", "Game-Sym", 0.0965, 0.8145),
    ("LLaMA-70B", "TriviaQA", "Game-Exp", 0.0886, 0.8123),
    ("Qwen-7B", "MMLU-Pro", "Uncalibrated", 0.6396, 0.3086),
    ("Qwen-7B", "MMLU-Pro", "Self-Cal", 0.6344, 0.3181),
    ("Qwen-7B", "MMLU-Pro", "Game-Sym", 0.5438, 0.2795),
    ("Qwen-7B", "MMLU-Pro", "Game-Exp", 0.5100, 0.2929),
    ("Qwen-72B", "MMLU-Pro", "Uncalibrated", 0.5168, 0.4460),
    ("Qwen-72B", "MMLU-Pro", "Self-Cal", 0.5035, 0.4624),
    ("Qwen-72B", "MMLU-Pro", "Game-Sym", 0.3876, 0.4222),
    ("Qwen-72B", "MMLU-Pro", "Game-Exp", 0.4095, 0.4136),
    ("Qwen-7B", "TriviaQA", "Uncalibrated", 0.4526, 0.4964),
    ("Qwen-7B", "TriviaQA", "Self-Cal", 0.4325, 0.5278),
    ("Qwen-7B", "TriviaQA", "Game-Sym", 0.3868, 0.5014),
    ("Qwen-7B", "TriviaQA", "Game-Exp", 0.3446, 0.5014),
    ("Qwen-72B", "TriviaQA", "Uncalibrated", 0.2049, 0.7572),
    ("Qwen-72B", "TriviaQA", "Self-Cal", 0.1975, 0.7672),
    ("Qwen-72B", "TriviaQA", "Game-Sym", 0.1477, 0.7451),
    ("Qwen-72B", "TriviaQA", "Game-Exp", 0.1531, 0.7431),
    ("GPT-oss-20B", "MMLU-Pro", "Uncalibrated", 0.1128, 0.8687),
    ("GPT-oss-20B", "MMLU-Pro", "Self-Cal", 0.1520, 0.8280),
    ("GPT-oss-20B", "MMLU-Pro", "Game-Sym", 0.0631, 0.8738),
    ("GPT-oss-20B", "MMLU-Pro", "Game-Exp", 0.0693, 0.8739),
    ("GPT-oss-20B", "TriviaQA", "Uncalibrated", 0.3241, 0.6475),
    ("GPT-oss-20B", "TriviaQA", "Self-Cal", 0.3223, 0.6505),
    ("GPT-oss-20B", "TriviaQA", "Game-Sym", 0.3047, 0.6476),
    ("GPT-oss-20B", "TriviaQA", "Game-Exp", 0.3094, 0.6448),
]

# Build baseline lookup
baselines = {}
for model, dataset, method, ece, acc in raw:
    if method == "Uncalibrated":
        baselines[(model, dataset)] = (ece, acc)

# Compute deltas relative to Uncalibrated
deltas = []
for model, dataset, method, ece, acc in raw:
    if method == "Uncalibrated":
        continue
    base_ece, base_acc = baselines[(model, dataset)]
    delta_ece = base_ece - ece
    delta_acc = acc - base_acc
    deltas.append((model, dataset, method, delta_ece, delta_acc))

method_styles = {
    "Self-Cal":  {"color": "#e74c3c", "marker": "s"},
    "Game-Sym":  {"color": "#2ecc71", "marker": "^"},
    "Game-Exp":  {"color": "#3498db", "marker": "D"},
}

fig, ax = plt.subplots(figsize=(7, 5.5))

# Only plot mean markers, no individual scatter points
for method, style in method_styles.items():
    xs = [d[3] for d in deltas if d[2] == method]
    ys = [d[4] for d in deltas if d[2] == method]
    mx, my = np.mean(xs), np.mean(ys)

    ax.scatter(mx, my,
               c=style["color"], marker=style["marker"],
               s=250, edgecolors='black', linewidths=1.5,
               zorder=5, label=method)

    offsets = {"Self-Cal": (12, 12), "Game-Sym": (-15, -35), "Game-Exp": (15, 15)}
    ox, oy = offsets[method]
    ax.annotate(f'{method}\nECE Reduction: {mx:.3f}  Acc: {my:+.3f}',
                xy=(mx, my), xytext=(ox, oy),
                textcoords='offset points', fontsize=8,
                color=style["color"], fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=style["color"], lw=0.8),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          alpha=0.9, edgecolor=style["color"]))

# Mark origin as baseline
ax.plot(0, 0, 'o', color='#666666', markersize=8, zorder=4)
ax.annotate('Uncalibrated\n(baseline)', xy=(0, 0), xytext=(15, -20),
            textcoords='offset points', fontsize=8, color='#666666',
            arrowprops=dict(arrowstyle='->', color='#999999', lw=0.8))

ax.set_xlim(-0.05, 0.14)
ax.set_ylim(-0.05, 0.04)
ax.set_xlabel('ECE Reduction (higher is better)', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy Change', fontsize=12, fontweight='bold')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.0)
ax.spines['bottom'].set_linewidth(1.0)
ax.tick_params(labelsize=10)

ax.legend(fontsize=11, frameon=True, fancybox=True, edgecolor='#cccccc',
          loc='upper left', markerscale=0.5)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "pareto_ece_accuracy.pdf"),
            bbox_inches='tight', dpi=300)
plt.savefig(os.path.join(OUT_DIR, "pareto_ece_accuracy.png"),
            bbox_inches='tight', dpi=300)
print("Pareto plot saved.")
