import re
import os
import numpy as np
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
})

STYLE = {
    'llama3.1-8b':  {'color': '#4878CF', 'label': 'LLaMA 3.1-8B',  'ls': '-',  'marker': 'o'},
    'llama3.1-70b': {'color': '#E8792B', 'label': 'LLaMA 3.1-70B', 'ls': '-',  'marker': 's'},
    'Qwen2.5-7b':  {'color': '#6ACC65', 'label': 'Qwen 2.5-7B',   'ls': '--', 'marker': '^'},
    'Qwen2.5-72b': {'color': '#D65F5F', 'label': 'Qwen 2.5-72B',  'ls': '--', 'marker': 'D'},
}


def extract_scores(file_path):
    scores = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            m = re.search(r'Total Score:\s+(-?\d+)', line)
            if m:
                scores.append(int(m.group(1)))
    return scores


def smooth(x, y, factor=5):
    x_new = np.linspace(x.min(), x.max(), len(x) * factor)
    spl = make_interp_spline(x, y, k=3)
    return x_new, spl(x_new)


def extract_model_key(filename):
    basename = os.path.basename(filename)
    return os.path.splitext(basename)[0].split('_')[0]


def plot_line(ax, file_list, title):
    for file_path in file_list:
        scores = extract_scores(file_path)
        key = extract_model_key(file_path)
        s = STYLE.get(key, {'color': '#333', 'label': key, 'ls': '-'})
        x = np.arange(1, len(scores) + 1)
        y = np.array(scores)
        xs, ys = smooth(x, y)

        ax.plot(xs, ys, color=s['color'], linewidth=1.0,
                linestyle=s['ls'], label=s['label'], zorder=2)
        ax.scatter(x[::5], y[::5], color=s['color'], s=28, alpha=0.85,
                   marker=s.get('marker', 'o'),
                   edgecolors='white', linewidths=0.6, zorder=3)

        # End-point annotation
        ax.annotate(f"{scores[-1]:,}",
                    xy=(x[-1], scores[-1]),
                    xytext=(5, 0), textcoords='offset points',
                    fontsize=8, fontweight='bold', color=s['color'],
                    va='center')

    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Question Number', fontsize=12, fontweight='bold', labelpad=5)
    ax.set_ylabel('Cumulative Score', fontsize=12, fontweight='bold', labelpad=5)
    ax.tick_params(axis='both', labelsize=10, direction='out', length=3)
    ax.axhline(y=0, color='#bbbbbb', linewidth=0.7, zorder=0)
    ax.yaxis.grid(True, linestyle=':', alpha=0.4, linewidth=0.5)
    ax.xaxis.grid(False)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(1.2)
        ax.spines[spine].set_color('#333333')


def main():
    hist = os.environ.get('GAME_HISTORY_DIR', './game_history')
    out  = os.environ.get('FIG_OUT_DIR', './figures')
    os.makedirs(out, exist_ok=True)

    sym_files = [
        os.path.join(hist, 'llama3.1-8b_20250609_202539.txt'),
        os.path.join(hist, 'llama3.1-70b_20250611_201414.txt'),
        os.path.join(hist, 'Qwen2.5-7b_20250616_184248.txt'),
        os.path.join(hist, 'Qwen2.5-72b_20250616_184812.txt'),
    ]

    exp_files = [
        os.path.join(hist, 'llama3.1-8b_20250701_204316_exp_score.txt'),
        os.path.join(hist, 'llama3.1-70b_20250706_180836_exp_score.txt'),
        os.path.join(hist, 'Qwen2.5-7b_20250701_203838_exp_score.txt'),
        os.path.join(hist, 'Qwen2.5-72b_20250706_181839_exp_score.txt'),
    ]

    # ── Side-by-side, wide aspect ratio for single-column layout ──
    # COLM text width is ~5.5 inches; figure at \linewidth = 5.5in
    # We render at 11in wide so when scaled to 5.5in, fonts stay readable
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    plot_line(ax1, sym_files, 'Symmetric Scoring')
    plot_line(ax2, exp_files, 'Exponential Scoring')

    # Shared legend between the two panels
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='upper center', ncol=4, fontsize=9.5,
               frameon=False, bbox_to_anchor=(0.5, 1.02),
               columnspacing=1.8, handlelength=2.0)

    fig.tight_layout(rect=[0, 0, 1, 0.92], w_pad=3.0)

    for ext in ['png', 'pdf']:
        path = os.path.join(out, f'score_comparison_combined.{ext}')
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f'[Saved] {path}')
    plt.close(fig)

    # ── Individual figures ──
    for file_list, title, name in [
        (sym_files, 'Symmetric Scoring', 'score_comparison_across_models_symm_large'),
        (exp_files, 'Exponential Scoring', 'score_comparison_across_models_expo_large'),
    ]:
        fig_s, ax_s = plt.subplots(figsize=(5.5, 4.8))
        plot_line(ax_s, file_list, title)
        ax_s.legend(fontsize=9, frameon=False, loc='upper left', handlelength=2.0)
        fig_s.tight_layout(pad=1.0)
        for ext in ['png', 'pdf']:
            fig_s.savefig(os.path.join(out, f'{name}.{ext}'),
                          dpi=300, bbox_inches='tight')
            print(f'[Saved] {name}.{ext}')
        plt.close(fig_s)


if __name__ == '__main__':
    main()
