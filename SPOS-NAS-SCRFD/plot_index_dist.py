import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def load_data(filepath):
    rows = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                'C2_idx': int(row['C2_idx']),
                'C3_idx': int(row['C3_idx']),
                'C4_idx': int(row['C4_idx']),
                'C5_idx': int(row['C5_idx'])
            })
    return rows

rows_main = load_data('nas_leaderboard-dynamic.csv')
top100_main = rows_main[:100]

rows_edge = load_data('nas_leaderboard_edge_focus.csv')
top100_edge = rows_edge[:100]

# We will plot a 2x2 grid for C2, C3, C4, C5 showing Top 100 Index Frequency.
fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor='#0d1117')

stages = ['C2', 'C3', 'C4', 'C5']
colors_main = '#58a6ff'
colors_edge = '#f85149'

for i, stage in enumerate(stages):
    ax = axes[i // 2, i % 2]
    ax.set_facecolor('#161b22')
    
    # Get index values (0-59)
    idx_main = [r[f'{stage}_idx'] for r in top100_main]
    idx_edge = [r[f'{stage}_idx'] for r in top100_edge]
    
    bins = np.arange(-0.5, 60.5, 1)
    
    ax.hist(idx_main, bins=bins, color=colors_main, alpha=0.7, label='Center-Focus (Main)', edgecolor='none', density=False)
    ax.hist(idx_edge, bins=bins, color=colors_edge, alpha=0.5, label='Edge-Focus (Appendix)', edgecolor='none', density=False)
    
    ax.set_title(f'{stage} Index Configuration Frequency (Top-100)', fontsize=12, fontweight='bold', color='#e6edf3', pad=10)
    ax.set_xlabel('Configuration Index (0 to 59)', color='#e6edf3', fontsize=10)
    ax.set_ylabel('Frequency (Count)', color='#e6edf3', fontsize=10)
    
    ax.tick_params(colors='#8b949e', labelsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#30363d')
    ax.spines['left'].set_color('#30363d')
    ax.grid(True, alpha=0.2, color='#8b949e', linestyle='--')
    ax.legend(facecolor='#21262d', edgecolor='#30363d', labelcolor='#e6edf3')

plt.tight_layout(pad=3.0)
fig.savefig('fig_index_distribution.png', dpi=150, facecolor='#0d1117')
print("Successfully generated fig_index_distribution.png")
