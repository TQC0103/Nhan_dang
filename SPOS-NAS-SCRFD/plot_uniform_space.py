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
print(f"Loaded {len(rows_main)} architectures from Main.")

fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor='#0d1117')

stages = ['C2', 'C3', 'C4', 'C5']
color_all = '#bc8cff'  # Purple-ish for the general search space

# Check if space is uniform
for i, stage in enumerate(stages):
    ax = axes[i // 2, i % 2]
    ax.set_facecolor('#161b22')
    
    # Get index values (0-59) for ALL 1000 models
    idx_main = [r[f'{stage}_idx'] for r in rows_main]
    
    bins = np.arange(-0.5, 60.5, 1)
    
    ax.hist(idx_main, bins=bins, color=color_all, alpha=0.8, edgecolor='#0d1117', linewidth=0.5)
    
    # Add a horizontal line representing perfect uniform distribution (1000 / 60 = 16.6)
    expected_uniform = len(rows_main) / 60
    ax.axhline(expected_uniform, color='#f85149', linestyle='--', alpha=0.8, label=f'Perfect Uniform ({expected_uniform:.1f})')
    
    ax.set_title(f'{stage} Search Space GFLOPs Distribution (All 1,000 Archs)', fontsize=12, fontweight='bold', color='#e6edf3', pad=10)
    ax.set_xlabel('Configuration Index (0 to 59)', color='#e6edf3', fontsize=10)
    ax.set_ylabel('Frequency (Count)', color='#e6edf3', fontsize=10)
    
    ax.tick_params(colors='#8b949e', labelsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#30363d')
    ax.spines['left'].set_color('#30363d')
    ax.grid(True, alpha=0.2, color='#8b949e', linestyle='--')
    ax.set_ylim(0, max(40, max(np.histogram(idx_main, bins=bins)[0]) + 10))
    ax.legend(facecolor='#21262d', edgecolor='#30363d', labelcolor='#e6edf3')

plt.tight_layout(pad=3.0)
fig.savefig('fig_search_space_uniformity.png', dpi=150, facecolor='#0d1117')
print("Successfully generated fig_search_space_uniformity.png")
