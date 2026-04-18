"""
SCRFD-SPOS NAS Visualization Suite
Generates all charts needed for the technical report.
"""
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# ─────────────────────────────────────────
# STYLE CONFIG
# ─────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'figure.facecolor': '#0d1117',
    'axes.facecolor':   '#161b22',
    'axes.edgecolor':   '#30363d',
    'axes.labelcolor':  '#e6edf3',
    'xtick.color':      '#8b949e',
    'ytick.color':      '#8b949e',
    'text.color':       '#e6edf3',
    'grid.color':       '#21262d',
    'grid.linestyle':   '--',
    'grid.linewidth':   0.6,
})

ACCENT = '#58a6ff'
GREEN  = '#3fb950'
ORANGE = '#d29922'
RED    = '#f85149'
PURPLE = '#bc8cff'
TEAL   = '#39d353'

# Known epoch training data (from notebook output)
TRAINING_DATA = [
    (1,  1.8963, 1.2034, 0.000199599),
    (2,  1.1431, 1.0095, 0.000198401),
    (3,  1.0626, 1.0545, 0.000196414),
    (4,  1.0082, 0.8887, 0.000193655),
    (5,  0.9576, 0.9139, 0.000190146),
    (6,  0.9217, 0.8545, 0.000185916),
    (7,  0.8865, 0.7827, 0.000180997),
    (8,  0.8622, 0.7813, 0.000175431),
    (9,  0.8435, 0.7591, 0.000169261),
    (10, 0.8292, 0.7508, 0.000162537),
    (11, 0.8003, 0.7475, 0.000155314),
    (12, 0.7861, 0.7004, 0.000147650),
    (13, 0.7657, 0.6739, 0.000139606),
    (14, 0.7548, 0.6820, 0.000131247),
    (15, 0.7465, 0.6741, 0.000122641),
    (16, 0.7333, 0.6548, 0.000113856),
    (17, 0.7188, 0.6604, 0.000104964),
    (18, 0.7213, 0.6519, 0.000096036),
]

# ─────────────────────────────────────────
# LOAD CSV
# ─────────────────────────────────────────
rows = []
with open(r'd:\Downloads\nhandangnotebook\nas_leaderboard-dynamic.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append({
            'arch_id': int(row['Arch_ID']),
            'val_loss': float(row['Val_Loss']),
            'C2_w': int(row['C2_w']), 'C2_d': int(row['C2_d']),
            'C3_w': int(row['C3_w']), 'C3_d': int(row['C3_d']),
            'C4_w': int(row['C4_w']), 'C4_d': int(row['C4_d']),
            'C5_w': int(row['C5_w']), 'C5_d': int(row['C5_d']),
        })

all_losses = [r['val_loss'] for r in rows]

# ─────────────────────────────────────────
# FIGURE 1: TRAINING CONVERGENCE CURVE
# ─────────────────────────────────────────
fig1, ax = plt.subplots(figsize=(12, 5), facecolor='#0d1117')
ax.set_facecolor('#161b22')

epochs = [d[0] for d in TRAINING_DATA]
train_losses = [d[1] for d in TRAINING_DATA]
val_losses   = [d[2] for d in TRAINING_DATA]
lrs          = [d[3] for d in TRAINING_DATA]

ax2 = ax.twinx()
ax2.set_facecolor('#161b22')

ax.plot(epochs, train_losses, 'o-', color=ACCENT, lw=2.2, ms=5, label='Train Loss', zorder=5)
ax.plot(epochs, val_losses,   's-', color=GREEN,  lw=2.2, ms=5, label='Val Loss',   zorder=5)
ax.fill_between(epochs, train_losses, val_losses, alpha=0.08, color=ACCENT)

ax2.plot(epochs, [lr*1e4 for lr in lrs], '--', color=ORANGE, lw=1.5, ms=4, label='LR (×10⁻⁴)', alpha=0.7)

# Mark best val loss
best_idx = val_losses.index(min(val_losses))
ax.annotate(f'  Best Val: {val_losses[best_idx]:.4f}\n  Epoch {epochs[best_idx]}',
            xy=(epochs[best_idx], val_losses[best_idx]),
            xytext=(epochs[best_idx]+1, val_losses[best_idx]+0.05),
            color=GREEN, fontsize=9,
            arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.2))

ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('Proxy Focal Loss', fontsize=11)
ax2.set_ylabel('Learning Rate (×10⁻⁴)', fontsize=10, color=ORANGE)
ax2.tick_params(colors=ORANGE)
ax.set_title('Supernet Training Convergence — Dynamic Weight Slicing (V2)', fontsize=13, fontweight='bold', pad=12, color='#e6edf3')
ax.grid(True, alpha=0.4)
ax.set_ylim(0.5, 2.2)
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, labels1+labels2, loc='upper right', facecolor='#21262d', edgecolor='#30363d', fontsize=9)

plt.tight_layout()
fig1.savefig(r'd:\Downloads\nhandangnotebook\fig1_training_curve.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print('Fig1 saved.')

# ─────────────────────────────────────────
# FIGURE 2: NAS LEADERBOARD — SCATTER + RANK DISTRIBUTION
# ─────────────────────────────────────────
fig2, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='#0d1117')
for ax in axes:
    ax.set_facecolor('#161b22')

# Left: scatter of arch_id vs val_loss (colored by rank)
ax_l = axes[0]
ranks = list(range(1, len(rows)+1))
colors = plt.cm.plasma(np.linspace(0.05, 0.95, len(rows)))
sc = ax_l.scatter([r['arch_id'] for r in rows], all_losses,
                  c=ranks, cmap='plasma_r', s=18, alpha=0.75, edgecolors='none')
# top 5 highlighted
for r in rows[:5]:
    ax_l.scatter(r['arch_id'], r['val_loss'], color=TEAL, s=80, zorder=10, edgecolors='white', lw=0.8)
ax_l.text(rows[0]['arch_id']+10, rows[0]['val_loss']-0.008, f"#1 Arch {rows[0]['arch_id']}\nLoss={rows[0]['val_loss']:.4f}",
          color=TEAL, fontsize=8)
ax_l.set_xlabel('Architecture ID (assigned by search order)', fontsize=10)
ax_l.set_ylabel('Validation Loss (Proxy Focal)', fontsize=10)
ax_l.set_title('1,000 Architecture Evaluation Scatter', fontsize=12, fontweight='bold', color='#e6edf3')
ax_l.grid(True, alpha=0.35)
cb = plt.colorbar(sc, ax=ax_l, pad=0.01)
cb.set_label('Rank', color='#8b949e', fontsize=9)
cb.ax.yaxis.set_tick_params(color='#8b949e')

# Right: histogram
ax_r = axes[1]
n, bins, patches = ax_r.hist(all_losses, bins=40, color=PURPLE, edgecolor='#0d1117', linewidth=0.4, alpha=0.85)
# color by loss magnitude
norm = plt.Normalize(bins.min(), bins.max())
cm = plt.cm.plasma_r
for patch, left in zip(patches, bins[:-1]):
    patch.set_facecolor(cm(norm(left)))
ax_r.axvline(min(all_losses), color=TEAL, lw=1.8, linestyle='--', label=f'Best: {min(all_losses):.4f}')
ax_r.axvline(sum(all_losses)/len(all_losses), color=ORANGE, lw=1.5, linestyle=':', label=f'Mean: {sum(all_losses)/len(all_losses):.4f}')
ax_r.set_xlabel('Validation Loss', fontsize=10)
ax_r.set_ylabel('Architecture Count', fontsize=10)
ax_r.set_title('Loss Distribution across 1,000 Architectures', fontsize=12, fontweight='bold', color='#e6edf3')
ax_r.legend(facecolor='#21262d', edgecolor='#30363d', fontsize=9)
ax_r.grid(True, alpha=0.35)

plt.tight_layout()
fig2.savefig(r'd:\Downloads\nhandangnotebook\fig2_nas_leaderboard.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print('Fig2 saved.')

# ─────────────────────────────────────────
# FIGURE 3: CHANNEL WIDTH DISTRIBUTION (FUNNEL ANALYSIS)
# ─────────────────────────────────────────
top100 = rows[:100]
bot100 = rows[-100:]

stages = ['C2 (160×160\nStride-4)', 'C3 (80×80\nStride-8)', 'C4 (40×40\nStride-16)', 'C5 (20×20\nStride-32)']
keys   = ['C2_w', 'C3_w', 'C4_w', 'C5_w']

top_means = [np.mean([r[k] for r in top100]) for k in keys]
bot_means = [np.mean([r[k] for r in bot100]) for k in keys]

top_stds = [np.std([r[k] for r in top100]) for k in keys]
bot_stds = [np.std([r[k] for r in bot100]) for k in keys]

fig3, ax = plt.subplots(figsize=(11, 6), facecolor='#0d1117')
ax.set_facecolor('#161b22')

x = np.arange(len(stages))
w = 0.35
bars1 = ax.bar(x-w/2, top_means, w, color=GREEN,  alpha=0.88, label='Top-100 Winners',   yerr=top_stds, capsize=4, error_kw=dict(color='#8b949e', lw=1.2))
bars2 = ax.bar(x+w/2, bot_means, w, color=RED,    alpha=0.70, label='Bottom-100 Losers', yerr=bot_stds, capsize=4, error_kw=dict(color='#8b949e', lw=1.2))

for bar, val in zip(bars1, top_means):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5, f'{val:.0f}', ha='center', va='bottom', fontsize=9, color=GREEN)
for bar, val in zip(bars2, bot_means):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5, f'{val:.0f}', ha='center', va='bottom', fontsize=9, color=RED)

ax.set_xticks(x)
ax.set_xticklabels(stages, fontsize=10)
ax.set_ylabel('Mean Channel Width (Output Channels)', fontsize=11)
ax.set_title('Computation Redistribution (CR) — Funnel Architecture Trend\nWinners vs. Losers: Channel Width by Stage', fontsize=12, fontweight='bold', pad=12, color='#e6edf3')
ax.legend(facecolor='#21262d', edgecolor='#30363d', fontsize=10, loc='upper right')
ax.grid(True, alpha=0.4, axis='y')
ax.set_ylim(0, 120)

# Annotate CR principle
ax.annotate('CR Principle:\nFunnel inward →\nThinner C5', xy=(3.3, 85), fontsize=9, color=ORANGE,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#21262d', edgecolor=ORANGE, alpha=0.9))

plt.tight_layout()
fig3.savefig(r'd:\Downloads\nhandangnotebook\fig3_funnel_analysis.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print('Fig3 saved.')

# ─────────────────────────────────────────
# FIGURE 4: TOP-5 ARCHITECTURE PROFILE (Radar / Bar)
# ─────────────────────────────────────────
fig4, ax = plt.subplots(figsize=(13, 5), facecolor='#0d1117')
ax.set_facecolor('#161b22')

top5 = rows[:5]
labels_arch = [f"Arch {r['arch_id']}\n({r['val_loss']:.4f})" for r in top5]
stage_labels = ['C2_w','C3_w','C4_w','C5_w','C2_d','C3_d','C4_d','C5_d']

x = np.arange(len(top5))
bar_w = 0.1
offset_base = np.linspace(-0.35, 0.35, 8)
colors_stages = [ACCENT, GREEN, ORANGE, PURPLE, '#79c0ff', '#56d364', '#e3b341', '#d2a8ff']

for si, (sk, sc) in enumerate(zip(stage_labels, colors_stages)):
    vals = [r[sk] for r in top5]
    ax.bar(x + offset_base[si], vals, bar_w, color=sc, alpha=0.85, label=sk)

ax.set_xticks(x)
ax.set_xticklabels(labels_arch, fontsize=10)
ax.set_ylabel('Channel Width / Depth', fontsize=11)
ax.set_title('Top-5 NAS Winning Architectures — Stage-wise Configuration Profile', fontsize=12, fontweight='bold', pad=12, color='#e6edf3')
ax.legend(fontsize=8, facecolor='#21262d', edgecolor='#30363d', ncol=4, loc='upper right')
ax.grid(True, alpha=0.4, axis='y')

plt.tight_layout()
fig4.savefig(r'd:\Downloads\nhandangnotebook\fig4_top5_profiles.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print('Fig4 saved.')

# ─────────────────────────────────────────
# FIGURE 5: C2 WIDTH PREFERENCE (TOP vs ALL)
# ─────────────────────────────────────────
fig5, axes5 = plt.subplots(1, 2, figsize=(13, 5), facecolor='#0d1117')
for a in axes5:
    a.set_facecolor('#161b22')

# C2 width distribution for TOP-100 vs ALL
c2_all  = [r['C2_w'] for r in rows]
c2_top  = [r['C2_w'] for r in top100]
c5_all  = [r['C5_w'] for r in rows]
c5_top  = [r['C5_w'] for r in top100]
widths  = sorted(set(c2_all))

ax5l = axes5[0]
c2_count_all = [c2_all.count(w)/len(c2_all)*100 for w in widths]
c2_count_top = [c2_top.count(w)/len(c2_top)*100 for w in widths]
ax5l.plot(widths, c2_count_all, 'o-', color='#8b949e', lw=1.5, ms=4, label='All 1,000 Archs', alpha=0.7)
ax5l.plot(widths, c2_count_top, 's-', color=GREEN,     lw=2.2, ms=5, label='Top-100 Winners')
ax5l.fill_between(widths, c2_count_top, alpha=0.15, color=GREEN)
ax5l.set_xlabel('C2 Channel Width (Stride-4, 160×160)', fontsize=10)
ax5l.set_ylabel('Frequency (%)', fontsize=10)
ax5l.set_title('SR Effect: C2 Width Preference\nTop-100 vs All Architectures', fontsize=11, fontweight='bold', color='#e6edf3')
ax5l.legend(facecolor='#21262d', edgecolor='#30363d', fontsize=9)
ax5l.grid(True, alpha=0.35)

# C5 width distribution
ax5r = axes5[1]
c5_widths_u = sorted(set(c5_all))
c5_count_all = [c5_all.count(w)/len(c5_all)*100 for w in c5_widths_u]
c5_count_top = [c5_top.count(w)/len(c5_top)*100 for w in c5_widths_u]
ax5r.plot(c5_widths_u, c5_count_all, 'o-', color='#8b949e', lw=1.5, ms=4, label='All 1,000 Archs', alpha=0.7)
ax5r.plot(c5_widths_u, c5_count_top, 's-', color=RED,       lw=2.2, ms=5, label='Top-100 Winners')
ax5r.fill_between(c5_widths_u, c5_count_top, alpha=0.15, color=RED)
ax5r.set_xlabel('C5 Channel Width (Stride-32, 20×20)', fontsize=10)
ax5r.set_ylabel('Frequency (%)', fontsize=10)
ax5r.set_title('CR Effect: C5 Width Preference\nTop-100 Winners Prefer Thinner C5', fontsize=11, fontweight='bold', color='#e6edf3')
ax5r.legend(facecolor='#21262d', edgecolor='#30363d', fontsize=9)
ax5r.grid(True, alpha=0.35)

plt.tight_layout()
fig5.savefig(r'd:\Downloads\nhandangnotebook\fig5_c2_c5_distributions.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print('Fig5 saved.')

print('\nAll figures saved successfully!')
