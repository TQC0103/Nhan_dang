import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calc_flops_breakdown(c2_w, c2_d, c3_w, c3_d, c4_w, c4_d, c5_w, c5_d):
    stages = [
        {'w': c2_w, 'd': c2_d, 'h_in': 320, 'w_in': 320, 'stride': 2},
        {'w': c3_w, 'd': c3_d, 'h_in': 160, 'w_in': 160, 'stride': 2},
        {'w': c4_w, 'd': c4_d, 'h_in': 80,  'w_in': 80,  'stride': 2},
        {'w': c5_w, 'd': c5_d, 'h_in': 40,  'w_in': 40,  'stride': 2}
    ]
    
    prev_w = 16
    stage_macs = [0, 0, 0, 0]
    out_resolutions = []
    
    for i, stage in enumerate(stages):
        w_in = prev_w
        w_out = stage['w']
        d = stage['d']
        h_in = stage['h_in']
        win_col = stage['w_in']
        
        h_out = h_in // stage['stride']
        w_out_res = win_col // stage['stride']
        out_resolutions.append((h_out, w_out_res))
        
        smacs = 0
        smacs += h_in * win_col * (4 * w_in) * w_in * 1
        smacs += h_out * w_out_res * (4 * w_in) * 1 * 9
        smacs += h_out * w_out_res * w_out * (4 * w_in) * 1
        
        for _ in range(1, d):
            smacs += h_out * w_out_res * (4 * w_out) * w_out * 1
            smacs += h_out * w_out_res * (4 * w_out) * 1 * 9
            smacs += h_out * w_out_res * w_out * (4 * w_out) * 1
            
        stage_macs[i] += smacs
        prev_w = w_out
        
    for i in range(4):
        res = out_resolutions[i]
        w = stages[i]['w']
        lat_macs = res[0] * res[1] * 32 * w * 1
        stage_macs[i] += lat_macs
        
    H, W = 640, 640
    fixed_macs = (H//2) * (W//2) * 16 * 3 * 9
    res2, res3, res4, res5 = out_resolutions
    fixed_macs += res3[0]*res3[1] * 32 * 32 * 9
    fixed_macs += res4[0]*res4[1] * 32 * 32 * 9
    fixed_macs += res5[0]*res5[1] * 32 * 32 * 9
    for res in [res3, res4, res5]:
        fixed_macs += res[0]*res[1] * 96 * 32 * 9
        fixed_macs += res[0]*res[1] * 1 * 96 * 1
        
    total_gflops = (sum(stage_macs) + fixed_macs) / 1e9
    return total_gflops, stage_macs[0]/1e9, stage_macs[1]/1e9, stage_macs[2]/1e9, stage_macs[3]/1e9

df = pd.read_csv('nas_leaderboard_edge_focus.csv')
gflops_list = []
c2_gf, c3_gf, c4_gf, c5_gf = [], [], [], []

for i, row in df.iterrows():
    tot, g2, g3, g4, g5 = calc_flops_breakdown(
        row['C2_w'], row['C2_d'], 
        row['C3_w'], row['C3_d'], 
        row['C4_w'], row['C4_d'], 
        row['C5_w'], row['C5_d']
    )
    gflops_list.append(tot)
    c2_gf.append(g2)
    c3_gf.append(g3)
    c4_gf.append(g4)
    c5_gf.append(g5)

df['Math_GFLOPs'] = gflops_list
df['C2_GFLOPs'] = c2_gf
df['C3_GFLOPs'] = c3_gf
df['C4_GFLOPs'] = c4_gf
df['C5_GFLOPs'] = c5_gf

top_100 = df.sort_values('Val_Loss', ascending=True).head(100)
mean_gflops_winners = top_100['Math_GFLOPs'].mean()

tol = 0.1
comparable = df[(df['Math_GFLOPs'] >= mean_gflops_winners - tol) & (df['Math_GFLOPs'] <= mean_gflops_winners + tol)]

if len(comparable) > 100:
    bottom_comparable = comparable.sort_values('Val_Loss', ascending=False).head(100)
else:
    bottom_comparable = comparable.sort_values('Val_Loss', ascending=False).head(max(len(comparable) // 2, 1))

print(f"Mean GFLOPs of Winners: {mean_gflops_winners:.3f}")
print(f"Mean GFLOPs of Losers:  {bottom_comparable['Math_GFLOPs'].mean():.3f}")

def format_delta(top, bot):
    delta = top - bot
    return f"{delta:+.3f}"

print("| Stage | Spatial Res. | Edge-Focus Winners (GFLOPs) | Equivalent-GFLOP Losers (GFLOPs) | Delta (Winners - Losers) |")
print("|---|---|---|---|---|")
print(f"| C2 | 160×160 | {top_100['C2_GFLOPs'].mean():.3f} | {bottom_comparable['C2_GFLOPs'].mean():.3f} | {format_delta(top_100['C2_GFLOPs'].mean(), bottom_comparable['C2_GFLOPs'].mean())} |")
print(f"| C3 | 80×80  | {top_100['C3_GFLOPs'].mean():.3f} | {bottom_comparable['C3_GFLOPs'].mean():.3f} | {format_delta(top_100['C3_GFLOPs'].mean(), bottom_comparable['C3_GFLOPs'].mean())} |")
print(f"| C4 | 40×40  | {top_100['C4_GFLOPs'].mean():.3f} | {bottom_comparable['C4_GFLOPs'].mean():.3f} | {format_delta(top_100['C4_GFLOPs'].mean(), bottom_comparable['C4_GFLOPs'].mean())} |")
print(f"| C5 | 20×20  | {top_100['C5_GFLOPs'].mean():.3f} | {bottom_comparable['C5_GFLOPs'].mean():.3f} | {format_delta(top_100['C5_GFLOPs'].mean(), bottom_comparable['C5_GFLOPs'].mean())} |")

# Plot
stages = ['C2\n(160×160)', 'C3\n(80×80)', 'C4\n(40×40)', 'C5\n(20×20)']
top_means = [top_100['C2_GFLOPs'].mean(), top_100['C3_GFLOPs'].mean(), top_100['C4_GFLOPs'].mean(), top_100['C5_GFLOPs'].mean()]
top_stds = [top_100['C2_GFLOPs'].std(), top_100['C3_GFLOPs'].std(), top_100['C4_GFLOPs'].std(), top_100['C5_GFLOPs'].std()]

bot_means = [bottom_comparable['C2_GFLOPs'].mean(), bottom_comparable['C3_GFLOPs'].mean(), bottom_comparable['C4_GFLOPs'].mean(), bottom_comparable['C5_GFLOPs'].mean()]
bot_stds = [bottom_comparable['C2_GFLOPs'].std(), bottom_comparable['C3_GFLOPs'].std(), bottom_comparable['C4_GFLOPs'].std(), bottom_comparable['C5_GFLOPs'].std()]

x = np.arange(len(stages))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, top_means, width, yerr=top_stds, label='Top-100 (Edge-Focus Winners)', 
        capsize=5, color='#9b59b6', alpha=0.9, edgecolor='black')
plt.bar(x + width/2, bot_means, width, yerr=bot_stds, label=f'Comparable Losers\n(~{mean_gflops_winners:.2f} GFLOPs)', 
        capsize=5, color='#34495e', alpha=0.9, edgecolor='black')

plt.ylabel('Stage Computation Cost (GFLOPs)', fontsize=12)
plt.title('Computation Cost: Edge-Detection Proxy Task', fontsize=14)
plt.xticks(x, stages, fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('fig_appendix_edge_funnel.png', dpi=300)
print('Saved fig_appendix_edge_funnel.png')
