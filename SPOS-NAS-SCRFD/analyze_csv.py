import csv

rows = []
with open(r'd:\Downloads\nhandangnotebook\nas_leaderboard-dynamic.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

losses = [float(r['Val_Loss']) for r in rows]
print(f'Total architectures: {len(rows)}')
print(f'Best Val Loss: {min(losses):.6f}')
print(f'Worst Val Loss: {max(losses):.6f}')
print(f'Mean Val Loss: {sum(losses)/len(losses):.6f}')
print()
print('Top 5:')
for r in rows[:5]:
    c2w = r['C2_w']; c2d = r['C2_d']
    c3w = r['C3_w']; c3d = r['C3_d']
    c4w = r['C4_w']; c4d = r['C4_d']
    c5w = r['C5_w']; c5d = r['C5_d']
    print(f"  Arch {r['Arch_ID']}: C2=({c2w},{c2d}) C3=({c3w},{c3d}) C4=({c4w},{c4d}) C5=({c5w},{c5d}) Loss={float(r['Val_Loss']):.6f}")

top50 = rows[:50]
c2_widths = [int(r['C2_w']) for r in top50]
c5_widths = [int(r['C5_w']) for r in top50]
c3_widths = [int(r['C3_w']) for r in top50]
c4_widths = [int(r['C4_w']) for r in top50]
print(f'\nTop 50 avg C2_width: {sum(c2_widths)/len(c2_widths):.1f}')
print(f'Top 50 avg C3_width: {sum(c3_widths)/len(c3_widths):.1f}')
print(f'Top 50 avg C4_width: {sum(c4_widths)/len(c4_widths):.1f}')
print(f'Top 50 avg C5_width: {sum(c5_widths)/len(c5_widths):.1f}')

bot50 = rows[-50:]
c2_bot = [int(r['C2_w']) for r in bot50]
c3_bot = [int(r['C3_w']) for r in bot50]
c4_bot = [int(r['C4_w']) for r in bot50]
c5_bot = [int(r['C5_w']) for r in bot50]
print(f'\nBottom 50 avg C2_width: {sum(c2_bot)/len(c2_bot):.1f}')
print(f'Bottom 50 avg C3_width: {sum(c3_bot)/len(c3_bot):.1f}')
print(f'Bottom 50 avg C4_width: {sum(c4_bot)/len(c4_bot):.1f}')
print(f'Bottom 50 avg C5_width: {sum(c5_bot)/len(c5_bot):.1f}')

# depth analysis
c2_depths_top = [int(r['C2_d']) for r in top50]
c5_depths_top = [int(r['C5_d']) for r in top50]
print(f'\nTop 50 avg C2_depth: {sum(c2_depths_top)/len(c2_depths_top):.2f}')
print(f'Top 50 avg C5_depth: {sum(c5_depths_top)/len(c5_depths_top):.2f}')

# Loss distribution by decile
import statistics
print('\nLoss distribution (every 100):')
for dec in range(0, 1000, 100):
    chunk = losses[dec:dec+100]
    print(f'  Rank {dec+1}-{dec+100}: {min(chunk):.4f} - {max(chunk):.4f}')
