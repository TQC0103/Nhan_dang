import json

with open('spos-supernet-for-scrf-dynamic-block.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

code_cells = []
for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        source = "".join(cell.get('source', []))
        code_cells.append(source)

with open('cells_train.py', 'w', encoding='utf-8') as f:
    f.write('\n\n# --- CELL ---\n\n'.join(code_cells))
