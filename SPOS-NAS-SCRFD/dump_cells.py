import json

with open('scrfd-spos-nas-and-inference-dynamic-block (2).ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

code_cells = []
for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        source = "".join(cell.get('source', []))
        code_cells.append(source)

with open('cells.py', 'w', encoding='utf-8') as f:
    f.write('\n\n# --- CELL ---\n\n'.join(code_cells))
