import json
import sys
sys.stdout.reconfigure(encoding='utf-8')

with open('curve_fitter.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
cell7 = ''.join(cells[7]['source'])
# Find the PICWave expression generation part
idx = cell7.find('picwave_expr')
if idx >= 0:
    print(cell7[max(0,idx-500):idx+5000])
