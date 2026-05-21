#!/usr/bin/env python
"""Aggregate 9 per-config full14 summary JSONs into a single 9-way × 14-metric
markdown table + combined summary JSON."""
import json
import os

R = '/public/home/maoyaoxin/zhangt/xxt/SF-v3/results/alpha_540'
configs = [
    'E0_new_code',
    'E3_cosine',
    'E4_reverse',
    'E4_reverse_clamped',
    'E4_sigmoid_mid',
    'E4_sigmoid_mid_clamped',
    'pathB_p1_iter0',
    'pathB_p1_iter500',
    'pathB_p1_iter1000',
    'pathB_p1_iter1500',
    'pathB_p1_iter2000',
]
METRIC_ORDER = [
    'FVD', 'EPE',
    'SSIM', 'PSNR', 'Hue-PCC',
    'CLIP', 'CTC', 'DTC', 'CLIP-PCC', 'VIFI',
    'Img-2way', 'Img-50way', 'Vid-2way', 'Vid-50way',
]
LOWER_BETTER = {'FVD', 'EPE'}

combined = {'experiments': {}, 'metrics': METRIC_ORDER}
for name in configs:
    p = f'{R}/summary_{name}_full14.json'
    if not os.path.exists(p):
        print(f'MISSING: {name}')
        continue
    d = json.load(open(p))
    # JSON may be {name: {metrics}} or {experiments: {name: {metrics}}}
    key = list(d.keys())[0]
    inner = d[key] if isinstance(d[key], dict) else d['experiments'][key]
    combined['experiments'][name] = inner

out_path = f'{R}/summary_11way_pathB_full14.json'
with open(out_path, 'w') as f:
    json.dump(combined, f, indent=2)
print(f'wrote {out_path} with {len(combined["experiments"])} entries')

# Markdown table (long-form)
print()
print('| Experiment | ' + ' | '.join(METRIC_ORDER) + ' |')
print('|---|' + '---:|' * len(METRIC_ORDER))
for name in configs:
    r = combined['experiments'].get(name, {})
    cells = []
    for m in METRIC_ORDER:
        if m in r:
            cells.append(f'{r[m]:.4f}')
        else:
            cells.append('n/a')
    print(f'| {name} | ' + ' | '.join(cells) + ' |')

# Delta table vs E4_reverse baseline
print('\n## Delta vs E4_reverse (Path A winner)\n')
baseline = combined['experiments'].get('E4_reverse', {})
print('| Experiment | ' + ' | '.join(METRIC_ORDER) + ' |')
print('|---|' + '---:|' * len(METRIC_ORDER))
for name in configs:
    if name == 'E4_reverse':
        continue
    r = combined['experiments'].get(name, {})
    cells = []
    for m in METRIC_ORDER:
        if m in r and m in baseline:
            d = r[m] - baseline[m]
            better = (d < 0) if m in LOWER_BETTER else (d > 0)
            mark = '✓' if better else '✗'
            cells.append(f'{d:+.4f} {mark}')
        else:
            cells.append('n/a')
    print(f'| {name} | ' + ' | '.join(cells) + ' |')
