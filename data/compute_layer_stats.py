"""
Compute per-layer attention statistics from attn JSON files.
Outputs a compact JSON for the frontend EntropyStats component.

Metrics:
  - entropy: normalized entropy H / log(N)  (0=focused, 1=uniform)
  - concentration: 1 - entropy  (0=uniform, 1=focused)
  - sparsity: Gini coefficient  (0=uniform, 1=one-hot)
  - top5_ratio: fraction of mass in top-5% patches
  - spatial maps: avg attention per key layer (for small multiples)
  - center_of_mass: weighted attention center per layer (for gaze trajectory)
  - spatial_spread: weighted std of attention positions
"""
import json
import os
import sys
import math
from collections import defaultdict
import numpy as np

ATTN_DIR = 'public/data/attn'
OUTPUT = 'public/data/layer_stats.json'
N_KEY_LAYERS = 8  # number of evenly-spaced layers to store full attention maps for


def normalized_entropy(vec):
    """H / log(N) — 0 = all mass on one patch, 1 = uniform"""
    v = np.asarray(vec, dtype=np.float64)
    v = np.maximum(v, 0)
    s = v.sum()
    if s < 1e-12:
        return 1.0
    p = v / s
    mask = p > 1e-12
    if mask.sum() < 2:
        return 0.0
    H = -np.sum(p[mask] * np.log(p[mask]))
    return float(H / math.log(len(v)))


def gini(vec):
    """Gini coefficient — 0 = uniform, 1 = one-hot"""
    v = np.asarray(vec, dtype=np.float64)
    v = np.maximum(v, 0)
    s = v.sum()
    if s < 1e-12:
        return 0.0
    v = np.sort(v / s)
    n = len(v)
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * v) - (n + 1)) / n)


def topk_ratio(vec, k=0.05):
    """Fraction of total mass in top k% patches"""
    v = np.asarray(vec, dtype=np.float64)
    v = np.maximum(v, 0)
    s = v.sum()
    if s < 1e-12:
        return float(k)
    n_top = max(1, int(len(v) * k))
    top = np.sort(v)[-n_top:]
    return float(top.sum() / s)


def attention_center_of_mass(vec, grid_w, grid_h):
    """Weighted center of mass of attention on a 2D grid.
    Returns (cx, cy) normalized to [0,1], and spatial spread.
    Handles mismatches between vec length and grid dimensions gracefully."""
    v = np.asarray(vec, dtype=np.float64)
    v = np.maximum(v, 0)
    n_patches = grid_w * grid_h
    # Truncate or pad to match grid dimensions
    if len(v) > n_patches:
        v = v[:n_patches]
    elif len(v) < n_patches:
        v = np.pad(v, (0, n_patches - len(v)), constant_values=0)
    s = v.sum()
    if s < 1e-12:
        return 0.5, 0.5, 0.5
    v = v / s
    xs = np.tile(np.arange(grid_w), grid_h)
    ys = np.repeat(np.arange(grid_h), grid_w)
    cx = float(np.sum(xs * v) / (grid_w - 1)) if grid_w > 1 else 0.5
    cy = float(np.sum(ys * v) / (grid_h - 1)) if grid_h > 1 else 0.5
    dx = xs / max(grid_w - 1, 1) - cx
    dy = ys / max(grid_h - 1, 1) - cy
    spread = float(np.sqrt(np.sum(v * (dx**2 + dy**2))))
    return cx, cy, spread


def infer_grid(vec_len, stored_grid):
    """Infer correct grid dimensions from attention vector length.
    Some models store a grid that doesn't match the actual attention
    vector length (features pooled to fixed size, or CLS token added)."""
    gw, gh = stored_grid.get('w', 1), stored_grid.get('h', 1)
    if gw * gh == vec_len:
        return gw, gh, 0  # (w, h, offset) — offset trims leading special tokens
    # Handle CLS token: vec_len = grid_w*grid_h + 1 (e.g., BLIP: 576+1=577)
    if vec_len == gw * gh + 1:
        return gw, gh, 1
    # Try factorizations close to original aspect ratio
    for w in range(1, int(vec_len ** 0.5) + 2):
        if vec_len % w == 0:
            h = vec_len // w
            if gh > 0 and h > 0:
                err = abs(float(w) / h - float(gw) / gh)
                if err < 0.5:
                    return w, h, 0
    # Fallback: any factorization
    for w in range(int(vec_len ** 0.5), 0, -1):
        if vec_len % w == 0:
            return w, vec_len // w, 0
    # Last resort: check if vec_len-1 is factorable (CLS token)
    for w in range(int((vec_len - 1) ** 0.5), 0, -1):
        if (vec_len - 1) % w == 0:
            return w, (vec_len - 1) // w, 1
    return gw, gh, 0


def process_file(filepath):
    """Extract per-layer statistics and spatial data from one attn file."""
    with open(filepath) as f:
        data = json.load(f)

    tokens = []
    for section in ['question', 'answer']:
        for t in data.get(section, {}).values():
            if t.get('vis_attn'):
                tokens.append(t['vis_attn'])

    if not tokens:
        return None

    stored_grid = data.get('image', {}).get('grid', {})
    # Infer actual grid from first token's first layer vector length
    first_vec = list(tokens[0].values())[0]
    grid_w, grid_h, grid_offset = infer_grid(len(first_vec), stored_grid)

    layer_keys = sorted(
        [k for k in tokens[0].keys() if k.startswith('layer_')],
        key=lambda x: int(x.split('_')[1])
    )
    if not layer_keys:
        return None

    per_layer_entropy = []
    per_layer_sparsity = []
    per_layer_top5 = []
    per_layer_avg_attn = []   # average attention vector per layer
    per_layer_com = []         # (cx, cy, spread) per layer

    for lk in layer_keys:
        entropies = []
        sparsities = []
        top5s = []
        attn_vecs = []
        for tok in tokens:
            vec = tok.get(lk)
            if vec and len(vec) > 1:
                entropies.append(normalized_entropy(vec))
                sparsities.append(gini(vec))
                top5s.append(topk_ratio(vec))
                attn_vecs.append(np.asarray(vec, dtype=np.float64))

        per_layer_entropy.append(float(np.mean(entropies)) if entropies else 0)
        per_layer_sparsity.append(float(np.mean(sparsities)) if sparsities else 0)
        per_layer_top5.append(float(np.mean(top5s)) if top5s else 0)

        if attn_vecs:
            avg_vec = np.mean(attn_vecs, axis=0)
            # Strip CLS/special token offset
            if grid_offset > 0 and len(avg_vec) > grid_offset:
                avg_vec = avg_vec[grid_offset:]
            per_layer_avg_attn.append(avg_vec.tolist())
            cx, cy, spread = attention_center_of_mass(avg_vec, grid_w, grid_h)
            per_layer_com.append((cx, cy, spread))
        else:
            per_layer_avg_attn.append([])
            per_layer_com.append((0.5, 0.5, 0.5))

    return {
        'layers_entropy': per_layer_entropy,
        'layers_sparsity': per_layer_sparsity,
        'layers_top5': per_layer_top5,
        'mean_entropy': float(np.mean(per_layer_entropy)),
        'mean_sparsity': float(np.mean(per_layer_sparsity)),
        'mean_top5': float(np.mean(per_layer_top5)),
        'n_layers': len(layer_keys),
        'grid': {'w': grid_w, 'h': grid_h},
        'layer_keys': layer_keys,
        'avg_attn': per_layer_avg_attn,
        'center_of_mass': per_layer_com,
    }


def select_key_layers(n_layers, n_key=N_KEY_LAYERS):
    """Select evenly-spaced layer indices for spatial map storage."""
    if n_layers <= n_key:
        return list(range(n_layers))
    indices = []
    for i in range(n_key):
        idx = int(round(i * (n_layers - 1) / (n_key - 1)))
        indices.append(idx)
    return indices


def main():
    if not os.path.exists(ATTN_DIR):
        print(f'Error: {ATTN_DIR} not found')
        sys.exit(1)

    files = sorted([f for f in os.listdir(ATTN_DIR) if f.endswith('.json')])
    print(f'Found {len(files)} attention files')

    per_sample = []
    model_layers = defaultdict(lambda: defaultdict(list))
    model_summary = defaultdict(lambda: {'samples': 0, 'correct': 0, 'wrong': 0,
                                           'entropies': [], 'sparsities': [], 'top5s': [],
                                           'correct_entropies': [], 'wrong_entropies': []})
    # Spatial aggregation: model -> layer_idx -> list of avg_attn vectors
    model_spatial_attn = defaultdict(lambda: defaultdict(list))
    model_spatial_com = defaultdict(lambda: defaultdict(list))
    model_grids = {}

    for i, fname in enumerate(files):
        base = fname.replace('.json', '')
        parts = base.rsplit('_', 2)
        if len(parts) >= 3:
            model = '_'.join(parts[:-2])
            img_id = parts[-2]
            qid = parts[-1]
        else:
            model = base
            img_id = ''
            qid = ''

        filepath = os.path.join(ATTN_DIR, fname)
        result = process_file(filepath)
        if result is None:
            print(f'  SKIP {fname}: no vis_attn data')
            continue

        # Store grid info (use inferred grid; all samples of same model share grid dims)
        if model not in model_grids:
            model_grids[model] = result['grid']

        # Correctness
        with open(filepath) as f:
            raw = json.load(f)
            correct = raw.get('correct', None)
            if isinstance(correct, bool):
                correct = correct
            elif isinstance(correct, (int, float)):
                correct = bool(correct)
            else:
                correct = None

        entry = {
            'model': model,
            'img_id': img_id,
            'qid': qid,
            'correct': correct,
            'layers_entropy': result['layers_entropy'],
            'layers_sparsity': result['layers_sparsity'],
            'layers_top5': result['layers_top5'],
            'mean_entropy': result['mean_entropy'],
            'mean_sparsity': result['mean_sparsity'],
            'mean_top5': result['mean_top5'],
            'n_layers': result['n_layers'],
        }
        per_sample.append(entry)

        # Aggregate statistics
        ms = model_summary[model]
        ms['samples'] += 1
        ms['entropies'].append(result['mean_entropy'])
        ms['sparsities'].append(result['mean_sparsity'])
        ms['top5s'].append(result['mean_top5'])
        if correct is True:
            ms['correct'] += 1
            ms['correct_entropies'].append(result['mean_entropy'])
        elif correct is False:
            ms['wrong'] += 1
            ms['wrong_entropies'].append(result['mean_entropy'])

        # Layer-wise entropy aggregation
        for j, ent in enumerate(result['layers_entropy']):
            model_layers[model]['all'].append((j, ent))
            if correct is True:
                model_layers[model]['correct'].append((j, ent))
            elif correct is False:
                model_layers[model]['wrong'].append((j, ent))

        # Spatial aggregation: accumulate avg_attn and com
        for j, (attn, com) in enumerate(zip(result['avg_attn'], result['center_of_mass'])):
            if attn:
                model_spatial_attn[model][j].append(attn)
            model_spatial_com[model][j].append(com)

        if (i + 1) % 20 == 0:
            print(f'  Processed {i+1}/{len(files)}')

    # Compute layer evolution (mean entropy per layer index)
    layer_evolution = {}
    for model, groups in model_layers.items():
        le = {'all': {}, 'correct': {}, 'wrong': {}}
        for group_name, pairs in groups.items():
            by_layer = defaultdict(list)
            for idx, ent in pairs:
                by_layer[idx].append(ent)
            le[group_name] = {
                str(layer): float(np.mean(ents))
                for layer, ents in sorted(by_layer.items())
            }
        layer_evolution[model] = le

    # Compute summary
    summary = {}
    for model, ms in model_summary.items():
        ents = ms['entropies']
        spars = ms['sparsities']
        top5s = ms['top5s']
        summary[model] = {
            'n_samples': ms['samples'],
            'n_correct': ms['correct'],
            'n_wrong': ms['wrong'],
            'n_layers': 0,
            'mean_entropy': float(np.mean(ents)) if ents else 0,
            'std_entropy': float(np.std(ents)) if ents else 0,
            'mean_sparsity': float(np.mean(spars)) if spars else 0,
            'mean_top5': float(np.mean(top5s)) if top5s else 0,
            'correct_mean_entropy': float(np.mean(ms['correct_entropies'])) if ms['correct_entropies'] else None,
            'wrong_mean_entropy': float(np.mean(ms['wrong_entropies'])) if ms['wrong_entropies'] else None,
        }
    for s in per_sample:
        m = s['model']
        if m in summary:
            summary[m]['n_layers'] = max(summary[m]['n_layers'], s['n_layers'])

    # Compute spatial evolution: average attention maps at key layers
    spatial_evolution = {}
    for model, layer_attns in model_spatial_attn.items():
        grid = model_grids.get(model, {'w': 1, 'h': 1})
        n_patches = grid['w'] * grid['h']
        n_layers = summary[model]['n_layers']
        key_indices = select_key_layers(n_layers)

        se_layers = {}
        for idx in key_indices:
            if idx in layer_attns:
                attn_list = layer_attns[idx]
                if attn_list:
                    # Truncate/pad each vector to match grid, then average
                    aligned = []
                    for a in attn_list:
                        arr = np.asarray(a, dtype=np.float64)
                        if len(arr) > n_patches:
                            arr = arr[:n_patches]
                        elif len(arr) < n_patches:
                            arr = np.pad(arr, (0, n_patches - len(arr)), constant_values=0)
                        aligned.append(arr)
                    avg = np.mean(aligned, axis=0)
                    se_layers[str(idx)] = avg.tolist()
        spatial_evolution[model] = {
            'grid': grid,
            'key_layers': key_indices,
            'layers': se_layers,
        }

    # Compute center-of-mass trajectory
    center_of_mass = {}
    spatial_spread = {}
    for model, layer_coms in model_spatial_com.items():
        n_layers = summary[model]['n_layers']
        com_dict = {}
        spread_dict = {}
        for idx in range(n_layers):
            if idx in layer_coms:
                com_list = layer_coms[idx]
                if com_list:
                    cxs = [c[0] for c in com_list]
                    cys = [c[1] for c in com_list]
                    spreads = [c[2] for c in com_list]
                    com_dict[str(idx)] = [float(np.mean(cxs)), float(np.mean(cys))]
                    spread_dict[str(idx)] = float(np.mean(spreads))
        center_of_mass[model] = com_dict
        spatial_spread[model] = spread_dict

    output = {
        'layer_evolution': layer_evolution,
        'per_sample': per_sample,
        'summary': summary,
        'spatial_evolution': spatial_evolution,
        'center_of_mass': center_of_mass,
        'spatial_spread': spatial_spread,
    }

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, 'w') as f:
        json.dump(output, f)
    print(f'\nDone! Output: {OUTPUT}')
    print(f'  Per-sample entries: {len(per_sample)}')
    print(f'  Models: {list(summary.keys())}')
    print(f'  Spatial maps: { {m: len(d["key_layers"]) for m, d in spatial_evolution.items()} }')
    print(f'  File size: {os.path.getsize(OUTPUT) / 1024:.1f} KB')


if __name__ == '__main__':
    main()
