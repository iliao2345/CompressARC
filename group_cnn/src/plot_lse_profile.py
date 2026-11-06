"""Plot JSONL from profile_lse.py; requires matplotlib and numpy."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


METHODS = ('phi', 'diagonal_phi')
COLORS = {'phi': '#007e87', 'diagonal_phi': '#c04473'}
STYLES = {'phi': '-', 'diagonal_phi': ':'}
MARKERS = ('o', '^', 's', 'D', 'v', 'P')


def save(fig, destination, name):
    fig.savefig(destination / f'{name}.png', dpi=180, bbox_inches='tight')
    fig.savefig(destination / f'{name}.svg', bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, default=Path('lse_plots'))
    parser.add_argument('--batch', type=int, help='B for operation breakdown; default largest B.')
    parser.add_argument('--height', type=int, help='H for operation breakdown.')
    parser.add_argument('--width', type=int, help='W for operation breakdown.')
    args = parser.parse_args()
    records = [json.loads(line) for line in args.input.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    metadata = next((r for r in records if r.get('type') == 'metadata'), {})
    rows = [r for r in records if r.get('type') == 'measurement' and r.get('status') == 'ok']
    if not rows:
        parser.error('No successful measurements found')
    keys = [(r['B'], r['H'], r['W'], r['method']) for r in rows]
    if len(set(keys)) != len(keys):
        parser.error('Duplicate measurements found; provide one sweep per input file')
    heights = sorted({r['H'] for r in rows})
    widths = sorted({r['W'] for r in rows})
    batches = sorted({r['B'] for r in rows})
    markers = {b: MARKERS[i % len(MARKERS)] for i, b in enumerate(batches)}
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    subtitle = f"{metadata.get('gpu', 'GPU')} | {metadata.get('mode', 'forward')} | median forward ms"

    # H/W map: batch facets avoid overlapping labels at identical coordinates.
    fig, axes = plt.subplots(2, len(batches), figsize=(5 * len(batches), 10), squeeze=False, layout='constrained')
    times = [r['cuda_median_ms'] for r in rows]
    norm = LogNorm(vmin=min(times), vmax=max(times) if max(times) > min(times) else min(times) * 1.01)
    for mi, method in enumerate(METHODS):
        for bi, batch in enumerate(batches):
            ax = axes[mi, bi]
            subset = [r for r in rows if r['method'] == method and r['B'] == batch]
            for r in subset:
                artist = ax.scatter(heights.index(r['H']), widths.index(r['W']),
                                    c=[r['cuda_median_ms']], norm=norm, cmap='viridis',
                                    marker=markers[batch], s=160, edgecolors='black', linewidths=.5)
                ax.annotate(f"{r['cuda_median_ms']:,.1f}", (heights.index(r['H']), widths.index(r['W'])),
                            xytext=(0, 11), textcoords='offset points', ha='center', fontsize=8)
            ax.set(xticks=range(len(heights)), xticklabels=heights,
                   yticks=range(len(widths)), yticklabels=widths,
                   xlim=(-.6, len(heights)-.4), ylim=(-.6, len(widths)-.3),
                   xlabel='H', ylabel='W', title=f'{method} | B={batch}')
            ax.grid(alpha=.15)
    fig.colorbar(artist, ax=axes.ravel().tolist(), label='Forward time (ms, logarithmic colour scale)', shrink=.75)
    fig.suptitle('LSE shape sweep\n' + subtitle + '\nBlank positions have no successful measurement', fontsize=14)
    save(fig, args.output, 'shape_map')

    # Fixed-H slices allow method line styles and batch symbols on the same axes.
    fig, axes = plt.subplots(len(heights), 1, figsize=(11, 3.5 * len(heights)), squeeze=False, layout='constrained')
    for ax, height in zip(axes[:, 0], heights):
        for method in METHODS:
            for bi, batch in enumerate(batches):
                subset = sorted((r for r in rows if r['H'] == height and r['B'] == batch and r['method'] == method), key=lambda r: r['W'])
                lookup = {r['W']: r for r in subset}
                values = [lookup[w]['cuda_median_ms'] if w in lookup else np.nan for w in widths]
                ax.plot(widths, values, STYLES[method], color=COLORS[method], marker=markers[batch],
                        markersize=6, linewidth=1.4, label=f'{method}, B={batch}')
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set(xticks=widths, xticklabels=widths, xlabel='W', ylabel='Forward time (ms)', title=f'H={height}')
        ax.grid(alpha=.2, which='both')
    axes[0, 0].legend(ncol=3, fontsize=8)
    fig.suptitle('Solid: phi | Dotted: diagonal_phi | Symbols: batch size\n' + subtitle, fontsize=13)
    save(fig, args.output, 'width_slices')

    batch = args.batch if args.batch is not None else max(batches)
    common = {(r['H'], r['W']) for r in rows if r['B'] == batch and r['method'] == 'phi'} & {
        (r['H'], r['W']) for r in rows if r['B'] == batch and r['method'] == 'diagonal_phi'}
    common = [shape for shape in common if (args.height is None or shape[0] == args.height)
              and (args.width is None or shape[1] == args.width)]
    if not common:
        parser.error('No matched phi/diagonal_phi shape for requested breakdown')
    height, width = max(common, key=lambda shape: (shape[0] * shape[1], shape))
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), layout='constrained')
    summary = []
    for ax, method in zip(axes, METHODS):
        row = next(r for r in rows if (r['B'], r['H'], r['W'], r['method']) == (batch, height, width, method))
        stages = sorted(row['stages'].items(), key=lambda pair: pair[1]['cuda_elapsed_ms'])
        values = [v['cuda_elapsed_ms'] for _, v in stages]
        bars = ax.barh([k.replace('_', ' ') for k, _ in stages], values, color=COLORS[method])
        ax.set_xscale('log')
        ax.set_xlim(left=max(min(v for v in values if v > 0) / 2, 1e-6), right=max(values) * 5)
        ax.bar_label(bars, labels=[f'{v:,.3f} ms' for v in values], padding=4, fontsize=8)
        ax.set(xlabel='Stage CUDA elapsed (ms, log scale)', title=f"{method} | total {row['cuda_median_ms']:,.1f} ms")
        summary.append({'method': method, 'total_ms': row['cuda_median_ms'], 'largest_stage': stages[-1][0],
                        'largest_stage_ms': values[-1], 'B': batch, 'H': height, 'W': width})
    fig.suptitle(f'Operation breakdown | B={batch}, H={height}, W={width}\n'
                 'Stage timings are separate instrumented measurements; include CPU dispatch gaps.', fontsize=13)
    save(fig, args.output, 'operation_breakdown')
    report = {'measurements': len(rows), 'metadata': metadata, 'breakdown': summary,
              'note': 'CUDA stage elapsed and CPU submission overlap; do not add them. No backward measured.'}
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
