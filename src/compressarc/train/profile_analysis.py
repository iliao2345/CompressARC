"""Attribute raw trace kernels to phases and forward/backward model components.

GPU launches use trace External id links, not operator-call counts. Backward
workers inherit the enclosing main-thread backward interval; sequence numbers
link their autograd nodes to forward components. Unknown links remain unassigned.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

PHASES = {'forward', 'loss', 'backward', 'optimizer', 'metrics',
          'progress', 'diagnostics', 'candidates', 'logging'}


def analyze_trace(path, prefix=None):
    path = Path(path)
    trace = json.loads(path.read_text(encoding='utf-8'))
    events = trace['traceEvents']
    ranges = [e for e in events if e.get('cat') == 'user_annotation'
              and e.get('name', '').startswith('ttt/') and e['name'][4:] in PHASES]

    def phase_at(event):
        matches = [p for p in ranges if p['pid'] == event['pid']
                   and p['ts'] <= event['ts'] < p['ts'] + p['dur']]
        return min(matches, key=lambda e:e['dur'])['name'][4:] if matches else 'unscoped'

    threads = defaultdict(list)
    for i, event in enumerate(events):
        if event.get('ph') == 'X' and event.get('cat') in {'cpu_op', 'user_annotation'}:
            threads[(event['pid'], event['tid'])].append(i)
    info, external = {}, {}
    seq_layers = defaultdict(set)
    seq_stages = defaultdict(set)
    for indices in threads.values():
        stack = []
        for i in sorted(indices, key=lambda i:(events[i]['ts'], -events[i]['dur'])):
            e = events[i]
            while stack and e['ts'] + e['dur'] > events[stack[-1]]['ts'] + events[stack[-1]]['dur'] + .01:
                stack.pop()
            parent = stack[-1] if stack else None
            inherited = info[parent] if parent is not None else {}
            phase = inherited.get('phase', 'unscoped')
            if e['name'].startswith('ttt/') and e['name'][4:] in PHASES:
                phase = e['name'][4:]
            if phase == 'unscoped':
                phase = phase_at(e)
            layer = e['name'][6:] if e['name'].startswith('layer/') else inherited.get('layer')
            share_stage = e['name'][6:] if e['name'].startswith('stage/') else inherited.get('share_stage')
            sequence = e.get('args', {}).get('Sequence number')
            backward_seq = inherited.get('backward_seq')
            if phase == 'backward' and sequence is not None and 'Backward' in e['name']:
                backward_seq = (e['pid'], sequence)
            backward_node = inherited.get('backward_node')
            if phase == 'backward' and e['name'].startswith('autograd::engine::evaluate_function: '):
                backward_node = e['name'].split(': ', 1)[1]
            optimizer_component = inherited.get('optimizer_component')
            if e['name'].startswith('ttt/optimizer/'):
                optimizer_component = e['name'].split('/')[-1]
            info[i] = dict(phase=phase, layer=layer, backward_seq=backward_seq,
                           self_cpu_us=float(e['dur']), parent=parent,
                           optimizer_component=optimizer_component, share_stage=share_stage, backward_node=backward_node)
            if parent is not None:
                info[parent]['self_cpu_us'] -= e['dur']
            if phase == 'forward' and layer and sequence is not None:
                seq_layers[(e['pid'], sequence)].add(layer)
                if share_stage:
                    seq_stages[(e['pid'],sequence)].add(share_stage)
            ext = e.get('args', {}).get('External id')
            if ext is not None:
                external[(e['pid'], ext)] = i
            stack.append(i)
    # Runtime API and GPU records share a correlation ID; use the runtime PID
    # to disambiguate External id across processes.
    runtimes = {e.get('args', {}).get('correlation'):e for e in events
                if e.get('cat') == 'cuda_runtime'}

    def owner(index):
        if index is None:
            return 'unscoped', 'unassigned'
        item = info[index]
        layer = item['layer']
        if not layer and item['phase'] == 'backward':
            owners = seq_layers.get(item['backward_seq'], set())
            if len(owners) == 1:
                layer = next(iter(owners))
        return item['phase'], layer or 'unassigned'

    def sharing_owner(index):
        if index is None:
            return None
        item = info[index]
        stage = item.get('share_stage')
        if not stage and item['phase'] == 'backward':
            owners = seq_stages.get(item['backward_seq'],set())
            if len(owners) == 1:
                stage = next(iter(owners))
        return stage

    def entry():
        return dict(kernel_launches=0, gpu_kernel_us=0., cpu_operator_self_us=0.,
                    host_range_us=0., synchronization_calls=0, synchronization_host_us=0.,
                    scalar_reads=0)
    layers, phases, operations, sync_apis, optimizer_parts = (defaultdict(entry) for _ in range(5))
    sharing = defaultdict(entry)
    backward_nodes = defaultdict(entry)
    for p in ranges:
        phases[p['name'][4:]]['host_range_us'] += p['dur']
    for i, item in info.items():
        e = events[i]
        phase, layer = owner(i)
        if e['name'].startswith('layer/'):
            layers[(phase, layer)]['host_range_us'] += e['dur']
        stage = sharing_owner(i)
        if e['name'].startswith('stage/'):
            sharing[(phase,layer,stage)]['host_range_us'] += e['dur']
        part = item.get('optimizer_component')
        if e['name'].startswith('ttt/optimizer/'):
            optimizer_parts[part]['host_range_us'] += e['dur']
        if e.get('cat') == 'cpu_op':
            targets = [layers[(phase, layer)], phases[phase]]
            if stage:
                targets.append(sharing[(phase,layer,stage)])
            if phase == 'optimizer':
                targets.append(optimizer_parts[part or 'unassigned'])
            if phase == 'backward':
                node = item.get('backward_node') or 'unassigned'
                targets.append(backward_nodes[(layer, stage or 'unassigned', node)])
            for row in targets:
                row['cpu_operator_self_us'] += max(0., item['self_cpu_us'])
                row['scalar_reads'] += e['name'] == 'aten::_local_scalar_dense'
    unmatched = 0
    kernel_durations, activity = [], []
    for e in events:
        category, args = e.get('cat'), e.get('args', {})
        if category in {'kernel', 'gpu_memcpy', 'gpu_memset'}:
            activity.append((e['ts'], e['ts']+e['dur']))
        if category == 'kernel':
            runtime = runtimes.get(args.get('correlation'))
            index = external.get((runtime['pid'], args.get('External id'))) if runtime else None
            phase, layer = owner(index)
            operation = events[index]['name'] if index is not None else 'unknown'
            unmatched += index is None
            kernel_durations.append(e['dur'])
            targets = [layers[(phase, layer)], phases[phase], operations[operation]]
            stage = sharing_owner(index)
            if stage:
                targets.append(sharing[(phase,layer,stage)])
            if phase == 'optimizer':
                part = info[index].get('optimizer_component') if index is not None else None
                targets.append(optimizer_parts[part or 'unassigned'])
            if phase == 'backward':
                node = info[index].get('backward_node') if index is not None else None
                targets.append(backward_nodes[(layer, stage or 'unassigned', node or 'unassigned')])
            for row in targets:
                row['kernel_launches'] += 1
                row['gpu_kernel_us'] += e['dur']
        elif category == 'cuda_runtime' and 'Synchronize' in e['name']:
            index = external.get((e['pid'], args.get('External id')))
            phase, layer = owner(index)
            if phase == 'unscoped':
                phase = phase_at(e)
            targets = [layers[(phase, layer)], phases[phase], sync_apis[(phase, e['name'])]]
            stage = sharing_owner(index)
            if stage:
                targets.append(sharing[(phase,layer,stage)])
            if phase == 'optimizer':
                part = info[index].get('optimizer_component') if index is not None else None
                targets.append(optimizer_parts[part or 'unassigned'])
            for row in targets:
                row['synchronization_calls'] += 1
                row['synchronization_host_us'] += e['dur']
    merged = []
    for start, end in sorted(activity):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    active = sum(end-start for start,end in merged)
    span = merged[-1][1]-merged[0][0] if merged else 0.
    layer_rows = [dict(phase=k[0], layer=k[1], **v) for k,v in layers.items()]
    layer_rows.sort(key=lambda r:r['kernel_launches'], reverse=True)
    components = defaultdict(entry)
    for row in layer_rows:
        component = row['layer'].rsplit('/', 1)[-1]
        for key in entry():
            components[(row['phase'], component)][key] += row[key]
    component_rows = sorted([dict(phase=k[0], component=k[1], **v) for k,v in components.items()],
                            key=lambda r:r['kernel_launches'], reverse=True)
    stage_totals = defaultdict(entry)
    for (phase,layer,stage),values in sharing.items():
        for key in entry():
            stage_totals[(phase,stage)][key] += values[key]
    stage_rows = sorted([dict(phase=k[0],component=k[1].split('/')[0],stage=k[1].split('/')[-1],**v)
                         for k,v in stage_totals.items()],key=lambda r:r['kernel_launches'],reverse=True)
    result = dict(
        operation_stages=stage_rows,
        backward_nodes=sorted([dict(layer=k[0], stage=k[1], node=k[2], **v) for k,v in backward_nodes.items()],
                              key=lambda r:r['cpu_operator_self_us'], reverse=True),
        sharing_stages=[r for r in stage_rows if r['component'] in ('share_up','share_down')],
        sharing_stage_attributed_launches=sum(r['kernel_launches'] for r in stage_rows if r['component'] in ('share_up','share_down')),
        sharing_layer_launches=sum(r['kernel_launches'] for r in layer_rows if r['layer'].endswith(('/share_up','/share_down'))),
        captured_forward_steps=sum(p['name']=='ttt/forward' for p in ranges),
        component_totals=component_rows,
        optimizer_components=[dict(component=k, **v) for k,v in optimizer_parts.items()],
        kernel_launches=len(kernel_durations), gpu_kernel_us=sum(kernel_durations),
        unmatched_kernel_launches=unmatched,
        layer_attributed_kernel_launches=sum(r['kernel_launches'] for r in layer_rows if r['layer']!='unassigned'),
        gpu_activity_union_us=active, gpu_activity_span_us=span,
        gpu_trace_activity_fraction=active/span if span else None,
        phases=[dict(phase=k, **v) for k,v in phases.items()], layers=layer_rows,
        operations=sorted([dict(operation=k, **v) for k,v in operations.items()], key=lambda r:r['kernel_launches'], reverse=True),
        synchronizations=[dict(phase=k[0], api=k[1], **v) for k,v in sync_apis.items()],
        caveats=[
            'Instrumented timings include profiler overhead. Use the separate clean pass for latency.',
            'Kernel durations sum GPU execution; host ranges include dispatch and waiting. Do not add them.',
            'Backward layer attribution uses autograd sequence numbers; unmatched work is unassigned.',
            'Backward CPU self time is attributed operator time, not an isolated layer wall-clock measurement.',
            'GPU trace activity fraction is trace coverage, not a hardware utilization measurement.',
            'Synchronization API counts include measurement barriers and do not imply equal waiting costs.'])
    if prefix is not None:
        prefix = str(prefix)
        Path(prefix+'.layers.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        with Path(prefix+'.layers.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, ['phase','layer',*entry().keys()])
            writer.writeheader()
            writer.writerows(layer_rows)
        lines = ['Instrumented capture: use separate clean timing for latency.', '',
                 f"Totals across {result['captured_forward_steps']} captured step(s).",
                 'Phase | kernels | GPU kernel ms | host range ms | sync calls']
        lines += [f"{r['phase']} | {r['kernel_launches']} | {r['gpu_kernel_us']/1000:.3f} | {r['host_range_us']/1000:.3f} | {r['synchronization_calls']}" for r in result['phases']]
        lines += ['', 'Optimizer subcomponent (included in optimizer phase) | kernels | GPU ms | host range ms | sync calls | scalar reads']
        lines += [f"{r['component']} | {r['kernel_launches']} | {r['gpu_kernel_us']/1000:.3f} | {r['host_range_us']/1000:.3f} | {r['synchronization_calls']} | {r['scalar_reads']}" for r in result['optimizer_components']]
        lines += ['', 'Sharing stage (included in sharing layer totals) | phase | kernels | GPU ms | CPU operator self ms | forward host range ms']
        lines += [f"{r['component']}/{r['stage']} | {r['phase']} | {r['kernel_launches']} | {r['gpu_kernel_us']/1000:.3f} | {r['cpu_operator_self_us']/1000:.3f} | {r['host_range_us']/1000:.3f}" for r in result['sharing_stages']]
        lines += ['', 'Operation stage | phase | kernels | GPU ms | attributed CPU self ms']
        lines += [f"{r['component']}/{r['stage']} | {r['phase']} | {r['kernel_launches']} | {r['gpu_kernel_us']/1000:.3f} | {r['cpu_operator_self_us']/1000:.3f}" for r in result['operation_stages']]
        lines += ['', 'Backward autograd node | layer | stage | kernels | GPU ms | attributed CPU self ms']
        lines += [f"{r['node']} | {r['layer']} | {r['stage']} | {r['kernel_launches']} | {r['gpu_kernel_us']/1000:.3f} | {r['cpu_operator_self_us']/1000:.3f}" for r in result['backward_nodes']]
        lines += ['', 'Component across depths | phase | kernels | GPU kernel ms | CPU operator self ms']
        lines += [f"{r['component']} | {r['phase']} | {r['kernel_launches']} | {r['gpu_kernel_us']/1000:.3f} | {r['cpu_operator_self_us']/1000:.3f}" for r in component_rows]
        lines += ['', 'Layer | phase | kernels | GPU kernel ms | CPU operator self ms']
        lines += [f"{r['layer']} | {r['phase']} | {r['kernel_launches']} | {r['gpu_kernel_us']/1000:.3f} | {r['cpu_operator_self_us']/1000:.3f}" for r in layer_rows]
        lines += ['', *result['caveats']]
        Path(prefix+'.layers.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return result
