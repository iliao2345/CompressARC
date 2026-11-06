from pathlib import Path
root=Path(r'C:/Users/teeds/Documents/GitHub/ARC-fafo')
p=root/'src/compressarc/train/objective.py';s=p.read_text();a=s.index('    logits = torch.cat');b=s.index('    if phase_callback is not None:\n        phase_callback("backward")',a)
body=s[a:b];helper='''def compute_objective(task, model_outputs, train_step, config):
    """Build the unchanged ARC loss from model outputs, without backward/update."""
    logits, x_mask, y_mask, kl_amounts, kl_names = model_outputs
'''+body+'''    return loss, total_kl, reconstruction_error, kl_sums, kl_names, (logits, x_mask, y_mask)


'''
s=s[:a]+'''    loss, total_kl, reconstruction_error, kl_sums, kl_names, outputs = compute_objective(
        task, (logits, x_mask, y_mask, kl_amounts, kl_names), train_step, config)
    logits, x_mask, y_mask = outputs
'''+s[b:];s=s.replace('def take_step(',helper+'def take_step(',1);p.write_text(s)
p=root/'src/profile_backward.py';s=p.read_text().replace("COMPONENTS = ('share_up', 'share_down', 'direction_share', 'shift', 'cummax')", "COMPONENTS = ('share_up', 'share_down', 'direction_share', 'shift', 'cummax',\n              'softmax', 'nonlinear', 'normalize', 'latent_decoder', 'output_heads', 'loss')")
a=s.index('@contextmanager\ndef capture_components');b=s.index('\n\ndef synchronize',a)
s=s[:a]+'''@contextmanager
def capture_components(model, selected=COMPONENTS, capture_connected=False):
    """Capture all boundaries, including direct KL outputs and the loss boundary."""
    from compressarc.model import model as model_module
    from compressarc.train import objective
    records, patches, handles, counts = [], [], [], {}
    def wrap(name, fn, input_indices=(0,), loss_only=False):
        def captured(*args, **kwargs):
            index = counts.get(name, 0)
            counts[name] = index + 1
            replay_args = list(args)
            memo = {}
            for i in input_indices:
                replay_args[i] = detach_inputs(args[i], memo)
            output = fn(*args, **kwargs)
            observed = output[0] if loss_only else output
            outputs = [x for x in unique_tensors(observed) if x.requires_grad]
            replay = (lambda *a, **kw: fn(*a, **kw)[0]) if loss_only else fn
            root = name in ('latent_decoder', 'output_heads', 'loss', 'connected_model')
            record = dict(name=name if root else f'depth_{index}/{name}', call=replay,
                          args=tuple(replay_args), kwargs=kwargs,
                          boundary_inputs=[replay_args[i] for i in input_indices],
                          upstream=[None] * len(outputs))
            for i, tensor in enumerate(outputs):
                def save(gradient, record=record, i=i):
                    record['upstream'][i] = gradient.detach().clone()
                handles.append(tensor.register_hook(save))
            records.append(record)
            return output
        return captured
    try:
        for name in [*selected, *(['connected_model'] if capture_connected else [])]:
            indices, loss_only = (0,), False
            if name in ('share_up', 'share_down'):
                owner, attribute = model.share, name
            elif name in ('nonlinear', 'normalize'):
                owner, attribute = model_module, name
            elif name == 'loss':
                owner, attribute, indices, loss_only = objective, 'compute_objective', (1,), True
            elif name == 'connected_model':
                owner, attribute, indices = model, 'forward', ()
            else:
                owner, attribute = model, name
                if name == 'latent_decoder':
                    indices = ()  # Parameters remain the original tied leaf tensors.
            fn = getattr(owner, attribute)
            replacement = patch.object(owner, attribute, wrap(name, fn, indices, loss_only))
            replacement.start()
            patches.append(replacement)
        yield records
    finally:
        for handle in handles:
            handle.remove()
        for replacement in reversed(patches):
            replacement.stop()
'''+s[b:]
s=s.replace("inputs = unique_tensors(record['args'][0])", "inputs = unique_tensors(record.get('boundary_inputs', record['args'][:1]))")
s=s.replace('with capture_components(model, args.components) as records:', 'with capture_components(model, args.components, capture_connected=True) as records:')
s=s.replace('                rows = []', '                rows, connected = [], None')
s=s.replace('                    rows.append(row)', "                    if row['name'] == 'connected_model':\n                        connected = row\n                    else:\n                        rows.append(row)")
s=s.replace('measured_steps=args.measured_steps, full_backward=full,', 'measured_steps=args.measured_steps, full_backward=full,\n                  connected_model_backward=connected,\n                  requested_components=args.components,\n                  expected_boundary_count=sum(1 if c in (\'latent_decoder\', \'output_heads\', \'loss\') else model.n_layers for c in args.components),\n                  observed_boundary_count=len(rows),')
s=s.replace("'No profiler, optimiser updates, numerical diagnostics or W&B logging inside timed intervals.'", "'Connected-model VJP excludes loss arithmetic but includes direct KL gradients and all model layers.',\n                      'Loss VJP detaches logits, masks and KL amounts; model VJP and loss VJP are separate experiments, not an additive partition.',\n                      'No profiler, optimiser updates, numerical diagnostics or W&B logging inside timed intervals.'")
s=s.replace("lines = [f\"Full backward median: {full['wall_median_seconds']*1000:.3f} ms\", '',", "lines = [f\"Full backward median: {full['wall_median_seconds']*1000:.3f} ms\",\n             f\"Connected model VJP median: {connected['wall_median_seconds']*1000:.3f} ms (loss arithmetic excluded)\",\n             f\"Boundary coverage: {len(rows)}/{report['expected_boundary_count']}\", '',")
s=s.replace("'d4-conv-backward-latency')", "'d4-conv-backward-coverage')")
p.write_text(s)
