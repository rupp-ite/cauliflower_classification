#!/usr/bin/env python3
"""Validate and summarize collected inference logs. Python 3.10+, stdlib only.

Reads efficiency_log produced by 07_collect_efficiency_logs.py. No model loading,
training, GPU execution or new FLOP tracing is performed. Default: preview only.
"""
import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
import sys
import tempfile
from pathlib import Path

MODELS = ('resnet50', 'densenet121', 'mobilenet_v3_large', 'efficientnet_v2_s',
          'xception', 'convnext_tiny', 'vit_base_patch16_224', 'twins_svt_base')
SIZES = {'baseline': 224, 'E1': 224, 'E2': 224, 'E3': 300, 'E4': 300, 'E5': 384, 'E6': 384}
SEEDS = (42, 43, 44)


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def csv_read(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def csv_write(path, rows, fields=None):
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def number(x):
    x = float(x)
    if not math.isfinite(x):
        raise ValueError('Nonfinite numeric measurement.')
    return x


def close(a, b, context):
    if not math.isclose(number(a), number(b), rel_tol=1e-8, abs_tol=1e-9):
        raise ValueError(f'Recorded measurement disagrees with recalculation: {context}')


def percentile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=Path('efficiency_log'))
    p.add_argument('--output', type=Path, default=Path('efficiency_analysis'))
    p.add_argument('--apply', action='store_true')
    a = p.parse_args()
    root, output = a.input.resolve(), a.output.resolve()
    if output.exists():
        raise ValueError('Output exists; choose a new --output. Nothing is overwritten.')
    if output == root or output in root.parents or root in output.parents:
        raise ValueError('Input and output must be separate directories.')
    manifest_path = root / 'collection_manifest.csv'
    manifest_rows = csv_read(manifest_path)
    manifest = {}
    for r in manifest_rows:
        if r['destination'] in manifest:
            raise ValueError('Duplicate row in collection manifest.')
        manifest[r['destination']] = r
    expected = {(ex, model, seed) for ex, size in SIZES.items() for model in MODELS
                if not (size == 300 and model == 'vit_base_patch16_224') for seed in SEEDS}
    expected_paths = {f'{ex}/{model}/seed_{seed}/efficiency.json' for ex, model, seed in expected}
    if set(manifest) != expected_paths:
        raise ValueError('Collection manifest must cover the 162 accepted runs exactly.')
    actual = {f.relative_to(root).as_posix() for f in root.glob('*/*/seed_*/efficiency.json')}
    if actual != expected_paths:
        raise ValueError(f'Missing/extra efficiency files: {len(expected_paths-actual)} missing, {len(actual-expected_paths)} extra.')
    per_run, unsupported, flags, protocols = [], [], [], []
    coverage = {}
    for ex in SIZES:
        size = SIZES[ex]
        for model in MODELS:
            if size == 300 and model == 'vit_base_patch16_224':
                continue
            for seed in SEEDS:
                rel = f'{ex}/{model}/seed_{seed}/efficiency.json'
                path = (root / rel).resolve()
                if root not in path.parents:
                    raise ValueError(f'Path escapes input root: {rel}')
                data, meta = load(path), manifest[rel]
                if hashlib.sha256(path.read_bytes()).hexdigest() != meta['sha256']:
                    raise ValueError(f'File changed after collection: {rel}')
                if data.get('model') != model or data.get('seed') != seed or data.get('image_size') != [size, size]:
                    raise ValueError(f'Identity/resolution mismatch: {rel}')
                training_precision = meta['training_precision']
                if training_precision != ('AMP_BF16' if model == 'efficientnet_v2_s' else 'AMP_FP16'):
                    raise ValueError(f'Unexpected training precision: {rel}')
                row = dict(experiment=ex, model=model, seed=seed, resolution=size,
                           training_precision=training_precision, gpu=data.get('gpu'),
                           device=data.get('device'), parameters=int(data['parameters']),
                           parameters_millions=int(data['parameters']) / 1e6,
                           checkpoint_mib=number(data['checkpoint_bytes']) / 2**20,
                           source_efficiency_file=rel, checkpoint_sha256=data['checkpoint_sha256'])
                comp = data.get('complexity', {})
                row['complexity_status'] = comp.get('status', 'missing')
                row['supported_gmacs'] = None
                row['derived_partial_gflops_2_per_mac'] = None
                row['fvcore_goperations_1_per_fma'] = None
                if row['complexity_status'] in ('partial', 'supported_ops_counted'):
                    macs = number(comp['supported_macs'])
                    native = number(comp['fvcore_fma1_operations'])
                    flops = 2 * macs + (native - macs)
                    if macs < 0 or native < macs:
                        raise ValueError(f'Invalid supported-operation counts: {rel}')
                    close(flops, comp['estimated_flops_2_per_mac'], rel + '/flops')
                    row.update(supported_gmacs=macs / 1e9,
                               derived_partial_gflops_2_per_mac=flops / 1e9,
                               fvcore_goperations_1_per_fma=native / 1e9)
                else:
                    flags.append(dict(experiment=ex, model=model, seed=seed, issue='complexity_unavailable', detail=comp.get('error', row['complexity_status'])))
                ops = comp.get('unsupported_operators', {})
                coverage[(ex, model, seed)] = dict(unsupported_operator_occurrences=ops,
                    attention_handler=comp.get('attention_handler', ''), convention=comp.get('convention', ''),
                    method=comp.get('method', ''), uncalled_modules=comp.get('uncalled_modules', []))
                row['unsupported_operators_json'] = json.dumps(ops, sort_keys=True)
                for op, count in sorted(ops.items()):
                    unsupported.append(dict(experiment=ex, model=model, seed=seed, resolution=size,
                                            operator=op, occurrences=int(count),
                                            note='Trace occurrences, not scalar operation counts.'))
                for batch in (1, 32):
                    prefix = f'batch_{batch}_'
                    measurement = data.get('batches', {}).get(str(batch), {})
                    status = measurement.get('status', 'missing')
                    row[prefix + 'status'] = status
                    row[prefix + 'precision'] = measurement.get('precision', '')
                    row[prefix + 'latency_ms_mean'] = None
                    row[prefix + 'latency_ms_repeat_sd'] = None
                    row[prefix + 'latency_ms_median'] = None
                    row[prefix + 'latency_ms_p95'] = None
                    row[prefix + 'throughput_images_per_second'] = None
                    row[prefix + 'amortized_ms_per_image'] = None
                    row[prefix + 'peak_allocated_mib'] = None
                    row[prefix + 'peak_reserved_mib'] = None
                    if status != 'completed':
                        flags.append(dict(experiment=ex, model=model, seed=seed, issue=prefix+'incomplete', detail=measurement.get('error', status)))
                        continue
                    if measurement.get('precision') != 'FP32' or measurement.get('batch_size') != batch:
                        raise ValueError(f'Unexpected inference precision/batch size: {rel}, batch {batch}')
                    times = [number(v) for v in measurement['timings_ms']]
                    if len(times) < 2 or len(times) != measurement['repeats'] or min(times) <= 0:
                        raise ValueError(f'Invalid timing repetitions: {rel}, batch {batch}')
                    mean = statistics.mean(times)
                    sd, median, p95 = statistics.stdev(times), statistics.median(times), percentile(times, .95)
                    throughput = 1000 * batch / mean
                    for name, value in [('batch_latency_ms_mean', mean), ('batch_latency_ms_sd', sd),
                                        ('batch_latency_ms_median', median), ('batch_latency_ms_p95', p95),
                                        ('throughput_images_per_second', throughput), ('amortized_ms_per_image', mean / batch)]:
                        close(value, measurement[name], rel + '/' + name)
                    row.update({prefix+'latency_ms_mean': mean, prefix+'latency_ms_repeat_sd': sd,
                                prefix+'latency_ms_median': median, prefix+'latency_ms_p95': p95,
                                prefix+'throughput_images_per_second': throughput,
                                prefix+'amortized_ms_per_image': mean / batch,
                                prefix+'peak_allocated_mib': number(measurement['peak_allocated_mib']),
                                prefix+'peak_reserved_mib': number(measurement['peak_reserved_mib'])})
                    protocols.append(dict(gpu=data.get('gpu'), device=measurement.get('device'),
                        precision=measurement['precision'], warmup=measurement['warmup'], repeats=measurement['repeats'],
                        timing_method=measurement['timing_method'], scope=measurement['scope'],
                        strict_deterministic=data.get('strict_deterministic'), tf32=data.get('tf32'),
                        torch_version=data.get('torch_version'), cuda_runtime=data.get('cuda_runtime')))
                per_run.append(row)
    protocol_variants = sorted({json.dumps(r, sort_keys=True) for r in protocols})
    if len(protocol_variants) != 1:
        flags.append(dict(experiment='all', model='all', seed='', issue='multiple_inference_protocols', detail=f'{len(protocol_variants)} distinct settings; inspect summary.json before comparing.'))
    groups, coverage_rows, long_rows = [], [], []
    numeric = ['supported_gmacs', 'derived_partial_gflops_2_per_mac', 'fvcore_goperations_1_per_fma']
    for batch in (1, 32):
        numeric += [f'batch_{batch}_{name}' for name in ['latency_ms_mean', 'throughput_images_per_second',
                                                      'amortized_ms_per_image', 'peak_allocated_mib', 'peak_reserved_mib']]
    for ex in SIZES:
        for model in MODELS:
            members = [r for r in per_run if r['experiment'] == ex and r['model'] == model]
            if not members:
                continue
            statuses = sorted({r['complexity_status'] for r in members})
            summary = dict(experiment=ex, model=model, resolution=SIZES[ex], seeds=len(members),
                           training_precision=members[0]['training_precision'], inference_precision='FP32',
                           parameters=members[0]['parameters'], parameters_millions=members[0]['parameters_millions'],
                           complexity_status=';'.join(statuses))
            if len({r['parameters'] for r in members}) != 1:
                raise ValueError(f'Parameter counts differ across seeds: {ex}/{model}')
            for metric in numeric:
                values = [r[metric] for r in members if r[metric] is not None]
                summary[metric + '_valid_runs'] = len(values)
                summary[metric + '_mean'] = statistics.mean(values) if values else None
                summary[metric + '_seed_sd'] = statistics.stdev(values) if len(values) > 1 else None
                long_rows.append(dict(experiment=ex, model=model, resolution=SIZES[ex], metric=metric,
                    valid_runs=len(values), mean=summary[metric+'_mean'], sample_sd_across_seeds=summary[metric+'_seed_sd']))
            groups.append(summary)
            variants = sorted({json.dumps(coverage[(ex, model, r['seed'])], sort_keys=True) for r in members})
            coverage_rows.append(dict(experiment=ex, model=model, resolution=SIZES[ex],
                                      coverage_identical_across_seeds=len(variants)==1,
                                      coverage_variants_json=json.dumps([json.loads(v) for v in variants])))
            if len(variants) != 1:
                flags.append(dict(experiment=ex, model=model, seed='', issue='coverage_differs_across_seeds', detail='Inspect operator_coverage.csv.'))
    print(f'Validated {len(per_run)} run records; summarized {len(groups)} experiment/model groups.')
    print(f'Inference protocol variants: {len(protocol_variants)}; audit flags: {len(flags)}.')
    print('MAC/FLOP values remain estimates over counted operations. Missing operations are not invented.')
    if not a.apply:
        print('PREVIEW ONLY. Add --apply to write CSV tables and the audit summary.')
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=output.name+'_building_', dir=output.parent))
    try:
        for name, rows in [('efficiency_per_run.csv', per_run), ('efficiency_summary.csv', groups),
                           ('efficiency_summary_long.csv', long_rows), ('operator_coverage.csv', coverage_rows)]:
            csv_write(stage/name, rows)
        csv_write(stage/'unsupported_operators.csv', unsupported,
                  ['experiment', 'model', 'seed', 'resolution', 'operator', 'occurrences', 'note'])
        csv_write(stage/'audit_flags.csv', flags, ['experiment', 'model', 'seed', 'issue', 'detail'])
        summary = dict(total_runs=len(per_run), experiment_model_groups=len(groups), audit_flags=len(flags),
            input_collection_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            observed_inference_protocols=[json.loads(v) for v in protocol_variants],
            notes=['Inference timings are recomputed from saved timing repetitions; no GPU workload was rerun.',
                   'Summary means and sample SDs are across the three seed-level measurements, not pooled timing repetitions.',
                   'Per-run repeat SD is separate from across-seed SD. Seeds with missing measurements are counted explicitly.',
                   'GMACs = supported_macs/1e9. Derived GFLOPs = (2*MACs + remaining counted operations)/1e9.',
                   'fvcore native operations count one fused multiply-add as one operation; they are not the same convention as the derived GFLOPs.',
                   'Unsupported operator counts describe trace occurrences, not missing scalar FLOPs. Explicitly ignored operations may not appear in that list.',
                   'Partial counts are not exact model totals. No missing-operation counts were added.',
                   'Batch-32 amortized time per image is not batch-1 single-image latency.',
                   'Peak allocated/reserved memory are PyTorch allocator measurements, not total Windows/NVIDIA device usage.',
                   'No confidence intervals or significance claims are calculated by this script.'])
        (stage/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
        if output.exists():
            raise ValueError('Output appeared during writing; refusing overwrite.')
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print(f'Saved {output}; input logs unchanged.')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f'STOPPED: {exc}', file=sys.stderr)
        sys.exit(1)
