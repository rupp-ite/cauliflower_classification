#!/usr/bin/env python3
"""Combine baseline/E1-E6 FP16 reports with replacement EfficientNet BF16 reports.

Does not move checkpoints, images or prediction files. Default is preview only.
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
SEEDS = (42, 43, 44)
METRICS = ('accuracy', 'macro_precision', 'macro_recall', 'macro_f1',
           'weighted_precision', 'weighted_recall', 'weighted_f1', 'loss')

FOLDERS = {
    'baseline': ('baseline_224_raw_no_aug', 'baseline_efficientnet_bf16', 224, False, False),
    'E1': ('E1_224_offline_no_online', 'E1_efficientnet_bf16', 224, True, False),
    'E2': ('E2_224_offline_online', 'E2_efficientnet_bf16', 224, True, True),
    'E3': ('E3_300_offline_no_online', 'E3_efficientnet_bf16', 300, True, False),
    'E4': ('E4_300_offline_online', 'E4_efficientnet_bf16', 300, True, True),
    'E5': ('E5_384_offline_no_online', 'E5_efficientnet_bf16', 384, True, False),
    'E6': ('E6_384_offline_online', 'E6_efficientnet_bf16', 384, True, True),
}


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError('Nonfinite numeric result in source reports.')
    return number


def agree(a, b, location):
    if not math.isclose(finite(a), finite(b), rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError(f'Inconsistent statistics or reports: {location}')


def protocol(cfg):
    hp = dict(cfg['hyperparameters'])
    hp.pop('amp_dtype', None)
    keys = ('experiment', 'image_size', 'class_names', 'counts', 'pretrained',
            'offline_augmentation', 'online_augmentation', 'split_manifest_sha256',
            'augmentation_manifest_sha256', 'augmentation_policy_sha256',
            'training_device', 'benchmark_protocol')
    return {k: cfg[k] for k in keys} | {'hyperparameters_except_amp_dtype': hp}


def aggregate_metrics(rows, models):
    result = []
    for model in models:
        members = [r for r in rows if r['model'] == model]
        for metric in METRICS:
            values = [r[metric] for r in members]
            result.append(dict(model=model, training_precision=members[0]['training_precision'],
                               metric=metric, seeds=len(values), mean=statistics.mean(values),
                               sample_sd=statistics.stdev(values),
                               unit='cross_entropy' if metric == 'loss' else 'fraction'))
    return result


def check_source_means(path, rows):
    table = read_csv(path)
    expected = {(m, k) for m, _ in rows for k in METRICS}
    seen = set()
    for r in table:
        key = r['model'], r['metric']
        if key in seen or key not in expected:
            raise ValueError(f'Duplicate/unexpected mean row: {path}')
        seen.add(key)
        values = [finite(v[r['metric']]) for (m, _), v in rows.items() if m == r['model']]
        if int(r['seeds']) != len(values):
            raise ValueError(f'Seed count mismatch: {path}')
        agree(r['mean'], statistics.mean(values), str(path))
        agree(r['sample_sd'], statistics.stdev(values), str(path))
    if seen != expected:
        raise ValueError(f'Missing mean rows: {path}')


def combine_experiment(args, experiment):
    primary, bf16, size, offline, online = FOLDERS[experiment]
    models = tuple(m for m in MODELS if not (size == 300 and m == 'vit_base_patch16_224'))
    non_eff = set(models) - {'efficientnet_v2_s'}
    train_count = 7191 if offline else 2756
    sources = [(args.results_root / primary).resolve(), (args.results_root / bf16).resolve()]
    output = (args.output_root / f'{experiment}_FinalReports').resolve()
    if len(set(sources)) != len(sources):
        raise ValueError('Repeated input folder.')
    if output.exists():
        raise ValueError('Output already exists. Choose a new --output; no overwriting.')
    if any(output == s or output in s.parents or s in output.parents for s in sources):
        raise ValueError('Final output must be separate from the source folders.')
    combined = {'test': [], 'val': [], 'benchmark': []}
    reference = env_reference = map_reference = None
    jobs, provenance, hashes = set(), [], {}
    for source_index, source in enumerate(sources):
        print(f'Checking {source}', flush=True)
        cfg = load(source / 'experiment.json')
        if cfg.get('experiment') != experiment or not cfg.get('completed') or not cfg.get('test_completed'):
            raise ValueError(f'Finish {experiment} training AND evaluation first: {source}')
        if sorted(cfg['seeds']) != list(SEEDS) or len(set(cfg['models'])) != len(cfg['models']) or not set(cfg['models']) <= set(models):
            raise ValueError(f'Unexpected models/seeds: {source}')
        source_models = set(cfg['models'])
        if source_index == 0:
            if source_models - {'efficientnet_v2_s'} != non_eff:
                raise ValueError(f'Primary folder must contain all {len(non_eff)} non-EfficientNet models: {source}')
            chosen_models = source_models - {'efficientnet_v2_s'}
        else:
            if source_models != {'efficientnet_v2_s'}:
                raise ValueError(f'BF16 replacement folder must contain only EfficientNet: {source}')
            chosen_models = source_models
        if cfg['image_size'] != [size, size] or cfg['offline_augmentation'] is not offline or cfg['online_augmentation'] is not online or cfg['pretrained'] is not True:
            raise ValueError(f'Unexpected {experiment} protocol: {source}')
        if cfg['counts'] != {'train': train_count, 'val': 579, 'test': 579} or len(cfg['class_names']) != 14:
            raise ValueError(f'Unexpected dataset counts/classes: {source}')
        hp = cfg['hyperparameters']
        if hp.get('amp') is not True:
            raise ValueError(f'Expected AMP training: {source}')
        dtype = hp.get('amp_dtype', 'float16')
        if dtype not in ('float16', 'bfloat16'):
            raise ValueError('Unknown AMP dtype.')
        for model in chosen_models:
            if dtype != ('bfloat16' if model == 'efficientnet_v2_s' else 'float16'):
                raise ValueError(f'Unexpected training precision for {model}')
        precision = 'AMP_BF16' if dtype == 'bfloat16' else 'AMP_FP16'
        env, mapping, current = load(source / 'environment.json'), load(source / 'class_to_index.json'), protocol(cfg)
        if mapping != {name: i for i, name in enumerate(cfg['class_names'])}:
            raise ValueError(f'Class-index mapping differs from class order: {source}')
        if reference is None:
            reference, env_reference, map_reference = current, env, mapping
        elif reference != current or env_reference != env or map_reference != mapping:
            raise ValueError('Inputs differ in data, core protocol, class mapping or software environment. Inspect before combining.')
        source_name = cfg.get('output', str(source))
        provenance.append(dict(results_directory=source_name, models_present=cfg['models'],
                               selected_models=sorted(chosen_models),
                               excluded_models=sorted(source_models - chosen_models), seeds=cfg['seeds'],
                               training_precision=precision, script_sha256=cfg['script_sha256'],
                               runner_version=cfg.get('runner_version'), memory_cleanup=cfg.get('memory_cleanup', 'not recorded in experiment.json')))
        expected_source = {(m, s) for m in cfg['models'] for s in SEEDS}
        expected = {(m, s) for m in chosen_models for s in SEEDS}
        if source_models - chosen_models:
            print(f'Exclude superseded EfficientNet FP16 rows from {source.name}; select BF16 replacement instead.')
        if jobs & expected:
            raise ValueError(f'Duplicate model/seed across folders: {sorted(jobs & expected)}')
        jobs |= expected
        tables = {}
        filenames = ['experiment.json', 'environment.json', 'class_to_index.json',
                     'test_runs.csv', 'val_runs.csv', 'test_mean_sd.csv', 'val_mean_sd.csv',
                     'benchmark_summary.csv']
        for filename in filenames:
            hashes[source_name.rstrip('/\\') + '/' + filename] = hashlib.sha256((source / filename).read_bytes()).hexdigest()
        for kind in ('test', 'val', 'benchmark'):
            filename = 'benchmark_summary.csv' if kind == 'benchmark' else f'{kind}_runs.csv'
            keyed = {}
            for r in read_csv(source / filename):
                key = r['model'], int(r['seed'])
                if key in keyed:
                    raise ValueError(f'Duplicate job in {filename}: {key}')
                keyed[key] = r
            if set(keyed) != expected_source:
                raise ValueError(f'Missing/extra jobs: {source / filename}')
            tables[kind] = keyed
        for kind in ('test', 'val'):
            check_source_means(source / f'{kind}_mean_sd.csv', tables[kind])
        for key in sorted(expected):
            model, seed = key
            extra = dict(training_precision=precision, source_results_directory=source_name,
                         source_runner_sha256=cfg['script_sha256'],
                         split_manifest_sha256=cfg['split_manifest_sha256'])
            for kind in ('test', 'val'):
                r = tables[kind][key]
                clean = dict(model=model, seed=seed)
                for metric in METRICS:
                    clean[metric] = finite(r[metric])
                    if metric != 'loss' and not 0 <= clean[metric] <= 1:
                        raise ValueError('Expected fractional metrics, not percentages.')
                # Accuracy must correspond to an integer number of correct images.
                count = cfg['counts'][kind]
                agree(clean['accuracy'] * count, round(clean['accuracy'] * count), f'{model}/{seed}/{kind}/support')
                combined[kind].append(clean | extra)
            bench = dict(tables['benchmark'][key])
            if bench['experiment'] != experiment or bench['split_manifest_sha256'] != cfg['split_manifest_sha256']:
                raise ValueError('Benchmark experiment/data provenance mismatch.')
            for field, value in [('image_size', size), ('training_images', train_count), ('validation_images', 579), ('test_images', 579)]:
                agree(bench[field], value, field)
            if bench['training_amp'].casefold() != 'true' or bench.get('benchmark_precision') != 'FP32':
                raise ValueError('Unexpected training or inference precision in benchmark.')
            if bench.get('training_amp_dtype', dtype) != dtype:
                raise ValueError('Benchmark AMP dtype differs from experiment.')
            for kind in ('test', 'val'):
                for metric in METRICS:
                    agree(bench[f'{kind}_{metric}'], tables[kind][key][metric], f'{model}/{seed}/{kind}/{metric}')
            for field in ('inference_batch_1_status', 'inference_batch_32_status'):
                if bench.get(field) != 'completed':
                    raise ValueError(f'Finish inference benchmarks before combining: {model}/{seed}')
            bench.update(extra, training_amp_dtype=dtype)
            combined['benchmark'].append(bench)
    if jobs != {(m, s) for m in models for s in SEEDS}:
        raise ValueError(f'Expected {len(models)} models and three seeds ({len(models)*3} unique runs).')
    for rows in combined.values():
        rows.sort(key=lambda r: (MODELS.index(r['model']), int(r['seed'])))
    means = {kind: aggregate_metrics(combined[kind], models) for kind in ('test', 'val')}
    benchmark_means = []
    for model in models:
        members = [r for r in combined['benchmark'] if r['model'] == model]
        for metric in dict.fromkeys(k for r in members for k in r):
            if metric in ('seed', 'image_size'):
                continue
            try:
                values = [finite(r[metric]) for r in members]
            except (ValueError, KeyError, TypeError):
                continue
            benchmark_means.append(dict(experiment=experiment, model=model, training_precision=members[0]['training_precision'],
                                        metric=metric, runs=len(values), mean=statistics.mean(values), sample_sd=statistics.stdev(values)))
    print(f'Validated {experiment}: {len(models)*3} unique runs; seeds 42/43/44.')
    for r in means['test']:
        if r['metric'] == 'accuracy':
            print(f"{r['model']}: {100*r['mean']:.2f} +/- {100*r['sample_sd']:.2f}% ({r['training_precision']})")
    if not args.apply:
        print('PREVIEW ONLY. Add --apply to create the final reports.')
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=output.name + '_building_', dir=output.parent))
    try:
        for filename, rows in [('test_runs.csv', combined['test']), ('val_runs.csv', combined['val']),
                               ('test_mean_sd.csv', means['test']), ('val_mean_sd.csv', means['val']),
                               ('benchmark_summary.csv', combined['benchmark']), ('benchmark_summary_mean_sd.csv', benchmark_means)]:
            write_csv(stage / filename, rows)
        summary = dict(experiment=experiment, completed=True, models=list(models), seeds=list(SEEDS), total_runs=len(jobs),
                       common_protocol=reference, source_experiments=provenance,
                       notes=['Report consolidation only. Original checkpoints, images and predictions remain in their source folders.',
                              'EfficientNet uses BF16 AMP; other models use FP16 AMP. Inference benchmarks use FP32.',
                              'Any older EfficientNet rows in the primary folder are excluded and replaced by the BF16 group, never averaged together.',
                              'Means and sample SDs are recomputed from individual seed rows; classification metrics are fractions.',
                              'Different runner hashes are recorded, not assumed identical. Numerical protocol checks exclude only the declared AMP dtype difference.',
                              'Validation covers supplied configs and report consistency; checkpoint bytes, test predictions and image bytes were not reverified.',
                              'Operation counts retain their coverage status. Partial estimates are not exact totals.'])
        for filename, value in [('summary.json', summary), ('input_file_hashes.json', hashes),
                                 ('environment.json', env_reference), ('class_to_index.json', map_reference)]:
            (stage / filename).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
        if output.exists():
            raise ValueError('Output appeared during writing; refusing overwrite.')
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print(f'Saved {output}. Source folders unchanged.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', choices=['all'] + list(FOLDERS), default='all')
    parser.add_argument('--results-root', type=Path, default=Path('Results'))
    parser.add_argument('--output-root', type=Path, default=Path('FinalReports'))
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    selected = list(FOLDERS) if args.experiment == 'all' else [args.experiment]
    # Validate every requested pair before writing any output folder.
    requested_write = args.apply
    args.apply = False
    for experiment in selected:
        combine_experiment(args, experiment)
    if requested_write:
        args.apply = True
        for experiment in selected:
            combine_experiment(args, experiment)
        print(f'Completed {len(selected)} experiment report groups. Originals unchanged.')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f'STOPPED: {exc}', file=sys.stderr)
        sys.exit(1)
