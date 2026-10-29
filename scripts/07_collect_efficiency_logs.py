#!/usr/bin/env python3
"""Copy accepted efficiency.json files to efficiency_log/experiment/model/seed.

Python 3.10+, standard library only. Default is preview; add --apply to copy.
Uses explicit final experiment folders, never scans archives or diagnostics.
"""
import argparse
import csv
import hashlib
import json
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

MODELS = ('resnet50', 'densenet121', 'mobilenet_v3_large', 'efficientnet_v2_s',
          'xception', 'convnext_tiny', 'vit_base_patch16_224', 'twins_svt_base')
SEEDS = (42, 43, 44)
FOLDERS = {
    'baseline': ('baseline_224_raw_no_aug', 'baseline_efficientnet_bf16', 224),
    'E1': ('E1_224_offline_no_online', 'E1_efficientnet_bf16', 224),
    'E2': ('E2_224_offline_online', 'E2_efficientnet_bf16', 224),
    'E3': ('E3_300_offline_no_online', 'E3_efficientnet_bf16', 300),
    'E4': ('E4_300_offline_online', 'E4_efficientnet_bf16', 300),
    'E5': ('E5_384_offline_no_online', 'E5_efficientnet_bf16', 384),
    'E6': ('E6_384_offline_online', 'E6_efficientnet_bf16', 384),
}


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', type=Path, default=Path('Results'))
    parser.add_argument('--output', type=Path, default=Path('efficiency_log'))
    parser.add_argument('--experiment', choices=['all'] + list(FOLDERS), default='all')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    root, output = args.results_root.resolve(), args.output.resolve()
    if not root.is_dir():
        raise ValueError(f'Results root not found: {root}')
    if output.exists():
        raise ValueError(f'Output already exists: {output}. Choose a new --output; nothing overwritten.')
    if output == root or root in output.parents or output in root.parents:
        raise ValueError('Output must be separate from, and not contain, Results.')
    selected = list(FOLDERS) if args.experiment == 'all' else [args.experiment]
    plan, metadata_hashes, destinations = [], {}, set()
    for experiment in selected:
        ordinary, efficientnet, size = FOLDERS[experiment]
        expected_models = set(MODELS) - ({'vit_base_patch16_224'} if size == 300 else set())
        for folder_name, is_bf16 in [(ordinary, False), (efficientnet, True)]:
            folder = (root / folder_name).resolve()
            if root not in folder.parents:
                raise ValueError(f'Source folder escapes Results: {folder}')
            cfg_path = folder / 'experiment.json'
            cfg = load(cfg_path)
            if cfg.get('experiment') != experiment or not cfg.get('completed') or not cfg.get('test_completed'):
                raise ValueError(f'Expected a completed {experiment} run: {folder}')
            if sorted(cfg.get('seeds', [])) != list(SEEDS) or cfg.get('image_size') != [size, size]:
                raise ValueError(f'Unexpected seeds or resolution: {folder}')
            present = cfg.get('models', [])
            if len(present) != len(set(present)) or not set(present) <= expected_models:
                raise ValueError(f'Unexpected/repeated models: {folder}')
            wanted = {'efficientnet_v2_s'} if is_bf16 else expected_models - {'efficientnet_v2_s'}
            if (set(present) if is_bf16 else set(present) - {'efficientnet_v2_s'}) != wanted:
                raise ValueError(f'Model coverage does not match final selection: {folder}')
            hp = cfg['hyperparameters']
            dtype = hp.get('amp_dtype', 'float16')
            if hp.get('amp') is not True or dtype != ('bfloat16' if is_bf16 else 'float16'):
                raise ValueError(f'Unexpected training precision: {folder}')
            metadata_hashes[str(cfg_path)] = digest(cfg_path)
            if not is_bf16 and 'efficientnet_v2_s' in present:
                print(f'{experiment}: skip superseded EfficientNet FP16 files; select BF16 replacements.')
            for model in MODELS:
                if model not in wanted:
                    continue
                for seed in SEEDS:
                    directory = folder / model / f'seed_{seed}'
                    source = directory / 'efficiency.json'
                    resolved = source.resolve()
                    if folder not in resolved.parents:
                        raise ValueError(f'Efficiency file escapes its experiment folder: {source}')
                    data = load(source)
                    if data.get('model') != model or data.get('seed') != seed or data.get('image_size') != [size, size]:
                        raise ValueError(f'Efficiency identity/resolution mismatch: {source}')
                    status = load(directory / 'status.json')
                    if not status.get('completed') or status.get('model') != model or status.get('seed') != seed:
                        raise ValueError(f'Incomplete or mismatched training status: {directory}')
                    if not data.get('checkpoint_sha256') or data['checkpoint_sha256'] != status.get('checkpoint_sha256'):
                        raise ValueError(f'Efficiency/checkpoint provenance mismatch: {source}')
                    relative = f'{experiment}/{model}/seed_{seed}/efficiency.json'
                    if relative.casefold() in destinations:
                        raise ValueError(f'Duplicate destination: {relative}')
                    destinations.add(relative.casefold())
                    batch1 = data.get('batches', {}).get('1', {})
                    batch32 = data.get('batches', {}).get('32', {})
                    # Collect failure/partial records too: this is an audit package,
                    # not an attempt to conceal unsuccessful measurements.
                    plan.append(dict(experiment=experiment, model=model, seed=seed,
                                     training_precision='AMP_BF16' if is_bf16 else 'AMP_FP16',
                                     source=str(source), destination=relative, sha256=digest(source),
                                     checkpoint_sha256=data['checkpoint_sha256'],
                                     complexity_status=data.get('complexity', {}).get('status', 'missing'),
                                     batch_1_status=batch1.get('status', 'missing'),
                                     batch_32_status=batch32.get('status', 'missing'),
                                     batch_1_precision=batch1.get('precision', 'missing'),
                                     batch_32_precision=batch32.get('precision', 'missing')))
        count = sum(r['experiment'] == experiment for r in plan)
        if count != len(expected_models) * len(SEEDS):
            raise ValueError(f'Wrong final run count: {experiment}')
        print(f'{experiment}: validated {count} efficiency files.')
    warnings = [r['destination'] for r in plan if
                r['batch_1_status'] != 'completed' or r['batch_32_status'] != 'completed'
                or r['batch_1_precision'] != 'FP32' or r['batch_32_precision'] != 'FP32']
    print(f'Total: {len(plan)} files. Benchmark status/precision flags: {len(warnings)}.')
    if not args.apply:
        print('PREVIEW ONLY: no files copied. Add --apply to collect the logs.')
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=output.name + '_building_', dir=output.parent))
    try:
        for row in plan:
            destination = stage / row['destination']
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(row['source'], destination)
            if digest(destination) != row['sha256']:
                raise ValueError(f'Copy differs from validated input: {row["source"]}')
        with (stage / 'collection_manifest.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(plan[0]))
            writer.writeheader()
            writer.writerows(plan)
        summary = dict(experiments=selected, total_files=len(plan),
                       files_per_experiment=dict(Counter(r['experiment'] for r in plan)),
                       source_experiment_json_sha256=metadata_hashes,
                       benchmark_status_or_precision_flags=warnings,
                       notes=['Only explicit final experiment folders were read; no diagnostics or archives were scanned.',
                              'EfficientNet BF16 replaces earlier FP16 versions. ViT is absent at 300 pixels.',
                              'Files are copied byte-for-byte, never moved or rewritten.',
                              'Partial/failed benchmark records are retained and flagged, not hidden.',
                              'Checkpoint hash strings are checked against status.json; checkpoint bytes are not rehashed.',
                              'This collector packages the records; operator coverage and detailed timing still require the audit.'])
        (stage / 'collection_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        if output.exists():
            raise ValueError('Output appeared during copying; refusing overwrite.')
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print(f'Copied {len(plan)} files to {output}. Original folders unchanged.')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f'STOPPED: {exc}', file=sys.stderr)
        sys.exit(1)
