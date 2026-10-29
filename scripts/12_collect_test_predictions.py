#!/usr/bin/env python3
"""Collect final baseline-E6 test predictions, using only BF16 EfficientNet runs.
Python 3.10+, standard library only. Default preview; --apply copies files.
Source defaults to Results; destination is Result/test_prediction (singular Result).
"""
import argparse
import csv
import hashlib
import json
import math
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


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path, hashes):
    data = path.read_bytes()
    hashes[str(path)] = hashlib.sha256(data).hexdigest()
    return json.loads(data.decode('utf-8-sig'))


def validate_predictions(path, labels):
    expected = {f'probability_{i}' for i in range(len(labels))}
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        required = {'image_path', 'cluster_id', 'true_index', 'true_label',
                    'predicted_index', 'predicted_label', 'correct'} | expected
        require(required <= set(fields), f'Missing prediction columns: {path}')
        require({f for f in fields if f.startswith('probability_')} == expected,
                f'Unexpected probability columns: {path}')
        truths, paths, correct = {}, set(), 0
        for row in reader:
            identity = row['cluster_id']
            require(identity and identity not in truths and row['image_path'] not in paths,
                    f'Empty/repeated image identity: {path}')
            ti, pi = int(row['true_index']), int(row['predicted_index'])
            require(0 <= ti < len(labels) and 0 <= pi < len(labels), f'Class index out of range: {path}')
            require(labels[ti] == row['true_label'] and labels[pi] == row['predicted_label'],
                    f'Class mapping mismatch: {path}')
            probs = [float(row[f'probability_{i}']) for i in range(len(labels))]
            require(all(math.isfinite(v) and 0 <= v <= 1 for v in probs)
                    and abs(sum(probs) - 1) < 1e-5, f'Invalid probabilities: {path}')
            require(max(range(len(labels)), key=lambda i: probs[i]) == pi,
                    f'Predicted index differs from probability argmax: {path}')
            require(row['correct'].strip().lower() in ('true', 'false'), f'Invalid correctness flag: {path}')
            require((row['correct'].strip().lower() == 'true') == (ti == pi), f'Incorrect correctness flag: {path}')
            truths[identity] = (row['image_path'], ti, row['true_label'])
            paths.add(row['image_path'])
            correct += ti == pi
    require(bool(truths), f'Empty predictions: {path}')
    require({v[1] for v in truths.values()} == set(range(len(labels))), f'Missing test class: {path}')
    return truths, correct / len(truths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', type=Path, default=Path('Results'))
    parser.add_argument('--output', type=Path, default=Path('Result/test_prediction'))
    parser.add_argument('--experiment', choices=['all'] + list(FOLDERS), default='all')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    root, output = args.results_root.resolve(), args.output.resolve()
    require(root.is_dir(), f'Results root not found: {root}')
    require(not output.exists(), f'Output exists: {output}; choose a new --output. Nothing overwritten.')
    require(output != root and output not in root.parents, 'Output must not contain the source Results folder.')
    selected = list(FOLDERS) if args.experiment == 'all' else [args.experiment]
    copies, runs, metadata = [], [], {}
    common_map, common_truths, common_split = None, None, None
    for experiment in selected:
        ordinary, bf16, size = FOLDERS[experiment]
        wanted = set(MODELS) - ({'vit_base_patch16_224'} if size == 300 else set())
        maps = []
        count_before = len(runs)
        for folder_name, is_bf16 in [(ordinary, False), (bf16, True)]:
            folder = (root / folder_name).resolve()
            require(root in folder.parents, 'Source folder escapes Results.')
            require(output != folder and folder not in output.parents and output not in folder.parents,
                    'Output must be separate from experiment source folders.')
            cfg = read_json(folder / 'experiment.json', metadata)
            require(cfg.get('experiment') == experiment and cfg.get('completed') and cfg.get('test_completed'),
                    f'Expected completed training and testing: {folder}')
            require(cfg.get('image_size') == [size, size] and sorted(cfg.get('seeds', [])) == list(SEEDS),
                    f'Unexpected resolution/seeds: {folder}')
            models = cfg.get('models', [])
            require(len(models) == len(set(models)) and set(models) <= wanted, f'Unexpected models: {folder}')
            chosen = {'efficientnet_v2_s'} if is_bf16 else wanted - {'efficientnet_v2_s'}
            require((set(models) if is_bf16 else set(models) - {'efficientnet_v2_s'}) == chosen,
                    f'Missing/extra final models: {folder}')
            hp = cfg['hyperparameters']
            require(hp.get('amp') is True and hp.get('amp_dtype', 'float16') == ('bfloat16' if is_bf16 else 'float16'),
                    f'Unexpected precision; EfficientNet must use BF16: {folder}')
            mapping_path = folder / 'class_to_index.json'
            mapping = read_json(mapping_path, metadata)
            require(len(mapping) == 14 and all(type(v) is int for v in mapping.values())
                    and sorted(mapping.values()) == list(range(14)), f'Expected contiguous 14-class mapping: {folder}')
            labels = sorted(mapping, key=mapping.get)
            require(cfg.get('class_names') == labels, f'Experiment class names differ from mapping: {folder}')
            if common_map is None:
                common_map = mapping
                common_split = cfg['split_manifest_sha256']
            require(mapping == common_map and cfg['split_manifest_sha256'] == common_split,
                    f'Class mapping or split differs across experiments: {folder}')
            maps.append((mapping_path, digest(mapping_path)))
            if not is_bf16 and 'efficientnet_v2_s' in models:
                print(f'{experiment}: superseded EfficientNet FP16 predictions skipped.')
            for model in MODELS:
                if model not in chosen:
                    continue
                for seed in SEEDS:
                    directory = folder / model / f'seed_{seed}'
                    status = read_json(directory / 'status.json', metadata)
                    rc = read_json(directory / 'config.json', metadata)
                    ev = read_json(directory / 'test_evaluation.json', metadata)
                    require(status.get('completed') and status.get('test_evaluated')
                            and status.get('model') == model and status.get('seed') == seed,
                            f'Incomplete/mismatched run: {directory}')
                    require(rc.get('model') == model and rc.get('seed') == seed and rc.get('labels') == labels
                            and rc.get('image_size') == [size, size], f'Run config mismatch: {directory}')
                    rhp = rc['hyperparameters']
                    require(rhp.get('amp') is True and rhp.get('amp_dtype', 'float16') == ('bfloat16' if is_bf16 else 'float16'),
                            f'Run precision mismatch: {directory}')
                    require(bool(status.get('checkpoint_sha256')) and ev.get('checkpoint_sha256') == status['checkpoint_sha256']
                            and ev.get('split_manifest_sha256') == common_split, f'Test/checkpoint provenance mismatch: {directory}')
                    prediction = directory / 'test_predictions.csv'
                    require(folder in prediction.resolve().parents, 'Prediction path escapes source experiment.')
                    truths, accuracy = validate_predictions(prediction, labels)
                    if common_truths is None:
                        common_truths = truths
                    require(truths == common_truths, f'Test images or true labels differ: {prediction}')
                    require(len(truths) == cfg['counts']['test'], f'Test count differs from experiment: {prediction}')
                    metrics = read_json(directory / 'test_metrics.json', metadata)
                    require(math.isclose(accuracy, metrics['accuracy'], abs_tol=1e-12), f'Accuracy differs from saved metrics: {directory}')
                    destination = f'{experiment}/{model}/seed_{seed}/test_predictions.csv'
                    file_hash = digest(prediction)
                    copies.append(dict(source=str(prediction), destination=destination, sha256=file_hash))
                    runs.append(dict(experiment=experiment, model=model, seed=seed,
                        training_precision='AMP_BF16' if is_bf16 else 'AMP_FP16', test_images=len(truths),
                        source=str(prediction), destination=destination, sha256=file_hash,
                        split_manifest_sha256=common_split, checkpoint_sha256=status['checkpoint_sha256'],
                        test_accuracy=accuracy))
        # Ordinary and BF16 mappings must agree semantically; copy the ordinary file unchanged.
        copies.append(dict(source=str(maps[0][0]), destination=f'{experiment}/class_to_index.json', sha256=maps[0][1]))
        require(len(runs) - count_before == len(wanted)*3, f'Wrong run coverage: {experiment}')
        print(f'{experiment}: verified {len(runs)-count_before} prediction files; class mappings agree.')
    require(len({r['destination'].casefold() for r in copies}) == len(copies), 'Duplicate destination.')
    print(f'Total: {len(runs)} prediction files + {len(selected)} class mappings; {len(common_truths)} matched test images per run.')
    print('Destination: ' + str(output))
    if not args.apply:
        print('PREVIEW ONLY: nothing written. Add --apply to copy.')
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=output.name+'_building_', dir=output.parent))
    try:
        for path, expected in metadata.items():
            require(digest(Path(path)) == expected, f'Metadata changed during collection: {path}')
        for item in copies:
            dst = stage / item['destination']
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item['source'], dst)
            require(digest(dst) == item['sha256'], f'Copy integrity check failed: {item["source"]}')
        with (stage / 'collection_manifest.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(runs[0]))
            writer.writeheader()
            writer.writerows(runs)
        summary = dict(experiments=selected, prediction_files=len(runs), class_mapping_files=len(selected),
            runs_per_experiment=dict(Counter(r['experiment'] for r in runs)), matched_test_images=len(common_truths),
            split_manifest_sha256=common_split, source_metadata_sha256=metadata, copied_files=copies,
            notes=['EfficientNet predictions come exclusively from the explicit BF16 experiment folders.',
                   'Other models come from explicit final main experiment folders; archives/diagnostics are not scanned.',
                   'ViT is absent from E3/E4 at 300 pixels.',
                   'Predictions and mappings are copied byte-for-byte. Originals are unchanged.',
                   'All selected runs share image IDs, paths, true labels, split hash and class-index mapping.',
                   'Checkpoint hash metadata is compared; checkpoint bytes and test image contents are not reverified.',
                   'This is a collection/consistency check, not the statistical significance analysis.'])
        (stage / 'collection_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
        require(not output.exists(), 'Output appeared during copying; refusing overwrite.')
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print('Collection completed. Original results unchanged.')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'STOPPED: {exc}', file=sys.stderr)
        sys.exit(1)
