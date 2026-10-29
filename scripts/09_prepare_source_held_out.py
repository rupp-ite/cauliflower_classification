"""Build audited source-held-out folds from cleaned originals (no augmentation).

Requires Pillow: python -m pip install Pillow
Preview validates every image and prints counts without writing output.
Use --apply to write reports and copy images. Existing output is never overwritten.
Validation fraction is 0.15 of each eligible development class (floor, min 1).
The remaining development images are training images. No internal test partition.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import random
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

FOLD_NAMES = ['fold_1_source_1', 'fold_2_source_2', 'fold_3_cauli_net',
              'fold_4_vegnet', 'fold_5_source_5']
SOURCES = ['1_Cauliflower_Diseases_Identification_Image_Dataset',
           '2_Cauliflower_Dataset', '3_Cauli-Net', '4_VegNet',
           '5_Cauliflower_Image_Dataset']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def json_file(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def csv_file(path, rows, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def safe_relative(value):
    p = PurePosixPath(value.replace('\\', '/'))
    require(bool(value) and not p.is_absolute() and all(x not in ('.', '..') and ':' not in x for x in p.parts),
            'Unsafe relative path: ' + value)
    return p


def load_manifest(path, root, check_images=True):
    raw = path.read_bytes()
    rows = list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
    require(bool(rows), 'Manifest is empty.')
    required = {'cluster_id', 'final_label', 'destination_path', 'file_sha256',
                'pixel_sha256', 'all_sources', 'all_source_paths', 'group_ids'}
    require(required <= rows[0].keys(), 'Missing manifest columns: ' + str(required - rows[0].keys()))
    seen = {k: set() for k in ('cluster_id', 'destination_path', 'file_sha256', 'pixel_sha256')}
    groups = set()
    if check_images:
        from PIL import Image
    for i, row in enumerate(rows, 2):
        for key, values in seen.items():
            value = row[key].strip()
            require(value and value not in values, f'Row {i}: empty or repeated {key}: {value}')
            values.add(value)
        sources = json.loads(row['all_sources'])
        require(isinstance(sources, list) and sources and all(isinstance(s, str) for s in sources),
                f'Row {i}: invalid sources')
        require(len(sources) == len(set(sources)) and set(sources) <= set(SOURCES), f'Row {i}: unexpected source')
        row['_sources'] = set(sources)
        original_paths = json.loads(row['all_source_paths'])
        require(isinstance(original_paths, list) and bool(original_paths), f'Row {i}: invalid source paths')
        require({safe_relative(p).parts[0] for p in original_paths} == row['_sources'],
                f'Row {i}: source provenance mismatch')
        group_ids = json.loads(row['group_ids'])
        require(isinstance(group_ids, list) and all(isinstance(g, str) and g for g in group_ids),
                f'Row {i}: invalid duplicate groups')
        require(not (set(group_ids) & groups), f'Row {i}: duplicate group spans retained images; review before splitting')
        groups.update(group_ids)
        relative = safe_relative(row['destination_path'])
        require(len(relative.parts) == 2 and relative.parts[0] == row['final_label'],
                f'Row {i}: expected class/filename destination')
        image_path = root.joinpath(*relative.parts).resolve()
        require(image_path.is_relative_to(root), f'Row {i}: image escapes dataset root')
        row['_path'] = image_path
        if check_images:
            require(image_path.is_file(), f'Missing image: {image_path}')
            content = image_path.read_bytes()
            require(hashlib.sha256(content).hexdigest() == row['file_sha256'], f'File hash mismatch: {image_path}')
            with Image.open(io.BytesIO(content)) as image:
                image.verify()
            with Image.open(io.BytesIO(content)) as image:
                image.convert('RGB').load()
        if check_images and (i - 1) % 500 == 0:
            print(f'Checked {i - 1}/{len(rows)} images', flush=True)
    require(set.union(*(r['_sources'] for r in rows)) == set(SOURCES), 'Expected all five sources')
    return rows, hashlib.sha256(raw).hexdigest()


def make_plans(rows, seed, fraction):
    plans = []
    for source, name in zip(SOURCES, FOLD_NAMES):
        development = Counter(r['final_label'] for r in rows if source not in r['_sources'])
        test = Counter(r['final_label'] for r in rows if r['_sources'] == {source})
        classes = sorted(development.keys() & test.keys())
        require(bool(classes), 'No shared classes for ' + source)
        train_ids, val_ids = set(), set()
        counts = []
        for label in classes:
            candidates = sorted((r['cluster_id'] for r in rows
                                 if r['final_label'] == label and source not in r['_sources']))
            require(len(candidates) >= 2, f'{name}/{label}: insufficient development images')
            local_seed = int.from_bytes(hashlib.sha256(f'{seed}|{source}|{label}'.encode()).digest(), 'big')
            random.Random(local_seed).shuffle(candidates)
            n_val = max(1, math.floor(len(candidates) * fraction))
            require(n_val < len(candidates), 'Validation split leaves no training images')
            val_ids.update(candidates[:n_val])
            train_ids.update(candidates[n_val:])
            counts.append(dict(final_label=label, development=len(candidates),
                               train=len(candidates)-n_val, val=n_val, test=test[label],
                               warning='Small held-out class support (<20)' if test[label] < 20 else ''))
        assignments = []
        for row in rows:
            if source in row['_sources'] and len(row['_sources']) > 1:
                partition, reason = 'excluded', 'shared_with_held_out_source'
            elif row['final_label'] not in classes:
                partition, reason = 'excluded', 'class_not_shared'
            elif row['_sources'] == {source}:
                partition, reason = 'test', ''
            else:
                partition = 'train' if row['cluster_id'] in train_ids else 'val'
                require(row['cluster_id'] in train_ids | val_ids, 'Unassigned development image')
                require(source not in row['_sources'], 'Source leakage detected')
                reason = ''
            assignments.append(dict(cluster_id=row['cluster_id'], final_label=row['final_label'],
                                    partition=partition, exclusion_reason=reason,
                                    destination_path=row['destination_path'], all_sources=row['all_sources'],
                                    file_sha256=row['file_sha256'], pixel_sha256=row['pixel_sha256'],
                                    group_ids=row['group_ids']))
        summary = dict(fold=name, held_out_source=source, eligible_classes=len(classes),
                       train=sum(c['train'] for c in counts), val=sum(c['val'] for c in counts),
                       test=sum(c['test'] for c in counts),
                       excluded_shared=sum(a['exclusion_reason']=='shared_with_held_out_source' for a in assignments),
                       excluded_class=sum(a['exclusion_reason']=='class_not_shared' for a in assignments))
        require(sum(summary[k] for k in ('train', 'val', 'test', 'excluded_shared', 'excluded_class')) == len(rows),
                'Fold counts do not reconcile')
        plans.append((summary, classes, counts, assignments))
    return plans


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--image-root', type=Path, default=Path(r'D:\Projects\CauliflowerV2\Dataset\harmonized_dataset'))
    parser.add_argument('--output', type=Path, default=Path(r'D:\Projects\CauliflowerV2\Dataset\source_held_out'))
    parser.add_argument('--seed', type=int, default=42, help='Split seed, separate from training seeds')
    parser.add_argument('--val-fraction', type=float, default=0.15)
    parser.add_argument('--manifest-only', action='store_true', help='Preview counts without accessing images; cannot use --apply')
    parser.add_argument('--apply', action='store_true', help='Write reports and copy images after validation')
    args = parser.parse_args()
    require(0 < args.val_fraction < 1, 'Validation fraction must be between 0 and 1')
    require(not (args.manifest_only and args.apply), '--manifest-only cannot create folds')
    root, output = args.image_root.resolve(), args.output.resolve()
    require(not output.is_relative_to(root) and not root.is_relative_to(output), 'Input and output folders must be separate')
    if args.apply:
        require(not output.exists() or (output.is_dir() and not any(output.iterdir())),
                'Output must be absent or empty. Existing output is never overwritten.')
    rows, digest = load_manifest(args.manifest, root, not args.manifest_only)
    plans = make_plans(rows, args.seed, args.val_fraction)
    print('\nFold                       classes   train    val   test   excluded')
    for summary, classes, counts, assignments in plans:
        print(f"{summary['fold']:27} {len(classes):7} {summary['train']:7} {summary['val']:6} {summary['test']:6} {summary['excluded_shared']+summary['excluded_class']:10}")
        for count in counts:
            if count['warning']:
                print(f"  WARNING: {count['final_label']}: {count['test']} held-out images")
    if not args.apply:
        print('\nPreview only; no output written.' + (' Image files NOT checked.' if args.manifest_only else ' All image files checked.'))
        return
    output.mkdir(parents=True, exist_ok=True)
    json_file(output/'BUILD_IN_PROGRESS.json', {'status':'incomplete', 'manifest_sha256':digest})
    lookup = {r['cluster_id']:r for r in rows}
    for summary, classes, counts, assignments in plans:
        fold = output/summary['fold']
        reports = fold/'reports'
        reports.mkdir(parents=True)
        for partition in ('train', 'val', 'test'):
            for label in classes:
                (fold/partition/label).mkdir(parents=True)
        for a in assignments:
            if a['partition'] == 'excluded':
                continue
            relative = safe_relative(a['destination_path'])
            target = fold/a['partition']/Path(*relative.parts)
            shutil.copy2(lookup[a['cluster_id']]['_path'], target)
            require(hashlib.sha256(target.read_bytes()).hexdigest() == a['file_sha256'], 'Copy integrity check failed')
        csv_file(reports/'assignments.csv', assignments, list(assignments[0]))
        csv_file(reports/'class_counts.csv', counts, list(counts[0]))
        json_file(reports/'class_to_index.json', {label:i for i,label in enumerate(classes)})
        json_file(reports/'summary.json', summary)
        print('Created and verified: ' + str(fold), flush=True)
    summaries = [p[0] for p in plans]
    csv_file(output/'eligibility_summary.csv', summaries, list(summaries[0]))
    json_file(output/'protocol.json', dict(status='complete', created_utc=datetime.now(timezone.utc).isoformat(),
        manifest=str(args.manifest.resolve()), manifest_sha256=digest, image_root=str(root), images=len(rows),
        split_seed=args.seed, validation_fraction=args.val_fraction,
        validation_rule='per class: max(1, floor(development_count * validation_fraction))',
        test_rule='exclusive held-out provenance; classes shared with development',
        mixed_source_rule='exclude images associated with held-out and any other source',
        augmentation='none; apply offline augmentation to train only in a subsequent step',
        limitations=['Repository separation does not establish farm/plant/session independence',
                     'Shared-class tasks differ between folds; small class supports require caution']))
    shutil.copy2(args.manifest, output/'retained_manifest_input.csv')
    (output/'BUILD_IN_PROGRESS.json').unlink()
    print('Completed. Originals unchanged. No augmentation or training performed.')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'STOPPED: {exc}', file=sys.stderr)
        sys.exit(1)
