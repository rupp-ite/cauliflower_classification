#!/usr/bin/env python3
"""Offline augmentation for the five source-held-out folds. Default: plan only.
Same transformations, probabilities, seeding and minimum-500 policy as 04_augment_train.py.
Requires completed output from 09_prepare_source_held_out.py. Python 3.10+.
Use the original augmentation environment (albumentations==2.0.8).
Plan writes reports only; --apply creates <fold>/augmented_train.
Existing outputs are never overwritten. Run --folds NAME to process selected folds.
"""
import argparse
import csv
import hashlib
import io
import json
import os
import platform
import random
import shutil
import sys
import tempfile
from collections import Counter,defaultdict
from datetime import datetime,timezone
from importlib.metadata import version
from pathlib import Path,PurePosixPath

EXTENSIONS={'.jpg','.jpeg','.jfif','.png','.bmp','.tif','.tiff','.webp','.gif'}


def sha_file(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def safe(root,relative):
    p=PurePosixPath(relative)
    if p.is_absolute() or not p.parts or any(x in {'.','..'} or ':' in x or '\\' in x for x in p.parts):raise ValueError(f'Unsafe path: {relative}')
    path=root.joinpath(*p.parts).resolve()
    if root not in path.parents:raise ValueError(f'Path escapes root: {relative}')
    return path


def write_csv(path,rows,fields):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)


def derive_seed(*parts):
    return int.from_bytes(hashlib.sha256('\n'.join(map(str,parts)).encode()).digest()[:4],'big')


FOLDS = ['fold_1_source_1', 'fold_2_source_2', 'fold_3_cauli_net', 'fold_4_vegnet', 'fold_5_source_5']
SOURCES = ['1_Cauliflower_Diseases_Identification_Image_Dataset', '2_Cauliflower_Dataset',
           '3_Cauli-Net', '4_VegNet', '5_Cauliflower_Image_Dataset']


def load_fold(root):
    """Validate provenance against the original retained manifest, not just counts."""
    parent = root.parent
    if (parent/'BUILD_IN_PROGRESS.json').exists():
        raise ValueError('Source-held-out preparation is incomplete.')
    protocol = json.loads((parent/'protocol.json').read_text(encoding='utf-8-sig'))
    if protocol.get('status') != 'complete':
        raise ValueError('Expected completed source-held-out protocol.')
    retained = parent/'retained_manifest_input.csv'
    if sha_file(retained) != protocol['manifest_sha256']:
        raise ValueError('Retained manifest hash differs from source-held-out protocol.')
    with retained.open(encoding='utf-8-sig', newline='') as f:
        originals = list(csv.DictReader(f))
    if len(originals) != protocol['images']:
        raise ValueError('Retained manifest count differs from protocol.')
    for field in ('cluster_id', 'destination_path', 'file_sha256', 'pixel_sha256'):
        values = [r[field] for r in originals]
        if not all(values) or len(set(values)) != len(values):
            raise ValueError('Empty or repeated retained ' + field)
    lookup = {r['cluster_id']:r for r in originals}
    groups = set()
    for row in originals:
        gs = set(json.loads(row['group_ids']))
        if gs & groups:
            raise ValueError('Duplicate group spans retained images.')
        groups.update(gs)
    report = root/'reports'
    summary = json.loads((report/'summary.json').read_text(encoding='utf-8-sig'))
    if root.name not in FOLDS or summary['fold'] != root.name:
        raise ValueError('Unexpected fold name.')
    held_out = SOURCES[FOLDS.index(root.name)]
    if summary['held_out_source'] != held_out:
        raise ValueError('Held-out source does not match fold name.')
    development, testing = set(), set()
    for row in originals:
        sources = set(json.loads(row['all_sources']))
        if not sources or not sources <= set(SOURCES):
            raise ValueError('Invalid source provenance.')
        if held_out not in sources:
            development.add(row['final_label'])
        if sources == {held_out}:
            testing.add(row['final_label'])
    classes = sorted(development & testing)
    labels = json.loads((report/'class_to_index.json').read_text(encoding='utf-8-sig'))
    if labels != {label:i for i,label in enumerate(classes)}:
        raise ValueError('Class mapping differs from shared classes.')
    manifest = report/'assignments.csv'
    with manifest.open(encoding='utf-8-sig', newline='') as f:
        assignments = list(csv.DictReader(f))
    if len(assignments) != len(originals) or {r['cluster_id'] for r in assignments} != set(lookup):
        raise ValueError('Assignments do not cover every retained image exactly once.')
    all_rows = []
    exclusions = Counter()
    for row in assignments:
        original = lookup[row['cluster_id']]
        for field in ('final_label', 'destination_path', 'file_sha256', 'pixel_sha256', 'all_sources', 'group_ids'):
            if row[field] != original[field]:
                raise ValueError('Assignment differs from retained manifest: ' + field)
        sources = set(json.loads(original['all_sources']))
        reason = ''
        if held_out in sources and len(sources) > 1:
            expected, reason = 'excluded', 'shared_with_held_out_source'
        elif original['final_label'] not in classes:
            expected, reason = 'excluded', 'class_not_shared'
        elif sources == {held_out}:
            expected = 'test'
        else:
            expected = 'development'
        part = row['partition']
        if (expected == 'development' and part not in ('train', 'val')) or (expected != 'development' and part != expected):
            raise ValueError('Partition violates source-held-out rules: ' + row['cluster_id'])
        if row['exclusion_reason'] != reason:
            raise ValueError('Incorrect exclusion reason.')
        if part == 'excluded':
            exclusions[reason] += 1
            continue
        rel = PurePosixPath(original['destination_path'])
        if len(rel.parts) != 2 or rel.parts[0] != original['final_label']:
            raise ValueError('Unexpected original class path.')
        converted = dict(original, split=part, split_path=f'{part}/{rel.as_posix()}')
        safe(root, converted['split_path'])
        all_rows.append(converted)
    totals = Counter(r['split'] for r in all_rows)
    if any(totals[s] != summary[s] for s in ('train', 'val', 'test')) or summary['eligible_classes'] != len(classes):
        raise ValueError('Fold summary differs from assignments.')
    if summary['excluded_shared'] != exclusions['shared_with_held_out_source'] or summary['excluded_class'] != exclusions['class_not_shared']:
        raise ValueError('Exclusion counts differ from assignments.')
    # Reproduce the split builder's per-class deterministic train/validation assignment.
    import math
    for label in classes:
        ids = sorted(r['cluster_id'] for r in originals
                     if r['final_label'] == label and held_out not in set(json.loads(r['all_sources'])))
        local_seed = int.from_bytes(hashlib.sha256(f"{protocol['split_seed']}|{held_out}|{label}".encode()).digest(), 'big')
        random.Random(local_seed).shuffle(ids)
        n = max(1, math.floor(len(ids)*protocol['validation_fraction']))
        actual_val = {r['cluster_id'] for r in all_rows if r['final_label']==label and r['split']=='val'}
        if actual_val != set(ids[:n]):
            raise ValueError('Validation assignments differ from frozen split rule.')
    for split in ('train', 'val', 'test'):
        expected = {safe(root,r['split_path']) for r in all_rows if r['split']==split}
        actual = {x.resolve() for x in (root/split).rglob('*') if x.is_file() and x.suffix.lower() in EXTENSIONS}
        if actual != expected:
            raise ValueError(f'{split} folder differs from assignments.')
        actual_classes = {p.name for p in (root/split).iterdir() if p.is_dir()}
        if actual_classes != set(classes):
            raise ValueError(f'{split} class folders differ from class mapping.')
    # Both preview and apply verify originals. Decoding is also checked during apply.
    for row in all_rows:
        if sha_file(safe(root,row['split_path'])) != row['file_sha256']:
            raise ValueError('Frozen image changed: ' + row['split_path'])
    return report, manifest, all_rows, originals, held_out


def build_transform(A,cv2):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.RandomBrightnessContrast(brightness_limit=0.2,contrast_limit=0.2,p=0.5),
        A.GaussianBlur(blur_limit=(3,3),p=0.05),
        A.CLAHE(clip_limit=4.0,tile_grid_size=(8,8),p=0.2),
        A.GaussNoise(std_range=(0.02,0.05),mean_range=(0.0,0.0),p=0.15),
        A.Affine(scale=(0.9,1.1),translate_percent=(-0.05,0.05),rotate=(-15,15),
                 shear=0,keep_ratio=True,interpolation=cv2.INTER_LINEAR,
                 border_mode=cv2.BORDER_REFLECT_101,p=0.5),
        A.HueSaturationValue(hue_shift_limit=5,sat_shift_limit=10,val_shift_limit=10,p=0.2),
    ],seed=42,save_applied_params=True)


def seed_candidate(transform, seed):
    # Compose propagates its seed to each child. Give each operation a separate
    # deterministic stream so their first probability draws are not identical.
    transform.set_random_seed(seed)
    for index, operation in enumerate(transform.transforms):
        operation.set_random_seed(derive_seed(seed, 'operation', index, type(operation).__name__))


def run_fold(argv):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--splits',type=Path,required=True)
    p.add_argument('--output',type=Path,help='Default: <splits>/augmented_train')
    p.add_argument('--reports',type=Path,help='Default: <fold>/offline_augmentation_reports')
    p.add_argument('--min-per-class',type=int,default=500)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--max-attempts',type=int,default=30,help='Maximum attempts per generated variant')
    p.add_argument('--apply',action='store_true',help='Generate and copy images; default plan only')
    a=p.parse_args(argv);root=a.splits.resolve()
    if not root.is_dir():raise ValueError(f'Split root not found: {root}')
    output=a.output.resolve() if a.output else root/'augmented_train'
    if output.exists():raise ValueError(f'Output already exists: {output}; it will not be overwritten.')
    if a.min_per_class<1 or a.max_attempts<1:raise ValueError('Counts and max-attempts must be positive.')
    raw=[root/s for s in ('train','val','test')]
    if any(output==s or s in output.parents or output in s.parents for s in raw):raise ValueError('Output must be separate from raw train/val/test folders.')
    report,manifest,all_rows,original_rows,held_out=load_fold(root)
    train=[r for r in all_rows if r['split']=='train'];by_class=defaultdict(list)
    for r in train:by_class[r['final_label']].append(r)
    targets=[dict(class_name=label,original=len(rs),target=max(len(rs),a.min_per_class),generated=max(0,a.min_per_class-len(rs))) for label,rs in sorted(by_class.items())]
    reports=a.reports.resolve() if a.reports else root/'offline_augmentation_reports'
    if any(reports==s or s in reports.parents or reports in s.parents for s in raw+[output]):raise ValueError('Reports must be separate from raw images and augmentation output.')
    reports.mkdir(parents=True,exist_ok=True)
    run=Path(tempfile.mkdtemp(prefix=datetime.now(timezone.utc).strftime('run_%Y%m%dT%H%M%SZ_'),dir=reports))
    write_csv(run/'class_counts.csv',targets,['class_name','original','target','generated'])
    summary=dict(script_version="source_held_out_1",held_out_source=held_out,mode='apply' if a.apply else 'plan_only',split_root=str(root),split_report=str(report),output=str(output),report_directory=str(run),
                 seed=a.seed,minimum_per_class=a.min_per_class,original_training_images=len(train),
                 planned_generated_images=sum(x['generated'] for x in targets),planned_total_images=sum(x['target'] for x in targets),
                 classes=len(targets),split_manifest_sha256=sha_file(manifest),completed=False,
                 notes=['Output includes unchanged original training images plus accepted offline variants.',
                        'No original is removed or downsampled; target is max(original_count, minimum).',
                        'No validation or test image is used as an augmentation parent.',
                        'Source-held-out assignments are checked against retained provenance and deterministic split rules.',
                        'Duplicate rejection includes all retained original pixel hashes, including fold exclusions.',
                        'Transform parameters follow the previous code except explicitly configured noise and modern Affine implementation.',
                        'No-op outputs and exact decoded-pixel matches to any original split or earlier generated output are retried.',
                        'Augmented variants are not independent observations; keep raw train for the no-offline baseline.',
                        'Inspect generated examples to assess preservation of disease symptoms.'])
    (run/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)
    if not a.apply:
        summary['completed']=True;(run/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        print('PLAN ONLY: no images copied/generated. Rerun with --apply.');return
    os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE','1')
    try:
        import albumentations as A
        import cv2
        import numpy as np
        from PIL import Image,ImageOps
    except ImportError as exc:raise ValueError('Use the same augmentation environment as before; albumentations==2.0.8, OpenCV, numpy and Pillow are required.') from exc
    if A.__version__!='2.0.8':raise ValueError(f'Use albumentations==2.0.8 for the documented pipeline; installed: {A.__version__}')
    cv2.setNumThreads(1)
    transform=build_transform(A,cv2)
    policy=A.to_dict(transform)
    policy.update(script_version=2,operation_seeding='each operation receives derive_seed(candidate_seed, operation, index, class_name); separate RNG streams',generated_format='JPEG',jpeg_quality=95,seed_derivation='first 4 SHA-256 bytes of master seed, class, variant index and attempt separated by newline',
                  acceptance='reject exact no-op before encoding and decoded-pixel duplicates after encoding; retry with a new seed',
                  parent_sampling='shuffled cycles without replacement within each cycle; only raw training images')
    (run/'augmentation_policy.json').write_text(json.dumps(policy,indent=2),encoding='utf-8')
    versions={}
    for k in ['albumentations','albucore','opencv-python-headless','opencv-python','numpy','Pillow','scipy','pydantic']:
        try: versions[k]=version(k)
        except Exception: versions[k]='not installed under this distribution name'
    versions.update(python=sys.version,platform=platform.platform())
    (run/'environment.json').write_text(json.dumps(versions,indent=2),encoding='utf-8')
    print('Verifying frozen train/val/test files...',flush=True)
    for i,r in enumerate(all_rows,1):
        if i==1 or i%100==0 or i==len(all_rows):print(f'  Verify {i}/{len(all_rows)}',flush=True)
        if sha_file(safe(root,r['split_path']))!=r['file_sha256']:raise ValueError(f'Split image changed since splitting: {r["split_path"]}')
    def pixel_sha(array):
        height,width=array.shape[:2]
        return hashlib.sha256(str((width,height)).encode()+b'|'+array.tobytes()).hexdigest()
    def load_rgb(path):
        with Image.open(path) as im:return np.array(ImageOps.exif_transpose(im).convert('RGB'))
    known_pixels={r['pixel_sha256'] for r in original_rows}
    output.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix=output.name+'_building_',dir=output.parent))
    records=[];retry_counts=Counter();parent_uses=Counter()
    def record(parent,rel,kind,file_sha,pix_sha,seed='',attempt='',applied=''):
        records.append(dict(output_path=rel,kind=kind,final_label=parent['final_label'],
                            parent_cluster_id=parent['cluster_id'],parent_split_path=parent['split_path'],
                            parent_file_sha256=parent['file_sha256'],parent_pixel_sha256=parent['pixel_sha256'],
                            output_file_sha256=file_sha,output_pixel_sha256=pix_sha,generation_seed=seed,attempt=attempt,
                            applied_transform_names=applied,all_sources=parent['all_sources'],all_source_paths=parent['all_source_paths']))
    try:
        for target in targets:
            label=target['class_name'];parents=sorted(by_class[label],key=lambda r:r['split_path'])
            destdir=staging/label;destdir.mkdir()
            print(f'{label}: originals={len(parents)}, generate={target["generated"]}, target={target["target"]}',flush=True)
            for parent in parents:
                src=safe(root,parent['split_path']);rel=f'{label}/{PurePosixPath(parent["split_path"]).name}';dst=safe(staging,rel)
                shutil.copy2(src,dst)
                if sha_file(dst)!=parent['file_sha256']:raise ValueError('Original copy verification failed.')
                record(parent,rel,'original',parent['file_sha256'],parent['pixel_sha256'])
            cycle=[]
            for index in range(target['generated']):
                if index%len(parents)==0:
                    cycle=list(parents);random.Random(derive_seed(a.seed,label,'parent_cycle',index//len(parents))).shuffle(cycle)
                parent=cycle[index%len(parents)];image=load_rgb(safe(root,parent['split_path']))
                if pixel_sha(image)!=parent['pixel_sha256']:raise ValueError(f'Decoded pixels differ from audit: {parent["split_path"]}')
                for attempt in range(a.max_attempts):
                    seed=derive_seed(a.seed,label,index,attempt)
                    seed_candidate(transform,seed);cv2.setRNGSeed(seed & 0x7fffffff)
                    result=transform(image=image);aug=result['image']
                    if np.array_equal(aug,image):retry_counts['no_op']+=1;continue
                    ok,encoded=cv2.imencode('.jpg',cv2.cvtColor(aug,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,95])
                    if not ok:raise ValueError('JPEG encoding failed.')
                    data=encoded.tobytes()
                    with Image.open(io.BytesIO(data)) as decoded:output_array=np.array(decoded.convert('RGB'))
                    pix_sha=pixel_sha(output_array)
                    if pix_sha in known_pixels:retry_counts['decoded_duplicate']+=1;continue
                    known_pixels.add(pix_sha)
                    token=hashlib.sha256(f'{a.seed}|{label}|{index}|{parent["cluster_id"]}'.encode()).hexdigest()[:16]
                    rel=f'{label}/aug_{index:05d}_{token}.jpg';dst=safe(staging,rel)
                    if dst.exists():raise ValueError(f'Output filename collision: {rel}')
                    dst.write_bytes(data);file_sha=hashlib.sha256(data).hexdigest()
                    if sha_file(dst)!=file_sha:raise ValueError('Generated image write verification failed.')
                    names=json.dumps([name for name,_ in result.get('applied_transforms',[])])
                    record(parent,rel,'offline_variant',file_sha,pix_sha,str(seed),str(attempt),names)
                    parent_uses[parent['split_path']]+=1
                    break
                else:raise ValueError(f'Cannot generate a nonduplicate variant after {a.max_attempts} attempts: {parent["split_path"]}')
                if index==0 or (index+1)%50==0 or index+1==target['generated']:print(f'  Generated {index+1}/{target["generated"]}',flush=True)
            if len(list(destdir.iterdir()))!=target['target']:raise ValueError(f'Count mismatch: {label}')
        if len(records)!=summary['planned_total_images']:raise ValueError('Total image count mismatch.')
        if output.exists():raise ValueError('Output appeared during processing; refusing overwrite.')
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging,ignore_errors=True);raise
    fields=['output_path','kind','final_label','parent_cluster_id','parent_split_path','parent_file_sha256','parent_pixel_sha256','output_file_sha256','output_pixel_sha256','generation_seed','attempt','applied_transform_names','all_sources','all_source_paths']
    write_csv(run/'augmentation_manifest.csv',records,fields)
    write_csv(run/'parent_usage.csv',[dict(parent_split_path=r['split_path'],class_name=r['final_label'],generated_variants=parent_uses[r['split_path']]) for r in sorted(train,key=lambda r:r['split_path'])],['parent_split_path','class_name','generated_variants'])
    applied_counts=Counter(name for r in records if r['kind']=='offline_variant' for name in json.loads(r['applied_transform_names']))
    generated_total=sum(x['generated'] for x in targets)
    applied_frequencies=[dict(transform_name=type(operation).__name__,nominal_probability=operation.p,
                              applied_count=applied_counts[type(operation).__name__],
                              accepted_variants=generated_total,
                              observed_fraction=applied_counts[type(operation).__name__]/generated_total if generated_total else None)
                         for operation in transform.transforms]
    write_csv(run/'transform_frequencies.csv',applied_frequencies,['transform_name','nominal_probability','applied_count','accepted_variants','observed_fraction'])
    summary.update(completed=True,copied_originals=len(train),generated_variants=sum(x['generated'] for x in targets),
                   final_total=len(records),retry_counts=dict(retry_counts),applied_transform_counts=dict(applied_counts),
                   source_verification='all_frozen_split_files_match_manifest',output_verification='original copies and generated JPEG writes verified; no accepted exact decoded matches to original splits or generated variants')
    (run/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2));print('Offline training collection created. Raw train/val/test unchanged.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(r'D:\Projects\CauliflowerV2\Dataset\source_held_out'))
    p.add_argument('--folds',nargs='+',choices=['all']+FOLDS,default=['all'])
    p.add_argument('--min-per-class',type=int,default=500)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--max-attempts',type=int,default=30)
    p.add_argument('--apply',action='store_true')
    a=p.parse_args()
    if a.min_per_class<1 or a.max_attempts<1:
        raise ValueError('Counts and maximum attempts must be positive.')
    if 'all' in a.folds and len(a.folds)!=1:
        raise ValueError('Use --folds all alone, or specify fold names.')
    selected=FOLDS if a.folds==['all'] else list(dict.fromkeys(a.folds))
    root=a.root.resolve()
    print('Preflight: checking all selected folds, provenance and original file hashes...',flush=True)
    for name in selected:
        fold=root/name
        if (fold/'augmented_train').exists():
            raise ValueError(f'Augmentation output already exists: {fold / "augmented_train"}; select only unfinished folds.')
        load_fold(fold)
        print('  Passed: '+name,flush=True)
    if a.apply:
        os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE','1')
        import albumentations as A
        import cv2
        import numpy
        from PIL import Image
        if A.__version__!='2.0.8':
            raise ValueError('Use albumentations==2.0.8, matching the original policy.')
        build_transform(A,cv2)
    for name in selected:
        print('\nProcessing '+name,flush=True)
        args=['--splits',str(root/name),'--min-per-class',str(a.min_per_class),
              '--seed',str(a.seed),'--max-attempts',str(a.max_attempts)]
        if a.apply: args.append('--apply')
        run_fold(args)
    print('\nCompleted all selected folds. Validation and test images unchanged.')


if __name__=='__main__':
    try:main()
    except Exception as exc:
        print(f'STOPPED: {exc}',file=sys.stderr);sys.exit(1)
