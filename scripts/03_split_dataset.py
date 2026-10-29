#!/usr/bin/env python3
"""Reproducible class-stratified splitting of a cleaned, audited image collection.
Python 3.10+. Standard library only. Default is plan-only; --apply copies images.
"""
import argparse
import csv
import hashlib
import json
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

SPLITS = ('train', 'val', 'test')
RATIOS = (70, 15, 15)
EXTENSIONS = {'.jpg','.jpeg','.jfif','.png','.bmp','.tif','.tiff','.webp','.gif'}


def sha_file(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def relative_file(root,relative):
    rel=PurePosixPath(relative)
    if rel.is_absolute() or not rel.parts or any(x in {'.','..'} or ':' in x or '\\' in x for x in rel.parts):
        raise ValueError(f'Unsafe relative path: {relative}')
    result=root.joinpath(*rel.parts).resolve()
    if root not in result.parents:raise ValueError(f'Path escapes dataset root: {relative}')
    return result


def allocate(n):
    # Equal validation/test counts per class; training receives the remainder.
    # Integer arithmetic implements floor(15% of n) exactly.
    evaluation_count = n * 15 // 100
    if evaluation_count < 1:
        raise ValueError(f'Class size {n} leaves validation/test empty under the 15% floor rule.')
    return {'train': n - 2 * evaluation_count, 'val': evaluation_count, 'test': evaluation_count}


def write_csv(path,rows,fields):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)


def choose_report(dataset,explicit):
    if explicit:
        report=explicit.resolve()
        s=json.loads((report/'summary.json').read_text(encoding='utf-8-sig'))
        if s.get('mode')!='apply' or not s.get('completed'):
            raise ValueError('The cleaning report must be from a completed APPLY run.')
        if Path(s['output']).resolve()!=dataset:
            raise ValueError('Cleaning report output path does not match --dataset.')
        return report,s
    root=dataset.with_name(dataset.name+'_reports')
    matches=[]
    if root.is_dir():
        for p in sorted(root.glob('*/summary.json')):
            s=json.loads(p.read_text(encoding='utf-8-sig'))
            if s.get('mode')=='apply' and s.get('completed') and Path(s['output']).resolve()==dataset:
                matches.append((p.parent,s))
    if len(matches)!=1:
        raise ValueError(f'Found {len(matches)} matching completed cleaning reports. Supply --cleaning-report with the correct run directory.')
    return matches[0]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=Path('Dataset/harmonized_dataset'))
    p.add_argument('--cleaning-report',type=Path,help='Completed cleaning APPLY report folder; auto-detected when unique')
    p.add_argument('--output',type=Path,default=Path('Dataset/harmonized_splits'))
    p.add_argument('--reports',type=Path,help='Report parent; default harmonized_splits_reports beside output')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--apply',action='store_true',help='Copy split images; default writes a plan only')
    a=p.parse_args()
    dataset,output=a.dataset.resolve(),a.output.resolve()
    if not dataset.is_dir():raise ValueError(f'Cleaned dataset not found: {dataset}')
    if output.exists():raise ValueError(f'Output already exists: {output}. It will not be overwritten.')
    if output==dataset or dataset in output.parents or output in dataset.parents:
        raise ValueError('Split output must be separate from the cleaned dataset.')
    report,clean_summary=choose_report(dataset,a.cleaning_report)
    print(f'Using cleaning report: {report}',flush=True)
    manifest_path=report/'retained_manifest.csv'
    with manifest_path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        required={'cluster_id','final_label','source_path','destination_path','file_sha256','pixel_sha256','all_sources','all_source_paths','group_ids'}
        if not required<=set(reader.fieldnames or []):raise ValueError('Cleaning manifest is missing required fields.')
        rows=list(reader)
    if not rows or len(rows)!=clean_summary['retained_images']:
        raise ValueError('Retained manifest count does not match the cleaning summary.')
    for field in ['cluster_id','destination_path','file_sha256','pixel_sha256']:
        if any(not r[field] for r in rows) or len({r[field].casefold() for r in rows})!=len(rows):
            raise ValueError(f'Missing/repeated {field}. Investigate before splitting.')
    by_class=defaultdict(list)
    expected_paths=set()
    for r in rows:
        rel=PurePosixPath(r['destination_path'])
        if len(rel.parts)!=2 or rel.parts[0]!=r['final_label']:
            raise ValueError(f'Unexpected class/path: {r["destination_path"]}')
        path=relative_file(dataset,r['destination_path'])
        if not path.is_file():raise ValueError(f'Missing cleaned image: {path}')
        expected_paths.add(path)
        by_class[r['final_label']].append(r)
    actual={x.resolve() for x in dataset.rglob('*') if x.is_file() and x.suffix.lower() in EXTENSIONS}
    if actual!=expected_paths:
        raise ValueError(f'Folder differs from cleaning manifest: {len(actual-expected_paths)} extra images, {len(expected_paths-actual)} missing images.')
    if dict(Counter(r['final_label'] for r in rows))!=clean_summary['retained_by_class']:
        raise ValueError('Class counts differ from cleaning summary.')
    assignments=[];counts=[]
    for label,members in sorted(by_class.items()):
        sizes=allocate(len(members))
        # Stable seeded pseudo-random ordering; independent of folder enumeration,
        # Python random implementation, and training random seeds.
        ordered=sorted(members,key=lambda r:(hashlib.sha256(f'{a.seed}\n{r["destination_path"]}'.encode('utf-8')).digest(),r['destination_path']))
        start=0
        for split in SPLITS:
            stop=start+sizes[split]
            for r in ordered[start:stop]:
                assignments.append(dict(r,split=split,cleaned_path=r['destination_path'],split_path=f'{split}/{r["destination_path"]}'))
            start=stop
        assert start==len(members)
        counts.append(dict(class_name=label,total=len(members),**sizes))
    for field in ['cluster_id','pixel_sha256','file_sha256','cleaned_path']:
        membership=defaultdict(set)
        for r in assignments:membership[r[field]].add(r['split'])
        if any(len(v)>1 for v in membership.values()):raise ValueError(f'Cross-split overlap detected for {field}')
    if len(assignments)!=len(rows):raise ValueError('Allocation count mismatch.')
    totals=Counter(r['split'] for r in assignments)
    reports=a.reports.resolve() if a.reports else output.with_name(output.name+'_reports')
    if any(reports==x or x in reports.parents or reports in x.parents for x in [dataset,output]):
        raise ValueError('Reports must be separate from source images and split output.')
    reports.mkdir(parents=True,exist_ok=True)
    run=Path(tempfile.mkdtemp(prefix=datetime.now(timezone.utc).strftime('run_%Y%m%dT%H%M%SZ_'),dir=reports))
    fields=['cluster_id','final_label','split','cleaned_path','split_path','source_path','file_sha256','pixel_sha256','all_sources','all_original_labels','all_source_paths','group_ids']
    assignments.sort(key=lambda r:r['split_path'])
    write_csv(run/'split_manifest.csv',assignments,fields)
    for split in SPLITS:write_csv(run/f'{split}_manifest.csv',[r for r in assignments if r['split']==split],fields)
    write_csv(run/'class_counts.csv',counts,['class_name','total','train','val','test'])
    protocol=dict(seed=a.seed,target_percentages=dict(zip(SPLITS,RATIOS)),
                  allocation='Per class: val = test = floor(15*n/100); train = n - val - test.',
                  ordering='SHA-256 of seed, newline, cleaned relative path; ascending digest, then path.',
                  split_manifest_sha256=sha_file(run/'split_manifest.csv'),
                  cleaning_manifest_sha256=sha_file(manifest_path))
    (run/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    summary=dict(mode='apply' if a.apply else 'plan_only',dataset=str(dataset),output=str(output),
                 cleaning_report=str(report),report_directory=str(run),seed=a.seed,
                 total_images=len(rows),classes=len(by_class),split_totals=dict(totals),
                 percentages={s:round(100*totals[s]/len(rows),3) for s in SPLITS},
                 overlap_checks='passed: cluster IDs, audited pixel hashes, file hashes and cleaned paths',
                 byte_verification='not_run_in_plan_mode',
                 notes=['No augmentation or source-held-out evaluation is performed.',
                        'Keep these assignments fixed across model comparisons and training seeds.',
                        'The script cannot infer plant/session identities. Known related captures must be grouped before using this protocol.',
                        'Hash checks and prior review do not prove absence of every possible related capture or near-duplicate.'])
    (run/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    if a.apply:
        print('Verifying cleaned image bytes against retained manifest...',flush=True)
        for i,r in enumerate(assignments,1):
            if i==1 or i%100==0 or i==len(assignments):print(f'  Verify {i}/{len(assignments)}',flush=True)
            if sha_file(relative_file(dataset,r['cleaned_path']))!=r['file_sha256']:
                raise ValueError(f'Cleaned file changed since cleaning: {r["cleaned_path"]}')
        output.parent.mkdir(parents=True,exist_ok=True)
        stage=Path(tempfile.mkdtemp(prefix=output.name+'_building_',dir=output.parent))
        try:
            for i,r in enumerate(assignments,1):
                if i==1 or i%100==0 or i==len(assignments):print(f'  Copy {i}/{len(assignments)}',flush=True)
                src=relative_file(dataset,r['cleaned_path']);dest=relative_file(stage,r['split_path'])
                dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
                if sha_file(dest)!=r['file_sha256']:raise ValueError(f'Copy verification failed: {r["split_path"]}')
            if output.exists():raise ValueError('Output appeared during copying; refusing to overwrite.')
            stage.rename(output)
        except BaseException:
            shutil.rmtree(stage,ignore_errors=True);raise
        summary.update(byte_verification='all_source_and_destination_files_verified',copied_and_verified=len(rows))
    summary['completed']=True
    (run/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
    print('Split images copied and verified.' if a.apply else 'PLAN ONLY: no images copied. Rerun with --apply to create the splits.')


if __name__=='__main__':
    try:main()
    except (ValueError,OSError,KeyError,json.JSONDecodeError) as exc:
        print(f'STOPPED: {exc}',file=sys.stderr);sys.exit(1)
