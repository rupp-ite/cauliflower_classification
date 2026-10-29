#!/usr/bin/env python3
"""Plan or create a harmonized dataset from an audited manifest and reviewed decisions.
Python 3.10+. Standard library only. Downloads and decision files are never changed.
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


def fail(message):
    raise ValueError(message)


def read_csv(path, required):
    if not path.is_file():
        fail(f'Missing input file: {path}')
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not set(required) <= set(reader.fieldnames or []):
            fail(f'{path.name}: missing columns {sorted(set(required)-set(reader.fieldnames or []))}')
        return list(reader)


def write_csv(path, rows, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def normalize(value):
    return ''.join(c for c in value.casefold() if c.isalnum())


def get_json_list(row, key):
    result = json.loads(row[key])
    if not isinstance(result, list) or any(not isinstance(x, str) for x in result):
        fail(f'{key} must be a JSON list of strings')
    return result


def safe_path(root, relative):
    # Audit CSV paths are portable, root-relative POSIX paths, also on Windows.
    parts = PurePosixPath(relative)
    if parts.is_absolute() or not parts.parts or any(x in {'..', '.'} or ':' in x or '\\' in x for x in parts.parts):
        fail(f'Unsafe relative path: {relative}')
    path = root.joinpath(*parts.parts).resolve()
    if root not in path.parents:
        fail(f'Path escapes download root: {relative}')
    return path


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--downloads', type=Path, default=Path('Dataset/downloaded_dataset'))
    parser.add_argument('--audit-dir', type=Path, required=True, help='Latest audit folder containing download_manifest.csv')
    here = Path(__file__).resolve().parent
    parser.add_argument('--mapping', type=Path, default=here/'source_label_mapping.csv')
    parser.add_argument('--exact', type=Path, default=here/'exact_duplicate_groups_reviewed.csv')
    parser.add_argument('--near', type=Path, default=here/'near_duplicate_candidates_reviewed.csv')
    parser.add_argument('--output', type=Path, default=Path('Dataset/harmonized_dataset'))
    parser.add_argument('--reports', type=Path, help='Report parent; defaults to <output_name>_reports beside output')
    parser.add_argument('--apply', action='store_true', help='Actually copy images; default only writes a cleaning plan')
    args = parser.parse_args()
    root, output, audit = args.downloads.resolve(), args.output.resolve(), args.audit_dir.resolve()
    if not root.is_dir():
        fail(f'Download directory not found: {root}')
    if output.exists():
        fail(f'Output already exists: {output}. Choose a new --output; existing datasets are never overwritten.')
    if output == root or root in output.parents or output in root.parents:
        fail('Output must be separate from, and not contain, the downloads.')
    if output == audit or output in audit.parents:
        fail('Output must not contain the audit inputs.')
    reports = args.reports.resolve() if args.reports else output.with_name(output.name+'_reports')
    if reports == root or root in reports.parents or reports == output or output in reports.parents or reports in output.parents:
        fail('Reports must be separate from downloads and cleaned image output.')
    mapping_rows = read_csv(args.mapping, ['source_folder','original_label','final_label','include'])
    mapping = {}
    for row in mapping_rows:
        key = (row['source_folder'].strip(), normalize(row['original_label']))
        include = row['include'].strip().casefold()
        label = row['final_label'].strip()
        if include not in {'yes','no'} or not key[1] or (include=='yes' and not label):
            fail(f'Invalid mapping entry: {row}')
        if label and (label in {'.','..'} or any(c in label for c in '<>:"/\\|?*') or label.endswith(('.', ' '))):
            fail(f'Invalid class-folder name: {label}')
        value = (label, include)
        if key in mapping and mapping[key] != value:
            fail(f'Conflicting mapping key: {key}')
        mapping[key] = value
    manifest_path = audit/'download_manifest.csv'
    manifest = read_csv(manifest_path, ['path','source_id','original_label','final_label','include','layout_error','error','file_sha256','pixel_sha256','exact_group','width','height'])
    exact = read_csv(args.exact, ['group_id','copies','paths','decision','review_notes'])
    near = read_csv(args.near, ['group_a','group_b','paths_a','paths_b','decision','review_notes'])
    if not manifest:
        fail('Empty manifest')
    groups, by_path = defaultdict(list), {}
    print('Validating manifest, source files, mapping and decisions...', flush=True)
    for row in manifest:
        path = row['path']
        if path in by_path:
            fail(f'Repeated manifest path: {path}')
        if row['error'] or row['layout_error']:
            fail(f'Unresolved decode/layout issue: {path}')
        key = (row['source_id'], normalize(row['original_label']))
        mapped = mapping.get(key)
        if mapped is None:
            fail(f'Unmapped label: {key}')
        label, include = mapped
        if (row['final_label'].strip(),row['include'].strip().casefold()) != mapped:
            fail(f'Stale manifest mapping for {path}. Rerun audit with the supplied mapping first.')
        sha = row['pixel_sha256']
        if len(sha)!=64 or any(c not in '0123456789abcdef' for c in sha):
            fail(f'Missing/invalid pixel hash: {path}')
        if row['exact_group'] != 'PX_'+sha[:16]:
            fail(f'Group/hash mismatch: {path}')
        if len(row['file_sha256'])!=64:
            fail(f'Missing file hash: {path}')
        if int(row['width'])<=0 or int(row['height'])<=0:
            fail(f'Invalid dimensions: {path}')
        if not safe_path(root,path).is_file():
            fail(f'Source file missing: {path}')
        row['_include'],row['_label'] = include,label
        by_path[path]=row;groups[row['exact_group']].append(row)
    for gid,members in groups.items():
        if len({r['pixel_sha256'] for r in members})!=1:
            fail(f'Truncated-hash collision: {gid}')
    expected_exact = {g for g,rs in groups.items() if len(rs)>1}
    seen, blocked = set(), set()
    for row in exact:
        gid = row['group_id']
        if gid in seen or gid not in groups:
            fail(f'Duplicate or unknown reviewed group: {gid}')
        seen.add(gid)
        if set(get_json_list(row,'paths')) != {r['path'] for r in groups[gid]} or int(row['copies'])!=len(groups[gid]):
            fail(f'Exact decision does not match current manifest: {gid}')
        decision = row['decision'].strip().casefold()
        if decision not in {'keep_one','exclude_conflict','exclude_out_of_scope'}:
            fail(f'Unresolved/invalid exact decision for {gid}: {decision!r}')
        labels = {r['_label'] for r in groups[gid] if r['_include']=='yes'}
        states = {r['_include'] for r in groups[gid]}
        if decision=='keep_one' and (len(labels)!=1 or states!={'yes'}):
            fail(f'keep_one is inconsistent with labels or inclusion states: {gid}')
        if decision=='exclude_out_of_scope' and 'yes' in states:
            fail(f'Out-of-scope decision contains included files: {gid}')
        if decision=='exclude_conflict':
            blocked.add(gid)
    if seen != expected_exact:
        fail('Reviewed exact report does not cover precisely the duplicate groups in this manifest.')
    parent = {g:g for g in groups}
    def find(g):
        while parent[g]!=g:
            parent[g]=parent[parent[g]];g=parent[g]
        return g
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[b]=a
    seen_pairs=set();independent=[]
    for row in near:
        a,b=row['group_a'],row['group_b']
        if a not in groups or b not in groups or a==b:
            fail(f'Unknown/self-linked near group: {a}, {b}')
        key=frozenset((a,b))
        if key in seen_pairs:fail(f'Duplicate reviewed pair: {a}, {b}')
        seen_pairs.add(key)
        for gid,side in [(a,'a'),(b,'b')]:
            if set(get_json_list(row,'paths_'+side)) != {r['path'] for r in groups[gid]}:
                fail(f'Near decision paths differ from manifest: {gid}')
        decision=row['decision'].strip().casefold()
        if decision=='duplicate':union(a,b)
        elif decision=='independent':independent.append((a,b))
        else:fail(f'Unresolved/unsupported near decision {decision!r}: {a}, {b}. This package expects reviewed duplicate/independent decisions.')
    for a,b in independent:
        if find(a)==find(b):fail(f'Inconsistent decisions: independent pair connected through duplicate links: {a}, {b}')
    # Verify the reviewed file covers the complete latest candidate report.
    baseline = read_csv(audit/'near_duplicate_candidates.csv',['group_a','group_b'])
    baseline_pairs=[frozenset((r['group_a'],r['group_b'])) for r in baseline]
    if len(baseline_pairs)!=len(set(baseline_pairs)) or set(baseline_pairs)!=seen_pairs:
        fail('Reviewed near report does not cover precisely the latest audit candidates.')
    components=defaultdict(list)
    for gid in groups:components[find(gid)].append(gid)
    image_plan=[];cluster_plan=[];retained=[]
    for gids in components.values():
        gids=sorted(gids)
        cluster='CL_'+hashlib.sha256('\n'.join(gids).encode()).hexdigest()[:20]
        members=[r for g in gids for r in groups[g]]
        eligible=[r for r in members if r['_include']=='yes']
        labels=sorted({r['_label'] for r in eligible})
        block=sorted(set(gids)&blocked)
        reasons=[]
        if block:reasons.append('contains_previously_excluded_conflict_group')
        if len(labels)>1:reasons.append('competing_included_labels_in_confirmed_duplicate_cluster')
        if eligible and any(r['_include']=='no' for r in members):
            reasons.append('included_and_excluded_source_labels_in_same_duplicate_cluster')
        if not eligible:action='exclude_out_of_scope'
        elif reasons:action='exclude_conflict'
        else:action='keep_one'
        chosen=None;destination=''
        if action=='keep_one':
            # Deterministic convenience rule, not a diagnosis or quality assessment.
            chosen=min(eligible,key=lambda r:(-int(r['width'])*int(r['height']),r['path'].casefold(),r['path']))
            extension=Path(chosen['path']).suffix.lower()
            destination=f'{labels[0]}/{cluster}{extension}'
            retained.append(dict(cluster_id=cluster,final_label=labels[0],source_path=chosen['path'],destination_path=destination,
                                 file_sha256=chosen['file_sha256'],pixel_sha256=chosen['pixel_sha256'],
                                 all_sources=json.dumps(sorted({r['source_id'] for r in members})),
                                 all_original_labels=json.dumps(sorted({r['original_label'] for r in members})),
                                 all_source_paths=json.dumps(sorted(r['path'] for r in members)),group_ids=json.dumps(gids)))
        cluster_plan.append(dict(cluster_id=cluster,action=action,file_count=len(members),included_files=len(eligible),
                                 included_labels=json.dumps(labels),reason='; '.join(reasons) or ('outside_scope' if not eligible else 'consistent_label'),
                                 excluded_group_ids=json.dumps(block),group_ids=json.dumps(gids),retained_source=chosen['path'] if chosen else '',
                                 all_source_paths=json.dumps(sorted(r['path'] for r in members))))
        for row in members:
            if row['_include']=='no':status='exclude_out_of_scope'
            elif action=='exclude_conflict':status='exclude_conflict'
            elif row is chosen:status='retain'
            else:status='exclude_redundant_copy'
            image_plan.append(dict(path=row['path'],source_id=row['source_id'],original_label=row['original_label'],
                                   final_label=row['_label'],include=row['_include'],exact_group=row['exact_group'],cluster_id=cluster,
                                   status=status,reason='; '.join(reasons) or ('mapping_exclusion' if row['_include']=='no' else 'one_representative_per_confirmed_duplicate_cluster'),
                                   retained_source=chosen['path'] if chosen else '',destination_path=destination))
    if not retained:fail('No retained images; inspect mapping and decisions.')
    if len({r['destination_path'].casefold() for r in retained})!=len(retained):fail('Destination collision')
    if len({r['pixel_sha256'] for r in retained})!=len(retained):fail('Repeated decoded pixel content among representatives')
    label_counts=Counter(r['final_label'] for r in retained)
    missing_labels=sorted({r['_label'] for r in manifest if r['_include']=='yes'}-set(label_counts))
    summary=dict(mode='apply' if args.apply else 'plan_only',download_root=str(root),output=str(output),
                 input_files=len(manifest),included_before_cleaning=sum(r['_include']=='yes' for r in manifest),
                 exact_review_groups=len(exact),near_review_pairs=len(near),
                 exact_decisions=dict(Counter(r['decision'].strip().casefold() for r in exact)),
                 near_decisions=dict(Counter(r['decision'].strip().casefold() for r in near)),
                 statuses=dict(Counter(r['status'] for r in image_plan)),
                 retained_images=len(retained),retained_classes=len(label_counts),retained_by_class=dict(sorted(label_counts.items())),
                 zero_remaining_classes=missing_labels,source_file_verification='not_run_in_plan_mode',
                 representative_rule='Largest decoded pixel area, then lexicographically first eligible source path; not a quality or diagnostic claim.',
                 notes=['All source copies of retained images are recorded in retained_manifest.csv.',
                        'All duplicate-linked copies of excluded conflict groups remain excluded.',
                        'Mixed included/excluded clusters are excluded conservatively.',
                        'No train/validation/test split or augmentation is created.',
                        'Zero remaining classes must be investigated before training.',
                        'No claim that unflagged images are free of near-duplicates or related captures.'])
    reports.mkdir(parents=True,exist_ok=True)
    reportdir=Path(tempfile.mkdtemp(prefix=datetime.now(timezone.utc).strftime('run_%Y%m%dT%H%M%SZ_'),dir=reports))
    for name,path in [('mapping.csv',args.mapping),('exact_decisions.csv',args.exact),('near_decisions.csv',args.near)]:shutil.copyfile(path,reportdir/name)
    input_hashes={str(path.resolve()):sha256_file(path) for path in [manifest_path,args.mapping,args.exact,args.near,audit/'near_duplicate_candidates.csv']}
    (reportdir/'input_file_hashes.json').write_text(json.dumps(input_hashes,indent=2),encoding='utf-8')
    write_csv(reportdir/'image_decisions.csv',sorted(image_plan,key=lambda r:r['path']),['path','source_id','original_label','final_label','include','exact_group','cluster_id','status','reason','retained_source','destination_path'])
    write_csv(reportdir/'cluster_decisions.csv',sorted(cluster_plan,key=lambda r:r['cluster_id']),['cluster_id','action','file_count','included_files','included_labels','reason','excluded_group_ids','group_ids','retained_source','all_source_paths'])
    write_csv(reportdir/'retained_manifest.csv',sorted(retained,key=lambda r:r['destination_path']),['cluster_id','final_label','source_path','destination_path','file_sha256','pixel_sha256','all_sources','all_original_labels','all_source_paths','group_ids'])
    write_csv(reportdir/'class_counts.csv',[{'class':k,'retained_images':v} for k,v in sorted(label_counts.items())],['class','retained_images'])
    summary['report_directory']=str(reportdir)
    (reportdir/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    if args.apply:
        print('Verifying every downloaded file against its audited SHA-256...',flush=True)
        for i,row in enumerate(manifest,1):
            if i==1 or i%100==0 or i==len(manifest):print(f'  Verification {i}/{len(manifest)}',flush=True)
            if sha256_file(safe_path(root,row['path']))!=row['file_sha256']:
                fail(f'File changed since audit: {row["path"]}. No cleaned dataset was published; rerun the audit.')
        output.parent.mkdir(parents=True,exist_ok=True)
        staging=Path(tempfile.mkdtemp(prefix=output.name+'_building_',dir=output.parent))
        try:
            for i,row in enumerate(retained,1):
                if i==1 or i%100==0 or i==len(retained):print(f'  Copying {i}/{len(retained)}',flush=True)
                destination=safe_path(staging,row['destination_path'])
                destination.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(safe_path(root,row['source_path']),destination)
                if sha256_file(destination)!=row['file_sha256']:fail(f'Copy verification failed: {row["source_path"]}')
            if output.exists():fail('Output appeared during copying; refusing to overwrite it.')
            staging.rename(output)
        except BaseException:
            # Remove only this run's temporary copies; original files remain untouched.
            shutil.rmtree(staging,ignore_errors=True)
            raise
        summary.update(source_file_verification='all_downloaded_files_match_audit',copied_and_verified=len(retained),completed=True)
    else:
        summary['completed']=True
    (reportdir/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
    print('Images copied and verified.' if args.apply else 'PLAN ONLY: no images copied. Review reports, then rerun with --apply.')


if __name__=='__main__':
    try:
        main()
    except (ValueError,OSError,KeyError,json.JSONDecodeError) as exc:
        print(f'STOPPED: {exc}',file=sys.stderr)
        sys.exit(1)
