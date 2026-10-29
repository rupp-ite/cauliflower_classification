#!/usr/bin/env python3
"""Read-only, source-aware audit. Python 3.10+ and Pillow."""
import argparse
import csv
import hashlib
import html
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageOps

SOURCES = (
    '1_Cauliflower_Diseases_Identification_Image_Dataset',
    '2_Cauliflower_Dataset',
    '3_Cauli-Net',
    '4_VegNet',
    '5_Cauliflower_Image_Dataset', 
)
EXTENSIONS = {
    '.jpg', '.jpeg', '.jfif', '.png',
    '.bmp', '.tif', '.tiff', '.webp', '.gif'
}

def normalize(value):
    return ''.join(c for c in value.casefold() if c.isalnum())


def write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def image_info(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    with Image.open(path) as opened:
        frames = getattr(opened, 'n_frames', 1)
        im = ImageOps.exif_transpose(opened).convert('RGB')
        im.load()
        pixels = im.convert('L').resize((9, 8), Image.Resampling.LANCZOS).tobytes()
        value = 0
        for y in range(8):
            for x in range(8):
                value = (value << 1) | (pixels[y * 9 + x] > pixels[y * 9 + x + 1])
        return dict(file_sha256=h.hexdigest(),
                    pixel_sha256=hashlib.sha256(str(im.size).encode() + b'|' + im.tobytes()).hexdigest(),
                    dhash64=f'{value:016x}', width=im.width, height=im.height, frames=frames)


def load_mapping(path):
    result = {}
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        required = {'source_folder', 'original_label', 'final_label', 'include'}
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f'Mapping CSV must contain {sorted(required)}')
        for row in reader:
            key = (row['source_folder'].strip(), normalize(row['original_label']))
            if not key[1]:
                raise ValueError('Empty original_label in mapping')
            flag = row['include'].strip().casefold()
            if flag not in {'yes', 'no'}:
                raise ValueError(f'include must be yes/no: {row}')
            target = row['final_label'].strip()
            if flag == 'yes' and not target:
                raise ValueError(f'Included label has no final_label: {row}')
            value = (target, flag)
            if key in result and result[key] != value:
                raise ValueError(f'Conflicting normalized mapping: {key}')
            result[key] = value
    return result


def class_from_path(source, relative):
    parts = relative.parts
    if source == SOURCES[1]:
        if len(parts) < 3 or normalize(parts[0]) not in {'cauliflowercategories', 'leavescategories'}:
            return '', 'Expected Cauliflower Categories/<class>/image or Leaves Categories/<class>/image'
        return parts[1], ''
    if len(parts) < 2:
        return '', 'Expected <class>/image'
    return parts[0], ''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--downloads', type=Path, default=Path('Dataset/downloaded_dataset'))
    parser.add_argument('--mapping', type=Path, default=Path(__file__).with_name('source_label_mapping.csv'))
    parser.add_argument('--harmonized', type=Path, help='Optional existing original folder for exact provenance matching')
    parser.add_argument('--output', type=Path, default=Path('source_audit_results'))
    parser.add_argument('--distance', type=int, default=6)
    parser.add_argument('--previews', type=int, default=100, help='Near-candidate previews; all label conflicts are previewed separately')
    args = parser.parse_args()
    root, out = args.downloads.resolve(), args.output.resolve()
    if not root.is_dir():
        parser.error(f'Download root not found: {root}')
    if not args.mapping.is_file():
        parser.error(f'Mapping not found: {args.mapping}')
    if not 0 <= args.distance <= 64 or args.previews < 0:
        parser.error('distance must be 0-64 and previews must be nonnegative')
    protected = [root] + ([args.harmonized.resolve()] if args.harmonized else [])
    if any(out == p or p in out.parents for p in protected):
        parser.error('Output must be outside the image directories')
    if args.harmonized and not args.harmonized.is_dir():
        parser.error(f'Harmonized folder not found: {args.harmonized}')
    present = [s for s in SOURCES if (root / s).is_dir()]
    if not present:
        parser.error('No expected source folders found. Check folder spellings and --downloads path.')
    mapping = load_mapping(args.mapping)
    if out.exists():
        parser.error('Output exists; use a new --output name to preserve earlier review decisions')
    out.mkdir(parents=True)
    (out / 'previews').mkdir()
    rows, ignored = [], []
    for source in present:
        paths = sorted(p for p in (root / source).rglob('*') if p.is_file())
        for index, path in enumerate(paths, 1):
            if index == 1 or index % 100 == 0 or index == len(paths):
                print(f'{source}: {index}/{len(paths)} files', flush=True)
            rel = path.relative_to(root).as_posix()
            if path.suffix.casefold() not in EXTENSIONS:
                ignored.append({'path': rel, 'reason': 'Unsupported extension'})
                continue
            label, layout_error = class_from_path(source, path.relative_to(root / source))
            final, include = mapping.get((source, normalize(label)), ('', 'unknown'))
            row = dict(image_id=hashlib.sha256(rel.encode()).hexdigest()[:16], path=rel,
                       source_id=source, original_label=label, final_label=final, include=include,
                       mapping_status='unmapped' if include == 'unknown' else 'mapped',
                       layout_error=layout_error, error='', exact_group='', file_sha256='',
                       pixel_sha256='', dhash64='', width='', height='', frames='')
            try:
                row.update(image_info(path))
            except Exception as exc:
                row['error'] = f'{type(exc).__name__}: {exc}'
            rows.append(row)
    valid = [r for r in rows if not r['error']]
    groups = defaultdict(list)
    for r in valid:
        groups[r['pixel_sha256']].append(r)
    exact, conflicts = [], []
    for sha, members in sorted(groups.items()):
        group = 'PX_' + sha[:16]
        for r in members:
            r['exact_group'] = group
        # Conflict means competing included harmonized targets, not mere spelling differences.
        labels = sorted({r['final_label'] for r in members if r['include'] == 'yes'})
        sources = sorted({r['source_id'] for r in members})
        item = dict(group_id=group, copies=len(members), cross_source=len(sources) > 1,
                    included_label_conflict=len(labels) > 1, sources=json.dumps(sources),
                    original_labels=json.dumps(sorted({r['original_label'] for r in members})),
                    final_labels=json.dumps(labels), include_states=json.dumps(sorted({r['include'] for r in members})),
                    paths=json.dumps([r['path'] for r in members]), decision='', review_notes='')
        if len(members) > 1:
            exact.append(item)
        if len(labels) > 1:
            conflicts.append(item)
    fields = ['image_id','path','source_id','original_label','final_label','include','mapping_status',
              'layout_error','width','height','frames','file_sha256','pixel_sha256','dhash64','exact_group','error']
    write_csv(out / 'download_manifest.csv', rows, fields)
    write_csv(out / 'mapping_or_layout_review.csv', [r for r in rows if r['mapping_status'] == 'unmapped' or r['layout_error']], fields)
    write_csv(out / 'corrupt_images.csv', [r for r in rows if r['error']], fields)
    write_csv(out / 'ignored_files.csv', ignored, ['path','reason'])
    group_fields = ['group_id','copies','cross_source','included_label_conflict','sources','original_labels','final_labels','include_states','paths','decision','review_notes']
    write_csv(out / 'exact_duplicate_groups.csv', exact, group_fields)
    write_csv(out / 'label_conflicts.csv', conflicts, group_fields)
    counts = Counter((r['source_id'],r['original_label'],r['final_label'],r['include']) for r in rows)
    write_csv(out / 'source_class_counts.csv', [dict(source_id=k[0],original_label=k[1],final_label=k[2],include=k[3],count=v) for k,v in sorted(counts.items())], ['source_id','original_label','final_label','include','count'])
    reps = [(m[0],m,int(m[0]['dhash64'],16)) for m in groups.values()]
    tree, near = None, []
    for i,(a,am,ah) in enumerate(reps):
        if i % 100 == 0 or i+1 == len(reps):
            print(f'Similarity screening: {i+1}/{len(reps)} unique images; {len(near)} candidates',flush=True)
        stack = [tree] if tree else []
        while stack:
            node = stack.pop()
            distance = (ah ^ node[0]).bit_count()
            if distance <= args.distance:
                for j in node[1]:
                    b,bm,_ = reps[j]
                    near.append(dict(group_a=a['exact_group'],group_b=b['exact_group'],distance=distance,
                                     cross_source=any(x['source_id']!=y['source_id'] for x in am for y in bm),
                                     labels_a=json.dumps(sorted({x['final_label'] or '[UNMAPPED]' for x in am})),
                                     labels_b=json.dumps(sorted({x['final_label'] or '[UNMAPPED]' for x in bm})),
                                     paths_a=json.dumps([x['path'] for x in am]),paths_b=json.dumps([x['path'] for x in bm]),
                                     decision='',review_notes='',_members=am+bm))
            stack.extend(child for edge,child in node[2].items() if distance-args.distance<=edge<=distance+args.distance)
        if tree is None:
            tree=[ah,[i],{}]
        else:
            node=tree
            while True:
                distance=(ah^node[0]).bit_count()
                if distance==0:
                    node[1].append(i);break
                if distance not in node[2]:
                    node[2][distance]=[ah,[i],{}];break
                node=node[2][distance]
    near.sort(key=lambda r:(not r['cross_source'],r['distance'],r['group_a'],r['group_b']))
    write_csv(out/'near_duplicate_candidates.csv',near,['group_a','group_b','distance','cross_source','labels_a','labels_b','paths_a','paths_b','decision','review_notes'])
    def gallery(filename,title,items):
        cards=[]
        for n,(heading,members) in enumerate(items,1):
            if n==1 or n%25==0:
                print(f'{title}: preview {n}',flush=True)
            figures=[]
            for k,r in enumerate(members):
                thumbname=f'{Path(filename).stem}_{n:04d}_{k:02d}.jpg'
                with Image.open(root/r['path']) as im:
                    im=ImageOps.exif_transpose(im).convert('RGB');im.thumbnail((320,240));im.save(out/'previews'/thumbname)
                caption=f"{r['source_id']} / {r['original_label']} → {r['final_label'] or '[unmapped]'}; include={r['include']}"
                figures.append(f'<figure><img src="previews/{thumbname}"><figcaption>{html.escape(caption)}<br>{html.escape(r["path"])}</figcaption></figure>')
            cards.append(f'<section><h2>{html.escape(heading)}</h2><div class="images">'+''.join(figures)+'</div></section>')
        (out/filename).write_text('<!doctype html><meta charset="utf-8"><title>'+html.escape(title)+'</title><style>body{font-family:sans-serif;margin:24px}section{border-top:1px solid #bbb;padding:12px 0}.images{display:flex;flex-wrap:wrap;gap:12px}figure{margin:0;width:320px}img{max-width:320px}figcaption{overflow-wrap:anywhere}</style><h1>'+html.escape(title)+'</h1><p>Record decisions in the corresponding CSV; no images are deleted or relabeled.</p>'+''.join(cards),encoding='utf-8')
    by_id={m[0]['exact_group']:m for m in groups.values()}
    gallery('label_conflicts.html','Exact-content label conflicts: all groups',[(x['group_id'],by_id[x['group_id']]) for x in conflicts])
    gallery('near_duplicates.html','Similarity candidates: not confirmed duplicates',[(f"Candidate {i+1}; distance={r['distance']}; cross-source={r['cross_source']}",r['_members']) for i,r in enumerate(near[:args.previews])])
    match_counts=Counter()
    if args.harmonized:
        matches=[]
        paths=sorted(p for p in args.harmonized.rglob('*') if p.is_file() and p.suffix.casefold() in EXTENSIONS)
        for n,path in enumerate(paths,1):
            if n==1 or n%100==0 or n==len(paths):
                print(f'Exact provenance matching: {n}/{len(paths)}',flush=True)
            match=dict(harmonized_path=path.relative_to(args.harmonized).as_posix(),status='',matching_sources='',download_paths='',error='')
            try:
                info=image_info(path);members=groups.get(info['pixel_sha256'],[])
                match.update(status='exact_content_match' if members else 'unresolved',
                             matching_sources=json.dumps(sorted({r['source_id'] for r in members})),
                             download_paths=json.dumps([r['path'] for r in members]))
            except Exception as exc:
                match.update(status='decode_error',error=str(exc))
            match_counts[match['status']]+=1;matches.append(match)
        write_csv(out/'harmonized_source_matches.csv',matches,['harmonized_path','status','matching_sources','download_paths','error'])
    summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),download_root=str(root),
                 present_sources=present,missing_sources=sorted(set(SOURCES)-set(present)),
                 unexpected_top_level_entries=[p.name for p in root.iterdir() if p.name not in SOURCES],
                 images_by_source=dict(Counter(r['source_id'] for r in rows)),
                 total_image_files=len(rows),decode_failures=sum(bool(r['error']) for r in rows),
                 unmapped_images=sum(r['mapping_status']=='unmapped' for r in rows),
                 layout_review_images=sum(bool(r['layout_error']) for r in rows),
                 included_images_before_cleaning=sum(r['include']=='yes' for r in valid),
                 excluded_images_before_cleaning=sum(r['include']=='no' for r in valid),
                 unique_decoded_contents=len(groups),exact_duplicate_groups=len(exact),
                 cross_source_exact_groups=sum(r['cross_source'] for r in exact),
                 included_label_conflict_groups=len(conflicts),near_candidate_pairs=len(near),
                 cross_source_near_candidate_pairs=sum(r['cross_source'] for r in near),
                 distance=args.distance,near_preview_pairs=min(args.previews,len(near)),
                 harmonized_matching=dict(match_counts),
                 limitations=['No deletion, relabeling, split creation, or diagnosis is performed.',
                              'Mapping is provisional and follows the supplied workbook, not expert validation.',
                              'Resolve unmapped labels before treating conflict counts as complete.',
                              'dHash misses some edits and related captures; zero candidates does not prove independence.',
                              'RGB decoded comparison normalizes EXIF orientation, ignores alpha and compares first frame only.',
                              'A repository is a source identifier, not proof of independent collection.',
                              'Multiple exact matching sources are all preserved, not assigned arbitrarily.',
                              'No fuzzy provenance assignments; unmatched images require further review.'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2));print(f'Reports: {out}')


if __name__=='__main__':
    main()
