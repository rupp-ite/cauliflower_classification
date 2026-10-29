#!/usr/bin/env python3
"""Source-held-out training: four models x five folds x seeds 42/43/44 = 60 runs.
Input: completed 09_prepare_source_held_out.py and 10_augment_source_held_out.py outputs.
224x224, offline minimum 500 + online augmentation, pretrained ImageNet initialization.
Preserves the main runner's heads, optimizer, scheduler, loss and checkpoint rule.
Default: validate data and show plan. --run trains and evaluates held-out tests.
Dependencies: same torch/torchvision environment as the main runs, timm==1.0.30,
numpy, Pillow, scikit-learn, matplotlib. fvcore only for optional --benchmark.
No overwrite or training resume. --folds selects specific folds.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PurePosixPath

MODELS = {
    'resnet50': {'provider':'torchvision','name':'resnet50','weights':'ResNet50_Weights.IMAGENET1K_V2'},
    'densenet121': {'provider':'torchvision','name':'densenet121','weights':'DenseNet121_Weights.IMAGENET1K_V1'},
    'mobilenet_v3_large': {'provider':'torchvision','name':'mobilenet_v3_large','weights':'MobileNet_V3_Large_Weights.IMAGENET1K_V2'},
    'efficientnet_v2_s': {'provider':'torchvision','name':'efficientnet_v2_s','weights':'EfficientNet_V2_S_Weights.IMAGENET1K_V1'},
    'xception': {'provider':'timm','name':'legacy_xception.tf_in1k'},
    'convnext_tiny': {'provider':'torchvision','name':'convnext_tiny','weights':'ConvNeXt_Tiny_Weights.IMAGENET1K_V1'},
    'vit_base_patch16_224': {'provider':'timm','name':'vit_base_patch16_224.augreg_in1k'},
    'twins_svt_base': {'provider':'timm','name':'twins_svt_base.in1k'},
}
PRESETS = {
    'baseline': {'size':224,'offline':False,'online':False},
    'E1': {'size':224,'offline':True,'online':False},
    'E2': {'size':224,'offline':True,'online':True},
    'E3': {'size':300,'offline':True,'online':False},
    'E4': {'size':300,'offline':True,'online':True},
    'E5': {'size':384,'offline':True,'online':False},
    'E6': {'size':384,'offline':True,'online':True},
}

EXTENSIONS={'.jpg','.jpeg','.jfif','.png','.bmp','.tif','.tiff','.webp','.gif'}


def sha_file(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def json_write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def json_safe(value):
    return json.loads(json.dumps(value,default=str))


def write_csv(path,rows,fields):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)


def safe_path(root,relative):
    rel=PurePosixPath(relative)
    if rel.is_absolute() or not rel.parts or any(x in {'.','..'} or ':' in x or '\\' in x for x in rel.parts):
        raise ValueError(f'Unsafe relative image path: {relative}')
    result=root.joinpath(*rel.parts).resolve()
    if root not in result.parents:raise ValueError(f'Image escapes dataset root: {relative}')
    return result


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
        safe_path(root, converted['split_path'])
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
        expected = {safe_path(root,r['split_path']) for r in all_rows if r['split']==split}
        actual = {x.resolve() for x in (root/split).rglob('*') if x.is_file() and x.suffix.lower() in EXTENSIONS}
        if actual != expected:
            raise ValueError(f'{split} folder differs from assignments.')
        actual_classes = {p.name for p in (root/split).iterdir() if p.is_dir()}
        if actual_classes != set(classes):
            raise ValueError(f'{split} class folders differ from class mapping.')
    # Both preview and apply verify originals. Decoding is also checked during apply.
    for row in all_rows:
        if sha_file(safe_path(root,row['split_path'])) != row['file_sha256']:
            raise ValueError('Frozen image changed: ' + row['split_path'])
    return report, manifest, all_rows, originals, held_out


def load_splits(root):
    report,manifest,rows,originals,held_out=load_fold(root)
    labels=sorted({r['final_label'] for r in rows})
    parts={s:sorted([r for r in rows if r['split']==s],key=lambda r:r['split_path']) for s in ('train','val','test')}
    return parts,labels,report,sha_file(manifest),originals,held_out


class ImageRecords:
    """Map-style dataset; top-level class is compatible with Windows worker spawning."""
    def __init__(self,root,records,class_to_idx,transform):
        self.root=root;self.records=records;self.class_to_idx=class_to_idx;self.transform=transform
    def __len__(self):return len(self.records)
    def __getitem__(self,index):
        from PIL import Image,ImageOps
        row=self.records[index]
        with Image.open(safe_path(self.root,row['split_path'])) as im:
            image=ImageOps.exif_transpose(im).convert('RGB');tensor=self.transform(image)
        return tensor,self.class_to_idx[row['final_label']],index


def seed_worker(worker_id):
    import torch
    import numpy as np
    seed=torch.initial_seed()%2**32;random.seed(seed);np.random.seed(seed)


def seed_all(seed,strict):
    import torch
    import numpy as np
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=strict
    torch.use_deterministic_algorithms(strict)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    # Prefer math attention for strict deterministic CUDA execution.
    torch.backends.cuda.enable_flash_sdp(not strict)
    torch.backends.cuda.enable_mem_efficient_sdp(not strict)
    torch.backends.cuda.enable_math_sdp(True)


def build_model(key,num_classes,pretrained=True):
    import torch.nn as nn
    import torchvision.models as tv
    import timm
    spec=MODELS[key]
    def head(n):
        return nn.Sequential(nn.Dropout(.3),nn.Linear(n,512),nn.ReLU(inplace=True),nn.BatchNorm1d(512),nn.Dropout(.3),nn.Linear(512,num_classes))
    if spec['provider']=='timm':
        model=timm.create_model(spec['name'],pretrained=pretrained,num_classes=num_classes)
        cfg=dict(model.pretrained_cfg)
        mean=cfg.get('mean',(0.485,0.456,0.406));std=cfg.get('std',(0.229,0.224,0.225))
        info=dict(spec,pretrained=pretrained,pretrained_config=json_safe(cfg),head='timm linear classifier')
    else:
        enum_name,member=spec['weights'].split('.');weights=getattr(getattr(tv,enum_name),member)
        model=tv.get_model(spec['name'],weights=weights if pretrained else None)
        recommended=weights.transforms();mean=recommended.mean;std=recommended.std
        if key=='resnet50':model.fc=head(model.fc.in_features)
        elif key=='densenet121':model.classifier=head(model.classifier.in_features)
        elif key=='mobilenet_v3_large':
            n=model.classifier[0].in_features
            model.classifier=nn.Sequential(nn.Linear(n,512),nn.Hardswish(inplace=True),nn.Dropout(.3),nn.Linear(512,num_classes))
        elif key=='efficientnet_v2_s':model.classifier=head(model.classifier[-1].in_features)
        else:model.classifier[-1]=head(model.classifier[-1].in_features)
        info=dict(spec,pretrained=pretrained,weights_url=weights.url,head='512-unit task-specific head; see script')
    info['recommended_mean']=list(mean);info['recommended_std']=list(std)
    return model,info


def preprocessing(mean,std,image_size=224,online=False):
    from torchvision import transforms as T
    operations=[T.Resize((image_size,image_size),interpolation=T.InterpolationMode.BILINEAR,antialias=True)]
    if online:
        operations += [T.RandomHorizontalFlip(p=.5),
                       T.RandomRotation(10,interpolation=T.InterpolationMode.BILINEAR,fill=0),
                       T.ColorJitter(brightness=.1,contrast=.1,saturation=.1)]
    operations += [T.ToTensor(),T.Normalize(mean=mean,std=std)]
    return T.Compose(operations)


def resize_model(model,key,size):
    # Always initialize/load the SAME 224px checkpoint before positional interpolation.
    # This preserves the classifier initialization across resolutions for a given seed.
    if key=='vit_base_patch16_224' and size!=224:
        if size==300:raise ValueError('ViT is excluded at 300px to preserve the declared E3/E4 protocol.')
        model.set_input_size(img_size=(size,size))
    return model


def load_augmented_training(root,raw_train,report_arg,manifest_hash):
    folder=root/'augmented_train'
    if not folder.is_dir():raise ValueError(f'Offline-augmented training folder not found: {folder}')
    candidates=[report_arg.resolve()] if report_arg else [p.parent for p in sorted((root/'offline_augmentation_reports').glob('*/summary.json'))]
    found=[]
    for directory in candidates:
        summary=json.loads((directory/'summary.json').read_text(encoding='utf-8-sig'))
        if summary.get('mode')=='apply' and summary.get('completed') and Path(summary['output']).resolve()==folder:
            found.append((directory,summary))
    if len(found)!=1:raise ValueError('Supply --augmentation-report for the correct completed augmentation APPLY run.')
    report,summary=found[0]
    if summary.get('script_version')!='source_held_out_1':raise ValueError('Use 10_augment_source_held_out.py outputs.')
    if summary.get('minimum_per_class')!=500 or summary.get('seed')!=42:raise ValueError('Expected minimum-500 offline augmentation with generation seed 42.')
    if summary['split_manifest_sha256']!=manifest_hash:raise ValueError('Offline augmentation used another split manifest.')
    with (report/'augmentation_manifest.csv').open(encoding='utf-8-sig',newline='') as f:
        rd=csv.DictReader(f);required={'output_path','kind','final_label','parent_cluster_id','parent_split_path','parent_file_sha256','parent_pixel_sha256','output_file_sha256','output_pixel_sha256'}
        if not required<=set(rd.fieldnames or []):raise ValueError('Augmentation manifest lacks required fields.')
        augmented=list(rd)
    if len(augmented)!=summary['final_total']:raise ValueError('Augmented count differs from completed summary.')
    parents={r['split_path']:r for r in raw_train};expected=set();records=[];original_parent_paths=[]
    for field in ['output_path','output_pixel_sha256','output_file_sha256']:
        if len({r[field] for r in augmented})!=len(augmented):raise ValueError(f'Repeated augmented {field}.')
    for i,row in enumerate(augmented,1):
        if i==1 or i%200==0 or i==len(augmented):print(f'  Verify augmented {i}/{len(augmented)}',flush=True)
        parent=parents.get(row['parent_split_path'])
        if parent is None:raise ValueError('An augmentation parent is not in frozen raw training.')
        if row['final_label']!=parent['final_label'] or row['parent_cluster_id']!=parent['cluster_id'] or row['parent_file_sha256']!=parent['file_sha256'] or row['parent_pixel_sha256']!=parent['pixel_sha256']:
            raise ValueError('Augmentation parent metadata mismatch.')
        rel=PurePosixPath(row['output_path'])
        if len(rel.parts)!=2 or rel.parts[0]!=row['final_label']:raise ValueError('Unexpected augmented class path.')
        path=safe_path(folder,row['output_path']);expected.add(path)
        if not path.is_file() or sha_file(path)!=row['output_file_sha256']:raise ValueError(f'Changed/missing augmentation output: {path}')
        if row['kind']=='original':
            if row['output_file_sha256']!=parent['file_sha256']:raise ValueError('Original augmented-folder copy differs from raw parent.')
            original_parent_paths.append(row['parent_split_path'])
        elif row['kind']!='offline_variant':raise ValueError('Unknown augmentation kind.')
        records.append(dict(parent,split='train',split_path='augmented_train/'+row['output_path'],file_sha256=row['output_file_sha256'],pixel_sha256=row['output_pixel_sha256'],parent_split_path=row['parent_split_path']))
    actual={p.resolve() for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS}
    if actual!=expected:raise ValueError('Augmented folder differs from its manifest.')
    if len(original_parent_paths)!=len(parents) or set(original_parent_paths)!=set(parents):raise ValueError('Augmented folder must contain every raw original exactly once.')
    if summary['original_training_images']!=len(parents):raise ValueError('Raw training count differs from augmentation summary.')
    return sorted(records,key=lambda r:r['split_path']),report,sha_file(report/'augmentation_manifest.csv')


def state_digest(model):
    h=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        h.update(name.encode());h.update(str(tensor.dtype).encode());h.update(str(tuple(tensor.shape)).encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def make_loader(root,records,labels,transform,args,seed,shuffle):
    import torch
    data=ImageRecords(root,records,{x:i for i,x in enumerate(labels)},transform)
    generator=torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(data,batch_size=args.batch_size,shuffle=shuffle,num_workers=args.workers,
                                      pin_memory=args.device.startswith('cuda'),drop_last=False,
                                      worker_init_fn=seed_worker,generator=generator,persistent_workers=False)


def predict(model,loader,device,criterion,amp):
    import torch
    import numpy as np
    model.eval();loss=0.;count=0;ys=[];ps=[];indices=[]
    with torch.inference_mode():
        for images,targets,idx in loader:
            images=images.to(device,non_blocking=True);targets=targets.to(device,non_blocking=True)
            with torch.autocast(device_type=device.type,enabled=amp):
                logits=model(images);batch_loss=criterion(logits,targets)
            if not torch.isfinite(logits).all() or not torch.isfinite(batch_loss):
                raise ValueError('Nonfinite evaluation logits/loss; stopping before checkpoint selection or reporting.')
            probabilities=logits.float().softmax(1)
            loss+=float(batch_loss)*len(targets);count+=len(targets)
            ys.extend(targets.cpu().tolist());ps.append(probabilities.cpu().numpy());indices.extend(idx.tolist())
    if not count:raise ValueError('Empty evaluation partition.')
    return loss/count,np.asarray(ys),np.concatenate(ps),indices


def score(y,probs,labels):
    from sklearn.metrics import accuracy_score,precision_recall_fscore_support,classification_report,confusion_matrix
    pred=probs.argmax(1);ids=list(range(len(labels)))
    out={'accuracy':float(accuracy_score(y,pred))}
    for average in ['macro','weighted']:
        precision,recall,f1,_=precision_recall_fscore_support(y,pred,labels=ids,average=average,zero_division=0)
        out.update({f'{average}_precision':float(precision),f'{average}_recall':float(recall),f'{average}_f1':float(f1)})
    report=classification_report(y,pred,labels=ids,target_names=labels,output_dict=True,zero_division=0)
    matrix=confusion_matrix(y,pred,labels=ids)
    return out,report,matrix


def save_predictions(directory,prefix,records,labels,y,probs,indices,loss):
    metrics,report,matrix=score(y,probs,labels);metrics['loss']=loss
    predictions=[]
    for truth,p,index in zip(y,probs,indices):
        row=records[index];pred=int(p.argmax())
        predictions.append(dict(image_path=row['split_path'],cluster_id=row['cluster_id'],true_index=int(truth),true_label=labels[int(truth)],predicted_index=pred,predicted_label=labels[pred],correct=int(truth)==pred,**{f'probability_{i}':float(v) for i,v in enumerate(p)}))
    write_csv(directory/f'{prefix}_predictions.csv',predictions,['image_path','cluster_id','true_index','true_label','predicted_index','predicted_label','correct']+[f'probability_{i}' for i in range(len(labels))])
    json_write(directory/f'{prefix}_metrics.json',metrics);json_write(directory/f'{prefix}_classification_report.json',report)
    write_csv(directory/f'{prefix}_per_class.csv',[dict(class_name=label,**report[label]) for label in labels],['class_name','precision','recall','f1-score','support'])
    write_csv(directory/f'{prefix}_confusion_counts.csv',[dict(true_class=label,**{str(i):int(v) for i,v in enumerate(matrix[j])}) for j,label in enumerate(labels)],['true_class']+[str(i) for i in range(len(labels))])
    plot_confusion(directory,prefix,matrix,labels)
    return metrics


def cleanup_between_runs(args,base,label):
    """Release unreachable Python objects and unused CUDA cache between jobs.

    This is not a device reset: live tensors and CUDA context allocations remain.
    No random-number generators or numerical training settings are changed.
    """
    import gc
    import torch
    collected=gc.collect()
    record=dict(boundary=label,collected_python_objects=collected,
                timestamp_utc=datetime.now(timezone.utc).isoformat())
    if args.device.startswith('cuda'):
        device=torch.device(args.device)
        torch.cuda.synchronize(device)
        record.update(allocated_before_mib=torch.cuda.memory_allocated(device)/2**20,
                      reserved_before_mib=torch.cuda.memory_reserved(device)/2**20)
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
        free,total=torch.cuda.mem_get_info(device)
        record.update(allocated_after_mib=torch.cuda.memory_allocated(device)/2**20,
                      reserved_after_mib=torch.cuda.memory_reserved(device)/2**20,
                      device_free_mib=free/2**20,device_total_mib=total/2**20)
        print(f"CUDA cleanup {label}: allocated={record['allocated_after_mib']:.1f} MiB, "
              f"reserved={record['reserved_after_mib']:.1f} MiB, device free={record['device_free_mib']:.1f} MiB",flush=True)
    if base.is_dir():
        with (base/'memory_cleanup.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(record,allow_nan=False)+'\n')


def train_one(args,key,seed,root,parts,labels,base):
    import torch
    seed_all(seed,not args.allow_nondeterministic)
    directory=base/key/f'seed_{seed}';directory.mkdir(parents=True,exist_ok=False)
    json_write(directory/'status.json',{'completed':False,'stage':'initializing'})
    model,model_info=build_model(key,len(labels),not args.scratch)
    mean=model_info['recommended_mean'];std=model_info['recommended_std']
    if args.normalization=='imagenet':mean=[.485,.456,.406];std=[.229,.224,.225]
    init_before_resize=state_digest(model)
    model=resize_model(model,key,args.image_size)
    train_transform=preprocessing(mean,std,args.image_size,args.online)
    eval_transform=preprocessing(mean,std,args.image_size,False)
    init_hash=state_digest(model)
    runconfig=dict(fold=args.fold,held_out_source=args.held_out_source,precision=('FP16 AMP' if args.device.startswith('cuda') and not args.no_amp else 'FP32'),model=key,seed=seed,model_info=model_info,labels=labels,mean=mean,std=std,image_size=[args.image_size,args.image_size],experiment=args.experiment,
                   offline_augmentation=args.offline,online_augmentation=args.online,
                   augmentation=('horizontal flip .5, rotation +/-10 degrees, color jitter .1' if args.online else 'no online image augmentation'),
                   initial_state_before_resize_sha256=init_before_resize,initial_state_sha256=init_hash,
                   training_preprocessing=repr(train_transform),evaluation_preprocessing=repr(eval_transform),
                   checkpoint_rule='strictly highest validation accuracy; retain earliest epoch on ties',
                   hyperparameters=base_config(args))
    json_write(directory/'config.json',runconfig)
    device=torch.device(args.device);model=model.to(device)
    amp=device.type=='cuda' and not args.no_amp
    scaler=torch.amp.GradScaler('cuda',enabled=amp)
    trainloader=make_loader(root,parts['train'],labels,train_transform,args,seed,True)
    valloader=make_loader(root,parts['val'],labels,eval_transform,args,seed+1,False)
    criterion=torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay,betas=(0.9,0.999),eps=1e-8)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=args.lr_factor,patience=args.lr_patience)
    history=[];best=-1.;best_epoch=0;stale=0;updates=0;seen=0;start=time.perf_counter()
    training_peak_allocated=0;training_peak_reserved=0
    for epoch in range(1,args.epochs+1):
        model.train();train_loss=0.;correct=0;total=0;epstart=time.perf_counter();used_lr=optimizer.param_groups[0]['lr']
        if device.type=='cuda':torch.cuda.synchronize(device);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(device)
        for batch_index,(images,targets,_) in enumerate(trainloader,1):
            images=images.to(device,non_blocking=True);targets=targets.to(device,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=amp):logits=model(images);loss=criterion(logits,targets)
            if not torch.isfinite(loss):raise ValueError(f'Nonfinite loss: {key}, seed {seed}, epoch {epoch}')
            scale_before=scaler.get_scale();scaler.scale(loss).backward();scaler.step(optimizer);scaler.update()
            if not amp or scaler.get_scale()>=scale_before:updates+=1
            train_loss+=float(loss.detach())*len(targets);correct+=int((logits.argmax(1)==targets).sum());total+=len(targets);seen+=len(targets)
            if args.progress_every and (batch_index==1 or batch_index%args.progress_every==0 or batch_index==len(trainloader)):
                elapsed=time.perf_counter()-epstart
                eta=elapsed*(len(trainloader)-batch_index)/batch_index
                print(f'  {key} seed={seed} epoch={epoch}/{args.epochs} batch={batch_index}/{len(trainloader)} loss={train_loss/total:.4f} acc={correct/total:.4f} elapsed={elapsed:.1f}s ETA={eta:.1f}s',flush=True)
        training_epoch_seconds=time.perf_counter()-epstart
        epoch_peak_allocated=None;epoch_peak_reserved=None
        if device.type=='cuda':
            torch.cuda.synchronize(device)
            epoch_peak_allocated=torch.cuda.max_memory_allocated(device)/2**20
            epoch_peak_reserved=torch.cuda.max_memory_reserved(device)/2**20
            training_peak_allocated=max(training_peak_allocated,epoch_peak_allocated)
            training_peak_reserved=max(training_peak_reserved,epoch_peak_reserved)
        val_loss,y,probs,_=predict(model,valloader,device,criterion,amp)
        metrics,_,_=score(y,probs,labels);val_acc=metrics['accuracy'];scheduler.step(val_loss)
        improved=val_acc>best
        if improved:
            best=val_acc;best_epoch=epoch;stale=0
            torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},directory/'best.pt')
            json_write(directory/'best_selection.json',dict(epoch=epoch,validation_accuracy=best,validation_loss=val_loss))
        else:stale+=1
        entry=dict(epoch=epoch,training_epoch_seconds=training_epoch_seconds,training_peak_allocated_mib=epoch_peak_allocated,training_peak_reserved_mib=epoch_peak_reserved,train_loss=train_loss/total,train_accuracy=correct/total,val_loss=val_loss,val_accuracy=val_acc,val_macro_f1=metrics['macro_f1'],lr_used=used_lr,lr_next=optimizer.param_groups[0]['lr'],optimizer_updates=updates,training_examples_seen=seen,epoch_seconds=time.perf_counter()-epstart)
        history.append(entry);write_csv(directory/'history.csv',history,list(entry))
        print(f'{key} seed={seed} epoch={epoch}/{args.epochs}: train_loss={train_loss/total:.4f} train_acc={correct/total:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} macroF1={metrics["macro_f1"]:.4f} lr={used_lr:.3g} best_epoch={best_epoch} stale={stale}/{args.patience} seconds={entry["epoch_seconds"]:.1f}',flush=True)
        if args.patience>0 and stale>=args.patience:break
    training_elapsed=time.perf_counter()-start
    plot_history(directory)
    model.load_state_dict(torch.load(directory/'best.pt',map_location='cpu',weights_only=True))
    val_loss,y,probs,indices=predict(model,valloader,device,criterion,amp)
    val_metrics=save_predictions(directory,'val',parts['val'],labels,y,probs,indices,val_loss)
    result=dict(completed=True,model=key,seed=seed,best_epoch=best_epoch,epochs_trained=len(history),
                parameters=sum(p.numel() for p in model.parameters()),optimizer_updates=updates,training_examples_seen=seen,
                training_and_validation_seconds=training_elapsed,training_only_seconds=sum(r["training_epoch_seconds"] for r in history),checkpoint_sha256=sha_file(directory/'best.pt'),validation=val_metrics,
                test_evaluated=False,pretrained=not args.scratch,training_peak_allocated_mib=training_peak_allocated if device.type=='cuda' else None,training_peak_reserved_mib=training_peak_reserved if device.type=='cuda' else None)
    json_write(directory/'status.json',result)
    # Scheduler holds the optimizer; optimizer and last-batch tensors can retain
    # CUDA allocations until all references are released.
    optimizer.zero_grad(set_to_none=True)
    del scheduler,optimizer,scaler,criterion,trainloader,valloader
    del images,targets,logits,loss,model
    return result


def base_config(args):
    return dict(epochs=args.epochs,batch_size=args.batch_size,learning_rate=args.lr,weight_decay=args.weight_decay,
                label_smoothing=args.label_smoothing,early_stopping_patience=args.patience,
                scheduler='ReduceLROnPlateau on validation loss',lr_patience=args.lr_patience,lr_factor=args.lr_factor,
                optimizer='AdamW',normalization=args.normalization,workers=args.workers,
                strict_deterministic=not args.allow_nondeterministic,amp=('enabled_if_cuda' if args.device=='auto' and not args.no_amp else not args.no_amp and args.device.startswith('cuda')))


def aggregate(base,prefix,models,seeds):
    import numpy as np
    rows=[];summary=[]
    for key in models:
        members=[]
        for seed in seeds:
            path=base/key/f'seed_{seed}'/f'{prefix}_metrics.json'
            metrics=json.loads(path.read_text());rows.append(dict(model=key,seed=seed,**metrics));members.append(metrics)
        for metric in members[0]:
            values=[r[metric] for r in members]
            summary.append(dict(model=key,metric=metric,seeds=len(values),mean=float(np.mean(values)),sample_sd=float(np.std(values,ddof=1)) if len(values)>1 else '',unit='fraction' if metric!='loss' else 'cross_entropy'))
    write_csv(base/f'{prefix}_runs.csv',rows,list(rows[0]));write_csv(base/f'{prefix}_mean_sd.csv',summary,['model','metric','seeds','mean','sample_sd','unit'])


def evaluate_all(args,base,root,parts,labels,manifest_hash):
    import torch
    config=json.loads((base/'experiment.json').read_text())
    if not config.get('completed'):raise ValueError('Training experiment is not complete; finish the planned runs before test evaluation.')
    if config['experiment']!=args.experiment:raise ValueError('Evaluation --experiment does not match the stored training experiment.')
    if config['split_manifest_sha256']!=manifest_hash or config['class_names']!=labels:raise ValueError('Evaluation data differ from training protocol.')
    jobs=[(key,seed) for key in config['models'] for seed in config['seeds']]
    for key,seed in jobs:
        directory=base/key/f'seed_{seed}'
        status=json.loads((directory/'status.json').read_text())
        if not status.get('completed'):raise ValueError(f'Training is incomplete: {key}/{seed}')
        if (directory/'test_metrics.json').exists():
            if not (directory/'test_evaluation.json').exists():raise ValueError(f'Incomplete existing test outputs in {directory}; archive them before reevaluation.')
            saved=json.loads((directory/'test_evaluation.json').read_text())
            if saved['checkpoint_sha256']!=status['checkpoint_sha256'] or saved['split_manifest_sha256']!=manifest_hash:raise ValueError('Existing test result provenance differs.')
        if sha_file(directory/'best.pt')!=status['checkpoint_sha256']:raise ValueError('Checkpoint changed after training.')
    for key,seed in jobs:
        directory=base/key/f'seed_{seed}'
        if (directory/'test_metrics.json').exists():
            print(f'Skipping existing test results: {key}/{seed}',flush=True);continue
        runconfig=json.loads((directory/'config.json').read_text())
        cleanup_between_runs(args,base,f'before test {key} seed={seed}')
        seed_all(seed,not args.allow_nondeterministic)
        model,_=build_model(key,len(labels),False)
        model=resize_model(model,key,runconfig['image_size'][0])
        model.load_state_dict(torch.load(directory/'best.pt',map_location='cpu',weights_only=True));model=model.to(args.device)
        transform=preprocessing(runconfig['mean'],runconfig['std'],runconfig['image_size'][0],False)
        loader=make_loader(root,parts['test'],labels,transform,args,seed+2,False)
        criterion=torch.nn.CrossEntropyLoss(label_smoothing=runconfig['hyperparameters']['label_smoothing'])
        amp=args.device.startswith('cuda') and not args.no_amp
        loss,y,probs,indices=predict(model,loader,torch.device(args.device),criterion,amp)
        metrics=save_predictions(directory,'test',parts['test'],labels,y,probs,indices,loss)
        json_write(directory/'test_evaluation.json',dict(checkpoint_sha256=sha_file(directory/'best.pt'),split_manifest_sha256=manifest_hash,device=args.device,amp=amp,evaluated_utc=datetime.now(timezone.utc).isoformat()))
        status=json.loads((directory/'status.json').read_text());status['test_evaluated']=True;json_write(directory/'status.json',status)
        print(f'{key} seed={seed}: test accuracy={metrics["accuracy"]:.4f}, macro F1={metrics["macro_f1"]:.4f}',flush=True)
        del model,loader,criterion
        cleanup_between_runs(args,base,f'after test {key} seed={seed}')
    aggregate(base,'test',config['models'],config['seeds'])
    config['test_completed']=True;json_write(base/'experiment.json',config)


class Tee:
    def __init__(self,stream,log):self.stream=stream;self.log=log
    def write(self,text):
        self.stream.write(text);self.log.write(text);self.log.flush()
        return len(text)
    def flush(self):self.stream.flush();self.log.flush()
    def isatty(self):return False
    @property
    def encoding(self):return self.stream.encoding


def install_log(base,name):
    log=(base/name).open('a',encoding='utf-8',buffering=1)
    sys.stdout=Tee(sys.stdout,log);sys.stderr=Tee(sys.stderr,log)
    # Keep open until process exit so uncaught exceptions are captured, too.
    print(f'[{datetime.now(timezone.utc).isoformat()}] Command: {sys.argv}',flush=True)


def pyplot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def save_figure(fig,path):
    fig.savefig(path.with_suffix('.png'),dpi=200,bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight')
    pyplot().close(fig)


def plot_history(directory):
    import numpy as np
    plt=pyplot()
    with (directory/'history.csv').open(newline='',encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    epochs=[int(r['epoch']) for r in rows]
    best=json.loads((directory/'best_selection.json').read_text())['epoch']
    for field,title,ylabel in [('loss','Loss curves','Cross-entropy (label smoothing)'),('accuracy','Accuracy curves','Accuracy (%)')]:
        fig,ax=plt.subplots(figsize=(7,4.5))
        for prefix,label in [('train','Training'),('val','Validation')]:
            values=[float(r[f'{prefix}_{field}'])*(100 if field=='accuracy' else 1) for r in rows]
            ax.plot(epochs,values,label=label,marker=".")
        ax.axvline(best,color='gray',linestyle='--',label=f'Selected epoch {best}')
        ax.set(xlabel='Epoch',ylabel=ylabel,title=title);ax.legend();ax.grid(alpha=.2)
        save_figure(fig,directory/f'{field}_curve')
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for prefix,label in [('train','Training'),('val','Validation')]:
        axes[0].plot(epochs,[float(r[f'{prefix}_loss']) for r in rows],label=label,marker=".")
        axes[1].plot(epochs,[100*float(r[f'{prefix}_accuracy']) for r in rows],label=label,marker=".")
    for ax in axes:ax.axvline(best,color='gray',linestyle='--');ax.set_xlabel('Epoch');ax.grid(alpha=.2);ax.legend()
    axes[0].set_ylabel('Cross-entropy');axes[1].set_ylabel('Accuracy (%)')
    fig.tight_layout();save_figure(fig,directory/'training_curves')
    fig,ax=plt.subplots(figsize=(7,4));ax.plot(epochs,[float(r['lr_used']) for r in rows],marker='.');ax.set(xlabel='Epoch',ylabel='Learning rate',yscale='log');ax.grid(alpha=.2)
    save_figure(fig,directory/'learning_rate_curve')


def plot_confusion(directory,prefix,matrix,labels):
    import numpy as np
    matrix=np.asarray(matrix);plt=pyplot()
    denominators=matrix.sum(axis=1,keepdims=True)
    percentages=np.divide(100.0*matrix,denominators,out=np.zeros_like(matrix,dtype=float),where=denominators!=0)
    write_csv(directory/f'{prefix}_confusion_percent.csv',[dict(true_class=label,**{str(i):float(v) for i,v in enumerate(percentages[j])}) for j,label in enumerate(labels)],['true_class']+[str(i) for i in range(len(labels))])
    for kind,data in [('counts',matrix),('percent',percentages)]:
        fig,ax=plt.subplots(figsize=(max(8,len(labels)*.72),max(7,len(labels)*.65)))
        im=ax.imshow(data,cmap='Blues',vmin=0,vmax=100 if kind=='percent' else None)
        ax.set_xticks(range(len(labels)),labels,rotation=60,ha='right',fontsize=8)
        ax.set_yticks(range(len(labels)),labels,fontsize=8)
        ax.set(xlabel='Predicted class',ylabel='True class',title=f'{prefix.capitalize()} confusion matrix ({"row %" if kind=="percent" else "counts"})')
        for i in range(len(labels)):
            for j in range(len(labels)):
                v=data[i,j];text=f'{v:.1f}' if kind=='percent' else str(int(v))
                ax.text(j,i,text,ha='center',va='center',fontsize=7,color='white' if v>data.max()*.55 else 'black')
        fig.colorbar(im,ax=ax,fraction=.046,pad=.04);save_figure(fig,directory/f'{prefix}_confusion_{kind}')


def regenerate_figures(base):
    import numpy as np
    config=json.loads((base/'experiment.json').read_text())
    for key in config['models']:
        for seed in config['seeds']:
            d=base/key/f'seed_{seed}'
            if (d/'history.csv').exists():plot_history(d)
            labels=json.loads((d/'config.json').read_text())['labels']
            for prefix in ['val','test']:
                path=d/f'{prefix}_confusion_counts.csv'
                if path.exists():
                    with path.open(newline='',encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
                    matrix=np.asarray([[int(r[str(i)]) for i in range(len(labels))] for r in rows])
                    plot_confusion(d,prefix,matrix,labels)


def complexity(model,size,device):
    """fvcore convention: one fused multiply-add = one operation.
    Separate supported conv/linear/matmul MACs from other operator estimates.
    Unsupported operators are always exposed; do not interpret partial counts as exact totals.
    """
    import torch
    from fvcore.nn import FlopCountAnalysis
    from fvcore.nn.jit_handles import get_shape
    def sdpa_macs(inputs,outputs):
        q,k,v=[get_shape(x) for x in inputs[:3]]
        if q is None or k is None or v is None:raise ValueError('Unknown attention tensor shape')
        return Counter({'matmul':math.prod(q[:-2])*q[-2]*k[-2]*(q[-1]+v[-1])})
    result={'status':'failed','method':'fvcore JIT tracing; FP32, batch 1',
            'attention_handler':'SDPA counts QK^T and attention*V MACs; excludes softmax, scaling, masking and dropout',
            'convention':'fvcore counts one fused multiply-add as one operation; MACs cover supported conv/linear/matmul/bmm/einsum operators; FLOPs estimate = 2*MACs + remaining supported operations'}
    try:
        analysis=FlopCountAnalysis(model,torch.zeros(1,3,size,size,device=device))
        analysis.set_op_handle('aten::scaled_dot_product_attention',sdpa_macs)
        analysis.unsupported_ops_warnings(False).uncalled_modules_warnings(False)
        total=float(analysis.total());operators=dict(analysis.by_operator());unsupported=dict(analysis.unsupported_ops())
        macs=sum(float(v) for k,v in operators.items() if k in {'conv','linear','matmul','bmm','einsum','addmm','mm'})
        result.update(status='partial' if unsupported else 'supported_ops_counted',fvcore_fma1_operations=total,
                      supported_macs=macs,estimated_flops_2_per_mac=2*macs+(total-macs),by_operator=operators,
                      unsupported_operators=unsupported,uncalled_modules=sorted(analysis.uncalled_modules()))
    except Exception as exc:result['error']=f'{type(exc).__name__}: {exc}'
    return result


def benchmark_forward(model,size,batch,args):
    import torch
    import numpy as np
    device=torch.device(args.device)
    result={'batch_size':batch,'status':'pending','precision':'FP32','scope':'model forward only; excludes loading, preprocessing and host/device transfers',
            'warmup':args.benchmark_warmup,'repeats':args.benchmark_repeats,'timing_method':'CUDA events, synchronized per repetition' if device.type=='cuda' else 'CPU perf_counter',
            'device':str(device)}
    x=None
    try:
        x=torch.zeros(batch,3,size,size,device=device)
        with torch.inference_mode():
            for _ in range(args.benchmark_warmup):model(x)
            if device.type=='cuda':
                torch.cuda.synchronize(device);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(device)
            timings=[]
            for _ in range(args.benchmark_repeats):
                if device.type=='cuda':
                    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                    start.record();model(x);end.record();end.synchronize();elapsed=start.elapsed_time(end)
                else:
                    start=time.perf_counter();model(x);elapsed=(time.perf_counter()-start)*1000
                timings.append(float(elapsed))
        mean=float(np.mean(timings))
        result.update(status='completed',batch_latency_ms_mean=mean,batch_latency_ms_sd=float(np.std(timings,ddof=1)),
                      batch_latency_ms_median=float(np.median(timings)),batch_latency_ms_p95=float(np.percentile(timings,95)),
                      amortized_ms_per_image=mean/batch,throughput_images_per_second=1000*batch/mean,
                      peak_allocated_mib=torch.cuda.max_memory_allocated(device)/2**20 if device.type=='cuda' else None,
                      peak_reserved_mib=torch.cuda.max_memory_reserved(device)/2**20 if device.type=='cuda' else None,
                      timings_ms=timings)
    except torch.cuda.OutOfMemoryError as exc:
        result.update(status='out_of_memory',error=str(exc))
    finally:
        del x
        if device.type=='cuda':torch.cuda.empty_cache()
    return result


def benchmark_all(args,base):
    import torch
    config=json.loads((base/'experiment.json').read_text())
    if not config.get('completed'):raise ValueError('Training incomplete; cannot benchmark this experiment.')
    for key in config['models']:
        for seed in config['seeds']:
            d=base/key/f'seed_{seed}'
            if (d/'efficiency.json').exists():
                previous=json.loads((d/'efficiency.json').read_text())
                if previous['checkpoint_sha256']!=sha_file(d/'best.pt'):raise ValueError('Checkpoint differs from existing efficiency result.')
                print(f'Skipping existing efficiency result: {key}/{seed}',flush=True);continue
            cfg=json.loads((d/'config.json').read_text());status=json.loads((d/'status.json').read_text())
            if not status.get('completed') or sha_file(d/'best.pt')!=status['checkpoint_sha256']:raise ValueError('Incomplete or changed checkpoint.')
            cleanup_between_runs(args,base,f'before benchmark {key} seed={seed}')
            seed_all(seed,not args.allow_nondeterministic)
            model,_=build_model(key,len(cfg['labels']),False);model=resize_model(model,key,cfg['image_size'][0])
            model.load_state_dict(torch.load(d/'best.pt',map_location='cpu',weights_only=True));model=model.to(args.device).float().eval()
            print(f'Benchmarking {key} seed={seed}: batch 1 and {args.benchmark_batch_size}, FP32',flush=True)
            measured={str(b):benchmark_forward(model,cfg['image_size'][0],b,args) for b in dict.fromkeys([1,args.benchmark_batch_size])}
            comp=complexity(model,cfg['image_size'][0],args.device)
            result=dict(model=key,seed=seed,checkpoint_sha256=sha_file(d/'best.pt'),device=args.device,
                        gpu=torch.cuda.get_device_name(torch.device(args.device)) if args.device.startswith('cuda') else None,
                        parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                        checkpoint_bytes=(d/'best.pt').stat().st_size,image_size=cfg['image_size'],batches=measured,complexity=comp,
                        torch_version=torch.__version__,cuda_runtime=torch.version.cuda,
                        strict_deterministic=not args.allow_nondeterministic,tf32=False,measured_utc=datetime.now(timezone.utc).isoformat())
            json_write(d/'efficiency.json',result)
            for batch,measurement in measured.items():
                if measurement['status']=='completed':write_csv(d/f'inference_timings_batch_{batch}.csv',[{'repetition':i+1,'batch_latency_ms':v} for i,v in enumerate(measurement['timings_ms'])],['repetition','batch_latency_ms'])
            if comp['status']!='supported_ops_counted':print(f'Complexity coverage for {key}: {comp["status"]}; inspect efficiency.json unsupported operators.',flush=True)
            del model
            cleanup_between_runs(args,base,f'after benchmark {key} seed={seed}')


def build_summaries(base):
    import numpy as np
    config=json.loads((base/'experiment.json').read_text());rows=[]
    for key in config['models']:
        for seed in config['seeds']:
            d=base/key/f'seed_{seed}';status=json.loads((d/'status.json').read_text())
            if not status.get('completed'):continue
            row=dict(fold=config['fold'],held_out_source=config['held_out_source'],classes=len(config['class_names']),experiment=config['experiment'],model=key,seed=seed,image_size=config['image_size'][0],split_manifest_sha256=config['split_manifest_sha256'],
                     offline_augmentation=config['offline_augmentation'],online_augmentation=config['online_augmentation'],training_images=config['counts']['train'],validation_images=config['counts']['val'],test_images=config['counts']['test'],training_amp=config['hyperparameters']['amp'],
                     best_epoch=status['best_epoch'],epochs_trained=status['epochs_trained'],parameters=status['parameters'],
                     training_and_validation_seconds=status['training_and_validation_seconds'],training_only_seconds=status.get('training_only_seconds'),optimizer_updates=status['optimizer_updates'],training_examples_seen=status['training_examples_seen'],
                     training_peak_allocated_mib=status.get('training_peak_allocated_mib'),training_peak_reserved_mib=status.get('training_peak_reserved_mib'),
                     metric_unit='fraction',pretrained=status['pretrained'])
            for prefix in ['val','test']:
                path=d/f'{prefix}_metrics.json'
                if path.exists():row.update({f'{prefix}_{k}':v for k,v in json.loads(path.read_text()).items()})
            if (d/'efficiency.json').exists():
                efficiency=json.loads((d/'efficiency.json').read_text());comp=efficiency['complexity']
                row.update(complexity_status=comp['status'],fvcore_fma1_operations=comp.get('fvcore_fma1_operations'),supported_macs=comp.get('supported_macs'),estimated_flops_2_per_mac=comp.get('estimated_flops_2_per_mac'),benchmark_device=efficiency['device'],benchmark_gpu=efficiency['gpu'],benchmark_precision='FP32')
                for batch,m in efficiency['batches'].items():
                    row[f'inference_batch_{batch}_status']=m['status']
                    for k in ['batch_latency_ms_mean','batch_latency_ms_sd','batch_latency_ms_p95','amortized_ms_per_image','throughput_images_per_second','peak_allocated_mib','peak_reserved_mib']:
                        row[f'inference_batch_{batch}_{k}']=m.get(k)
            rows.append(row)
    if not rows:return
    fields=list(dict.fromkeys(k for row in rows for k in row));write_csv(base/'benchmark_summary.csv',rows,fields)
    means=[]
    for key in config['models']:
        members=[r for r in rows if r['model']==key]
        for metric in fields:
            values=[r.get(metric) for r in members]
            if not values or any(isinstance(v,bool) or not isinstance(v,(int,float)) for v in values) or metric in ['seed','image_size']:continue
            means.append(dict(fold=config['fold'],held_out_source=config['held_out_source'],experiment=config['experiment'],model=key,metric=metric,runs=len(values),mean=float(np.mean(values)),sample_sd=float(np.std(values,ddof=1)) if len(values)>1 else ''))
    write_csv(base/'benchmark_summary_mean_sd.csv',means,['fold','held_out_source','experiment','model','metric','runs','mean','sample_sd'])
    # Rebuild from explicit experiment summaries only. Include provenance so different
    # split manifests remain visible and cannot be mistaken for matched comparisons.
    allrows=[]
    for path in [base.parent/name/'benchmark_summary.csv' for name in FOLDS if (base.parent/name/'benchmark_summary.csv').is_file()]:
        with path.open(newline='',encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):row['results_directory']=str(path.parent);allrows.append(row)
    if allrows:write_csv(base.parent/'all_folds_runs.csv',allrows,list(dict.fromkeys(k for row in allrows for k in row)))
    allmeans=[]
    for path in [base.parent/name/'benchmark_summary_mean_sd.csv' for name in FOLDS if (base.parent/name/'benchmark_summary_mean_sd.csv').is_file()]:
        with path.open(newline='',encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):row['results_directory']=str(path.parent);allmeans.append(row)
    if allmeans:write_csv(base.parent/'all_folds_mean_sd.csv',allmeans,list(allmeans[0]))


DEFAULT_MODELS = ['resnet50','mobilenet_v3_large','vit_base_patch16_224','twins_svt_base']


def restore_training_settings(args, stored):
    # Evaluation uses the saved training precision/loss rather than CLI defaults.
    hp=stored['hyperparameters']
    for attr,key in [('epochs','epochs'),('batch_size','batch_size'),('lr','learning_rate'),
                     ('weight_decay','weight_decay'),('label_smoothing','label_smoothing'),
                     ('patience','early_stopping_patience'),('lr_patience','lr_patience'),
                     ('lr_factor','lr_factor'),('normalization','normalization')]:
        setattr(args,attr,hp[key])
    args.no_amp=not hp['amp']
    args.allow_nondeterministic=not hp['strict_deterministic']
    args.scratch=not stored['pretrained']
    args.seeds=stored['seeds']


def validate_saved(base,manifest_hash,labels,augmentation_hash,held_out):
    config=json.loads((base/'experiment.json').read_text())
    if config.get('runner_version')!='source_held_out_1' or config['held_out_source']!=held_out:
        raise ValueError('Saved results belong to a different protocol/source.')
    if config['split_manifest_sha256']!=manifest_hash or config['class_names']!=labels:
        raise ValueError('Saved data provenance/class mapping differs.')
    if config['augmentation_manifest_sha256']!=augmentation_hash:
        raise ValueError('Saved augmentation manifest differs.')
    if config['image_size']!=[224,224] or not config['offline_augmentation'] or not config['online_augmentation']:
        raise ValueError('Saved settings differ from the fixed source-held-out study.')
    if not config.get('completed'):
        raise ValueError('Fold training is incomplete; evaluation/reporting is not permitted.')
    return config


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(r'D:\Projects\CauliflowerV2\Dataset\source_held_out'))
    p.add_argument('--output',type=Path,default=Path('Results/source_held_out_224_offline_online'))
    p.add_argument('--folds',nargs='+',choices=['all']+FOLDS,default=['all'])
    p.add_argument('--models',nargs='+',choices=DEFAULT_MODELS,default=DEFAULT_MODELS)
    p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44])
    p.add_argument('--stage',choices=['train','evaluate','report','benchmark'],default='train')
    p.add_argument('--run',action='store_true',help='Default is data-verified plan only')
    p.add_argument('--no-auto-test',action='store_true')
    p.add_argument('--benchmark',action='store_true',help='Optional extra FP32 efficiency measurements after training')
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--weight-decay',type=float,default=1e-4)
    p.add_argument('--label-smoothing',type=float,default=.1)
    p.add_argument('--patience',type=int,default=10)
    p.add_argument('--lr-patience',type=int,default=3)
    p.add_argument('--lr-factor',type=float,default=.3)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--cpu-threads',type=int,default=4)
    p.add_argument('--progress-every',type=int,default=10)
    p.add_argument('--device',default='auto')
    p.add_argument('--normalization',choices=['checkpoint','imagenet'],default='checkpoint')
    p.add_argument('--no-amp',action='store_true')
    p.add_argument('--allow-nondeterministic',action='store_true')
    p.add_argument('--scratch',action='store_true',help='Software diagnostic only; not for paper results')
    p.add_argument('--benchmark-batch-size',type=int,default=32)
    p.add_argument('--benchmark-warmup',type=int,default=20)
    p.add_argument('--benchmark-repeats',type=int,default=100)
    args=p.parse_args()
    if 'all' in args.folds and args.folds!=['all']:raise ValueError('Use all alone or individual fold names.')
    selected=FOLDS if args.folds==['all'] else args.folds
    if len(set(selected))!=len(selected) or len(set(args.models))!=len(args.models):raise ValueError('Repeated fold/model.')
    if len(set(args.seeds))!=len(args.seeds) or any(s<0 or s>=2**32 for s in args.seeds):raise ValueError('Invalid/repeated seed.')
    if min(args.epochs,args.batch_size,args.cpu_threads,args.benchmark_batch_size,args.benchmark_warmup)<1 or args.workers<0 or args.progress_every<0 or args.benchmark_repeats<2:
        raise ValueError('Invalid count argument.')
    if args.patience<0 or args.lr_patience<0 or args.lr<=0 or args.weight_decay<0 or not 0<=args.label_smoothing<1 or not 0<args.lr_factor<1:
        raise ValueError('Invalid optimizer/loss settings.')
    root=args.root.resolve();output=args.output.resolve()
    if output==root or root in output.parents or output in root.parents:raise ValueError('Results must be separate from datasets.')
    args.image_size=224;args.offline=True;args.online=True;args.experiment='source_held_out_224_offline_online'
    # Complete preflight for every selected fold before any expensive training.
    jobs=[]
    for name in selected:
        fold=root/name;base=output/name
        if args.stage=='train' and base.exists():
            raise ValueError(f'Results already exist: {base}; select unstarted folds or use a new --output. No overwrite/resume.')
        print('Checking '+name,flush=True)
        parts,labels,report,manifest_hash,originals,held_out=load_splits(fold)
        raw_train=list(parts['train'])
        parts['train'],aug_report,aug_hash=load_augmented_training(fold,raw_train,None,manifest_hash)
        forbidden={r['pixel_sha256'] for r in originals}-{r['pixel_sha256'] for r in raw_train}
        if any(r['pixel_sha256'] in forbidden for r in parts['train']):raise ValueError('Training pixels match a non-training original.')
        expected_counts={label:max(500,sum(r['final_label']==label for r in raw_train)) for label in labels}
        if dict(Counter(r['final_label'] for r in parts['train']))!=expected_counts:
            raise ValueError('Augmented class counts violate the minimum-500 policy.')
        models=list(args.models);seeds=list(args.seeds)
        stored=None
        if args.stage!='train':
            stored=validate_saved(base,manifest_hash,labels,aug_hash,held_out)
            models=stored['models'];seeds=stored['seeds']
        if args.stage=='train' and 'resnet50' in models and (args.batch_size==1 or len(parts['train'])%args.batch_size==1):
            raise ValueError('BatchNorm head requires batches >=2; avoid a singleton final batch.')
        print(f'{name}: {len(labels)} classes; augmented train={len(parts["train"])}, val={len(parts["val"])}, held-out test={len(parts["test"])}',flush=True)
        jobs.append((name,fold,base,parts,labels,report,manifest_hash,held_out,aug_report,aug_hash,models,seeds,stored,len(raw_train)))
    count=sum(len(j[10])*len(j[11]) for j in jobs)
    print(f'\nStage={args.stage}; model/seed jobs={count}; 224x224; offline=True; online=True.',flush=True)
    if not args.run:
        print('PLAN ONLY: no results written or models trained. Add --run to execute.')
        return
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import torch
    import torchvision
    import timm
    import numpy
    import sklearn
    import matplotlib
    if version('timm')!='1.0.30':raise ValueError('Use timm==1.0.30 to match the main experiments.')
    if args.benchmark or args.stage=='benchmark':import fvcore
    if args.device=='auto':args.device='cuda:0' if torch.cuda.is_available() else 'cpu'
    if args.device.startswith('cuda'):
        if not torch.cuda.is_available():raise ValueError('CUDA requested but unavailable.')
        torch.cuda.set_device(torch.device(args.device))
    torch.set_num_threads(args.cpu_threads)
    for name,fold,base,parts,labels,report,manifest_hash,held_out,aug_report,aug_hash,models,seeds,stored,raw_count in jobs:
        args.fold=name;args.held_out_source=held_out;args.seeds=seeds
        if stored is not None:restore_training_settings(args,stored)
        if args.stage=='train':
            base.mkdir(parents=True,exist_ok=False)
        original_stdout,original_stderr=sys.stdout,sys.stderr
        install_log(base,'execution.log' if args.stage=='train' else f'{args.stage}.log')
        try:
            if args.stage!='train':
                if args.stage=='evaluate':evaluate_all(args,base,fold,parts,labels,manifest_hash)
                elif args.stage=='benchmark':benchmark_all(args,base)
                else:regenerate_figures(base)
                build_summaries(base)
                continue
            settings=dict(runner_version='source_held_out_1',fold=name,held_out_source=held_out,
                experiment=args.experiment,models=models,seeds=seeds,class_names=labels,image_size=[224,224],
                counts={k:len(v) for k,v in parts.items()},raw_training_images=raw_count,
                train_folder='augmented_train',offline_augmentation=True,online_augmentation=True,
                precision='FP16 AMP' if args.device.startswith('cuda') and not args.no_amp else 'FP32',
                hyperparameters=base_config(args),pretrained=not args.scratch,completed=False,
                output=str(base),split_root=str(fold),split_report=str(report),split_manifest_sha256=manifest_hash,
                augmentation_report=str(aug_report),augmentation_manifest_sha256=aug_hash,
                augmentation_policy_sha256=sha_file(aug_report/'augmentation_policy.json'),
                script_sha256=sha_file(Path(__file__)),training_device=args.device,
                automatic_test=not args.no_auto_test,checkpoint_rule='highest validation accuracy; earliest ties',
                memory_cleanup='gc.collect and synchronized unused CUDA cache release between model-seed jobs',
                benchmark_protocol=dict(batch_size=args.benchmark_batch_size,warmup=args.benchmark_warmup,
                    repeats=args.benchmark_repeats,precision='FP32',scope='model forward only; synthetic device input'),
                interpretation='Shared-class source-held-out task; class sets and supports differ between folds. Do not tune on held-out test results.')
            json_write(base/'experiment.json',settings)
            json_write(base/'class_to_index.json',{x:i for i,x in enumerate(labels)})
            dependencies={}
            for dep in ['torch','torchvision','timm','numpy','Pillow','scikit-learn','matplotlib']:
                try:dependencies[dep]=version(dep)
                except PackageNotFoundError:dependencies[dep]='not installed'
            dependencies.update(python=sys.version,platform=platform.platform(),cuda_runtime=torch.version.cuda,
                cudnn_version=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(torch.device(args.device)) if args.device.startswith('cuda') else None)
            json_write(base/'environment.json',dependencies)
            print(json.dumps(settings,indent=2),flush=True)
            for key in models:
                for seed in seeds:
                    cleanup_between_runs(args,base,f'before training {key} seed={seed}')
                    train_one(args,key,seed,fold,parts,labels,base)
                    cleanup_between_runs(args,base,f'after training {key} seed={seed}')
            aggregate(base,'val',models,seeds)
            settings['completed']=True;json_write(base/'experiment.json',settings)
            if not args.no_auto_test:evaluate_all(args,base,fold,parts,labels,manifest_hash)
            if args.benchmark:benchmark_all(args,base)
            build_summaries(base)
            print('Completed fold: '+name,flush=True)
        finally:
            # Avoid nesting logs between folds and close this fold's log stream.
            log=getattr(sys.stdout,'log',None)
            sys.stdout,sys.stderr=original_stdout,original_stderr
            if log is not None:log.close()
    print('Completed all selected folds. Results: '+str(output),flush=True)


if __name__=='__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    try:main()
    except Exception as exc:
        traceback.print_exc()
        print(f'STOPPED: {exc}',file=sys.stderr);sys.exit(1)
