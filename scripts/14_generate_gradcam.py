#!/usr/bin/env python3
"""Revised qualitative Grad-CAM for final saved runs; no training.
Default: E6, seed 42, ResNet50/MobileNetV3-Large/ViT/Twins, fixed hash-based
selection of one common success and one any-model failure per true class.
Default preview validates files and writes nothing. --run generates FP32 CAMs.
Requires same torch/torchvision/timm==1.0.30 environment, numpy, Pillow, matplotlib.
No external grad-cam package is required. No silent layer fallback or blank-map replacement.
"""
import argparse,csv,gc,hashlib,json,math,sys,traceback
from pathlib import Path,PurePosixPath
from collections import Counter
from datetime import datetime,timezone
from importlib.metadata import version

DEFAULT_MODELS=['resnet50','mobilenet_v3_large','vit_base_patch16_224','twins_svt_base']


def require(ok,message):
    if not ok:raise ValueError(message)


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))


def csv_rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def write_json(p,value):p.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def write_csv(p,rows,fields=None):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)


def safe(root,value):
    p=PurePosixPath(value)
    require(not p.is_absolute() and p.parts and all(x not in ('.','..') and ':' not in x and '\\' not in x for x in p.parts),'Unsafe path: '+value)
    result=root.joinpath(*p.parts).resolve();require(root in result.parents,'Image escapes dataset root')
    return result

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

def json_safe(value):
    return json.loads(json.dumps(value,default=str))

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

def select_examples(predictions,labels,per_class,selection_seed):
    """Selection is made before CAM generation, without confidence or heatmap ranking."""
    maps={model:{r['cluster_id']:r for r in rows} for model,rows in predictions.items()}
    model_list=list(maps);reference=maps[model_list[0]]
    require(all(set(x)==set(reference) for x in maps.values()),'Models use different test images')
    for identity,row in reference.items():
        for rows in maps.values():
            require((rows[identity]['image_path'],rows[identity]['true_label'],rows[identity]['true_index'])==
                    (row['image_path'],row['true_label'],row['true_index']),'Models differ on true labels/image paths')
    selected=[];summary=[]
    for label in labels:
        candidates=[i for i,r in reference.items() if r['true_label']==label]
        require(bool(candidates),'Test class missing: '+label)
        successes=[i for i in candidates if all(x[i]['predicted_label']==label for x in maps.values())]
        failures=[i for i in candidates if any(x[i]['predicted_label']!=label for x in maps.values())]
        for category,ids in [('common_correct',successes),('any_model_error',failures)]:
            ranked=sorted(ids,key=lambda i:hashlib.sha256(f'{selection_seed}|{label}|{category}|{i}'.encode()).hexdigest())
            chosen=ranked[:per_class]
            summary.append(dict(true_label=label,category=category,available=len(ids),selected=len(chosen)))
            for identity in chosen:
                row=reference[identity]
                selected.append(dict(cluster_id=identity,image_path=row['image_path'],true_label=label,
                                     true_index=int(row['true_index']),selection_category=category))
    return selected,summary,maps


def target_layer(model,key):
    if key=='resnet50':return model.layer4[-1],'layer4[-1]'
    if key=='mobilenet_v3_large':return model.features[-1],'features[-1]'
    if key=='vit_base_patch16_224':return model.blocks[-1].norm1,'blocks[-1].norm1'
    if key=='twins_svt_base':return model.blocks[-1][-1].norm1,'blocks[-1][-1].norm1'
    raise ValueError('Unsupported model: '+key)


def spatial_tensor(tensor,key,prefix_tokens=0):
    """Explicit model-specific layout; never guess/remove a Twins class token."""
    if key in ('resnet50','mobilenet_v3_large'):
        require(tensor.ndim==4,'Expected CNN activation B,C,H,W')
        return tensor
    require(tensor.ndim==3,'Expected transformer activation B,N,C')
    if key=='vit_base_patch16_224':tensor=tensor[:,prefix_tokens:,:]
    elif key!='twins_svt_base':raise ValueError('Unknown transformer layout')
    batch,n,channels=tensor.shape;side=math.isqrt(n)
    require(side*side==n,'Non-square spatial token count: '+str(tuple(tensor.shape)))
    return tensor.transpose(1,2).reshape(batch,channels,side,side)


def normalize_positive(signed):
    import numpy as np
    require(np.isfinite(signed).all(),'Nonfinite signed CAM')
    positive=np.maximum(signed,0);lo=float(positive.min());hi=float(positive.max())
    scale=max(abs(lo),abs(hi),float(np.max(np.abs(signed))),1e-30)
    flat=(hi-lo)<=max(1e-30,1e-7*scale)
    if hi<=0:return np.zeros_like(positive),'no_positive_evidence'
    if flat:return np.zeros_like(positive),'constant_positive_map'
    return (positive-lo)/(hi-lo),'valid'


def prepare(args):
    run_root=args.run_root.resolve();experiment=js(run_root/'experiment.json')
    require(experiment.get('completed') and experiment.get('test_completed'),'Expected completed training and test evaluation')
    mapping=js(run_root/'class_to_index.json');labels=sorted(mapping,key=mapping.get)
    require(mapping=={label:i for i,label in enumerate(labels)} and experiment['class_names']==labels,'Invalid class mapping')
    require(args.seed in experiment['seeds'],'Seed absent from saved experiment')
    require(set(args.models)<=set(experiment['models']),'Requested model absent from experiment')
    root=args.splits.resolve() if args.splits else Path(experiment['split_root']).resolve()
    require(root.is_dir(),'Test split folder not found: '+str(root))
    report=Path(experiment['split_report'])
    if (report/'split_manifest.csv').is_file():
        manifest=report/'split_manifest.csv';rows=csv_rows(manifest)
        frozen={r['cluster_id']:r for r in rows if r['split']=='test'}
    else:
        manifest=root/'reports'/'assignments.csv';rows=csv_rows(manifest)
        frozen={r['cluster_id']:dict(r,split_path='test/'+r['destination_path']) for r in rows if r['partition']=='test'}
    require(digest(manifest)==experiment['split_manifest_sha256'],'Frozen split manifest differs from training')
    predictions={};configs={};metadata={}
    for model in args.models:
        directory=run_root/model/f'seed_{args.seed}'
        cfg=js(directory/'config.json');status=js(directory/'status.json');ev=js(directory/'test_evaluation.json')
        require(cfg['labels']==labels and cfg['model']==model and cfg['seed']==args.seed,'Run config identity differs')
        require(cfg['image_size']==experiment['image_size'],'Resolution differs from experiment')
        require(status.get('completed') and status.get('test_evaluated'),'Run incomplete')
        checkpoint=directory/'best.pt'
        require(digest(checkpoint)==status['checkpoint_sha256']==ev['checkpoint_sha256'],'Checkpoint hash mismatch')
        require(ev['split_manifest_sha256']==experiment['split_manifest_sha256'],'Test evaluation uses a different split')
        rows=csv_rows(directory/'test_predictions.csv');require(len(rows)==len(frozen),'Test prediction count differs')
        require(len({r['cluster_id'] for r in rows})==len(rows),'Repeated test image identity')
        require({r['cluster_id'] for r in rows}==set(frozen),'Test prediction IDs differ from frozen split')
        for row in rows:
            original=frozen[row['cluster_id']]
            require(row['true_label']==original['final_label']==labels[int(row['true_index'])],'True-label mismatch')
            require(row['image_path']==original['split_path'],'Test path mismatch')
            require(row['predicted_label']==labels[int(row['predicted_index'])],'Predicted-label mismatch')
        predictions[model]=rows;configs[model]=(directory,cfg)
        metadata[model]=dict(checkpoint_sha256=status['checkpoint_sha256'],
            config_sha256=digest(directory/'config.json'),predictions_sha256=digest(directory/'test_predictions.csv'))
    selected,summary,maps=select_examples(predictions,labels,args.per_class,args.selection_seed)
    for row in selected:
        image=safe(root,row['image_path'])
        require(image.is_file() and digest(image)==frozen[row['cluster_id']]['file_sha256'],'Image differs from frozen original: '+str(image))
    return root,labels,selected,summary,maps,configs,experiment,metadata


def save_panel(path,image,heat,title,blank_status):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    rgb=np.asarray(image,dtype=np.float32)/255
    # Use identical display pixels and attribution geometry. No heatmap-driven selection.
    colors=plt.get_cmap('jet')(heat)[...,:3]
    overlay=np.clip(.6*rgb+.4*colors,0,1)
    Image.fromarray(np.uint8(np.round(255*overlay))).save(path/'overlay.png')
    Image.fromarray(np.uint8(np.round(255*colors))).save(path/'heatmap.png')
    image.save(path/'original.png')
    fig,axes=plt.subplots(1,3,figsize=(10,3.5))
    axes[0].imshow(image);axes[0].set_title('Original')
    axes[1].imshow(heat,cmap='jet',vmin=0,vmax=1);axes[1].set_title('Grad-CAM: '+blank_status)
    axes[2].imshow(overlay);axes[2].set_title('Overlay')
    for ax in axes:ax.axis('off')
    fig.suptitle(title,fontsize=9);fig.tight_layout();fig.savefig(path/'panel.png',dpi=220,bbox_inches='tight');plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',type=Path,default=Path('Results/E6_384_offline_online'))
    p.add_argument('--splits',type=Path,help='Override saved split-root path; original image hashes must still match')
    p.add_argument('--output',type=Path,default=Path('Result/gradcam_E6_seed42'))
    p.add_argument('--models',nargs='+',choices=DEFAULT_MODELS,default=DEFAULT_MODELS)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--selection-seed',type=int,default=42)
    p.add_argument('--per-class',type=int,default=1)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--run',action='store_true')
    a=p.parse_args();require(a.per_class>=1,'per-class must be positive')
    require(len(a.models)==len(set(a.models)),'Repeated model')
    output=a.output.resolve();require(not output.exists(),'Output exists; choose a new --output')
    root,labels,selected,summary,maps,configs,experiment,metadata=prepare(a)
    require(not any(output==x or x in output.parents or output in x.parents for x in [root,a.run_root.resolve()]),'Output must be separate from data and saved results')
    print(json.dumps(dict(experiment=experiment['experiment'],seed=a.seed,models=a.models,classes=len(labels),image_size=experiment['image_size'],
        selected_images=len(selected),selection='fixed hash ranking within true class and common-success/any-error strata',
        available_cases=summary,cam_precision='FP32',test_prediction_precision='saved training/evaluation AMP configuration'),indent=2),flush=True)
    if not a.run:print('PREVIEW ONLY. Nothing written. Add --run to generate CAMs.');return
    import os
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import torch
    import numpy as np
    from PIL import Image,ImageOps
    require(version('timm')=='1.0.30','Use timm==1.0.30, matching the final runs')
    if a.device.startswith('cuda'):
        require(torch.cuda.is_available(),'CUDA unavailable');torch.cuda.set_device(torch.device(a.device))
    torch.manual_seed(a.seed);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    output.mkdir(parents=True)
    protocol=dict(completed=False,experiment=experiment['experiment'],run_root=str(a.run_root.resolve()),split_root=str(root),seed=a.seed,
        selection_seed=a.selection_seed,per_class_per_category=a.per_class,models=a.models,class_names=labels,image_size=experiment['image_size'],
        selected_images=len(selected),cam_precision='FP32',target_rule='predicted FP32 logit; additionally ground-truth logit for misclassified cases',
        selection='SHA256 ranking independent of confidence/heatmaps; one common success and one any-model failure per true class when available',
        source_metadata=metadata,split_manifest_sha256=experiment['split_manifest_sha256'],script_sha256=digest(Path(__file__)),
        software={name:version(name) for name in ['torch','torchvision','timm','numpy','Pillow','matplotlib']},
        notes=['CAMs are qualitative, not evidence of correct symptom localization or causal feature use.',
               'Model-specific late target layers differ in spatial resolution; colors are normalized separately per map and cannot compare absolute evidence across models.',
               'Blank/nonfinite/constant maps and saved-AMP vs FP32 prediction differences are recorded, never silently repaired or removed.',
               'Fixed seed 42 is used instead of choosing the seed with the most attractive heatmaps.',
               'No retraining, new checkpoint selection, label changes or test-set filtering are performed.'])
    write_json(output/'protocol.json',protocol);write_csv(output/'selection_manifest.csv',selected);write_csv(output/'selection_counts.csv',summary)
    results=[]
    for key in a.models:
        directory,cfg=configs[key];print('Loading '+key,flush=True)
        model,_=build_model(key,len(labels),False);model=resize_model(model,key,cfg['image_size'][0])
        model.load_state_dict(torch.load(directory/'best.pt',map_location='cpu',weights_only=True),strict=True)
        model=model.to(a.device).float().eval()
        transform=preprocessing(cfg['mean'],cfg['std'],cfg['image_size'][0],False)
        layer,layer_name=target_layer(model,key);capture={}
        def hook(_module,_inputs,output_tensor):
            require(isinstance(output_tensor,torch.Tensor),'Expected a tensor at CAM target layer')
            capture['activation']=output_tensor
        handle=layer.register_forward_hook(hook)
        try:
            for item in selected:
                identity=item['cluster_id'];saved=maps[key][identity]
                with Image.open(safe(root,item['image_path'])) as im:image=ImageOps.exif_transpose(im).convert('RGB')
                x=transform(image).unsqueeze(0).to(a.device)
                display=image.resize((cfg['image_size'][0],cfg['image_size'][0]),resample=Image.Resampling.BILINEAR)
                model.zero_grad(set_to_none=True);capture.clear()
                logits=model(x);require(torch.isfinite(logits).all(),'Nonfinite FP32 logits')
                require('activation' in capture,'Target layer hook was not invoked')
                activation=capture['activation'];pred=int(logits.argmax(1));truth=item['true_index']
                fp32_probs=logits.detach().softmax(1)[0]
                targets=[('predicted',pred)]+([('ground_truth',truth)] if pred!=truth else [])
                for j,(role,target) in enumerate(targets):
                    gradient=torch.autograd.grad(logits[0,target],activation,retain_graph=j<len(targets)-1)[0]
                    require(torch.isfinite(activation).all() and torch.isfinite(gradient).all(),'Nonfinite activations/gradients')
                    acts=spatial_tensor(activation,key,getattr(model,'num_prefix_tokens',0))
                    grads=spatial_tensor(gradient,key,getattr(model,'num_prefix_tokens',0))
                    signed=(grads.mean(dim=(-2,-1),keepdim=True)*acts).sum(1)[0].detach().cpu().numpy()
                    heat,status=normalize_positive(signed)
                    if float(grads.detach().abs().max())==0:status='zero_spatial_gradient'
                    cell=output/key/item['selection_category']/identity/role;cell.mkdir(parents=True)
                    # Keep raw low-resolution signed evidence, normalized map, and resized map.
                    np.save(cell/'signed_cam_native.npy',signed);np.save(cell/'normalized_cam_native.npy',heat)
                    resized=torch.nn.functional.interpolate(torch.from_numpy(heat)[None,None],size=(cfg['image_size'][0],cfg['image_size'][0]),mode='bilinear',align_corners=False)[0,0].numpy()
                    np.save(cell/'cam_display.npy',resized)
                    title=f'{key}, seed {a.seed} | true: {labels[truth]} | predicted: {labels[pred]} | target: {labels[target]}'
                    save_panel(cell,display,resized,title,status)
                    record=dict(model=key,seed=a.seed,cluster_id=identity,image_path=item['image_path'],selection_category=item['selection_category'],
                        true_label=labels[truth],saved_amp_prediction=saved['predicted_label'],fp32_prediction=labels[pred],
                        saved_vs_fp32_prediction_match=pred==int(saved['predicted_index']),fp32_predicted_probability=float(fp32_probs[pred]),
                        target_role=role,target_index=target,target_label=labels[target],layer=layer_name,
                        activation_shape=str(list(activation.shape)),spatial_shape=str(list(acts.shape)),
                        gradient_abs_mean=float(grads.detach().abs().mean()),gradient_abs_max=float(grads.detach().abs().max()),
                        signed_cam_min=float(signed.min()),signed_cam_max=float(signed.max()),
                        native_positive_fraction=float((signed>0).mean()),cam_status=status,output=str(cell.relative_to(output)))
                    results.append(record);write_json(cell/'metadata.json',record)
                    write_csv(output/'cam_audit.csv',results)
                del logits,activation,gradient,acts,grads,x
                capture.clear()
                print(f'{key}: {identity}, saved={saved["predicted_label"]}, FP32={labels[pred]}',flush=True)
        finally:
            handle.remove();capture.clear();del model;gc.collect()
            if a.device.startswith('cuda'):torch.cuda.synchronize();torch.cuda.empty_cache()
    protocol.update(completed=True,created_utc=datetime.now(timezone.utc).isoformat(),cam_maps=len(results),
        cam_status_counts=dict(Counter(r['cam_status'] for r in results)),
        saved_prediction_mismatch_images=len({(r['model'],r['cluster_id']) for r in results if not r['saved_vs_fp32_prediction_match']}))
    write_json(output/'protocol.json',protocol)
    print('Completed. Inspect cam_audit.csv and example panels before manuscript use.',flush=True)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        traceback.print_exc();print('STOPPED: '+str(exc),file=sys.stderr);sys.exit(1)
