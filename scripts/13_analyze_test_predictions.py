#!/usr/bin/env python3
"""Audit collected predictions; conditional stratified image bootstrap and paired randomization.
Requires Python 3.10+, numpy. Input is the collector ZIP; no training dependencies.
Outputs are percentages/percentage points. Never pool seed predictions as independent images.
"""
import argparse,csv,hashlib,io,itertools,json,math,platform,sys,zipfile
from pathlib import Path
import numpy as np

EXPERIMENTS=['baseline','E1','E2','E3','E4','E5','E6']
MODELS=['resnet50','densenet121','mobilenet_v3_large','efficientnet_v2_s','xception','convnext_tiny','vit_base_patch16_224','twins_svt_base']
SEEDS=[42,43,44]


def write_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def load_package(path):
    with zipfile.ZipFile(path) as z:
        names=[n for n in z.namelist() if not n.endswith('/') and '__MACOSX' not in n and not n.split('/')[-1].startswith('.')]
        assert len(names)==len(set(names)), 'Duplicate ZIP entries'
        summaries=[n for n in names if n.endswith('/collection_summary.json') or n=='collection_summary.json']
        assert len(summaries)==1, 'Expected one collection summary'
        prefix=summaries[0][:-len('collection_summary.json')]
        js=lambda rel:json.loads(z.read(prefix+rel))
        readcsv=lambda rel:list(csv.DictReader(io.StringIO(z.read(prefix+rel).decode('utf-8-sig'))))
        summary=js('collection_summary.json');manifest=readcsv('collection_manifest.csv')
        assert len(manifest)==162 and summary['prediction_files']==162
        for item in summary['copied_files']:
            assert hashlib.sha256(z.read(prefix+item['destination'])).hexdigest()==item['sha256'],item['destination']
        assert len(summary['copied_files'])==169
        mapping=js('baseline/class_to_index.json');assert sorted(mapping.values())==list(range(14))
        labels=sorted(mapping,key=mapping.get)
        for e in EXPERIMENTS:assert js(e+'/class_to_index.json')==mapping
        data={};truth=None;identities=None;checks=0;supports=None;rowsout=[]
        for e in EXPERIMENTS:
            for m in MODELS:
                if e in ['E3','E4'] and m=='vit_base_patch16_224':continue
                preds=[]
                for seed in SEEDS:
                    dest=f'{e}/{m}/seed_{seed}/test_predictions.csv'
                    meta=[r for r in manifest if r['destination']==dest];assert len(meta)==1
                    meta=meta[0];assert meta['experiment']==e and meta['model']==m and int(meta['seed'])==seed
                    assert meta['training_precision']==('AMP_BF16' if m=='efficientnet_v2_s' else 'AMP_FP16')
                    if m=='efficientnet_v2_s':assert 'efficientnet_bf16' in meta['source']
                    assert meta['split_manifest_sha256']==summary['split_manifest_sha256']
                    rows=sorted(readcsv(dest),key=lambda r:r['cluster_id']);assert len(rows)==579
                    ids=[r['cluster_id'] for r in rows];assert len(set(ids))==579
                    tr=np.array([int(r['true_index']) for r in rows]);pr=np.array([int(r['predicted_index']) for r in rows])
                    probs=np.array([[float(r[f'probability_{i}']) for i in range(14)] for r in rows])
                    assert np.isfinite(probs).all() and (probs>=0).all() and (probs<=1).all()
                    assert np.allclose(probs.sum(1),1,atol=1e-6) and np.array_equal(probs.argmax(1),pr)
                    assert all(r['true_label']==labels[int(r['true_index'])] and r['predicted_label']==labels[int(r['predicted_index'])] for r in rows)
                    assert all((r['correct'].lower()=='true')==(r['true_index']==r['predicted_index']) for r in rows)
                    signature=[(r['cluster_id'],r['image_path'],r['true_label']) for r in rows]
                    if truth is None:truth=tr;identities=signature;supports=np.bincount(tr,minlength=14)
                    assert signature==identities and np.array_equal(tr,truth)
                    assert abs((pr==tr).mean()-float(meta['test_accuracy']))<1e-12
                    preds.append(pr);checks+=len(rows)
                data[e,m]=np.array(preds)
        assert len(data)==54
    audit=dict(prediction_files=162,model_conditions=54,unique_test_images=579,classes=14,
        checked_prediction_rows=checks,class_supports=dict(zip(labels,map(int,supports))),
        efficientnet_precision='BF16 per collection manifest and explicit source paths',
        collection_hashes='all 169 copied files match collection metadata',
        labels_and_pairing='identical cluster IDs, image paths, true labels and class mapping in all 162 runs',
        split_manifest_sha256=summary['split_manifest_sha256'],
        limitation='Collector metadata checked; original checkpoint/image bytes and training source records are not present in this package.')
    return data,truth,labels,audit


def cm_scores(cm):
    # Last two axes are true/predicted classes. Any leading dimensions are preserved.
    tp=np.diagonal(cm,axis1=-2,axis2=-1)
    denom=cm.sum(-1)+cm.sum(-2)
    f1=np.divide(2*tp,denom,out=np.zeros_like(tp,dtype=float),where=denom>0).mean(-1)
    accuracy=tp.sum(-1)/cm.sum(axis=(-2,-1))
    return np.stack([accuracy,f1],axis=-1)


def original_scores(pred,y,k):
    return np.array([cm_scores(np.bincount(y*k+p,minlength=k*k).reshape(k,k)) for p in pred])


def bootstrap_scores(pred,y,indices,k,batch=200):
    out=np.empty((len(indices),2));n=len(y)
    for start in range(0,len(indices),batch):
        ix=indices[start:start+batch];b=len(ix);offset=np.arange(b)[:,None]*k*k
        scores=[]
        for p in pred:
            code=y[ix]*k+p[ix]+offset
            cm=np.bincount(code.ravel(),minlength=b*k*k).reshape(b,k,k)
            scores.append(cm_scores(cm))
        out[start:start+b]=np.mean(scores,axis=0)
    return out


def permutation_p(pred_a,pred_b,y,k,draws,rng,batch=400):
    # A single swap decision moves the entire three-seed vector for an image.
    observed=original_scores(pred_b,y,k).mean(0)-original_scores(pred_a,y,k).mean(0)
    mask=np.any(pred_a!=pred_b,axis=0);d=int(mask.sum());n=len(y)
    if d==0:return np.ones(2),d,1,True,np.zeros(2)
    exact=d<=15;total=(1<<d) if exact else draws
    base_a=np.array([np.bincount(y[~mask]*k+p[~mask],minlength=k*k).reshape(k,k) for p in pred_a])
    base_b=np.array([np.bincount(y[~mask]*k+p[~mask],minlength=k*k).reshape(k,k) for p in pred_b])
    a=pred_a[:,mask];b=pred_b[:,mask];yy=y[mask];tails=np.zeros(2,dtype=int)
    for start in range(0,total,batch):
        size=min(batch,total-start)
        swap=((np.arange(start,start+size,dtype=np.uint64)[:,None] >> np.arange(d,dtype=np.uint64)) & 1).astype(bool) if exact else rng.integers(0,2,(size,d),dtype=np.int8).astype(bool)
        offsets=np.arange(size)[:,None]*k*k;sa=[];sb=[]
        for seed in range(3):
            pa=np.where(swap,b[seed],a[seed]);pb=np.where(swap,a[seed],b[seed])
            ca=np.bincount((yy*k+pa+offsets).ravel(),minlength=size*k*k).reshape(size,k,k)+base_a[seed]
            cb=np.bincount((yy*k+pb+offsets).ravel(),minlength=size*k*k).reshape(size,k,k)+base_b[seed]
            sa.append(cm_scores(ca));sb.append(cm_scores(cb))
        delta=np.mean(sb,axis=0)-np.mean(sa,axis=0)
        tails+=np.sum(np.abs(delta)>=np.abs(observed)-1e-12,axis=0)
    p=tails/total if exact else (tails+1)/(total+1)
    se=np.zeros(2) if exact else np.sqrt(p*(1-p)/(total+1))
    return p,d,total,exact,se


def holm(pvalues):
    p=np.asarray(pvalues);order=np.argsort(p);adjusted=np.empty(len(p))
    adjusted[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    return adjusted


def comparisons(data):
    out=[]
    for m in MODELS:
        out.append(('offline_224','baseline',m,'E1',m))
    for a,b in [('E1','E2'),('E3','E4'),('E5','E6')]:
        for m in MODELS:
            if (a,m) in data:out.append(('online_'+{'E1':'224','E3':'300','E5':'384'}[a],a,m,b,m))
    # All E6 pairs, avoiding selection of just the numerically best architecture.
    for a,b in itertools.combinations(MODELS,2):out.append(('architecture_E6','E6',a,'E6',b))
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('statistical_analysis'))
    p.add_argument('--bootstrap',type=int,default=10000)
    p.add_argument('--permutations',type=int,default=50000)
    p.add_argument('--seed',type=int,default=20261025)
    args=p.parse_args();assert args.bootstrap>=1000 and args.permutations>=1000
    assert not args.output.exists(),'Output exists; choose a new folder'
    data,y,labels,audit=load_package(args.input);k=len(labels)
    rng=np.random.default_rng(args.seed)
    class_ids=[np.flatnonzero(y==i) for i in range(k)]
    indices=np.concatenate([rng.choice(ids,size=(args.bootstrap,len(ids)),replace=True) for ids in class_ids],axis=1)
    boots={};estimates=[]
    for index,(key,pred) in enumerate(data.items(),1):
        scores=original_scores(pred,y,k);bs=bootstrap_scores(pred,y,indices,k);boots[key]=bs
        for j,metric in enumerate(['accuracy','macro_f1']):
            low,high=np.quantile(bs[:,j],[.025,.975])
            estimates.append(dict(experiment=key[0],model=key[1],metric=metric,
                seed_42_percent=scores[0,j]*100,seed_43_percent=scores[1,j]*100,seed_44_percent=scores[2,j]*100,
                mean_percent=scores[:,j].mean()*100,seed_sample_sd_pp=scores[:,j].std(ddof=1)*100,
                ci95_low_percent=low*100,ci95_high_percent=high*100,
                test_images=len(y),seeds=3,ci_scope='conditional on these three fitted runs; fixed class supports'))
        if index%10==0:print(f'Bootstrap conditions: {index}/54',flush=True)
    effects=[]
    for index,(family,ea,ma,eb,mb) in enumerate(comparisons(data),1):
        a,b=(ea,ma),(eb,mb)
        delta=original_scores(data[b],y,k).mean(0)-original_scores(data[a],y,k).mean(0)
        ci=np.quantile(boots[b]-boots[a],[.025,.975],axis=0)
        pvalue,discordant,draws,exact,mcse=permutation_p(data[a],data[b],y,k,args.permutations,rng)
        for j,metric in enumerate(['accuracy','macro_f1']):
            effects.append(dict(comparison_group=family,experiment_A=ea,model_A=ma,experiment_B=eb,model_B=mb,
                metric=metric,delta_B_minus_A_pp=delta[j]*100,ci95_low_pp=ci[0,j]*100,ci95_high_pp=ci[1,j]*100,
                p_unadjusted=pvalue[j],p_monte_carlo_se=mcse[j],discordant_images=discordant,
                permutation_draws=draws,exact_enumeration=exact))
        if index%10==0:print(f'Paired comparisons: {index}/59',flush=True)
    adjusted=holm([r['p_unadjusted'] for r in effects])
    for r,pa in zip(effects,adjusted):
        r['p_holm_all_118']=float(pa);r['reject_at_0_05']=bool(pa<.05)
    args.output.mkdir(parents=True)
    write_csv(args.output/'confidence_intervals.csv',estimates)
    write_csv(args.output/'paired_comparisons.csv',effects)
    write_csv(args.output/'class_supports.csv',[dict(class_name=l,test_images=int((y==i).sum())) for i,l in enumerate(labels)])
    audit.update(input_zip_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        bootstrap_replicates=args.bootstrap,permutation_replicates=args.permutations,resampling_seed=args.seed,
        analysis='Exploratory analysis specified after benchmark results were available; no confirmatory preregistration.',
        bootstrap='Class-stratified percentile 95% intervals, sampling image IDs with replacement within each true class. All three seed predictions move together. Same draws for every condition.',
        statistic='Arithmetic mean of the three per-seed accuracies or per-seed macro-F1 scores; not a seed ensemble or pooled 1737-observation metric.',
        permutation='Two-sided absolute-difference paired randomization: swap A/B three-seed prediction vectors together within each image. Enumerate all swaps when <=15 discordant images; otherwise Monte Carlo with +1 numerator/denominator correction.',
        multiplicity='One Holm family over all 118 tests: 31 augmentation comparisons and 28 E6 architecture pairs, each on accuracy and macro-F1.',
        limitations=['Intervals are pointwise, not simultaneous; Holm correction applies to p-values only.',
            'Inference is conditional on three trained seeds; it does not quantify uncertainty over new training runs or independently resampled training datasets.',
            'Image-level independence is assumed; unknown farm/plant/session clusters are not accounted for.',
            'Fixed class supports and one fixed split; uncertainty in class prevalence, split selection and source shift is not captured.',
            'Only three seeds limits characterization of training randomness; seed SD is reported separately.',
            'Offline augmentation increases training set size and optimizer updates; its contrast concerns the entire training condition.',
            'A nonsignificant result does not establish equality/equivalence. CI exclusion of zero can coexist with a nonsignificant Holm-adjusted p-value.',
            'E6 architecture comparisons are only for 384px with offline+online augmentation; EfficientNet BF16 differs from other models FP16.'],
        software=dict(python=sys.version,numpy=np.__version__,platform=platform.platform()),
        supported_significant_tests=sum(r['reject_at_0_05'] for r in effects))
    (args.output/'analysis_protocol.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print('COMPLETE',json.dumps({g:sum(r['reject_at_0_05'] for r in effects if r['comparison_group']==g) for g in sorted({r['comparison_group'] for r in effects})}),flush=True)


if __name__=='__main__':main()
