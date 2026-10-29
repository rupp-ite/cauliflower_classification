#!/usr/bin/env python3
"""Compose final four-model Grad-CAM figures without inference or training.
Preview is default. --apply writes main PNG/PDF, full-selection supplementary
PNG pages/PDF, row manifests and captions. Requires Pillow, numpy, matplotlib.
Main rows are chosen by a fixed hash ranking before viewing maps, never confidence.
"""
import argparse,csv,hashlib,json,math,shutil,sys,tempfile,textwrap
from collections import Counter
from pathlib import Path,PurePosixPath

MODELS=['resnet50','mobilenet_v3_large','vit_base_patch16_224','twins_svt_base']
TITLES={'resnet50':'ResNet50','mobilenet_v3_large':'MobileNetV3-Large',
        'vit_base_patch16_224':'ViT-Base-Patch16','twins_svt_base':'Twins-SVT-Base'}


def require(ok,message):
    if not ok:raise ValueError(message)


def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))


def read_csv(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def write_csv(p,rows):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def safe(root,value):
    rel=PurePosixPath(value.replace('\\','/'))
    require(rel.parts and not rel.is_absolute() and all(x not in ('.','..') and ':' not in x for x in rel.parts),'Unsafe output path')
    path=root.joinpath(*rel.parts).resolve();require(root in path.parents,'CAM file escapes input folder')
    return path


def validate_input(root):
    from PIL import Image
    protocol=js(root/'protocol.json')
    require(protocol.get('completed') is True,'Grad-CAM generation is incomplete')
    require(set(protocol['models'])==set(MODELS),'Use the completed FOUR-model run, not the Twins-only diagnostic')
    require(protocol.get('cam_precision')=='FP32','Expected documented FP32 attribution')
    selected=read_csv(root/'selection_manifest.csv');audit=read_csv(root/'cam_audit.csv')
    require(selected and len(selected)==protocol['selected_images'],'Selection count mismatch')
    require(len({r['cluster_id'] for r in selected})==len(selected),'Repeated selected image ID')
    require(len(audit)==protocol['cam_maps'],'CAM audit count mismatch')
    require(dict(Counter(r['cam_status'] for r in audit))==protocol['cam_status_counts'],'CAM status totals mismatch')
    cells={};file_hashes={};warnings=[]
    for r in audit:
        require(r['model'] in MODELS,'Unexpected model in audit')
        require(int(r['seed'])==protocol['seed'],'Audit seed mismatch')
        if r['target_role']!='predicted':continue
        key=(r['cluster_id'],r['model'])
        require(key not in cells,'Duplicate predicted-class map')
        require(r['target_label']==r['fp32_prediction'],'Predicted CAM targets a different class')
        directory=safe(root,r['output']);metadata=js(directory/'metadata.json')
        for field in ['model','cluster_id','true_label','target_label','target_role','fp32_prediction','cam_status','image_path','selection_category']:
            require(metadata[field]==r[field],'Map metadata differs from audit: '+field)
        require(math.isclose(float(metadata['fp32_predicted_probability']),float(r['fp32_predicted_probability']),abs_tol=1e-10),'Confidence metadata differs')
        confidence=float(r['fp32_predicted_probability'])
        require(math.isfinite(confidence) and 0<=confidence<=1,'Invalid confidence')
        for name in ['original.png','overlay.png']:
            path=directory/name;require(path.is_file(),'Missing image: '+str(path))
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                require(list(image.size)==list(reversed(protocol['image_size'])),'Unexpected image size')
            file_hashes[str(path.relative_to(root))]=digest(path)
        if r['cam_status']!='valid':warnings.append(f"{r['model']}/{r['cluster_id']}: {r['cam_status']}; retained in figures")
        if r['saved_vs_fp32_prediction_match']!='True':warnings.append(f"{r['model']}/{r['cluster_id']}: saved AMP vs FP32 prediction differs; marked with dagger")
        cells[key]=dict(r,directory=directory)
    selected_ids={r['cluster_id'] for r in selected}
    require({key[0] for key in cells}==selected_ids,'Audit image coverage differs from selection')
    for r in selected:
        require(r['selection_category'] in ['common_correct','any_model_error'],'Unexpected selection category')
        originals=[]
        for model in MODELS:
            require((r['cluster_id'],model) in cells,'Missing model/image cell')
            cell=cells[r['cluster_id'],model]
            require(cell['true_label']==r['true_label'] and cell['image_path']==r['image_path']
                    and cell['selection_category']==r['selection_category'],'Image identity differs between models')
            originals.append(digest(cell['directory']/'original.png'))
        require(len(set(originals))==1,'Original display image differs across models: '+r['cluster_id'])
    return protocol,selected,cells,file_hashes,warnings


def choose_rows(selected,correct,error,seed,clusters=None):
    lookup={r['cluster_id']:r for r in selected}
    if clusters:
        require(len(clusters)==len(set(clusters)),'Repeated --clusters ID')
        require(all(c in lookup for c in clusters),'A requested cluster is absent from the selected image set')
        return [lookup[c] for c in clusters]
    rows=[]
    for category,count in [('common_correct',correct),('any_model_error',error)]:
        available=[r for r in selected if r['selection_category']==category]
        ranked=sorted(available,key=lambda r:hashlib.sha256(f"{seed}|{category}|{r['cluster_id']}".encode()).hexdigest())
        # Prioritize distinct true classes within each category, then fill if needed.
        chosen=[];seen=set()
        for r in ranked:
            if len(chosen)>=count:break
            if r['true_label'] not in seen:chosen.append(r);seen.add(r['true_label'])
        for r in ranked:
            if len(chosen)>=count:break
            if r not in chosen:chosen.append(r)
        rows.extend(chosen)
    require(bool(rows),'No examples available for requested categories')
    return rows


def render_figure(rows,cells,protocol,dpi):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from PIL import Image
    n=len(rows);height=3.25*n+1.5;fig,axes=plt.subplots(n,5,figsize=(15,height),squeeze=False)
    headers=['Original']+[TITLES[m] for m in MODELS]
    for j,title in enumerate(headers):axes[0,j].set_title(title,fontsize=12,fontweight='bold',pad=10)
    for i,row in enumerate(rows):
        reference=cells[row['cluster_id'],MODELS[0]]
        with Image.open(reference['directory']/'original.png') as image:axes[i,0].imshow(np.array(image.convert('RGB')))
        category='Common correct case' if row['selection_category']=='common_correct' else 'Case with a model error'
        original_label='True: '+row['true_label']+'\n'+category
        axes[i,0].text(.5,-.025,original_label,transform=axes[i,0].transAxes,ha='center',va='top',fontsize=9)
        axes[i,0].text(.01,.99,chr(97+i),transform=axes[i,0].transAxes,va='top',ha='left',fontsize=12,
                       bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
        for j,model in enumerate(MODELS,1):
            cell=cells[row['cluster_id'],model]
            with Image.open(cell['directory']/'overlay.png') as image:axes[i,j].imshow(np.array(image.convert('RGB')))
            mismatch=cell['saved_vs_fp32_prediction_match']!='True'
            prediction=textwrap.fill('Pred: '+cell['fp32_prediction']+(' †' if mismatch else ''),width=27)
            correct=cell['fp32_prediction']==cell['true_label']
            annotation=prediction+f"\nConfidence: {float(cell['fp32_predicted_probability'])*100:.1f}% | "+('Correct' if correct else 'Incorrect')
            if cell['cam_status']!='valid':annotation+='\nCAM: '+cell['cam_status']
            axes[i,j].text(.5,-.025,annotation,transform=axes[i,j].transAxes,ha='center',va='top',fontsize=8.5,
                           color=('#155e35' if correct else '#a52020'))
        for axis in axes[i]:axis.axis('off')
    fig.suptitle(f"Grad-CAM comparison — {protocol['experiment']}, seed {protocol['seed']}\nTargets: each model's predicted class; FP32 attribution",fontsize=13,y=.995)
    fig.subplots_adjust(left=.015,right=.995,top=1-.8/height,bottom=1.35/height,wspace=.035,hspace=.32)
    bar=fig.add_axes([.36,.55/height,.28,.12/height])
    colorbar=fig.colorbar(ScalarMappable(norm=Normalize(0,1),cmap='jet'),cax=bar,orientation='horizontal')
    colorbar.set_ticks([0,.5,1]);colorbar.ax.tick_params(labelsize=8)
    colorbar.set_label('Within-map normalized positive attribution',fontsize=9)
    fig.text(.5,.07/height,'Colors are normalized separately per map; they do not compare absolute evidence across models.',ha='center',fontsize=8)
    return fig


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path('Result/gradcam_E6_seed42'))
    p.add_argument('--output',type=Path,default=Path('Result/gradcam_comparison_E6_seed42'))
    p.add_argument('--correct-rows',type=int,default=2)
    p.add_argument('--error-rows',type=int,default=2)
    p.add_argument('--selection-seed',type=int,default=42)
    p.add_argument('--clusters',nargs='+',help='Optional explicit selected IDs, in display order; recorded as manual selection')
    p.add_argument('--rows-per-page',type=int,default=4)
    p.add_argument('--dpi',type=int,default=300)
    p.add_argument('--apply',action='store_true')
    args=p.parse_args();root=args.input.resolve();output=args.output.resolve()
    require(root.is_dir(),'Input folder not found. Complete the four-model Grad-CAM run first.')
    require(not output.exists(),'Output exists; choose a new --output. No overwrite.')
    require(root!=output and root not in output.parents and output not in root.parents,'Input and output must be separate folders')
    require(args.correct_rows>=0 and args.error_rows>=0 and args.rows_per_page>=1 and args.dpi>=72,'Invalid row count or DPI')
    protocol,selected,cells,hashes,warnings=validate_input(root)
    main_rows=choose_rows(selected,args.correct_rows,args.error_rows,args.selection_seed,args.clusters)
    main_counts=dict(Counter(r['selection_category'] for r in main_rows))
    print(json.dumps(dict(input=str(root),output=str(output),main_rows=len(main_rows),main_categories=main_counts,
        main_image_ids=[r['cluster_id'] for r in main_rows],supplement_rows=len(selected),
        supplement_pages=math.ceil(len(selected)/args.rows_per_page),warnings=warnings),indent=2))
    if not args.apply:print('PREVIEW ONLY. No files written. Add --apply to build figures.');return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    output.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix=output.name+'_building_',dir=output.parent))
    try:
        fig=render_figure(main_rows,cells,protocol,args.dpi)
        fig.savefig(stage/'gradcam_comparison_main.png',dpi=args.dpi,bbox_inches='tight',facecolor='white')
        fig.savefig(stage/'gradcam_comparison_main.pdf',bbox_inches='tight',facecolor='white');plt.close(fig)
        # Preserve the complete original selection, including blank or unfavorable maps.
        ordered=[r for category in ['common_correct','any_model_error'] for r in selected if r['selection_category']==category]
        with PdfPages(stage/'gradcam_comparison_supplement.pdf') as pdf:
            for page,start in enumerate(range(0,len(ordered),args.rows_per_page),1):
                fig=render_figure(ordered[start:start+args.rows_per_page],cells,protocol,args.dpi)
                fig.savefig(stage/f'supplement_page_{page:02d}.png',dpi=args.dpi,bbox_inches='tight',facecolor='white')
                pdf.savefig(fig,bbox_inches='tight',facecolor='white');plt.close(fig)
        write_csv(stage/'main_figure_rows.csv',[dict(display_row=i+1,**r) for i,r in enumerate(main_rows)])
        write_csv(stage/'supplement_figure_rows.csv',[dict(display_row=i+1,page=i//args.rows_per_page+1,**r) for i,r in enumerate(ordered)])
        caption=(f"Grad-CAM comparison for {protocol['experiment']} using training seed {protocol['seed']}. "
            'Each row uses the same original test image across ResNet50, MobileNetV3-Large, ViT-Base-Patch16 and Twins-SVT-Base. '
            f'The main figure contains {main_counts.get("common_correct",0)} common-correct and {main_counts.get("any_model_error",0)} any-model-error cases, selected from the previously frozen example set. '
            'Default row selection uses fixed SHA-256 ranking with class diversity, independently of confidence or heatmap appearance; an explicit --clusters override is documented in figure_protocol.json. '
            'Success/error selection categories refer to saved evaluation predictions. Cell labels show the FP32 prediction and its softmax probability; a dagger marks a difference from the saved AMP prediction. '
            'Each displayed CAM targets that model\'s predicted class logit. Ground-truth-target CAMs are excluded from this predicted-class comparison. '
            'Consequently a failure row may contain both correct and incorrect model predictions. '
            'Colors indicate positive attribution normalized within each map; absolute color strength cannot be compared between models. '
            'Target layers differ in spatial resolution. Non-valid maps are retained and labeled. '
            'The supplementary figure includes every example in the original selection. '
            'These qualitative maps do not establish causal feature use or lesion-localization accuracy.\n')
        (stage/'FIGURE_CAPTION.txt').write_text(caption,encoding='utf-8')
        build=dict(completed=True,input=str(root),experiment=protocol['experiment'],seed=protocol['seed'],models=MODELS,
            main_rows=len(main_rows),main_categories=main_counts,all_selection_rows=len(selected),
            row_selection='explicit IDs supplied by user' if args.clusters else 'SHA256 rank of seed/category/cluster; prefer distinct true classes within category',
            selection_seed=args.selection_seed,explicit_ids=args.clusters,target_role='predicted',confidence='FP32 softmax probability of predicted class',
            dpi=args.dpi,warnings=warnings,input_protocol_sha256=digest(root/'protocol.json'),
            input_selection_sha256=digest(root/'selection_manifest.csv'),input_audit_sha256=digest(root/'cam_audit.csv'),input_image_sha256=hashes,
            note='Composition only. No training, inference, CAM regeneration, map alteration or favorable-map filtering.')
        (stage/'figure_protocol.json').write_text(json.dumps(build,indent=2),encoding='utf-8')
        # Refuse publication if sources changed while composing the figures.
        for relative,expected in hashes.items():require(digest(root/relative)==expected,'Input image changed during figure generation')
        require(not output.exists(),'Output appeared during generation; refusing overwrite')
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage,ignore_errors=True);raise
    print('Completed: '+str(output))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        print('STOPPED: '+str(exc),file=sys.stderr);sys.exit(1)
