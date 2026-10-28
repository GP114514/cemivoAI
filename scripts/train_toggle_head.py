"""Train a separate randomly initialized neural toggle editor; no app mutation."""
from pathlib import Path
import argparse
from datetime import datetime, timezone
from io import BytesIO
import hashlib
import json
import sys
import time

import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from neural.toggle_head import ToggleHead, encode_batch, save_checkpoint
from neural.toggle_training_data import make_splits, measurable_state
from stable_edit import measure_toggle_state, compose_toggle, validate_toggle_transition


def encode_examples(rows,device):
    inputs=encode_batch([row['source'] for row in rows],[row['target_state'] for row in rows],device=device)
    targets=encode_batch([row['target'] for row in rows],[row['target_state'] for row in rows],device=device)
    return inputs,targets['pixels']


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=1800)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--validation-image',type=Path,required=True,
        help='A separately generated, native grayscale PNG with an off-state switch; not used for training.')
    parser.add_argument('--validation-bounds',type=int,nargs=4,required=True,
        metavar=('X0','Y0','X1','Y1'),help='Half-open switch bounds inside the validation image.')
    args=parser.parse_args()
    if not args.validation_image.is_file():
        parser.error('--validation-image must be an existing PNG file')
    full=args.validation_image.read_bytes()
    with Image.open(BytesIO(full)) as validation_image:
        if validation_image.format!='PNG' or validation_image.mode!='L':
            parser.error('--validation-image must be a native grayscale PNG (mode L)')
        image_width,image_height=validation_image.size
    bounds=list(args.validation_bounds)
    x0,y0,x1,y1=bounds
    if not(0<=x0<x1<=image_width and 0<=y0<y1<=image_height):
        parser.error('--validation-bounds must fit within the validation image')
    if measure_toggle_state(full,bounds)!='off':
        parser.error('the validation bounds must enclose a measurable off-state switch')
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(20261026)
    device=torch.device(args.device)
    if device.type=='cuda':
        free,_=torch.cuda.mem_get_info()
        if free<3*1024**3: raise RuntimeError('Keep main training priority: less than 3 GiB free')
        torch.cuda.set_per_process_memory_fraction(.06)
        torch.backends.cudnn.benchmark=False
    started=time.perf_counter()
    splits=make_splits(1600,160,seed=24)
    print(json.dumps({'stage':'data_ready','train':len(splits['train']),'val':len(splits['val'])}),flush=True)
    model=ToggleHead().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    batches=[encode_examples(splits['train'][start:start+32],device) for start in range(0,1600,32)]
    rng=np.random.default_rng(20261026)
    history=[]
    for step in range(1,args.steps+1):
        batch,target=batches[int(rng.integers(len(batches)))]
        prediction=model(batch['pixels'],batch['valid'],batch['target_states'])
        mask=batch['valid'].float()
        # Emphasize actual shape strokes while still learning page polarity.
        weights=mask*(1+2*(target.abs()<.95).float())
        loss=((prediction-target).abs()*weights).sum()/weights.sum()
        loss=loss+.25*((prediction-target).square()*mask).sum()/mask.sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1)
        optimizer.step()
        if step%100==0:
            item={'step':step,'loss':round(float(loss.detach()),6),'elapsed_seconds':round(time.perf_counter()-started,2)}
            history.append(item)
            print(json.dumps(item),flush=True)
        if step==args.steps//2:
            for group in optimizer.param_groups: group['lr']=.0003
    model.eval()
    val=[]
    grid=Image.new('L',(64*3,40*12),128)
    for index,row in enumerate(splits['val']):
        pred=model.predict_native(row['source'],row['target_state'])
        actual=measurable_state(pred,row['bounds'],row['background'])
        val.append({'index':index,'expected':row['target_state'],'detected':actual,'passed':actual==row['target_state']})
        if index<12:
            for column,pixels in enumerate((row['source'][0],row['target'][0],pred)):
                grid.paste(Image.fromarray(pixels),(column*64,index*40))
    grid.save(args.output/'synthetic-validation.png')
    pixels=np.asarray(Image.open(BytesIO(full)),dtype=np.uint8)
    original=pixels[y0:y1,x0:x1].copy()
    real=[]
    current=full
    for ordinal,(before_state,target_state) in enumerate((('off','on'),('on','off'))):
        current_pixels=np.asarray(Image.open(BytesIO(current)),dtype=np.uint8)
        pred=model.predict_native(current_pixels[y0:y1,x0:x1].copy(),target_state)
        candidate=current_pixels.copy();candidate[y0:y1,x0:x1]=pred
        buffer=BytesIO();Image.fromarray(candidate).save(buffer,format='PNG')
        delivered,exterior=compose_toggle(current,buffer.getvalue(),bounds)
        detected=measure_toggle_state(delivered,bounds)
        real.append({'from':before_state,'target':target_state,'detected':detected,'passed':detected==target_state,
            'outside_changed_pixels':exterior['outside_changed_pixels']})
        (args.output/f'actual-switch-{ordinal}-{target_state}.png').write_bytes(delivered)
        current=delivered
    passed=sum(item['passed'] for item in val)
    validated=passed>=.9*len(val) and all(item['passed'] for item in real)
    validation={'synthetic_count':len(val),'synthetic_passed':passed,'synthetic_details':val,
        'real_neural_source_sha256':hashlib.sha256(full).hexdigest(),'real_neural_crop_used_for_training':False,
        'real_neural_results':real,'geometry_gate':'unchanged production stable_edit detector',
        'outside_changed_pixels':0,'not_general_ui_quality':True}
    save_checkpoint(args.output/'inference.pt',model,args.steps,heldout_validated=validated,validation=validation)
    report={'created_at':datetime.now(timezone.utc).isoformat(),'random_initialization':True,'pretrained_weights':False,
        'parameters':model.parameter_count,'steps':args.steps,'train_samples':1600,'val_samples':160,
        'elapsed_seconds':round(time.perf_counter()-started,2),'heldout_validated':validated,
        'training_history':history,'validation':validation,'main_training_modified':False,'inference_uses_procedural_drawing':False}
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(args.output),'validated':validated,'synthetic_passed':passed,'real':real},ensure_ascii=True),flush=True)
    return 0 if validated else 2


if __name__=='__main__':
    sys.exit(main())
