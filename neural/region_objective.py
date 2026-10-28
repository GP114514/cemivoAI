"""Training-only region importance from real target metadata, never a renderer."""
from __future__ import annotations
import math

import numpy as np
import torch

from .readability import edge_importance,edge_weighted_flow_loss


def resolve_region_weight(value,checkpoint=None,field='text_roi_weight'):
    if value is None:
        value=checkpoint.get('training_args',{}).get(field,0.0) if checkpoint else 0.0
    if isinstance(value,bool):
        raise ValueError('Region loss weight must be finite and in 0..16')
    try:weight=float(value)
    except (ValueError,TypeError,OverflowError):
        raise ValueError('Region loss weight must be finite and in 0..16') from None
    if not math.isfinite(weight) or not 0<=weight<=16:
        raise ValueError('Region loss weight must be finite and in 0..16')
    return weight


def metadata_region_masks(metadata,height,width):
    """Native binary boxes from the visible text and declared state target ROI."""
    if not isinstance(metadata,dict) or type(height) is not int or type(width) is not int or min(height,width)<1:
        raise ValueError('Invalid native target metadata dimensions')
    text=np.zeros((1,height,width),dtype=np.bool_)
    state=np.zeros_like(text)
    def fill(mask,bounds):
        if (not isinstance(bounds,(list,tuple)) or len(bounds)!=4 or
                any(type(value) is not int for value in bounds)):
            raise ValueError('Target region requires four native integer coordinates')
        x0,y0,x1,y1=bounds
        if x1<=x0 or y1<=y0:
            raise ValueError('Target region must have positive extent')
        # Historical native glyph extents can meet a canvas edge. Weight only
        # the pixels that are actually present, never wrap negative indices.
        x0,y0,x1,y1=max(0,x0),max(0,y0),min(width,x1),min(height,y1)
        if x1>x0 and y1>y0:mask[:,y0:y1,x0:x1]=True
    labels=metadata.get('drawn_text')
    if labels is None:
        # V2 stores the actual title/button extents under its older audit key;
        # these are native visible boxes too, not inferred or redrawn labels.
        labels=metadata.get('caption_label_audit',[])
    if not isinstance(labels,list):raise ValueError('drawn_text must be a list')
    for label in labels:
        if not isinstance(label,dict):raise ValueError('Invalid target label metadata')
        if label.get('visible'):fill(text,label['bbox'])
    if metadata.get('caption_policy')=='verbatim-interaction-state' and metadata.get('target_bounds') is not None:
        fill(state,metadata['target_bounds'])
    if metadata.get('caption_policy') == 'verbatim-app-flow-state-v5':
        if metadata.get('dataset_version') != 'native256-app-flows-v5':
            raise ValueError('V5 state regions require the audited V5 dataset version')
        empty_regions = metadata.get('empty_regions')
        transition = metadata.get('transition_from_previous')
        if not isinstance(empty_regions, list) or (transition is not None and not isinstance(transition, dict)):
            raise ValueError('V5 needs explicit empty regions and transition metadata')
        effects = transition.get('effect_regions') if transition is not None else []
        if not isinstance(effects, list) or (transition is not None and not effects):
            raise ValueError('V5 transitions need a nonempty effect region list')
        for bounds in empty_regions + effects:
            if (not isinstance(bounds, list) or len(bounds) != 4
                    or any(type(value) is not int for value in bounds)
                    or not 0 <= bounds[0] < bounds[2] <= width
                    or not 0 <= bounds[1] < bounds[3] <= height):
                raise ValueError('V5 state regions must stay inside the native target')
            fill(state, bounds)
    return {'text':text,'state':state}


def region_weighted_flow_loss(prediction,target,images,regions,*,edge_weight=0.0,text_weight=0.0,state_weight=0.0):
    """Balance global, glyph-box and changed-region error per image.

    Each active ROI contributes its own pixel mean so a small composer/switch
    is not diluted by the 65,536-pixel image. Divide by active loss weights to
    keep scale bounded. Zero ROI strengths retain the exact historical path.
    """
    text_weight=resolve_region_weight(text_weight)
    state_weight=resolve_region_weight(state_weight)
    if text_weight==0 and state_weight==0:
        return edge_weighted_flow_loss(prediction,target,images,edge_weight)
    if (not isinstance(prediction,torch.Tensor) or prediction.ndim!=4 or prediction.shape[1]!=1 or
            not isinstance(target,torch.Tensor) or not isinstance(images,torch.Tensor) or
            prediction.shape!=target.shape or prediction.shape!=images.shape or
            prediction.device!=target.device or prediction.device!=images.device):
        raise ValueError('Region objective requires matching native Bx1xHxW tensors')
    error=(prediction.float()-target.float()).square()
    base=(error*edge_importance(images,edge_weight)).mean(dim=(1,2,3))
    numerator=base
    denominator=torch.ones_like(base)
    for name,strength in (('text',text_weight),('state',state_weight)):
        if not strength:continue
        mask=regions.get(name) if isinstance(regions,dict) else None
        if (not isinstance(mask,torch.Tensor) or mask.dtype!=torch.bool or mask.shape!=prediction.shape or
                mask.device!=prediction.device):
            raise ValueError('Region masks must be native boolean tensors matching the target')
        mask=mask.detach()
        count=mask.sum(dim=(1,2,3))
        present=count>0
        local=(error*mask).sum(dim=(1,2,3))/count.clamp_min(1)
        numerator=numerator+strength*local
        denominator=denominator+strength*present
    return (numerator/denominator).mean()
