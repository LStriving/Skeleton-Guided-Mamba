import numpy as np
import torch
import torch.nn as nn
from fairscale.nn.checkpoint import checkpoint_wrapper

from .meta_archs import PtTransformerClsHead, PtTransformerRegHead
from .models import register_two_tower, make_neck
from .blocks import PostNormCrossTransformerBlock, PreNormCrossMambaBlock, PreNormCrossTransformerBlock, PreNormDINOTransformerBlock


class TwoTower(nn.Module):
    def __init__(self, visual_tower, heatmap_tower, cfg1, cfg2, *args, **kwargs):
        super(TwoTower, self).__init__()
        self.visual_tower = visual_tower
        self.heatmap_tower = heatmap_tower
        self.cfg1 = cfg1
        self.cfg2 = cfg2
    
    @property
    def device(self):
        return list(set(p.device for p in self.parameters()))[0]

@register_two_tower('Convfusion')
class Convfusion(TwoTower):
    def __init__(self, visual_tower, heatmap_tower, cfg1, cfg2, include_ori_loss=False, *args, **kwargs):
        super(Convfusion, self).__init__(visual_tower, heatmap_tower, cfg1, cfg2)
        self.include_ori_loss = include_ori_loss
        model_cfg = cfg1['model']

        embd_dim = model_cfg['embd_dim'] + cfg2['model']['embd_dim']
        backbone_arch = model_cfg['backbone_arch']
        fpn_dim = model_cfg['fpn_dim']
        head_dim = model_cfg['head_dim']
        head_kernel_size = model_cfg['head_kernel_size']
        head_with_ln = model_cfg['head_with_ln']
        head_num_layers = model_cfg['head_num_layers']
        train_cfg = cfg1['train_cfg']
        
        self.fusion_module = make_neck(
            'fusion_fpn', 
            **{
                'in_channels': [embd_dim] * (backbone_arch[-1] + 1),
                'out_channel': fpn_dim, # reduce the dimension
                'start_level': 0,
                'end_level': backbone_arch[-1] + 1,
            })
        
        # create the classification and regression heads
        # get param from cfg1
        self.cls_head = PtTransformerClsHead(
            fpn_dim, head_dim, model_cfg['num_classes'],
            kernel_size=head_kernel_size,
            prior_prob=train_cfg['cls_prior_prob'],
            with_ln=head_with_ln,
            num_layers=head_num_layers,
            empty_cls=train_cfg['head_empty_cls']
        )
        self.reg_head = PtTransformerRegHead(
            fpn_dim, head_dim, len(model_cfg['regression_range']),
            kernel_size=head_kernel_size,
            num_layers=head_num_layers,
            with_ln=head_with_ln
        )

    def forward(self, input): # TODO: CHECK
        video_list = [i[0] for i in input]
        heatmap_list = [i[1] for i in input]
        # get the FPN features from the visual tower
        visual_fpn_feats, visual_fpn_masks, visual_points = self.get_fpn_feat(self.visual_tower, video_list)
        # get the FPN features from the heatmap tower
        heatmap_fpn_feats, heatmap_fpn_masks, heatmap_points = self.get_fpn_feat(self.heatmap_tower, heatmap_list)
        # fuse the features from the two towers
        ## 1. concatenate the features from the two towers
        fused_feats = []
        for visual_feat, heatmap_feat in zip(visual_fpn_feats, heatmap_fpn_feats):
            fused_feat = torch.cat([visual_feat, heatmap_feat], dim=1)
            fused_feats.append(fused_feat)
        fused_masks = []
        for visual_mask, heatmap_mask in zip(visual_fpn_masks, heatmap_fpn_masks):
            fused_mask = torch.cat([visual_mask, heatmap_mask], dim=1)
            fused_masks.append(fused_mask)
        ## 2. apply the fusion module to fuse the features
        fused_feats, fpn_masks = self.fusion_module(fused_feats, visual_fpn_masks)
        # to reg/cls head
        # out_cls: List[B, #cls + 1, T_i]
        out_cls_logits = self.cls_head(fused_feats, fpn_masks)
        # out_offset: List[B, 2, T_i]
        out_offsets = self.reg_head(fused_feats, fpn_masks)
        # permute the outputs
        # out_cls: F List[B, #cls, T_i] -> F List[B, T_i, #cls]
        out_cls_logits = [x.permute(0, 2, 1) for x in out_cls_logits]
        # out_offset: F List[B, 2 (xC), T_i] -> F List[B, T_i, 2 (xC)]
        out_offsets = [x.permute(0, 2, 1) for x in out_offsets]
        # fpn_masks: F list[B, 1, T_i] -> F List[B, T_i]
        fpn_masks = [x.squeeze(1) for x in fpn_masks]

        # return loss during training
        if self.training:
            # generate segment/lable List[N x 2] / List[N] with length = B
            assert video_list[0]['segments'] is not None, "GT action labels does not exist"
            assert video_list[0]['labels'] is not None, "GT action labels does not exist"
            # print(video_list)
            gt_segments = [x['segments'].to(self.device) for x in video_list]
            gt_labels = [x['labels'].to(self.device) for x in video_list]

            # compute the gt labels for cls & reg
            # list of prediction targets
            gt_cls_labels, gt_offsets = self.visual_tower.label_points(
                visual_points, gt_segments, gt_labels)

            # compute the loss and return
            losses = self.visual_tower.losses(
                fpn_masks,
                out_cls_logits, out_offsets,
                gt_cls_labels, gt_offsets
            )
            if self.include_ori_loss:   # TODO: 1. align heatmap & visual feature 2. use resting forward rather than whole forward function
                v_losses = self.visual_tower(video_list)
                h_losses = self.heatmap_tower(heatmap_list)
                losses['final_loss'] += v_losses['final_loss'] + h_losses['final_loss']
            return losses

        else:
            self.visual_tower.training = False
            # decode the actions (sigmoid / stride, etc)
            results = self.visual_tower.inference(
                video_list, visual_points, fpn_masks,
                out_cls_logits, out_offsets
            )
            return results 
    
    def get_fpn_feat(self, module, input_list):
        # batch the video list into feats (B, C, T) and masks (B, 1, T)
        batched_inputs, batched_masks = module.preprocessing(input_list)

        # forward the network (backbone -> neck -> heads)
        feats, masks = module.backbone(batched_inputs, batched_masks)
        fpn_feats, fpn_masks = module.neck(feats, masks)
        # fpn_feats [16, 256, 768] ..[16, 256, 384]..[16, 256, 24]

        # compute the point coordinate along the FPN
        # this is used for computing the GT or decode the final results
        # points: List[T x 4] with length = # fpn levels
        # (shared across all samples in the mini-batch)
        points = module.point_generator(fpn_feats)
        return fpn_feats, fpn_masks, points

@register_two_tower('LogitsAvg')
class LogitAvg(TwoTower):
    def __init__(self, visual_tower, heatmap_tower, cfg1, cfg2, vw=0.7, *args, **kwargs):
        super().__init__(visual_tower, heatmap_tower, cfg1, cfg2)
        self.vw = vw
        assert 0 <= vw <= 1, "late fusion weight should be in [0, 1]"

    def forward(self, input):
        video_list = [i[0] for i in input]
        heatmap_list = [i[1] for i in input]
        # data item check
        v_videos = [i['video_id'] for i in video_list]
        h_videos = [i['video_id'] for i in heatmap_list]
        assert v_videos == h_videos,\
        "video list should be the same, make sure the data loader is correct"

        if self.training:
            return self.train_forward(video_list, heatmap_list)
        else:
            return self.inference_forward(video_list, heatmap_list)

    def train_forward(self, video_list, heatmap_list):
        v_losses = self.visual_tower(video_list)
        h_losses = self.heatmap_tower(heatmap_list) # problem occur when second forward
        losses = v_losses
        losses['final_loss'] += h_losses['final_loss']
        return losses

    def inference_forward(self, video_list, heatmap_list):
        self.visual_tower.training = False
        self.heatmap_tower.training = False
        # fuse logits and offsets
        v_out_cls_logits, v_out_offsets, v_fpn_masks, v_points = self.visual_tower.logit_forward(video_list)
        h_out_cls_logits, h_out_offsets, h_fpn_masks, h_points = self.heatmap_tower.logit_forward(heatmap_list)
        #NOTE: may need to check, but okay for now
        out_cls_logits = [(1 - self.vw) * h + self.vw * v for v, h in zip(v_out_cls_logits, h_out_cls_logits)] 
        out_offsets = [(1 - self.vw) * h + self.vw * v for v, h in zip(v_out_offsets, h_out_offsets)]

        return self.visual_tower.inference(
            video_list, v_points, v_fpn_masks,
            out_cls_logits, out_offsets
        )

@register_two_tower('LogitsAvg_sepbranch')
class LogitAvg_sepbranch(LogitAvg):
    def __init__(self, visual_tower, heatmap_tower, cfg1, cfg2, vw=0.7, *args, **kwargs):
        super().__init__(visual_tower, heatmap_tower, cfg1, cfg2, vw)
    
    def train_forward(self, video_list, heatmap_list):
        v_losses = self.visual_tower(video_list)
        h_losses = self.heatmap_tower(heatmap_list)
        return v_losses, h_losses

# TODO: Implement LogitsAvg_List or just remove it
@register_two_tower('LogitsAvg_List')
class LogitAvg_List(TwoTower):
    def __init__(self, visual_tower, heatmap_tower, cfg1, cfg2, vws=np.linspace(0.,1.,num=11), *args, **kwargs):
        super().__init__(visual_tower, heatmap_tower, cfg1, cfg2)
        self.vws = vws
        # iterable
        assert isinstance(vws, (list, np.ndarray, tuple)), "late fusion weight should be iterable"
        assert all(0 <= vw <= 1 for vw in vws), "late fusion weight should be in [0, 1]"
        # assert 0 <= vw <= 1, "late fusion weight should be in [0, 1]"
        
    def forward(self, input):
        video_list = [i[0] for i in input]
        heatmap_list = [i[1] for i in input]
        # data item check
        v_videos = [i['video_id'] for i in video_list]
        h_videos = [i['video_id'] for i in heatmap_list]
        assert v_videos == h_videos,\
        "video list should be the same, make sure the data loader is correct"

        if self.training:
            v_losses = self.visual_tower(video_list)
            h_losses = self.heatmap_tower(heatmap_list) # problem occur when second forward
            losses = v_losses
            losses['final_loss'] += h_losses['final_loss']
            return losses
        else:
            self.visual_tower.training = False
            self.heatmap_tower.training = False
            # fuse logits and offsets
            v_out_cls_logits, v_out_offsets, v_fpn_masks, v_points = self.old_forward(self.visual_tower, video_list)
            h_out_cls_logits, h_out_offsets, h_fpn_masks, h_points = self.old_forward(self.heatmap_tower, heatmap_list)
            #NOTE: may need to check, but okay for now
            results = []
            for vw in self.vws:
                out_cls_logits = [(1 - vw) * h + vw * v for v, h in zip(v_out_cls_logits, h_out_cls_logits)] 
                out_offsets = [(1 - vw) * h + vw * v for v, h in zip(v_out_offsets, h_out_offsets)]

                res = self.visual_tower.inference(
                    video_list, v_points, v_fpn_masks,
                    out_cls_logits, out_offsets
                )
                results.append(res)
            return results


    def old_forward(self, module, video_list):
        # batch the video list into feats (B, C, T) and masks (B, 1, T)
        batched_inputs, batched_masks = module.preprocessing(video_list)

        # forward the network (backbone -> neck -> heads)
        feats, masks = module.backbone(batched_inputs, batched_masks)
        fpn_feats, fpn_masks = module.neck(feats, masks)
        # fpn_feats [16, 256, 768] ..[16, 256, 384]..[16, 256, 24]

        # compute the point coordinate along the FPN
        # this is used for computing the GT or decode the final results
        # points: List[T x 4] with length = # fpn levels
        # (shared across all samples in the mini-batch)
        points = module.point_generator(fpn_feats)

        # out_cls: List[B, #cls + 1, T_i]
        out_cls_logits = module.cls_head(fpn_feats, fpn_masks)
        # out_offset: List[B, 2, T_i]
        out_offsets = module.reg_head(fpn_feats, fpn_masks)

        # permute the outputs
        # out_cls: F List[B, #cls, T_i] -> F List[B, T_i, #cls]
        out_cls_logits = [x.permute(0, 2, 1) for x in out_cls_logits]
        # out_offset: F List[B, 2 (xC), T_i] -> F List[B, T_i, 2 (xC)]
        out_offsets = [x.permute(0, 2, 1) for x in out_offsets]
        # fpn_masks: F list[B, 1, T_i] -> F List[B, T_i]
        fpn_masks = [x.squeeze(1) for x in fpn_masks]
        return out_cls_logits, out_offsets, fpn_masks, points

class CrossBlockEarlyFusion(TwoTower):
    """
    Add cross module for both visual and heatmap towers
    The module is added before the (actionmamba or actionformer) backbone
    """
    def __init__(self, visual_tower, heatmap_tower, cfg1, cfg2, vw=0.8, num_layers=1, act_checkpoint=True, *args, **kwargs):
        super().__init__(visual_tower, heatmap_tower, cfg1, cfg2)
        self.vw = vw
        self.wrapper = checkpoint_wrapper if act_checkpoint else lambda x: x
        self.num_layers = num_layers
        v_input_dim = cfg1['model']['input_dim']
        h_input_dim = cfg2['model']['input_dim']
        print(f"v_input_dim={v_input_dim}, h_input_dim={h_input_dim}")

        self.h2v = nn.ModuleList( # query: visual, key/value: heatmap
            [self.wrapper(PostNormCrossTransformerBlock(
                v_input_dim,
                h_input_dim,
                cfg1['model']['n_head'],
            )) for _ in range(num_layers)]
        )
        self.v2h = nn.ModuleList(
            [self.wrapper(PostNormCrossTransformerBlock(
                h_input_dim,
                v_input_dim,
                cfg2['model']['n_head'],
            )) for _ in range(num_layers)]
        )

    def forward(self, input):
        video_list = [i[0] for i in input]
        heatmap_list = [i[1] for i in input]
        v_out, h_out = self.fused_logit_foward(video_list, heatmap_list)
        if self.training:
            v_losses = self.visual_tower.train_forward(video_list, *v_out)
            h_losses = self.heatmap_tower.train_forward(heatmap_list, *h_out)
            losses = v_losses
            losses['final_loss'] += h_losses['final_loss']
            losses['heamap_cls_loss'] = h_losses['cls_loss']
            losses['heamap_reg_loss'] = h_losses['reg_loss']
            return losses
        else:
            self.visual_tower.training = False
            self.heatmap_tower.training = False
            v_out_cls_logits, v_out_offsets, v_fpn_masks, v_points = v_out
            h_out_cls_logits, h_out_offsets, h_fpn_masks, h_points = h_out
            # fuse logits and offsets
            out_cls_logits = [(1 - self.vw) * h + self.vw * v for v, h in zip(v_out_cls_logits, h_out_cls_logits)]
            out_offsets = [(1 - self.vw) * h + self.vw * v for v, h in zip(v_out_offsets, h_out_offsets)]
            return self.visual_tower.inference(
                video_list, v_points, v_fpn_masks,
                out_cls_logits, out_offsets
            )

    def fused_logit_foward(self, video_list, heatmap_list):
        if getattr(self.visual_tower, 'pre_forward', None):
            video_list = self.visual_tower.pre_forward(video_list)
        if getattr(self.heatmap_tower, 'pre_forward', None):
            heatmap_list = self.heatmap_tower.pre_forward(heatmap_list)
        batched_v_inputs, batched_v_masks = self.visual_tower.preprocessing(video_list)
        batched_h_inputs, batched_h_masks = self.heatmap_tower.preprocessing(heatmap_list)

        # cross attention module here 
        for i in range(self.num_layers):
            original_v_inputs, original_v_masks = batched_v_inputs, batched_v_masks
            # (query: visual, key/value: heatmap)
            batched_v_inputs, batched_v_masks = self.h2v[i](query=batched_v_inputs, key=batched_h_inputs, value=batched_h_inputs, 
                                                            query_mask=batched_v_masks, kv_mask=batched_h_masks)
            # (query: heatmap, key/value: visual)
            batched_h_inputs, batched_h_masks = self.v2h[i](query=batched_h_inputs, key=original_v_inputs, value=original_v_inputs, 
                                                            query_mask=original_v_masks, kv_mask=original_v_masks)
        
        # forward the network (backbone -> neck -> heads)
        v_out_cls_logits, v_out_offsets, v_fpn_masks, v_points \
            = self.visual_tower._logit_processed_input_forward(batched_v_inputs, batched_v_masks)
        h_out_cls_logits, h_out_offsets, h_fpn_masks, h_points \
            = self.heatmap_tower._logit_processed_input_forward(batched_h_inputs, batched_h_masks)
        
        return (v_out_cls_logits, v_out_offsets, v_fpn_masks, v_points), \
                (h_out_cls_logits, h_out_offsets, h_fpn_masks, h_points)


@register_two_tower('CrossAttnEarlyFusion')
class CrossAttnEarlyFusion(CrossBlockEarlyFusion):
    """
    Add cross attention module for both visual and heatmap towers
    The module is added before the (actionmamba or actionformer) backbone
    """
    def __init__(self, visual_tower, heatmap_tower, cfg1, cfg2, vw=0.8, num_layers=1, act_checkpoint=True, *args, **kwargs):
        super().__init__(visual_tower, heatmap_tower, cfg1, cfg2)
        v_input_dim = cfg1['model']['input_dim']
        h_input_dim = cfg2['model']['input_dim']
        print(f"CrossAttnEarlyFusion: vw={vw}, num_layers={num_layers}")

        self.h2v = nn.ModuleList( # query: visual, key/value: heatmap
            [self.wrapper(PostNormCrossTransformerBlock(
                v_input_dim,
                h_input_dim,
                cfg1['model']['n_head'],
            )) for _ in range(num_layers)]
        )
        self.v2h = nn.ModuleList(
            [self.wrapper(PostNormCrossTransformerBlock(
                h_input_dim,
                v_input_dim,
                cfg2['model']['n_head'],
            )) for _ in range(num_layers)]
        )
    
@register_two_tower('CrossAttnEarlyFusion-PreNorm')
class CrossAttnEarlyFusionPreNorm(CrossBlockEarlyFusion):
    def __init__(self, visual_tower, heatmap_tower, cfg1, cfg2, vw=0.8, num_layers=1, act_checkpoint=True, *args, **kwargs):
        super().__init__(visual_tower, heatmap_tower, cfg1, cfg2, vw, num_layers, act_checkpoint)
        v_input_dim = cfg1['model']['input_dim']
        h_input_dim = cfg2['model']['input_dim']
        print(f"CrossAttnEarlyFusion-PreNorm: vw={vw}, num_layers={num_layers}")
        print(f"v_input_dim={v_input_dim}, h_input_dim={h_input_dim}")

        self.h2v = nn.ModuleList( # query: visual, key/value: heatmap
            [self.wrapper(PreNormCrossTransformerBlock(
                v_input_dim,
                h_input_dim,
                cfg1['model']['n_head'],
            )) for _ in range(num_layers)]
        )
        self.v2h = nn.ModuleList(
            [self.wrapper(PreNormCrossTransformerBlock(
                h_input_dim,
                v_input_dim,
                cfg2['model']['n_head'],
            )) for _ in range(num_layers)]
        )

@register_two_tower('CrossAttnEarlyFusion-PreNorm-LastLayerGamma')
class CrossAttnEarlyFusionPreNormLastLayerGamma(CrossBlockEarlyFusion):
    def __init__(self, visual_tower, heatmap_tower, cfg1, cfg2, vw=0.8, num_layers=3, act_checkpoint=True, *args, **kwargs):
        super().__init__(visual_tower, heatmap_tower, cfg1, cfg2, vw, num_layers, act_checkpoint)
        v_input_dim = cfg1['model']['input_dim']
        h_input_dim = cfg2['model']['input_dim']
        print(f"CrossAttnEarlyFusion-PreNorm-LastLayerGamma: vw={vw}, num_layers={num_layers}")
        print(f"v_input_dim={v_input_dim}, h_input_dim={h_input_dim}")

        assert num_layers > 1, "num_layers should be larger than 1"
        self.h2v = nn.ModuleList( # query: visual, key/value: heatmap
            [self.wrapper(PreNormCrossTransformerBlock(
                v_input_dim,
                h_input_dim,
                cfg1['model']['n_head'],
                gamma_num=0,
            )) for _ in range(num_layers - 1)]
        )
        self.h2v.append(self.wrapper(PreNormCrossTransformerBlock(
            v_input_dim,
            h_input_dim,
            cfg1['model']['n_head'],
            gamma_num=cfg1['two_tower']['gamma_num'],
            init_value=cfg1['two_tower']['init_value'],
        ))) # last layer with gamma
        self.v2h = nn.ModuleList(
            [self.wrapper(PreNormCrossTransformerBlock(
                h_input_dim,
                v_input_dim,
                cfg1['model']['n_head'],
                gamma_num=0,
            )) for _ in range(num_layers - 1)]
        )
        self.v2h.append(self.wrapper(PreNormCrossTransformerBlock(
            h_input_dim,
            v_input_dim,
            cfg1['model']['n_head'],
            gamma_num=cfg1['two_tower']['gamma_num'],
            init_value=cfg1['two_tower']['init_value'],
        ))) # last layer with gamma


@register_two_tower('CrossAttnEarlyFusion-PreNorm-Gamma')
class CrossAttnEarlyFusionPreNormLastLayerGamma(CrossBlockEarlyFusion):
    def __init__(self, visual_tower, heatmap_tower, cfg1, cfg2, vw=0.8, num_layers=3, act_checkpoint=True, *args, **kwargs):
        super().__init__(visual_tower, heatmap_tower, cfg1, cfg2, vw, num_layers, act_checkpoint)
        v_input_dim = cfg1['model']['input_dim']
        h_input_dim = cfg2['model']['input_dim']
        print(f"CrossAttnEarlyFusion-PreNorm-LastLayerGamma: vw={vw}, num_layers={num_layers}")
        print(f"v_input_dim={v_input_dim}, h_input_dim={h_input_dim}")

        assert num_layers > 1, "num_layers should be larger than 1"
        self.h2v = nn.ModuleList( # query: visual, key/value: heatmap
            [self.wrapper(PreNormCrossTransformerBlock(
                v_input_dim,
                h_input_dim,
                cfg1['model']['n_head'],
                gamma_num=cfg1['two_tower']['gamma_num'],
                init_value=cfg1['two_tower']['init_value'],
            )) for _ in range(num_layers)]
        )
        self.v2h = nn.ModuleList(
            [self.wrapper(PreNormCrossTransformerBlock(
                h_input_dim,
                v_input_dim,
                cfg1['model']['n_head'],
                gamma_num=cfg1['two_tower']['gamma_num'],
                init_value=cfg1['two_tower']['init_value'],
            )) for _ in range(num_layers)]
        )


@register_two_tower('DINOAttnEarlyFusion')
class DINOAttnEarlyFusionPreNorm(CrossBlockEarlyFusion):
    def __init__(self, visual_tower, heatmap_tower, cfg1, cfg2, vw=0.8, num_layers=1, act_checkpoint=True, *args, **kwargs):
        super().__init__(visual_tower, heatmap_tower, cfg1, cfg2, vw, num_layers, act_checkpoint)
        v_input_dim = cfg1['model']['input_dim']
        h_input_dim = cfg2['model']['input_dim']
        print(f"DINOAttnEarlyFusion: vw={vw}, num_layers={num_layers}")

        self.h2v = nn.ModuleList( # query: visual, key/value: heatmap
            [self.wrapper(PreNormDINOTransformerBlock(
                v_input_dim,
                h_input_dim,
                cfg1['model']['n_head'],
                path_pdrop=0.1,
                init_value=cfg1['two_tower']['init_value']
            )) for _ in range(num_layers)]
        )
        self.v2h = nn.ModuleList(
            [self.wrapper(PreNormDINOTransformerBlock(
                h_input_dim,
                v_input_dim,
                cfg2['model']['n_head'],
                path_pdrop=0.1,
                init_value=cfg1['two_tower']['init_value']
            )) for _ in range(num_layers)]
        )

@register_two_tower('CrossMambaEarlyFusion')
class CrossMambaEarlyFusion(CrossBlockEarlyFusion):
    """
    Add cross mamba module for both visual and heatmap towers
    The module is added before the (actionmamba or actionformer) backbone
    """
    def __init__(self, visual_tower, heatmap_tower, cfg1, cfg2, vw=0.7, num_layers=3, act_checkpoint=True, *args, **kwargs):
        super().__init__(visual_tower, heatmap_tower, cfg1, cfg2)
        self.vw = vw
        self.wrapper = checkpoint_wrapper if act_checkpoint else lambda x: x
        self.num_layers = num_layers
        v_input_dim = cfg1['model']['input_dim']
        h_input_dim = cfg2['model']['input_dim']
        print(f"CrossMambaEarlyFusion: vw={vw}, num_layers={num_layers}")

        self.h2v = nn.ModuleList( # query: visual, key/value: heatmap
            [self.wrapper(PreNormCrossMambaBlock(
                v_input_dim,
                h_input_dim,
                init_value=cfg1['two_tower']['init_value'],
                path_pdrop=0.1,
                channel_agg=cfg1['two_tower']['channel_agg'],
                ca_type=cfg1['two_tower']['ca_type'],
            )) for _ in range(num_layers)]
        )
        self.v2h = nn.ModuleList(
            [self.wrapper(PreNormCrossMambaBlock(
                h_input_dim,
                v_input_dim,
                init_value=cfg1['two_tower']['init_value'],
                path_pdrop=0.1,
                channel_agg=cfg1['two_tower']['channel_agg'],
                ca_type=cfg1['two_tower']['ca_type'],
            )) for _ in range(num_layers)]
        )

    def fused_logit_foward(self, video_list, heatmap_list):
        if getattr(self.visual_tower, 'pre_forward', None):
            video_list = self.visual_tower.pre_forward(video_list)
        if getattr(self.heatmap_tower, 'pre_forward', None):
            heatmap_list = self.heatmap_tower.pre_forward(heatmap_list)
        batched_v_inputs, batched_v_masks = self.visual_tower.preprocessing(video_list)
        batched_h_inputs, batched_h_masks = self.heatmap_tower.preprocessing(heatmap_list)

        # cross attention module here 
        for i in range(self.num_layers):
            original_v_inputs, original_v_masks = batched_v_inputs, batched_v_masks
            # (query: visual, key/value: heatmap)
            batched_v_inputs, batched_v_masks = self.h2v[i](query=batched_v_inputs, value=batched_h_inputs, 
                                                            query_mask=batched_v_masks, kv_mask=batched_h_masks)
            # (query: heatmap, key/value: visual)
            batched_h_inputs, batched_h_masks = self.v2h[i](query=batched_h_inputs, value=original_v_inputs, 
                                                            query_mask=original_v_masks, kv_mask=original_v_masks)
        
        # forward the network (backbone -> neck -> heads)
        v_out_cls_logits, v_out_offsets, v_fpn_masks, v_points \
            = self.visual_tower._logit_processed_input_forward(batched_v_inputs, batched_v_masks)
        h_out_cls_logits, h_out_offsets, h_fpn_masks, h_points \
            = self.heatmap_tower._logit_processed_input_forward(batched_h_inputs, batched_h_masks)
        
        return (v_out_cls_logits, v_out_offsets, v_fpn_masks, v_points), \
                (h_out_cls_logits, h_out_offsets, h_fpn_masks, h_points)
    
@register_two_tower('SelfMambaEarlyFusion')
class SelfMambaEarlyFusion(CrossBlockEarlyFusion):
    """
    Self-Mamba module for ablation study
    The module is added before the (actionmamba or actionformer) backbone
    """
    def __init__(self, visual_tower, heatmap_tower, cfg1, cfg2, vw=0.7, num_layers=3, act_checkpoint=True, *args, **kwargs):
        super().__init__(visual_tower, heatmap_tower, cfg1, cfg2)
        self.vw = vw
        self.wrapper = checkpoint_wrapper if act_checkpoint else lambda x: x
        self.num_layers = num_layers
        v_input_dim = cfg1['model']['input_dim']
        h_input_dim = cfg2['model']['input_dim']
        print(f"SelfMambaEarlyFusion: vw={vw}, num_layers={num_layers}")

        self.h2v = nn.ModuleList( # query: visual, key/value: heatmap
            [self.wrapper(PreNormCrossMambaBlock(
                v_input_dim,
                v_input_dim,
                init_value=cfg1['two_tower']['init_value'],
                path_pdrop=0.1,
                channel_agg=cfg1['two_tower']['channel_agg'],
                ca_type=cfg1['two_tower']['ca_type'],
            )) for _ in range(num_layers)]
        )
        self.v2h = nn.ModuleList(
            [self.wrapper(PreNormCrossMambaBlock(
                h_input_dim,
                h_input_dim,
                init_value=cfg1['two_tower']['init_value'],
                path_pdrop=0.1,
                channel_agg=cfg1['two_tower']['channel_agg'],
                ca_type=cfg1['two_tower']['ca_type'],
            )) for _ in range(num_layers)]
        )

    def fused_logit_foward(self, video_list, heatmap_list):
        if getattr(self.visual_tower, 'pre_forward', None):
            video_list = self.visual_tower.pre_forward(video_list)
        if getattr(self.heatmap_tower, 'pre_forward', None):
            heatmap_list = self.heatmap_tower.pre_forward(heatmap_list)
        batched_v_inputs, batched_v_masks = self.visual_tower.preprocessing(video_list)
        batched_h_inputs, batched_h_masks = self.heatmap_tower.preprocessing(heatmap_list)

        # cross attention module here 
        for i in range(self.num_layers):
            # (query: visual, key/value: heatmap)
            batched_v_inputs, batched_v_masks = self.h2v[i](query=batched_v_inputs, value=batched_v_inputs, 
                                                            query_mask=batched_v_masks, kv_mask=batched_v_masks)
            # (query: heatmap, key/value: visual)
            batched_h_inputs, batched_h_masks = self.v2h[i](query=batched_h_inputs, value=batched_h_inputs, 
                                                            query_mask=batched_h_masks, kv_mask=batched_h_masks)
        
        # forward the network (backbone -> neck -> heads)
        v_out_cls_logits, v_out_offsets, v_fpn_masks, v_points \
            = self.visual_tower._logit_processed_input_forward(batched_v_inputs, batched_v_masks)
        h_out_cls_logits, h_out_offsets, h_fpn_masks, h_points \
            = self.heatmap_tower._logit_processed_input_forward(batched_h_inputs, batched_h_masks)
        
        return (v_out_cls_logits, v_out_offsets, v_fpn_masks, v_points), \
                (h_out_cls_logits, h_out_offsets, h_fpn_masks, h_points)