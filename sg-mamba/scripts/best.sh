# heatmap
python eval2stage.py \
    --config configs/mamba_swallow_i3d_eval_stage1.yaml \
    --config2 configs/sg-mamba-stage2-heatmap.yaml \
    --ckpt ckpts/best_stage1/mamba_swallow_i3d_stage1_mamba_swallow_stage1_2_0.0001/epoch_024_0.82621.pth.tar \
    --ckpt2 ckpts/e2e_heatmap_stage2_video_mamba_l3_ep45_sigma4_hid576 \
    --heatmap \
    --heatmap_dir tmp/raw_heatmap_sigma4 \
    --re-extract \
    --cache_dir tmp/raw_heatmap_sigma4 \
    --flow_i3d ckpts/flow_imagenet.pt \
    --rgb_i3d ckpts/pretrained_swallow_i3d.pth \
    --video_root data/swallow/sliding_videos/ \
    --heatmap_branch none \
    --heatmap_size 56 \
    --seg_duration 4.004 \
    --heatmap_sigma 4 \
    --cropped_videos \
    --test_first_stage \
    --confidence 0.3 

# desired result: ~52.73% (0.1-0.7 avg mAP)


# stage 2
python eval2tower.py \
    --config configs/mamba_swallow_i3d_eval_stage1.yaml \
    --config2 configs/sg-mamba-stage2.yaml \
    --config3 configs/sg-mamba-stage2-heatmap.yaml \
    --ckpt ckpts/best_stage1/mamba_swallow_i3d_stage1_mamba_swallow_stage1_2_0.0001/epoch_024_0.82621.pth.tar \
    --ckpt2 ckpts/2tower_crossmamba_3layer_ep30_vw0.7_heatmap_channelagg \
    --confidence 0.23 \
    --re-extract \
    --cache_dir tmp/threshold0.23 \
    --flow_i3d ckpts/flow_imagenet.pt \
    --rgb_i3d ckpts/pretrained_swallow_i3d.pth \
    --flow_dir xxx \
    --video_root data/swallow/sliding_videos/ \
    --test_first_stage \
    --tower_name CrossMambaEarlyFusion \
    --heatmap_dir tmp/raw_heatmap_sigma4_p0.23 \
    --heatmap_branch none \
    --heatmap_size 56 \
    --seg_duration 4.004 \
    --image_size 128 \
    --heatmap_sigma 4

# desired result: ~64.30% (0.1-0.7 avg mAP)