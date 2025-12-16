CUDA_VISIBLE_DEVICES="0,1,2,3"
# split the video list into multiple parts for parallel processing
python ../tools/split_video_list.py \
    --input_dir data/swallow/sliding_videos \
    --output_dir  tmp/ \
    --num_splits 4
mkdir -p logs

# loop
for part_id in {0..3}
do
    # record command and script to log
    echo "Processing part ${part_id} ..."
    
    ################################################################################
    echo "########### Command ###########" >> logs/extract_features_part_${part_id}_total_4.log
    echo "CUDA_VISIBLE_DEVICES=$part_id python ../tools/get_features_batch.py \
        --videos_dir data/swallow/sliding_videos \
        --flow_dir data/swallow/flowframe_slided \
        --output_dir data/swallow/stage_2/rgb_flow_no_interplote/no_interplote \
        --filter_list tmp/split_${part_id}_total_4.txt \
        --batch_size -1 \
        --cuda > logs/extract_features_part_${part_id}_total_4.log 2>&1 &" >> logs/extract_features_part_${part_id}_total_4.log
    echo "#####################################" >> logs/extract_features_part_${part_id}_total_4.log

    echo "########### Script Content ###########" >> logs/extract_features_part_${part_id}_total_4.log
    cat ../tools/get_features_batch.py >> logs/extract_features_part_${part_id}_total_4.log
    echo "#####################################" >> logs/extract_features_part_${part_id}_total_4.log
    ################################################################################
    
    # run in background
    CUDA_VISIBLE_DEVICES=$part_id python ../tools/get_features_batch.py \
        --videos_dir data/swallow/sliding_videos \
        --flow_dir data/swallow/flowframe_slided \
        --output_dir data/swallow/stage_2/rgb_flow_no_interplote/no_interplote \
        --filter_list tmp/split_${part_id}_total_4.txt \
        --batch_size -1 \
        --cuda > logs/extract_features_part_${part_id}_total_4.log 2>&1 &
done