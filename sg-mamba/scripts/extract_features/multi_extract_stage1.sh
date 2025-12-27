# Set CUDA_VISIBLE_DEVICES to a specific GPU for each part
CUDA_VISIBLE_DEVICES="0,1,2,3"
# split the video list into multiple parts for parallel processing
python tools/split_list.py \
    --input_dir data/swallow/sliding_videos \
    --output_dir  tmp/stage1/ \
    --num_splits 4

# Create logs directory if it doesn't exist
mkdir -p logs/stage1

# Loop through each part
for part_id in {0..3}
do
    echo "Processing part ${part_id} ..."

    # Define the GPU ID for this part
    gpu_id=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f$((part_id+1)))

    # Record command and script to log
    log_file="logs/stage1/extract_features_part_${part_id}_total_4.log"
    echo "########### Command ###########" >> "$log_file"
    echo "CUDA_VISIBLE_DEVICES=$gpu_id python tools/get_features_batch.py \\
        --videos_dir data/swallow/sliding_videos \\
        --flow_dir data/swallow/flowframe_slided \\
        --output_dir data/swallow/stage1_features/rgb_flow_no_interplote/no_interplote \\
        --filter_list tmp/stage1/split_${part_id}_total_4.txt \\
        --batch_size -1 \\
        --cuda > $log_file 2>&1 &" >> "$log_file"
    echo "#####################################" >> "$log_file"

    echo "########### Script Content ###########" >> "$log_file"
    cat tools/get_features_batch.py >> "$log_file"
    echo "#####################################" >> "$log_file"

    # Run the command in the background
    CUDA_VISIBLE_DEVICES=$gpu_id python tools/get_features_batch.py \
        --videos_dir data/swallow/sliding_videos \
        --flow_dir data/swallow/flowframe_slided \
        --output_dir data/swallow/stage1_features/rgb_flow_no_interplote/no_interplote \
        --filter_list tmp/stage1/split_${part_id}_total_4.txt \
        --batch_size -1 \
        --cuda > "$log_file" 2>&1 &
done