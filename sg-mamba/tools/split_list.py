'''
Read the directory containing files to be processed, and split the list of files into multiple parts for parallel processing.
The results are saved into separate text files, each containing a subset of the original file list.
'''
import os
import argparse

def split_file_list(input_dir, output_dir, num_splits):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Read all file names in the input directory
    all_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    
    # Calculate the number of files per split
    total_files = len(all_files)
    files_per_split = (total_files + num_splits - 1) // num_splits  # Ceiling division
    print(f"Total files: {total_files}, Files per split: {files_per_split}")
    
    # Split the file list and save to separate text files
    for i in range(num_splits):
        split_files = all_files[i * files_per_split : (i + 1) * files_per_split]
        split_file_path = os.path.join(output_dir, f'split_{i + 1}_total_{num_splits}.txt')
        with open(split_file_path, 'w') as f:
            for file_name in split_files:
                f.write(f"{file_name}\n")
        print(f"Saved {len(split_files)} files to {split_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split file list for parallel processing")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing files to be processed')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save split file lists')
    parser.add_argument('--num_splits', type=int, default=4, help='Number of splits for parallel processing')
    
    args = parser.parse_args()
    split_file_list(args.input_dir, args.output_dir, args.num_splits)