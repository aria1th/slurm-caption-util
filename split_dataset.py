"""Utility script to split the dataset into multiple parts based on the sanitized indices."""

import os
import shutil
from tqdm import tqdm

def read_indices(file_path):
    with open(file_path, 'r') as f:
        return set(line.strip() for line in f)

def create_output_dirs(base_path, dataset_name="dataset_cropped", split_count=10):
    for i in range(split_count):  # Assuming indices go from 0 to 9
        os.makedirs(os.path.join(base_path, f'{dataset_name}_{i}'), exist_ok=True)

def move_files(source_dir, dest_base, index_dir, dataset_name="dataset_cropped", dry_run=False, split_count=10, max_idx=2):
    for idx in range(split_count):  # Assuming indices go from 0 to 9
        index_file = os.path.join(index_dir, f'sanitized_indices_{idx}.txt')
        indices = read_indices(index_file)
        dest_dir = os.path.join(dest_base, f'{dataset_name}_{idx}')
        with tqdm(total=len(indices)) as pbar:
            for root, _, files in os.walk(source_dir):
                for file in files:
                    if "_0" in file or "_1" in file: #<index>_0.jpg or <index>_1.jpg or maybe _<i for i in range(max_idx)>.jpg...
                        if "_tags" not in file and ".txt" in file:
                            continue
                        if ".txt" in file:
                            file_prefix = file.split('_tags.txt')[0]
                            file_prefix = file_prefix.rsplit('_',1)[0]
                        else:
                            file_prefix = file.lsplit('_',1)[0]
                    else:
                        file_prefix = file.rsplit('.',1)[0]
                    if file_prefix in indices:
                        src_path = os.path.join(root, file)
                        rel_path = os.path.relpath(src_path, source_dir)
                        dest_path = os.path.join(dest_dir, rel_path).replace("_tags", "")
                        if not dry_run:
                            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                            shutil.move(src_path, dest_path)
                        else:
                            print(f"Moved {src_path} to {dest_path}")
                        pbar.update(1)

if __name__ == "__main__":
    import argparse
    # source_dir = "/dataset/dataset_cropped/"
    # dest_base = "/dataset/"
    # index_dir = "/dataset/dataset_cropped/index/"
    # dataset_name = "dataset_cropped"
    # dry_run = False
    # split_count = 10
    
    parser = argparse.ArgumentParser(description="Split the sanitized indices.")
    parser.add_argument("--source_dir", type=str, help="The source directory path.")
    parser.add_argument("--dest_base", type=str, help="The destination base directory path.")
    parser.add_argument("--index_dir", type=str, help="The index directory path.")
    parser.add_argument("--dataset_name", type=str, default="dataset_cropped", help="The dataset name.")
    parser.add_argument("--dry_run", action="store_true", help="Dry run.")
    parser.add_argument("--split_count", type=int, default=10, help="The number of split files.")
    
    args = parser.parse_args()
    
    source_dir = args.source_dir
    dest_base = args.dest_base
    index_dir = args.index_dir
    dataset_name = args.dataset_name
    dry_run = args.dry_run
    split_count = args.split_count
    assert split_count >= 2, "Split count must be at least 2."
    
    create_output_dirs(dest_base, dataset_name, split_count)
    move_files(source_dir, dest_base, index_dir, "dataset_cropped", dry_run)
