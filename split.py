# if /scratch/aria4th/InternVL/konachan/dataset_cropped -> ensure it contains "cropped_images", "original_images" folder
# for each folder, it should contain the cropped images and text files
import os
from tqdm import tqdm
from numpy import array_split

IMG_EXTENSIONS = ['jpg', 'jpeg', 'png', 'webp']

def sanitize_cropped(dirpath, max_idx=2, skip_validate:bool=False):
    """
    Sanitize the cropped images and text files in the given directory. Then return the list of file indexes.
    
    dirpath: str
        The directory path containing the cropped images and text files.
    max_idx: int
        The maximum index of the files. Default is 2. 2 -> _0.jpg, _0.txt, _1.jpg, _1.txt should exist.
    """
    assert os.path.exists(os.path.join(dirpath, "cropped_images")), f"Directory path {dirpath} does not exist."
    
    cropped_dir = os.path.join(dirpath, "cropped_images")
    files = os.listdir(cropped_dir)
    file_indices = set()
    for file in tqdm(files, desc="getting indices..."):
        ext = os.path.splitext(file)[1][1:]
        if ext in IMG_EXTENSIONS:
            file_idx = file.rsplit("_", 1)[0]
            file_indices.add(file_idx)
    # check
    if skip_validate:
        return file_indices
    for file_idx in tqdm(list(file_indices), desc="validating cropped"):
        for i in range(0, max_idx):
            if not any(os.path.exists(os.path.join(cropped_dir, f"{file_idx}_{i}.{e}")) for e in IMG_EXTENSIONS):
                print(f"File {file_idx}_{i}.webp does not exist.")
                file_indices.remove(file_idx)
                break
            if not os.path.exists(os.path.join(cropped_dir, f"{file_idx}_{i}.txt")):
                print(f"File {file_idx}_{i}.txt does not exist.")
                file_indices.remove(file_idx)
                break
    return file_indices

def sanitize_original(dirpath,skip_validate:bool=False):
    """
    Sanitize the original images in the given directory. Then return the list of file indexes.
    
    dirpath: str
        The directory path containing the original images.
    """
    assert os.path.exists(os.path.join(dirpath, "original_images")), f"Directory path {dirpath} does not exist."
    
    original_dir = os.path.join(dirpath, "original_images")
    files = os.listdir(original_dir)
    file_indices = set()
    for file in tqdm(files):
        ext = os.path.splitext(file)[1][1:]
        if ext in IMG_EXTENSIONS:
            file_idx = file.rsplit(".", 1)[0]
            file_indices.add(file_idx)
    if skip_validate:
        return file_indices
    # check if .webp and .txt exist
    for file_idx in tqdm(list(file_indices)):
        if not any(os.path.exists(os.path.join(original_dir, f"{file_idx}.{e}")) for e in IMG_EXTENSIONS):
            print(f"File {file_idx}.webp does not exist.")
            file_indices.remove(file_idx)
        # if not os.path.exists(os.path.join(original_dir, f"{file_idx}.txt")):
        #     print(f"File {file_idx}.txt does not exist.")
        #     file_indices.remove(file_idx)
    return file_indices

def get_sanitized_result(dirpath, max_idx=2, skip_validate:bool=False):
    """
    Get the sanitized result of the given directory path.
    
    dirpath: str
        The directory path containing the cropped images and text files.
    max_idx: int
        The maximum index of the files. Default is 2. 2 -> _0.jpg, _0.txt, _1.jpg, _1.txt should exist.
    """
    cropped_indices = sanitize_cropped(dirpath, max_idx, skip_validate)
    print(f"Total cropped files: {len(cropped_indices)}")
    original_indices = sanitize_original(dirpath, skip_validate)
    print(f"Total original files: {len(original_indices)}")
    return cropped_indices.intersection(original_indices)

def write_to_file(file_indices, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for idx in file_indices:
            f.write(f"{idx}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Split the sanitized indices.")
    parser.add_argument("--dataset_dir", type=str, help="The dataset directory path.")
    parser.add_argument("--split_count", type=int, default=10, help="The number of split files.")
    parser.add_argument("--skip_validate", action="store_true")
    args = parser.parse_args()
    dataset_dir = args.dataset_dir
    #dataset_dir = "/scratch/aria4th/InternVL/konachan/dataset_cropped"
    indice_save_dir = f"{dataset_dir}/index"
    os.makedirs(indice_save_dir, exist_ok=True)
    result = get_sanitized_result(dataset_dir, skip_validate=args.skip_validate)
    print(list(result)[:10])
    print(f"Total sanitized files: {len(result)}")
    split_count = args.split_count
    split_indices = array_split(list(result), split_count)
    for i, indices in enumerate(split_indices):
        write_to_file(indices, f"{indice_save_dir}/sanitized_indices_{i}.txt")
    print(f"Split indices into {split_count} files.")
