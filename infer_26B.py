""" Standard inference script for InternVL2-26B model. Unfortunately LMDeploy didn't work. """

import math
import os
import json
from time import time_ns
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

def split_model(model_name):
    """
    Split the model into multiple GPUs.
    FOr 26B, you need 2x 24GB GPUs to load with 8bit. (or maybe A6000?)
    """
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        'InternVL2-1B': 24, 'InternVL2-2B': 24, 'InternVL2-4B': 32, 'InternVL2-8B': 32,
        'InternVL2-26B': 48, 'InternVL2-40B': 60, 'InternVL2-Llama3-76B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map

# change by yourself, but SSD/NVMe is recommended for faster loading, and it can be cached.
option = "26B"
path_76b = "~/InternVL/models/76B"
path_40b = "~/InternVL/models/40B"
path_26b = "~/InternVL/models/26B"
device_map_76b = split_model('InternVL2-Llama3-76B')
device_map_40b = split_model('InternVL2-40B')
device_map_26b = split_model('InternVL2-26B')

if option == "76B":
    path = path_76b
    device_map = device_map_76b
elif option == "40B":
    path = path_40b
    device_map = device_map_40b
elif option == "26B":
    path = path_26b
    device_map = device_map_26b

model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    load_in_8bit=True,
    device_map=device_map).eval()

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def load_text(text_file):
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    return text

def format_load_text(text_file):
    text = load_text(text_file)
    return f"You can use the following tags, but it may contain false positives.: {text}."
generation_config = dict(max_new_tokens=1024, do_sample=False)
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

def get_extension(fname):
    for exts in ["webp", "jpg", "png"]:
        if os.path.exists(fname.replace("webp", exts)):
            return fname.replace("webp", exts)
    raise FileNotFoundError(f"File {fname} not found")


def inference(dir_base_path, file_idx, verbose=True, skip_existing=False, output_dir=None):
    """
    Main inference function.
    
    Maybe change questions and orders by yourself.
    Here, we perform simplist & fastest & reliable gen"""
    file1 = get_extension(f'{dir_base_path}/original_images/{file_idx}.webp')
    text1 = f'{dir_base_path}/original_images/{file_idx}.txt'

    file1_split1 = get_extension(f'{dir_base_path}/cropped_images/{file_idx}_0.webp')
    text1_split1 = f'{dir_base_path}/cropped_images/{file_idx}_0.txt'

    file2_split1 = get_extension(f'{dir_base_path}/cropped_images/{file_idx}_1.webp')
    text2_split1 = f'{dir_base_path}/cropped_images/{file_idx}_1.txt'
    if output_dir is None:
        output_dir = f'{dir_base_path}/results'
    else:
        output_dir = output_dir.rstrip("/")
    os.makedirs(output_dir, exist_ok=True)

    result_filename = f'{output_dir}/{file_idx}.json'
    if skip_existing and os.path.exists(result_filename):
        print(f"File {result_filename} already exists. Skipping...")
        return
    os.makedirs(os.path.dirname(result_filename), exist_ok=True)
    # set the max number of tiles in `max_num`
    t = time_ns()
    collect_responses = {}
    pixel_values = load_image(file1_split1, max_num=12).to(torch.bfloat16).cuda()
    question = '<image>\nThis is the cropped part of the image. Describe only focused one person in detail with sentence, including gender and characteristic features of the image.'  + format_load_text(text1_split1)
    response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
    if verbose:
        print(f'User: {question}\nAssistant: {response}')
    collect_responses["response_A"] = str(response)
    pixel_values = load_image(file2_split1, max_num=12).to(torch.bfloat16).cuda()
    question = '<image>\nThis is the cropped part of the image. Describe only focused one person in detail with sentence, including gender and characteristic features of the imag.'  + format_load_text(text2_split1)
    response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=history, return_history=True)
    if verbose:
        print(f'User: {question}\nAssistant: {response}')
    collect_responses["response_B"] = str(response)
    # overall
    pixel_values = load_image(file1, max_num=12).to(torch.bfloat16).cuda()
    question = '<image>\nThe given image is original image of each cropped parts. Descrbe the image in sentence.'# + format_load_text(text1)
    response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=history, return_history=True)
    if verbose:
        print(f'User: {question}\nAssistant: {response}')
    collect_responses["response_all"] = str(response)

    # # additional QA - is it wide image? / is it tall image? / are there any additional text in the image?
    # question = '<image>\nIs the image photorealistic or illustration?'
    # response = model.chat(tokenizer, pixel_values, question, generation_config, history=history)
    # if verbose:
    #     print(f'User: {question}\nAssistant: {response}')
    # collect_responses["response_realistic"] = str(response)

    # question = '<image>\nWhat makes this image special? or is it common?'
    # response = model.chat(tokenizer, pixel_values, question, generation_config, history=history)
    # if verbose:
    #     print(f'User: {question}\nAssistant: {response}')
    # collect_responses["response_uncommon"] = str(response)

    question = '<image>\nIf any texts are found, describe the text with its location and style. If not, answer "No text found."'
    response = model.chat(tokenizer, pixel_values, question, generation_config, history=history)
    if verbose:
        print(f'User: {question}\nAssistant: {response}')
    collect_responses["response_text"] = str(response)

    end = time_ns()
    if verbose:
        print(f"Time: {(end - t) / 1e9} s")
    with open(result_filename, 'w', encoding='utf-8') as f:
        json.dump(collect_responses, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # read file to process
    parser.add_argument("--infer_file", type=str, default=None, help="File containing list of filenames(without extension or path) to process. Each line should contain one index.")
    parser.add_argument("--infer_dir", type=str, default=None) # as dir_base_path
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()
    print(f"Skip existing: {args.skip_existing}")
    assert os.path.exists(args.infer_dir), f"Directory {args.infer_dir} does not exist."
    assert os.path.exists(args.infer_file), f"File {args.infer_file} does not exist."
    
    with open(args.infer_file, "r", encoding="utf-8") as f:
        indices = f.readlines()
    indices = [idx.strip() for idx in indices] # remove leading/trailing whitespaces
    indices = set(indices) # remove duplicates
    
    if args.output_dir is not None:
        # get json files in output_dir and its names
        existing_files = os.listdir(args.output_dir)
        existing_files = set([f.split(".")[0] for f in existing_files if f.endswith(".json")])
        indices = indices - existing_files
    print(f"Number of files to process: {len(indices)}")
    
    for file_idx in tqdm(indices):
        dir_base_path = args.infer_dir.rstrip("/")
        inference(dir_base_path, file_idx, verbose=args.verbose, skip_existing=args.skip_existing, output_dir=args.output_dir)
