import os
import torch
import torchvision
import torchvision.datasets as datasets

ROOT_DATASET = '/content/drive/MyDrive/V2E/test/jester'

def return_jester(modality):
    print(f"Returning Jester dataset with modality: {modality}")
    filename_categories = '/content/MFF-pytorch/category.txt'
    filename_imglist_train = '/content/MFF-pytorch/train_videofolder.txt'
    filename_imglist_val = '/content/MFF-pytorch/val_videofolder.txt'
    if modality == 'RGB':
        prefix = '{:05d}.jpg'
        root_data = '/content/drive/MyDrive/V2E/test/jester'
    elif modality == 'RGBFlow':
        prefix = '{:05d}.jpg'
        root_data = '/content/drive/MyDrive/V2E/test/jester'
    else:
        print('No such modality: ' + modality)
        os.exit()
    print(f"Returning: {filename_categories}, {filename_imglist_train}, {filename_imglist_val}, {root_data}, {prefix}")
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_nvgesture(modality):
    print(f"Returning NVGesture dataset with modality: {modality}")
    filename_categories = 'nvgesture/category.txt'
    filename_imglist_train = 'nvgesture/train_videofolder.txt'
    filename_imglist_val = 'nvgesture/val_videofolder.txt'
    if modality == 'RGB':
        prefix = '{:05d}.jpg'
        root_data = '/data2/nvGesture'
    elif modality == 'RGBFlow':
        prefix = '{:05d}.jpg'
        root_data = '/data2/nvGesture'
    else:
        print('No such modality: ' + modality)
        os.exit()
    print(f"Returning: {filename_categories}, {filename_imglist_train}, {filename_imglist_val}, {root_data}, {prefix}")
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_chalearn(modality):
    print(f"Returning ChaLearn dataset with modality: {modality}")
    filename_categories = 'chalearn/category.txt'
    filename_imglist_train = 'chalearn/train_videofolder.txt'
    filename_imglist_val = 'chalearn/val_videofolder.txt'
    if modality == 'RGB':
        prefix = '{:05d}.jpg'
        root_data = '/data2/ChaLearn'
    elif modality == 'RGBFlow':
        prefix = '{:05d}.jpg'
        root_data = '/data2/ChaLearn'
    else:
        print('No such modality: ' + modality)
        os.exit()
    print(f"Returning: {filename_categories}, {filename_imglist_train}, {filename_imglist_val}, {root_data}, {prefix}")
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_dataset(dataset, modality):
    print(f"Getting dataset: {dataset} with modality: {modality}")
    dict_single = {'jester': return_jester, 'nvgesture': return_nvgesture, 'chalearn': return_chalearn}
    if dataset in dict_single:
        file_categories, file_imglist_train, file_imglist_val, root_data, prefix = dict_single[dataset](modality)
    else:
        raise ValueError('Unknown dataset ' + dataset)
    
    # These lines are overriding the values from the returned functions, so make sure it's intentional
    file_imglist_train = '/content/drive/MyDrive/V2E/test/jester/jester-v1-train.csv'
    file_imglist_val = '/content/drive/MyDrive/V2E/test/jester/jester-v1-validation.csv'
    file_categories = '/content/MFF-pytorch/category.txt'

    print(f"File paths for dataset '{dataset}':")
    print(f"  - file_categories: {file_categories}")
    print(f"  - file_imglist_train: {file_imglist_train}")
    print(f"  - file_imglist_val: {file_imglist_val}")
    
    with open(file_categories) as f:
        lines = f.readlines()
    
    categories = [item.rstrip() for item in lines]
    print(f"Categories read from file: {categories[:5]}...")  # Show only the first 5 categories for brevity
    
    return categories, file_imglist_train, file_imglist_val, root_data, prefix
