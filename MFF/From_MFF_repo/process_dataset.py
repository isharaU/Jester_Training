import os
import pdb

root = '/content/drive/MyDrive/V2E/test/jester'

# Read category labels
with open('%s/jester-v1-labels.csv' % root) as f:
    lines = f.readlines()

categories = [line.strip() for line in lines]
categories = sorted(categories)

# Save category list
with open('category.txt', 'w') as f:
    f.write('\n'.join(categories))

dict_categories = {category: i for i, category in enumerate(categories)}

files_input = ['%s/jester-v1-validation.csv' % root, '%s/jester-v1-train.csv' % root]
files_output = ['val_videofolder.txt', 'train_videofolder.txt']

for filename_input, filename_output in zip(files_input, files_output):
    with open(filename_input) as f:
        lines = f.readlines()

    folders = []
    idx_categories = []

    for line in lines:
        line = line.strip()
        items = line.split(';')

        if len(items) != 2:
            print(f"Skipping malformed line: {line}")
            continue

        folders.append(items[0])
        idx_categories.append(dict_categories[items[1]])  # Store as integer

    output = []
    for i in range(len(folders)):
        curFolder = folders[i]
        curIDX = idx_categories[i]

        # Corrected dataset path
        video_folder_path = os.path.join('%s/rgb' % root, curFolder)

        if not os.path.exists(video_folder_path):
            print(f"Warning: Folder {video_folder_path} does not exist, skipping...")
            continue

        num_frames = len(os.listdir(video_folder_path))

        output.append('%s %d %d' % (curFolder, num_frames, curIDX))  # Ensure correct format
        print('%d/%d Processed: %s' % (i + 1, len(folders), curFolder))

    with open(filename_output, 'w') as f:
        f.write('\n'.join(output))

    print(f"✅ Saved {filename_output} with {len(output)} entries.")
