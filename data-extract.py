import os
import lzma
from tqdm import tqdm

def xz_files_in_dir(directory):
    files = []
    for filename in os.listdir(directory):
        if filename.endswith('.xz') and os.path.isfile(os.path.join(directory, filename)):
            files.append(filename)
    return files

# Use the directory where this script lives, making the project portable
folder_path = os.path.dirname(os.path.abspath(__file__))
output_file_train = "output.train.txt"
output_file_val = "output.val.txt" 
vocab_file = "vocab.txt"
 
files = xz_files_in_dir(folder_path)
total_files = len(files)

if total_files == 0:
    print(f"WARNING: No .xz files found in {folder_path}. "
          "Output files will be empty. Place compressed data files in the project directory first.")

# Sort files for deterministic train/val splits across runs
files.sort()

# Calculate the split indices
split_index = int(total_files * 0.9)
files_train = files[:split_index]
files_val = files[split_index:]


# Process files for training and validation separately
vocab = set()

# Process the training files
with open(output_file_train, "w", encoding="utf-8") as outfile:
    for filename in tqdm(files_train, total=len(files_train)):
        file_path = os.path.join(folder_path, filename)
        with lzma.open(file_path, "rt", encoding="utf-8") as infile:
            # Read in chunks to avoid loading entire files into memory
            while True:
                chunk = infile.read(1024 * 1024)  # 1 MB at a time
                if not chunk:
                    break
                outfile.write(chunk)
                characters = set(chunk)
                vocab.update(characters)

# Process the validation files
with open(output_file_val, "w", encoding="utf-8") as outfile:
    for filename in tqdm(files_val, total=len(files_val)):
        file_path = os.path.join(folder_path, filename)
        with lzma.open(file_path, "rt", encoding="utf-8") as infile:
            # Read in chunks to avoid loading entire files into memory
            while True:
                chunk = infile.read(1024 * 1024)  # 1 MB at a time
                if not chunk:
                    break
                outfile.write(chunk)
                characters = set(chunk)
                vocab.update(characters)

# Write the vocabulary to vocab.txt in sorted order for deterministic mappings
with open(vocab_file, "w", encoding="utf-8") as vfile:
    for char in sorted(vocab):
        # Use repr() so special characters like newlines are unambiguous
        vfile.write(char + '\n') 
