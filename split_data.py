import os
import shutil

def main():
    # Base directory for the CUB dataset.
    base_dir = "./CUB_200_2011/CUB_200_2011"
    
    # Directories for the cropped images and the output splits.
    cropped_images_dir = os.path.join(base_dir, "images_cropped")
    train_split_file = os.path.join(base_dir, "train_test_split.txt")
    images_file = os.path.join(base_dir, "images.txt")
    
    # Output directories for training and test cropped images.
    train_output_dir = "./datasets/cub200_cropped/train_cropped"
    test_output_dir = "./datasets/cub200_cropped/test_cropped"
    
    # Create output directories if they do not exist.
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)
    
    # Read the image paths from images.txt.
    # Each line is of the form:
    #   <image_id> <relative_path>
    image_paths = {}
    with open(images_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                print(f"Skipping malformed line in images.txt: {line}")
                continue
            image_id = parts[0]
            rel_path = parts[1]
            image_paths[image_id] = rel_path
    
    # Read the train/test split from train_test_split.txt.
    # Each line is of the form:
    #   <image_id> <split_flag>
    # where a split_flag of 1 means the image is for training and 0 means test.
    split_dict = {}
    with open(train_split_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                print(f"Skipping malformed line in train_test_split.txt: {line}")
                continue
            image_id, flag = parts
            split_dict[image_id] = int(flag)
    
    # Loop over each image in the split file.
    for image_id, flag in split_dict.items():
        if image_id not in image_paths:
            print(f"Warning: Image ID {image_id} not found in images.txt; skipping.")
            continue
        
        rel_path = image_paths[image_id]
        source_path = os.path.join(cropped_images_dir, rel_path)
        
        # Determine the destination base directory based on the flag.
        # Flag 1 -> training; Flag 0 -> test.
        if flag == 1:
            dest_base = train_output_dir
        else:
            dest_base = test_output_dir
        
        dest_path = os.path.join(dest_base, rel_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        try:
            shutil.copy2(source_path, dest_path)
            print(f"Copied {source_path} to {dest_path}")
        except Exception as e:
            print(f"Error copying {source_path} to {dest_path}: {e}")

if __name__ == "__main__":
    main()

