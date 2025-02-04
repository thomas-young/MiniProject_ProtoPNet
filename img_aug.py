import Augmentor
import os

def makedir(path):
    """Create the directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)

# Convert paths to absolute paths.
datasets_root_dir = os.path.abspath('./datasets/cub200_cropped/')
train_dir = os.path.join(datasets_root_dir, 'train_cropped')
target_dir = os.path.join(datasets_root_dir, 'train_cropped_augmented')

makedir(target_dir)

# Get a list of class subdirectories.
walk = next(os.walk(train_dir))
subfolder_names = walk[1]  # List of immediate subdirectories (e.g., '005.Crested_Auklet', etc.)
folders = [os.path.join(train_dir, folder) for folder in subfolder_names]
target_folders = [os.path.join(target_dir, folder) for folder in subfolder_names]

for idx in range(len(folders)):
    source_folder = os.path.abspath(folders[idx])
    aug_target_folder = os.path.abspath(target_folders[idx])
    makedir(aug_target_folder)  # Ensure the target folder exists

    print(f"\nProcessing source folder: {source_folder}")
    files = os.listdir(source_folder)
    print(f"Files found: {files}")
    
    if not files:
        print(f"No images found in {source_folder}. Skipping augmentation for this folder.")
        continue

    # --- Rotation augmentation ---
    p = Augmentor.Pipeline(
        source_directory=source_folder,
        output_directory=aug_target_folder
    )
    p.rotate(probability=1, max_left_rotation=15, max_right_rotation=15)
    p.flip_left_right(probability=0.5)
    try:
        for _ in range(10):
            p.process()
    except ValueError as e:
        print("Skipping a rotation augmentation iteration due to:", e)
    del p

    # --- Skew augmentation (using a reduced magnitude) ---
    p = Augmentor.Pipeline(
        source_directory=source_folder,
        output_directory=aug_target_folder
    )
    p.skew(probability=1, magnitude=0.1)  # Reduced from 0.2 to 0.1
    p.flip_left_right(probability=0.5)
    try:
        for _ in range(10):
            p.process()
    except ValueError as e:
        print("Skipping a skew augmentation iteration due to:", e)
    del p

    # --- Shear augmentation (using reduced shear angles) ---
    p = Augmentor.Pipeline(
        source_directory=source_folder,
        output_directory=aug_target_folder
    )
    p.shear(probability=1, max_shear_left=5, max_shear_right=5)  # Reduced angles
    p.flip_left_right(probability=0.5)
    try:
        for _ in range(10):
            p.process()
    except ValueError as e:
        print("Skipping a shear augmentation iteration due to:", e)
    del p

    # (Optional) You can add other augmentation operations similarly.
