import os
from PIL import Image

def main():
    # Define the base directory for the dataset.
    base_dir = "./CUB_200_2011/CUB_200_2011"
    
    # Directories for original images and the cropped images.
    images_dir = os.path.join(base_dir, "images")
    cropped_dir = os.path.join(base_dir, "images_cropped")
    
    # File paths for the bounding boxes and image paths.
    bbox_file = os.path.join(base_dir, "bounding_boxes.txt")
    images_file = os.path.join(base_dir, "images.txt")
    
    # Read bounding box information.
    # Each line in bounding_boxes.txt is formatted as:
    # <image_id> <x> <y> <width> <height>
    bounding_boxes = {}
    with open(bbox_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                print(f"Skipping malformed line in bounding_boxes.txt: {line}")
                continue
            image_id = parts[0]
            x, y, w, h = map(float, parts[1:])
            bounding_boxes[image_id] = (x, y, w, h)
    
    # Read the image paths.
    # Each line in images.txt is formatted as:
    # <image_id> <relative_path>
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

    # Process each image based on its bounding box.
    for image_id, bbox in bounding_boxes.items():
        if image_id not in image_paths:
            print(f"Warning: No image path found for image id {image_id}")
            continue
        
        rel_path = image_paths[image_id]
        input_image_path = os.path.join(images_dir, rel_path)
        output_image_path = os.path.join(cropped_dir, rel_path)
        
        # Create the output directory (and any necessary parent directories).
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        
        try:
            with Image.open(input_image_path) as img:
                x, y, w, h = bbox
                # Calculate the crop box.
                # Pillow's crop box is (left, upper, right, lower).
                left = int(round(x))
                upper = int(round(y))
                right = int(round(x + w))
                lower = int(round(y + h))
                
                cropped_img = img.crop((left, upper, right, lower))
                cropped_img.save(output_image_path)
                print(f"Cropped image saved to {output_image_path}")
        except Exception as e:
            print(f"Error processing {input_image_path}: {e}")

if __name__ == "__main__":
    main()

