import os
import cv2
import yaml
import numpy as np

# Step 1: Load Dataset
dataset_dir = "C:/Work/Learning Lab 2024/datasets_2/yolo_full"
output_dir = "C:/Work/Learning Lab 2024/datasets_2/dataset_for_classification"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load data_config.yaml to get class names
data_config_file = os.path.join(dataset_dir, "data_config.yaml")
with open(data_config_file, "r") as f:
    data_config = yaml.safe_load(f)
class_names = data_config["names"]

# Step 2: Identify Class 3 Instances
def identify_class_3_instances(label_file):
    class_3_instances = []
    with open(label_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            class_index, *coordinates = line.split()
            if int(class_index) == 2:
                class_3_instances.append(list(map(float, coordinates)))
    return class_3_instances

# Step 3: Extract Segmented Areas
def extract_segmented_areas(image_file, class_3_instances):
    image = cv2.imread(image_file)
    if image is None:
        print(f"Failed to load image: {image_file}")
        return

    height, width = image.shape[:2]
    
    for coordinates in class_3_instances:
        # Denormalize coordinates
        points = np.array(coordinates).reshape(-1, 2)
        points[:, 0] *= width
        points[:, 1] *= height
        points = points.astype(np.int32)
        
        # Create a mask from the polygon coordinates
        mask = np.zeros(image.shape[:2], dtype=np.uint8)  # Create a single-channel mask
        cv2.fillPoly(mask, [points], 255)
        
        # Apply the mask to the image
        segmented_area = cv2.bitwise_and(image, image, mask=mask)
        
        # Crop the image to the bounding box of the polygon
        x, y, w, h = cv2.boundingRect(points)
        cropped_segmented_area = segmented_area[y:y+h, x:x+w]
        
        yield cropped_segmented_area

# Step 4: Create New Dataset
def create_new_dataset(dataset_dir, output_dir):
    for root, _, files in os.walk(os.path.join(dataset_dir, "labels")):
        for label_file in files:
            if label_file.endswith(".txt"):
                label_file_path = os.path.join(root, label_file)
                image_file_path = label_file_path.replace("labels", "images").replace(".txt", ".jpg")
                
                # Check if image loading was successful
                if not os.path.exists(image_file_path):
                    print(f"Image file not found: {image_file_path}")
                    continue
                
                class_3_instances = identify_class_3_instances(label_file_path)
                print(f"Found {len(class_3_instances)} class 3 instances in {label_file}.")

                # Extract segmented areas
                for idx, segmented_area in enumerate(extract_segmented_areas(image_file_path, class_3_instances)):
                    output_file = os.path.join(output_dir, f"{os.path.basename(label_file).split('.')[0]}_{idx}.jpg")
                    cv2.imwrite(output_file, segmented_area)
                        
# Step 5: Save New Dataset
create_new_dataset(dataset_dir, output_dir)

print("Dataset preprocessing completed.")
