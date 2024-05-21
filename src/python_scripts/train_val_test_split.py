import os
import shutil
from sklearn.model_selection import train_test_split


dataset_from_path = 'C:/Work/Learning Lab 2024/datasets_2/yolo_full'
images_from_path = os.path.join(dataset_from_path, 'images', 'train')
labels_from_path = os.path.join(dataset_from_path, 'labels', 'train')

dataset_to_path = 'C:/Work/Learning Lab 2024/datasets_2/yolo_split'
images_to_path = os.path.join(dataset_to_path, 'images')
labels_to_path = os.path.join(dataset_to_path, 'labels')

# Paths for training, validation, and testing
train_images_path = os.path.join(images_to_path, 'train')
val_images_path = os.path.join(images_to_path, 'val')
test_images_path = os.path.join(images_to_path, 'test')
train_labels_path = os.path.join(labels_to_path, 'train')
val_labels_path = os.path.join(labels_to_path, 'val')
test_labels_path = os.path.join(labels_to_path, 'test')

# Create directories if they do not exist
os.makedirs(train_images_path, exist_ok=True)
os.makedirs(val_images_path, exist_ok=True)
os.makedirs(test_images_path, exist_ok=True)
os.makedirs(train_labels_path, exist_ok=True)
os.makedirs(val_labels_path, exist_ok=True)
os.makedirs(test_labels_path, exist_ok=True)

# List all image files in the images directory
all_images = os.listdir(images_from_path)
all_labels = os.listdir(labels_from_path)

# Ensure images and labels are matched
all_images = sorted([f for f in all_images if f.endswith(('.jpg', '.jpeg', '.png'))])
all_labels = sorted([f for f in all_labels if f.endswith('.txt')])

# Split the dataset into training, validation, and testing sets
train_images, temp_images = train_test_split(all_images, test_size=0.20, random_state=42)
val_images, test_images = train_test_split(temp_images, test_size=0.25, random_state=42)  # 0.25 * 0.20 = 0.05

# Function to copy files to the appropriate directories
def copy_files(file_list, source_image_dir, source_label_dir, dest_image_dir, dest_label_dir):
    for file_name in file_list:
        # copy image file
        src_image_path = os.path.join(source_image_dir, file_name)
        dest_image_path = os.path.join(dest_image_dir, file_name)
        shutil.copy(src_image_path, dest_image_path)

        # copy corresponding label file
        label_name = file_name.replace(os.path.splitext(file_name)[1], '.txt')
        src_label_path = os.path.join(source_label_dir, label_name)
        dest_label_path = os.path.join(dest_label_dir, label_name)
        if os.path.exists(src_label_path):
            shutil.copy(src_label_path, dest_label_path)

# copy training files
copy_files(train_images, images_from_path, labels_from_path, train_images_path, train_labels_path)

# copy validation files
copy_files(val_images, images_from_path, labels_from_path, val_images_path, val_labels_path)

# copy testing files
copy_files(test_images, images_from_path, labels_from_path, test_images_path, test_labels_path)

print("Dataset split into training, validation, and testing sets successfully.")
