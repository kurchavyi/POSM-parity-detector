import os
import shutil
from sklearn.model_selection import train_test_split

# Define the paths to the dataset
dataset_path = 'C:/Users/deniskirbaba/Desktop/yolo8_dataset_full'
images_path = os.path.join(dataset_path, 'images')
labels_path = os.path.join(dataset_path, 'labels')

# Define the train and val directories
train_images_path = os.path.join(images_path, 'train')
val_images_path = os.path.join(images_path, 'val')
train_labels_path = os.path.join(labels_path, 'train')
val_labels_path = os.path.join(labels_path, 'val')

# Create directories if they do not exist
os.makedirs(train_images_path, exist_ok=True)
os.makedirs(val_images_path, exist_ok=True)
os.makedirs(train_labels_path, exist_ok=True)
os.makedirs(val_labels_path, exist_ok=True)

# List all image files in the images/train directory
all_images = os.listdir(train_images_path)
all_labels = os.listdir(train_labels_path)

# Ensure images and labels are matched
all_images = sorted([f for f in all_images if f.endswith(('.jpg', '.jpeg', '.png'))])
all_labels = sorted([f for f in all_labels if f.endswith('.txt')])

# Split the dataset into training and validation sets (80% train, 20% val)
train_images, val_images = train_test_split(all_images, test_size=0.15, random_state=42)

# Move images and their corresponding labels to the appropriate directories
def move_files(file_list, source_image_dir, source_label_dir, dest_image_dir, dest_label_dir):
    for file_name in file_list:
        # Move image file
        src_image_path = os.path.join(source_image_dir, file_name)
        dest_image_path = os.path.join(dest_image_dir, file_name)
        shutil.move(src_image_path, dest_image_path)

        # Move corresponding label file
        label_name = file_name.replace(os.path.splitext(file_name)[1], '.txt')
        src_label_path = os.path.join(source_label_dir, label_name)
        dest_label_path = os.path.join(dest_label_dir, label_name)
        if os.path.exists(src_label_path):
            shutil.move(src_label_path, dest_label_path)

# Move training files
move_files(train_images, train_images_path, train_labels_path, train_images_path, train_labels_path)

# Move validation files
move_files(val_images, train_images_path, train_labels_path, val_images_path, val_labels_path)

print("Dataset split into training and validation sets successfully.")
