import os
import shutil


def move_according_to_ann(ann_folder_path, img_from_folder_path, img_to_folder_path):
    for ann_file_name in os.listdir(ann_folder_path):
        img_file_name = ann_file_name[:-5]
        shutil.move(os.path.join(img_from_folder_path, img_file_name), os.path.join(img_to_folder_path, img_file_name))


ann_folder_path = "C:/Work/Learning Lab 2024/datasets_2/3/ann"
img_from_folder_path = "C:/Users/deniskirbaba/Downloads/297232_posm/dataset 2024-04-25 10_59_02/img"
img_to_folder_path = "C:/Work/Learning Lab 2024/datasets_2/3/img"
move_according_to_ann(ann_folder_path, img_from_folder_path, img_to_folder_path)