import os

def rename_files(directory):
    # Loop through all files in the given directory
    for filename in os.listdir(directory):
        # Check if the file contains '.jpg' and ends with '.json'
        if '.jpg' in filename and filename.endswith('.json'):
            # Create the new filename by removing '.jpg'
            new_filename = filename.replace('.jpg', '')
            # Construct full file paths
            old_file = os.path.join(directory, filename)
            new_file = os.path.join(directory, new_filename)
            # Rename the file
            os.rename(old_file, new_file)
            print(f'Renamed: {old_file} -> {new_file}')

# Specify the directory
directory_path = 'C:/Users/deniskirbaba/Desktop/Learning Lab 2024/data/dataset/ann'

# Call the function
rename_files(directory_path)
