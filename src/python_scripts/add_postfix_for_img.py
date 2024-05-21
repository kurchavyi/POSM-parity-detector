import os

def add_postfix_to_files(folder_path, postfix):
    # Check if the provided path is a directory
    if not os.path.isdir(folder_path):
        print(f"The path {folder_path} is not a valid directory.")
        return

    # Iterate over all files in the directory
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Check if it's a file
        if os.path.isfile(file_path):
            # Split the file name and its extension
            name, ext = os.path.splitext(filename)
            # Create a new file name with the postfix
            new_filename = f"{name}{postfix}{ext}"
            new_file_path = os.path.join(folder_path, new_filename)
            
            # Rename the file
            os.rename(file_path, new_file_path)
            print(f"Renamed {filename} to {new_filename}")

# Example usage
folder_path = "C:/Users/deniskirbaba/Desktop/add_data_2/img"
postfix = "_4"

add_postfix_to_files(folder_path, postfix)
