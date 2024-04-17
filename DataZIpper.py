import os
import zipfile
from tqdm import tqdm

def zip_folder_with_progress(folder_path, output_path):
    # Get a list of all files in the folder
    file_paths = [os.path.join(root, file) for root, dirs, files in os.walk(folder_path) for file in files]
    
    # Writing files to a zipfile
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Loop through each file
        for file in tqdm(file_paths, desc='Zipping', unit='file'):
            # Add file to zip
            zipf.write(file, os.path.relpath(file, folder_path))

# Usage
folder_to_zip = 'data'  # Folder to zip
zip_file_name = 'data.zip'  # Output zip file name
zip_folder_with_progress(folder_to_zip, zip_file_name)
