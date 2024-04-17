import zipfile
import os
from tqdm import tqdm

def unzip_file_with_progress(zip_path, extract_to='data2'):
    # Ensure the 'data' directory exists
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    
    # Open the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get the list of file names in the zip
        file_list = zip_ref.infolist()
        # Set up the progress bar
        with tqdm(total=len(file_list), unit='file', desc='Extracting') as pbar:
            for file in file_list:
                # Extract each file
                zip_ref.extract(file, extract_to)
                # Update the progress bar
                pbar.update(1)
        print(f"Files extracted to {extract_to}/")

# Replace 'your_file.zip' with the path to your zip file
unzip_file_with_progress('C:/Users/joeha/Downloads/archive.zip')
