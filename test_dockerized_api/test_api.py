import requests
import os
from pathlib import Path
from typing import List


# Get the project's root path
project_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Define the path to the directory containing image files
DATA_PATH = os.path.join(project_root_path, 'test_dockerized_api', 'data', 'pituitary')
# Max number of files per API call
MAX_FILES_PER_API_CALL = 1000



def check_api_response(response):
    # Check the status code and print the response
    if response.status_code == 200:
        pass
        #print('API endpoint call success:\n', response.json(), '\n')
    else:
        raise ValueError("Request failed with status code:\n", response.status_code, '\n', 
                         "Response:\n", response.json(), '\n')
        
        
def get_input_files_batchs(data_dir_path:str=DATA_PATH, max_files:int=MAX_FILES_PER_API_CALL) -> List:
    # Get a list of all .jpg and .png files in the directory
    file_paths = list(Path(data_dir_path).glob("*.jpg")) + list(Path(data_dir_path).glob("*.png"))
    # Prepare the files for upload
    files = [('files', (path.name, open(path, 'rb'), 'image/jpeg')) for path in file_paths]
    # Split the files into batches of a specified size
    return [files[x:x+max_files] for x in range(0, len(files), max_files)]


def predict_list(file_batches:List):
    # Define the URL for the predict-list API endpoint
    url = "http://127.0.0.1:8000/predict-list/"

    predictions = []
    # Send the files in batches
    for batch in file_batches:
        response = requests.post(url, files=batch)
        check_api_response(response)
        predictions.extend(response.json()['predictions'])
    return predictions
    
    
def upload_db(file_batches:List):
    # Define the URL for the upload-db API endpoint
    url = "http://127.0.0.1:8000/upload-db/"
    
    image_ids = []
    # Send the files in batches
    for batch in file_batches:
        # Get list of file IDs
        response = requests.post(url, files=batch)
        check_api_response(response)
        image_ids.extend(response.json()['image_ids'])
    return image_ids


def predict_db(list_ids:List[int]):
    # Define the URL for the predict-db API endpoint
    url = "http://127.0.0.1:8000/predict-db/"
    
    response = requests.post(url, json={"list_ids": list_ids})
    check_api_response(response)
    return response.json()['predictions']
    
    
    
    
if __name__ == "__main__":
         
    # Predict on list
    predictions = predict_list(get_input_files_batchs())
    print("predict-list endpoint's returned predictions:\n", predictions, '\n')
    
    # Upload in dabatase
    image_ids = upload_db(get_input_files_batchs())
    print("upload-db endpoint's returned image indexes:\n", image_ids, '\n')
    
    # Predict on uploaded files (via index)
    list_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Replace with the actual image IDs received from the upload-db endpoint
    predictions = predict_db(list_ids)
    print("predict-db endpoint's returned predictions:\n", predictions, '\n')
    