# dockerized_fastapi_inference

This repository contains a Dockerized FastAPI-based application for deploying a PyTorch model (for brain tumors classification based on MRIs) running on cuda GPU. The application provides endpoints for making predictions based on uploaded images list or to store images in a SQLite database prior to making predictions on them. The code is structured to ensure modularity, reusability, and clarity.

## Installation and app launch

Using the PyTorch NGC Container (pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime) requires the host system to have the following installed (per https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch):
- Docker Engine: https://docs.docker.com/get-docker/
- NVIDIA GPU Drivers/Cuda Toolkit: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/
- NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

To install and run this application, follow these steps:
- Clone the repository:

        git clone https://github.com/your-username/dockerized_fastapi_inference.git

- Cd to the cloned repository:

        cd dockerized_fastapi_inference

- Run the initialize.py file to build the Docker image and run the Docker container:

        python initialize.py
    
The application will be available at http://127.0.0.1:8000 or http://localhost:8000

## Usage (API calls)
In the test_dockerized_api folder (https://github.com/ValentinOzeel/dockerized_fastapi_inference/tree/main/test_dockerized_api) can de found:
- Some MRI data to test the API endpoints (test_dockerized_api/data/pituitary)
- A detailed exemple of how to use the API endpoints (test_dockerized_api/test_api.py)

Run the test_api.py file to perform the api calls (after having launched the dockerized app).

## Endpoints
- /predict-list/

Description: Make predictions based on a list of uploaded images.
Request: A list of .jpg or .png files.
Response: JSON object containing predictions.

- /upload-db/

Description: Upload images to the database.
Request: A list of .jpg or .png files.
Response: JSON object containing image IDs stored in the database.

- /predict-db/

Description: Make predictions based on image IDs fetched from the database.
Request: JSON object with a list of image IDs.
Response: JSON object containing predictions.


## Code Overview
- Dockerfile
Defines the Docker image for deploying the FastAPI application.

Key Components:
    Base Image: Uses the official PyTorch image with CUDA support.
    Dependencies: Installs required packages.
    Application Setup: Copies the application code and sets up the working directory.
    Exposes Port: Exposes port 8000 for the FastAPI application.
    
- fastapi_serving.py
This script sets up the FastAPI application, defines endpoints, and manages database interactions.

Key Components:
    Model Initialization: Initializes the MRI_CNN model and loads weights.
    Database Setup: Configures SQLite database and SQLAlchemy ORM for image storage.
    API Endpoints: Defines FastAPI endpoints for predictions and database operations.

- use_model.py
Handles model loading, data preprocessing, and inference.

Key Components:
    DeployTorchModel: Main class for model deployment.
        get_model(*args, **kwargs): Loads the model weights.
        get_dataloader(*args, **kwargs): Prepares the data loader for inference.
        inference(): Performs inference and returns predictions.

- datasets.py
Defines custom PyTorch datasets for handling images.

Components:
    PredictListDataset: Handles in-memory images for prediction.
    PredictDbDataset: Fetches images from the database for prediction.

- pydantic_input_validation.py
Provides input validation using Pydantic models.

Key Components:
    ImageFilePydantic: Validates uploaded image files (list).
    ListIdsPydantic: Validates JSON input for database image IDs.



