import os
from typing import Dict, List

import torch

from pytorch_vision_framework.model import MRI_CNN
from use_model import DeployTorchModel
from datasets import PredictListDataset, PredictDbDataset

from pydantic import ValidationError
from pydantic_input_validation import PayloadPydantic, FolderPathPydantic, ImageFilePydantic, ListIdsPydantic

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

import uvicorn

from sqlalchemy import create_engine, Column, Integer, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker


# Set up some paths
project_root_path = os.environ.get('container_workdir')
config_path = os.path.join(project_root_path, 'conf', 'deploy_config.yml')

# The model to deploy (will load the weights later on)
MODEL = MRI_CNN(input_shape=[16, 3, 256, 256],
                hidden_units=32,
                output_shape=4,
                activation_func=torch.nn.LeakyReLU)

# Initiate instance of DeployTorchModel used to load model with weights, get dataloaders, make inference
dtm = DeployTorchModel(config_path)
# Load the model with weights
dtm.get_model(model_instance_no_weight=MODEL, project_root_path=project_root_path)
    
    
# Database set up
DATABASE_URL = "sqlite:///./test.db"  # Example using SQLite

Base = declarative_base()

class ImageModel(Base):
    __tablename__ = "images"
    id = Column(Integer, primary_key=True, index=True)
    data = Column(LargeBinary)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)





# Set up API endpoints
app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Welcome to the FastAPI application!"}

@app.get("/favicon.ico")
async def favicon():
    return {"message": "No favicon available"}

# Make predictions based on input = List of .jpg and .png files
@app.post("/predict-list/")
async def predict_list(files: List[UploadFile] = File(...)):
    
    try:
        ImageFilePydantic(filenames=[file.filename for file in files])
    except ValidationError as e:
        return JSONResponse(status_code=400, content={"detail": str(e)})
    
    dtm.get_dataloader('predict-list', 
                       DatasetClass=PredictListDataset, 
                       inference_data=files)
    predictions = dtm.inference()
    
    return JSONResponse({"predictions": predictions})


# Upload input (List of .jpg and .png files) and load them in the database
@app.post("/upload-db/")
async def upload_images_to_db(files: list[UploadFile] = File(...)):

    try:
        ImageFilePydantic(filenames=[file.filename for file in files])
    except ValidationError as e:
        return JSONResponse(status_code=400, content={"detail": str(e)})
    
    db = SessionLocal()
    image_ids = []
    for file in files:
        contents = await file.read()
        image_record = ImageModel(data=contents)
        db.add(image_record)
        db.commit()
        db.refresh(image_record)
        image_ids.append(image_record.id)
        
    return JSONResponse({"image_ids": image_ids})


# Make predictions based on input = List of index to fetch in the database
@app.post("/predict-db/")
async def predict(json_input:Dict):
    try:
        ListIdsPydantic(json_input=json_input)
    except ValidationError as e:
        return JSONResponse(status_code=400, content={"detail": str(e)})
    
    dtm.get_dataloader('predict-db', 
                       DatasetClass=PredictDbDataset, 
                       inference_data=json_input['list_ids'], 
                       additional_dataset_kwargs={'sessionlocal_db_instance':SessionLocal(), 'declarative_base':ImageModel})
    predictions = dtm.inference()

    return JSONResponse({"predictions": predictions})


# Will not work in a dockerized environment
#@app.post("/predict-folder/")
#async def predict_folder(payload: Dict):
#    try:
#        PayloadPydantic(payload=payload)
#    except ValidationError as e:
#        print(e)
#    
#    try:
#        folder_path = payload['images_list']
#    except ValidationError as e:
#        print(f"There is no 'folder_path' key assigned in {payload}")
#        
#    try:
#        FolderPathPydantic(folder_path=folder_path)
#    except ValidationError as e:
#        print(e)
#    
#    dtm = DeployTorchModel(config_path)
#    dtm.get_model(model_instance_no_weight=MODEL, project_root_path=project_root_path)
#    dtm.get_dataloader(inference_data_path=folder_path)
#    predictions = dtm.inference()
#    return {"predictions": predictions}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)