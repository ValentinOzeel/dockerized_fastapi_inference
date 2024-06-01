import os
from typing import Dict, List
from pathlib import Path
from pydantic import BaseModel, field_validator, ValidationInfo


############ predict-folder endpoint ############

class PayloadPydantic(BaseModel):
    payload: Dict

    @field_validator('payload')
    @classmethod
    def validate_payload(cls, value, info: ValidationInfo):
        if not isinstance(value, Dict):
            raise ValueError(f"Input {info.field_name} -- {value} -- is not an instance of {Dict}.")
        return value

class FolderPathPydantic(BaseModel):
    folder_path: str

    @field_validator('folder_path')
    @classmethod
    def validate_folder_path(cls, value, info: ValidationInfo):
        if not os.path.isdir(value):
            raise ValueError(f"Input {info.field_name}: folder path -- {value} -- does not exist.")
        return value

    @field_validator('folder_path')
    @classmethod
    def validate_image_files(cls, value, info: ValidationInfo):
        folder_path = Path(value)
        if not list(folder_path.glob("*.jpg")) + list(folder_path.glob("*.png")):
            raise ValueError(f"Input {info.field_name}: no .jpg nor .png file have been found at provided path: -- {value} --")
        return value



############ predict-list and upload-db endpoint ############

class ImageFilePydantic(BaseModel):
    filenames: List[str]

    @field_validator('filenames')
    @classmethod
    def validate_image_file(cls, value, info: ValidationInfo):
        for val in value:
            if not (val.lower().endswith('.jpg') or val.lower().endswith('.png')):
                raise ValueError(f"Input {info.field_name}: file -- {val} -- must be a .jpg or .png image")
        return value
    
    
############ predict-db endpoint ############

class ListIdsPydantic(BaseModel):
    json_input: Dict

    @field_validator('json_input')
    @classmethod
    def validate_image_file(cls, value, info: ValidationInfo):
        if not isinstance(value, Dict):
            raise ValueError(f"Input {info.field_name}: -- {value} -- must be an instance of Dict")
        
        if not value.get('list_ids'):
            raise ValueError(f"Input {info.field_name}: -- {value} -- must have a key named 'list_ids'")
        
        if not isinstance(value['list_ids'], List):
            raise ValueError(f"Input {info.field_name} key = 'list_ids': -- {value} -- must be an instance of List")
        
        if not all(isinstance(id, int) for id in value['list_ids']):
            raise ValueError(f"Input {info.field_name} key = 'list_ids':  -- {value['list_ids']} -- must be a List of int")

        return value
      
    