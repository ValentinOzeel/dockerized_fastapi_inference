import os
from typing import List, Dict
import torch
from pytorch_vision_framework.secondary_module import ConfigLoad, decompress_model
from pytorch_vision_framework.data_loading import LoadOurData

class DeployTorchModel():
    def __init__(self, yml_config_path):
        self.config_load = ConfigLoad(path=yml_config_path)
        self.config = self.config_load.get_config()
        
        self.device = self._set_device()
        self.model = None
        self.data_loader = None
    
    def _set_device(self):
        if self.config['DEVICE'] != 'cpu':
            return self.config['DEVICE'] if torch.cuda.is_available() else "cpu"
        else:
            return 'cpu'  
        
    def get_model(self, model_instance_no_weight=None, project_root_path:str=None):
        def _get_path(dict_name:str, decompress:bool):
            if 'project_root' in self.config[dict_name]:
                if not project_root_path: raise ValueError('Please assign the project_root_path kwarg.')
                splitted_path = self.config[dict_name].split('/')
                splitted_path = [string for string in splitted_path if string != 'project_root']
                path = os.path.join(project_root_path, *splitted_path)
            else:
                path = self.config[dict_name]

            if path and decompress:
                decompress_model(path)
                path = path.replace('.gz', '')

            return path if path else None

        to_load = _get_path('TO_LOAD_PATH', self.config['COMPRESSED'])

        if self.config['MODEL_OPTION'] == 'state_dict':
            if not model_instance_no_weight: raise ValueError('Please assign the model_instance_no_weight kwarg.')
            model_instance_no_weight.load_state_dict(torch.load(to_load)) 
            self.model = model_instance_no_weight

        elif self.config['MODEL_OPTION'] == 'model':
            self.model = torch.load(to_load)
            
        self.model = self.model.to(self.device)
           
            
    def get_dataloader(self, endpoint:str, DatasetClass=None, inference_data_path:str=None, inference_data:List=None, additional_dataset_kwargs:Dict=None):
        
        if not DatasetClass:
            DatasetClass = self.config_load.get_dataset()
            
        inference_transform_steps = self.config_load.get_transform_steps(dataset_type='inference')
        data_loader_params = self.config['DATA_LOADER']
        
        if endpoint.lower() == 'predict-folder':
            load = LoadOurData(DatasetClass, data_dir_path=inference_data_path, inference_only=True)
            
            
        elif endpoint.lower() == 'predict-list':
            load = LoadOurData(DatasetClass, inference_input=inference_data, inference_only=True)
      
        elif endpoint.lower() == 'predict-db':
            load = LoadOurData(DatasetClass, inference_input=inference_data, inference_only=True, additional_dataset_kwargs=additional_dataset_kwargs)
                  
        self.data_loader = load.load_inference_data(inference_transform_steps, data_loader_params)

        
    def inference(self) -> List:
        """
        Perform inference
        Returns: List: Predicted classes.
        """
        
        # Model in eval mode
        self.model.eval()

        pred_classes = []

        # Inference mode (not to compute gradient)
        with torch.inference_mode():
            # Loop over batches
            for i, imgs in enumerate(self.data_loader):
                # Set data to device
                imgs = imgs.to(self.device)
                # Forward pass
                pred_logit = self.model(imgs)
                # Get predicted classes
                predicted_classes = pred_logit.argmax(dim=1)
                # Extend predictions lists
                pred_classes.extend(predicted_classes.cpu().numpy().tolist())
        return pred_classes
     
