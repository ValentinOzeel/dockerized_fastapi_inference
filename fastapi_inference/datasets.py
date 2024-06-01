from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
import io

class PredictListDataset(Dataset):
    def __init__(self, files, transform=None):
        self.files = files
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        contents = file.file.read()
        file.file.seek(0)  # Reset file pointer
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        return self.transform(image) if self.transform else image
    
    
class PredictDbDataset(Dataset):
    def __init__(self, image_ids, transform=None, sessionlocal_db_instance=None, declarative_base=None):
        self.image_ids = image_ids
        self.transform = transform
        self.db = sessionlocal_db_instance
        self.declarative_base = declarative_base

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_record = self.db.query(self.declarative_base).filter(self.declarative_base.id == image_id).first()
        if not image_record:
            raise ValueError(f"Image with ID {image_id} not found")

        image = Image.open(io.BytesIO(image_record.data)).convert('RGB')
        
        return self.transform(image) if self.transform else image
    
    
