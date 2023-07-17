import torchvision.transforms as tvt
from torch.utils.data import Dataset
from PIL import Image

class HW7Dataset(Dataset):
    '''
    This is a dataset class for the hw 7.
    '''
    IMAGE_SIZE = 64 # Image size
    def __init__(self, path, dataset) -> None:
        super().__init__()
        # Read meta data that stores ground truth bboxes and labels
        path = path / 'hw7_dataset'
        self.filenames = [] # Keep filename
        path = path / dataset # Either train / eval
        for filename in path.iterdir():
            self.filenames.append(filename)        
        self.augment = tvt.Compose([ 
            tvt.ToTensor(), # Convert to tensor
            tvt.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # Normalize
        ])

    def __len__(self):
        return len(self.filenames)


    def __getitem__(self, index):
        filename = self.filenames[index]
        image = Image.open(filename) # Load image from filename
        tensor = self.augment(image) # Apply transformation
        return {
            'filename': str(filename), # For debug
            'image': tensor,
        }