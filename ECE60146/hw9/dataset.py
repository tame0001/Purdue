import re
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as tvt

class HW4Dataset(Dataset):
    '''
    Borrow dataset from HW4
    '''
    LABELS = ['airplane', 'bus', 'cat', 'dog', 'pizza']
    def __init__(self, dataset) -> None:
        super().__init__()
        # define a folder
        self.folder = Path('/home/tam/git/ece60146/data/hw4_dataset')
        self.filenames = [] # keep filename
        for filename in self.folder.iterdir():
            if re.findall(r'(\w+)-(\w+)-(\d+)', filename.stem)[0][0] == dataset:
                self.filenames.append(filename)
        # Convert to tensor and normalize to 0 mean and unit variance
        self.transform = tvt.Compose([
            tvt.ToTensor(),
            tvt.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        
    def __len__(self):
        
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index] # Load image from filename
        image = Image.open(filename)
        if image.mode != 'RGB': # Convert to RGB if not RGB
            image = image.convert(mode='RGB')
        tensor = self.transform(image) # Apply transformation
        label = re.findall(r'(\w+)-(\w+)-(\d+)', filename.stem)[0][1]
        label = self.LABELS.index(label) # Class is embedded in filename
        return tensor, label