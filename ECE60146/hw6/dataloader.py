import re
import json
import utils
import torch
import torchvision.transforms as tvt
from torch.utils.data import Dataset
from PIL import Image
from typing import List

class HW6Dataset(Dataset):
    '''
    This is a dataset class for the hw 6.
    '''
    LABELS = ['bus', 'cat', 'pizza'] # Labels for this task
    IMAGE_SIZE = 256 # Image size
    def __init__(self, path, dataset, n_cell, anchor_ar) -> None:
        super().__init__()
        # Read meta data that stores ground truth bboxes and labels
        path = path / 'hw6_dataset'
        with open(path / 'metadata.json') as fp:
            self.meta = json.load(fp)
        self.image_folder = path / 'no_box' # Location for raw images
        self.filenames = [] # Keep filename
        for filename in self.image_folder.iterdir():
            if re.findall(r'(\w+)-(\d+)', filename.stem)[0][0] == dataset:
                self.filenames.append(filename)        
        self.augment = tvt.Compose([ 
            tvt.ToTensor(), # Convert to tensor
            tvt.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # Normalize
            ])
        self.n_cell = n_cell # Keep how many YOLO cell in 1 dimension
        self.anchor_ar = anchor_ar # Aspect ratios for anchor boxes
        
    def __len__(self):
        return len(self.filenames)

    def compose_yolo_vector(self, bbox: torch.Tensor, label: torch.Tensor):
        '''
        Compose 8-element YOLO vector for a given bbox and label.
        '''
        yolo_vector = torch.cat((
            torch.ones(1), 
            utils.compose_yolo_location_feature(bbox, self.n_cell),
            utils.compose_yolo_label_feature(label, self.LABELS)
            ), dim=0
        )
        
        return yolo_vector
    
    def compose_yolo_tensor(self, meta: List[any]):
        '''
        Compose YOLO tensor of that particular image.
        '''
        yolo_tensor = torch.zeros((
            self.n_cell**2, # n_cell is number of cell in one dimension
            len(self.anchor_ar), # Number of anchor boxes for each cell
            8
        ))
        for object_index in range(len(meta)):
            # Read bbox and label from meta data
            # Bbox comes as [[x1, x2] [y1, y2]] -> transpose then flatten
            # So it will be [x1, y1, x2, y2]
            bbox = torch.tensor(meta[object_index]['bbox01'],
                                dtype=torch.float).T.flatten() 
            label = self.LABELS.index(meta[object_index]['label'])+1
            yolo_vector = self.compose_yolo_vector(bbox, label)
            # Index of center of image
            i, j = utils.find_center_index(bbox, self.n_cell) 
            # The number of cell that has center of image
            center_index = i*self.n_cell + j 
            bbox_ar = utils.closest_aspect_ratio(bbox, self.anchor_ar)
            # Resize anchor box
            # yolo_vector[3] *= self.anchor_ar[bbox_ar]
            yolo_tensor[center_index, bbox_ar] = yolo_vector

        return yolo_tensor
    
    def get_ground_truth(self, index):
        '''
        Get the image ground truth bbox and labels using index
        '''
        filename = self.filenames[index]
        meta = self.meta[filename.stem] 
        n_object = len(meta)
        labels = []
        bboxes = torch.zeros(n_object, 4)
        for i in range(n_object):
            labels.append(self.LABELS.index(meta[i]['label']))
            bboxes[i] = torch.tensor(meta[i]['bbox01'], 
                                     dtype=torch.float).T.flatten()
            
        return {
            'filename': filename,
            'n_object': n_object,
            'bboxes': bboxes,
            'labels': labels
        }

    def __getitem__(self, index):
        filename = self.filenames[index]
        image = Image.open(filename) # Load image from filename
        tensor = self.augment(image) # Apply transformation
        meta = self.meta[filename.stem] 
        return {
            'filename': str(filename), # For debug
            'image': tensor,
            'yolo': self.compose_yolo_tensor(meta)
        }