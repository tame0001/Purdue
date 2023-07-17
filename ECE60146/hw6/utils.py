import torch
from typing import List, Dict

def validate_bbox(bbox: torch.Tensor):
    '''
    To make sure that 0 <= value <= 1 and x1 < x2 and y1 < y2
    '''
    bbox = torch.clamp(bbox, min=0, max=1)
    bbox[2] = max(bbox[0], bbox[2])
    bbox[3] = max(bbox[1], bbox[3])

    return bbox

def find_center_xy(bbox: torch.Tensor):
    '''
    Get center x and y position from bbox [x1, y1, x2, y2]
    '''
    x_center = (bbox[0] + bbox[2]) / 2
    y_center = (bbox[1] + bbox[3]) / 2

    return x_center, y_center

def find_center_index(bbox: torch.Tensor, n_cell: int):
    '''
    Calculate cell index i, j that center of image falls into.
    Assume that image size is 1 x 1.
    '''
    x, y = find_center_xy(bbox)
    i = int(x * n_cell) # Each cell is 1/n_cell wide and high
    j = int(y * n_cell) # Center devided by cell dimension tells cell index

    return i, j

def find_delta_position(bbox: torch.Tensor, n_cell: int):
    '''
    This function return the delta x and y that tells the center of the image
    relative to the top left cornor that the center cell. Normalized delta 
    x and y by the size of the cell.'''
    x, y = find_center_xy(bbox)
    i, j = find_center_index(bbox, n_cell)
    # Find center of the image by i, j coordinate
    i_center = x * n_cell
    j_center = y * n_cell
    # Calculate the delta position from top left cornor of the cell
    dx = i_center - i
    dy = j_center - j

    return dx, dy
    
def cal_bbox_size(bbox: torch.Tensor):
    '''
    Calculate bbox height and width from bbox [x1, y1, x2, y2].
    '''
    h = bbox[3] - bbox[1]
    w = bbox[2] - bbox[0]

    return h, w

def cal_bbox_size_ratio(bbox: torch.Tensor, n_cell: int):
    '''
    Calculate bh and bw which are relative bbox height and width ratio
    to the size of grid cell. Assume that image size is 1 x 1.
    '''
    h, w = cal_bbox_size(bbox)
    bh = torch.log(h * n_cell) # Each cell is 1/n_cell wide and high
    bw = torch.log(w * n_cell) # The ratio is h and w devided by cell size 

    return bh, bw

def compose_yolo_location_feature(bbox: torch.Tensor, n_cell: int):
    ''''
    Compose the YOLO location feature [dx dy bh bw]
    '''
    dx, dy = find_delta_position(bbox, n_cell)
    bh, bw = cal_bbox_size_ratio(bbox, n_cell)

    return torch.tensor([dx, dy, bh, bw])

def compose_yolo_label_feature(label: torch.Tensor, labels: List[str]):
    '''
    Encode the YOLO label vector. The size of vector equals to number of class.
    Class index 0 is for no_object.
    '''
    encoded_vector = torch.zeros(len(labels))
    label_index = int(label)
    encoded_vector[label_index-1] = 1 # Since label index starts from 1

    return encoded_vector

def get_aspect_ratio(bbox: torch.Tensor) -> float:
    '''
    Calculate aspect ratio which is height to width
    '''
    h, w = cal_bbox_size(bbox)

    return h / w

def closest_aspect_ratio(bbox: torch.Tensor, aspect_ratios: List[float]):
    '''
    Find the closest aspect ratio for a given bbox to the list of aspect ratios
    '''
    diff = torch.tensor(aspect_ratios) - get_aspect_ratio(bbox)
    diff = torch.abs(diff)
    index = torch.argmin(diff).item()
    
    return index

def print_losses(losses: Dict[str, float], batch: int):
    print(f'{batch:6}{losses["detection"]/(batch):15.6f}'
          f'{losses["localization"]/(batch):18.6f}'
          f'{losses["classification"]/(batch):20.6f}')

def print_losses_header():
     print(f'{"batch#":>6}{"detection":>15}'
           f'{"localization":>18}{"classification":>20}')