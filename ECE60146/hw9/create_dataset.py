import random
import skimage.io as io
from pycocotools.coco import COCO
from pathlib import Path
from PIL import Image
from tqdm import tqdm

TRAIN_SIZE = 1500
VAL_SIZE = 500
IMAGE_SIZE = (64, 64)

root = Path('D:') # Root location
# Load training dataset
data_type = 'train2014'
data_file = root / f'annotations/instances_{data_type}.json'
train_dataset = COCO(data_file)
# Load validation dataset
data_type = 'val2014'
data_file = root / f'annotations/instances_{data_type}.json'
val_dataset = COCO(data_file)
# Location for this hw dataset
dataset_location = root / 'hw4_dataset'
# Target class
classes = ['airplane', 'bus', 'cat', 'dog', 'pizza']
for class_name in classes:
    print(f'Load data from class {class_name}')
    
    # Create training dataset
    class_id = train_dataset.getCatIds(class_name) # Get class id
    img_ids = train_dataset.getImgIds(catIds=class_id) # Get all image
    ids = random.sample(img_ids, TRAIN_SIZE) # Sample ids 
    imgs = train_dataset.loadImgs(ids) # Load image's meta data
    print(f'Load training set {TRAIN_SIZE} images')
    for index, img in tqdm(enumerate(imgs), total=TRAIN_SIZE):
        img = io.imread(img['coco_url']) # Load image from url
        img = Image.fromarray(img) # Convert to PIL
        if img.mode != 'RGB': # Deal with gray scale images
            img = img.convert(mode='RGB')
        img = img.resize(IMAGE_SIZE) # Re size
        img.save(dataset_location / f'train-{class_name}-{index}.png') # Save
    
    # Create validation dataset
    class_id = val_dataset.getCatIds(class_name) # Get class id
    img_ids = val_dataset.getImgIds(catIds=class_id) # Get all image from class
    ids = random.sample(img_ids, VAL_SIZE) # Sample ids 
    imgs = val_dataset.loadImgs(ids) # Load image's meta data
    print(f'Load validataion set {VAL_SIZE} images')
    for index, img in tqdm(enumerate(imgs), total=VAL_SIZE):
        img = io.imread(img['coco_url']) # Load image from url
        img = Image.fromarray(img) # Convert to PIL
        if img.mode != 'RGB': # Deal with gray scale images
            img = img.convert(mode='RGB')
        img = img.resize(IMAGE_SIZE) # Re size
        img.save(dataset_location / f'val-{class_name}-{index}.png') # Save