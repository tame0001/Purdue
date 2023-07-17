import json
import numpy as np
import cv2 as cv
import skimage.io as io
from pycocotools.coco import COCO
from pathlib import Path
from PIL import Image
from tqdm import tqdm

MIN_SIZE = 200
TARGET_SIZE = 256
IMAGE_SIZE = (TARGET_SIZE, TARGET_SIZE)

# root = Path('/home/tam/Downloads') # Root location
root = Path('/home/tam/Downloads')
# Load training dataset
data_type = 'train2014'
data_file = root / f'annotations/instances_{data_type}.json'
train_dataset = COCO(data_file)
# Load validation dataset
data_type = 'val2014'
data_file = root / f'annotations/instances_{data_type}.json'
val_dataset = COCO(data_file)
# Location for this hw dataset
dataset_location = root / 'hw5_dataset'
# Target class
classes = [ 'bus', 'cat', 'pizza']
classes = dict(zip(train_dataset.getCatIds(classes), classes))
train_images = []
val_images = []
for class_id in classes.keys():
    # Get image id for each class and data set
    train_images.extend(train_dataset.getImgIds(catIds=class_id)) 
    val_images.extend(val_dataset.getImgIds(catIds=class_id))
# Remove duplicate
train_images = list(set(train_images))
val_images = list(set(val_images))
datasets = [
    {
    'name': 'training',
    'ids': train_images,
    'coco': train_dataset
    },
    {
    'name': 'testing',
    'ids': val_images,
    'coco': val_dataset
    }
]
target_class_id = list(classes.keys())
meta = {}
# Create dataset
for dataset in datasets:
    print(f'Creating {dataset["name"]} set ..........')
    count = 0
    source: COCO = dataset['coco']
    for id in tqdm(dataset['ids']):
        ann_ids = source.getAnnIds(imgIds=id, iscrowd=False)
        anns = source.loadAnns(ids=ann_ids) # Load annatations of that image
        biggest_size, biggest_ann = 0, None # Track the biggest box
        for ann in anns:
            x, y, w, h = ann['bbox']
            if (    w > MIN_SIZE and # Must bigger than 200 x 200 pixel
                    h > MIN_SIZE and
                    w*h > biggest_size and # Must bigger than previous
                    ann['category_id'] in target_class_id):
                biggest_ann = ann
                biggest_size = w*h
        if biggest_ann is not None:
            count += 1 # Counting number of images meet the conditions
            img = source.loadImgs(id)[0]
            img = io.imread(img['coco_url'])
            img = Image.fromarray(img)
            width, height = img.size # Save original shape
            img = img.resize(IMAGE_SIZE) # Resize
            if img.mode != 'RGB': # Deal with gray scale images
                img = img.convert(mode='RGB')
            filename = f'{dataset["name"]}-{count}'
            save_path = root / 'hw5_dataset' / 'no_box' / f'{filename}.png'
            img.save(save_path)
            img = np.array(img) # Convert back to array for opencv       
            scaler = cv.getPerspectiveTransform( # Scaling matrix
                np.float32([(0, 0), (width, 0), (width, height), [0, height]]),
                np.float32([
                    (0, 0), (TARGET_SIZE, 0), 
                    IMAGE_SIZE, [0, TARGET_SIZE]
                ]),
            )
            x, y, w, h = biggest_ann['bbox']
            bbox = np.float32([(x, y), (x+w, y+h)]).T # Top left & buttom right
            bbox = np.matmul(scaler[:2, :2], bbox) # Only need scaling part
            img = cv.rectangle(
                img, 
                tuple(bbox[:, 0].astype(int)), # Top left cornor
                tuple(bbox[:, 1].astype(int)), # Buttom right cornor
                (0, 255, 0), 1 # Just green
            )
            img = cv.putText(
                img,
                classes[biggest_ann['category_id']],
                (int(bbox[0, 0]), int(bbox[1,0]-10)),
                cv.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 1
            )
            save_path = root / 'hw5_dataset' / 'with_box' / f'{filename}.png'
            Image.fromarray(img).save(save_path)
            meta[filename] = { # Add meta data for that image
                'category_id': biggest_ann['category_id'],
                'label': classes[biggest_ann['category_id']],
                'bbox': bbox.tolist(),
                'bbox01': (bbox/TARGET_SIZE).tolist()
            }  

    print(f'{dataset["name"].title()} set has {count} images')
# Save mata data
with open(root / 'hw5_dataset' / 'metadata.json', 'w') as fp:
    json.dump(meta, fp)
