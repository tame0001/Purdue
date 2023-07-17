import random
import skimage.io as io
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pathlib import Path
from PIL import Image
from tqdm import tqdm

NUM_IMAGE = 3
IMAGE_SIZE = (64, 64)

root = Path('D:') # Root location
# Load validation dataset
data_type = 'val2014'
data_file = root / f'annotations/instances_{data_type}.json'
val_dataset = COCO(data_file)
# Target class
classes = ['airplane', 'bus', 'cat', 'dog', 'pizza']
fig, ax = plt.subplots(nrows=NUM_IMAGE, ncols=len(classes), figsize=(35, 15))
for col, class_name in enumerate(classes):
    print(f'Load data from class {class_name}')
    class_id = val_dataset.getCatIds(class_name) # Get class id
    img_ids = val_dataset.getImgIds(catIds=class_id) # Get all image from class
    ids = random.sample(img_ids, NUM_IMAGE) # Sample ids 
    imgs = val_dataset.loadImgs(ids) # Load image's meta data
    for row, img in tqdm(enumerate(imgs), total=NUM_IMAGE):
        img = io.imread(img['coco_url']) # Load image from url
        img = Image.fromarray(img) # Convert to PIL
        if img.mode != 'RGB': # Deal with gray scale images
            img = img.convert(mode='RGB')
        img = img.resize(IMAGE_SIZE) # Re size
        ax[row, col].imshow(img)

fig.tight_layout()
fig.savefig(root / 'sample_images.png')