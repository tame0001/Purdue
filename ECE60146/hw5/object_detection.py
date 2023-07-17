import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as tvt
from torchvision.ops import generalized_box_iou, generalized_box_iou_loss
from torchvision.utils import draw_bounding_boxes
from torchvision.io import read_image
from torchinfo import summary

import re
import random
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from sklearn import metrics
from tqdm import tqdm

# Choose numbers of iteration when execution
parser = argparse.ArgumentParser(description='ECE60146 HW5.')
parser.add_argument(
    '--epoch', '-n', type=int, default=10,
    help='number of training iterations'
)
args = vars(parser.parse_args())

'''
--------------------------------------------------------------------------------
Network structure with skip-block 
'''

class DownSampling(nn.Module):
    '''
    This class will half spatial resolutoin and double channels.
    '''
    def __init__(self, n_channel) -> None:
        super().__init__()
        out_ch = n_channel*2
        self.conv1 = nn.Conv2d(n_channel, out_ch, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))

        return x
    
class SkipBlock(nn.Module):
    '''
    This is a skip connection block. It will retain the original size.
    '''
    def __init__(self, n_channel) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(n_channel, n_channel, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(n_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_channel, n_channel, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(n_channel)

    def forward(self, x):
        original = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.relu(x + original)

        return x
    
class DownSamplingWithSkip(DownSampling):
    '''
    This class will half spatial resolutoin and double channels. 
    Convo original tensor then add at the end.
    '''
    def __init__(self, n_channel) -> None:
        super().__init__(n_channel)
        out_ch = n_channel*2
        self.down_conv = nn.Conv2d(n_channel, out_ch, 1, stride=2)
    
    def forward(self, x):
        skip = self.down_conv(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.relu(x + skip)

        return x
    
class HW5Net(nn.Module):
    '''
    Object detection and localization with skip blocks
    '''
    def __init__(self) -> None:
        super().__init__()
        # Start with 3x256x256
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 16, 7),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        ] # 16x256x256
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model.append(DownSamplingWithSkip(16*mult)) # 64x64x64
        self.model = nn.Sequential(*model)
        n_channel = 64

        model = [] # For classification
        n_downsampling = 4
        for _ in range(n_downsampling):
            model.extend([
                nn.Conv2d(n_channel, n_channel, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            ]) # 64x4x4
        self.classification = nn.Sequential(*model)
        self.fc1 = nn.Linear(1024, 64)
        self.fc2 = nn.Linear(64, 3)

        model = [] # For bouding box
        n_downsampling = 4
        for i in range(n_downsampling):
            mult = 2**i
            model.extend([
                SkipBlock(n_channel*mult),
                DownSamplingWithSkip(n_channel*mult)
            ]) # 1024x4x4
        self.bbox = nn.Sequential(*model)
        self.fc3 = nn.Linear(16384, 4096)
        self.fc4 = nn.Linear(4096, 256)
        self.fc5 = nn.Linear(256, 4)

    def forward(self, x):
        x = self.model(x) # Base block
        xc = self.classification(x) # Start classfication
        xc = xc.view(xc.shape[0], -1)
        xc = F.relu(self.fc1(xc))
        xc = self.fc2(xc) # Classification output with 3 numbers
        xb = self.bbox(x) # Start regeresion 
        xb = xb.view(xb.shape[0], -1)
        xb = F.relu(self.fc3(xb))
        xb = F.relu(self.fc4(xb))
        xb = self.fc5(xb) # Regression output with 4 numbers

        return xc, xb # Return two results

'''
--------------------------------------------------------------------------------
The dataset
'''

class HW5Dataset(Dataset):
    '''
    This is a dataset class for this hw.
    '''
    LABELS = ['bus', 'cat', 'pizza'] # Labels for this task
    TARGET_SIZE = 256 # Image size
    def __init__(self, path, dataset) -> None:
        super().__init__()
        # Read meta data that stores ground truth bboxes and labels
        path = path / 'hw5_dataset'
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
        
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        image = Image.open(filename) # Load image from filename
        tensor = self.augment(image) # Apply transformation
        meta = self.meta[filename.stem] 
        label = self.LABELS.index(meta['label']) # Read label
        return {
            'filename': str(filename), # For debug
            'image': tensor,
            # It comes as [[x1, x2] [y1, y2]] -> transpose then flatten
            # So it will be [x1, y1, x2, y2]
            'bbox': torch.tensor(meta['bbox01'], dtype=torch.float).T.flatten(),
            'label': label
        }

'''
--------------------------------------------------------------------------------
Training / testing loop
'''

def train(dataloader, model, loss_fn_c, loss_fn_b, optimizer):
    '''
    This is a traning loop. It will return a training loss at the end.
    '''
    model.train() # Set to training mode
    total_loss_c = 0 # Classfication loss
    total_loss_b = 0 # Regression loss
    for _, data in enumerate(dataloader):
        images = data['image'].to(device)
        labels = data['label'].to(device)
        bboxes = data['bbox'].to(device)
        pred_labels, pred_boxes = model(images) # Get prediction
        loss_c = loss_fn_c(pred_labels, labels) # Calculate classification loss
        total_loss_c += loss_c.item()
        loss_b = loss_fn_b(pred_boxes, bboxes) # Calculate regression loss
        total_loss_b += loss_b.item()
        optimizer.zero_grad() # Reset gradient
        loss_c.backward(retain_graph=True) # First backprop need extra setting
        loss_b.backward()
        optimizer.step() # Update parameters
    # Average loss over all batches
    total_loss_c /= len(dataloader)
    total_loss_b /= len(dataloader)
    return [total_loss_c, total_loss_b] # Return for ploting 

def test(dataloader, model, loss_fn_c, loss_fn_b):
    '''
    This is eval/testing mode. It will return testing loss and predictions.
    '''
    model.eval() # Enter evaluation mode
    test_loss_c = 0 # Classfication loss
    test_loss_b = 0 # Regression loss
    predictions = { # To keep the prediction result
        'label': [],
        'bbox': []
    }
    with torch.no_grad():
        for data in dataloader:
            images = data['image'].to(device)
            labels = data['label'].to(device)
            bboxes = data['bbox'].to(device)
            pred_labels, pred_boxes = model(images) # Return two results
            # Accumulating losses
            test_loss_b += loss_fn_b(pred_boxes, bboxes).item()
            test_loss_c += loss_fn_c(pred_labels, labels).item()
            # Save prediction results
            predictions['label'].extend(pred_labels.cpu().argmax(axis=1))
            predictions['bbox'].extend(pred_boxes.cpu())
    # Average loss over batches
    test_loss_c /= len(dataloader)
    test_loss_b /= len(dataloader)
    return { # Return result as dictnary with losses and predictions
        'label_loss': test_loss_c,
        'pred_labels': predictions['label'],
        'bbox_loss': test_loss_b,
        'pred_bboxes': predictions['bbox']
    }
'''
--------------------------------------------------------------------------------
Utility functions
'''
def validate_bbox(x):
    '''
    To make sure that 0 <= value <= 1 and x1 < x2 and y1 < y2
    '''
    x = torch.clamp(x, min=0, max=1)
    x[2] = max(x[0], x[2])
    x[3] = max(x[1], x[3])

    return x

'''
--------------------------------------------------------------------------------
Start the program
'''

path = Path('/home/tam/git/ece60146/data') # Define dataset location.
# Use cuda 1 because other people in my lab usually use cuda 0
if torch.cuda.is_available():
    device = "cuda:1"
    num_workers = 8
    batch_size = 16
else:
    device = 'cpu'
    num_workers = 2
    batch_size = 8 
# Checking message
print(f"Using {device} device with {num_workers} workers")
epochs = args['epoch']

# Load training dataset
train_dataset = HW5Dataset(path, 'training')
train_dataloader = DataLoader(
    train_dataset, 
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers
) 
# Load evalution dataset
test_dataset = HW5Dataset(path, 'testing')
test_dataloader = DataLoader(
    test_dataset, 
    batch_size=batch_size, 
    shuffle=False, # No need to shuffle. Able to reuse ground truth list
    num_workers=num_workers
)

# Sample 9 images. 3 images for each class
sample_images = {} # Save as a dictionary with class's name as a key
for label in range(len(HW5Dataset.LABELS)):
    samples = []
    n_sample = 0
    while n_sample < 3: # 3 images
        index = random.randint(0, len(test_dataset)) # Random select
        if test_dataset[index]['label'] == label:
            sample = test_dataset[index]
            sample['index'] = index
            samples.append(sample)
            n_sample += 1 
    sample_images[HW5Dataset.LABELS[label]] = samples
# Ploting the sample images and ground truth bboxes
fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
for i in range(3):
    for j in range(3):
        result = draw_bounding_boxes(
            read_image(sample_images[HW5Dataset.LABELS[i]][j]['filename']), 
            sample_images[HW5Dataset.LABELS[i]][j]['bbox'].reshape(-1, 4)*256, 
            labels=[HW5Dataset.LABELS[i]],
            colors=['green'], width=5
        )
        ax[i, j].imshow(tvt.functional.to_pil_image(result))
        ax[i, j].set(yticklabels=[])
        ax[i, j].set(xticklabels=[])
fig.tight_layout()
fig.savefig(f'sample-pre.png')
plt.close()

# Record ground truth
gt_labels = []
gt_bboxes = []
for data in test_dataloader:
    labels = data['label']
    bboxes = data['bbox']
    gt_labels.extend(labels.tolist())
    gt_bboxes.extend(bboxes.tolist())
# Printout the model summary. Final check before starting training
model = HW5Net().to(device)
num_layers = len(list(model.parameters()))
print(f"The number of layers in the model: {num_layers}")
with open(f'model-summary.txt', 'w') as fp:
    fp.write(str(summary(model, input_size=(batch_size, 3, 256, 256))))
    fp.write(f"\n\nThe number of layers in the model: {num_layers}")
# This hw will use two loss fucntion for bbox regression
loss_funtions = {
    'MSE': nn.MSELoss(),
    'G-IoU': generalized_box_iou_loss
}
master_record = {} # All result will be kept here for later visualizations
for loss_fn_name, loss_fn_b in loss_funtions.items(): # Start the training
    print(f'Using {loss_fn_name} for regression....')
    model = HW5Net().to(device) # Restart the model and optimizer
    loss_fn_c = nn.CrossEntropyLoss()
    loss_fn_b = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=3e-4,
        betas=(0.9, 0.99)
    )
    # Result for this configuration
    label_records = []
    bbox_records = []
    for epoch in tqdm(range(epochs)):
        # Training / testing cycle
        train_loss = train(
            train_dataloader, model, loss_fn_c, loss_fn_b, optimizer
        )
        test_result = test(test_dataloader, model, loss_fn_c, loss_fn_b)
        # Labeling result 
        record = {'epoch': epoch+1}
        pred_labels = test_result['pred_labels']
        record['training'] = train_loss[0]
        record['testing'] = test_result['label_loss']
        record['accuracy'] = metrics.accuracy_score(gt_labels, pred_labels)
        label_records.append(record) # Record per iteration
        # Bounding box result
        record = {'epoch': epoch+1}
        pred_bboxes = test_result['pred_bboxes']
        record['training'] = train_loss[1]
        record['testing'] = test_result['bbox_loss']
        record['iou'] = generalized_box_iou( # Calculate IoU
            torch.stack(list(map(validate_bbox, pred_bboxes))),
            torch.tensor(gt_bboxes)
        )
        record['iou'] = torch.diagonal(record['iou']).mean() # Average IoU
        bbox_records.append(record)
    master_record[loss_fn_name] = {} 
    report = metrics.classification_report( # Labeling report on terminal
        gt_labels,
        pred_labels,
        target_names=HW5Dataset.LABELS
    )
    with open(f'labeling-report-{loss_fn_name}.txt', 'w') as fp: # And file
        fp.write(report)
    print(report)
    # Plot confusion matrix
    plot = metrics.ConfusionMatrixDisplay.from_predictions(
        gt_labels,
        pred_labels,
        display_labels=HW5Dataset.LABELS,
        colorbar=False,
        normalize='true'
    )
    plot.plot()
    plt.title(f'Confusion matrix when using {loss_fn_name} for regression')
    plt.tight_layout()
    plt.savefig(f'confusion-{loss_fn_name}.png') # Save confusion matrix

    # Plot classfication loss for both training and testing
    records = pd.DataFrame(label_records)
    records.to_csv(f'classification-record-{loss_fn_name}.csv', index=False)
    master_record[loss_fn_name]['classification'] = records # Save into master
    fig, ax = plt.subplots()
    lns1 = ax.plot( 
        records['epoch'], 
        records['training'], 
        label=f'Training Loss'
    )
    lns2 = ax.plot(
        records['epoch'], 
        records['testing'], 
        label=f'Testing Loss'
    )
    ax2 = ax.twinx() # Accuracy will be on second y-axis
    lns3 = ax2.plot(
        records['epoch'], 
        records['accuracy']*100, 
        'r--', # Use dash line
        label=f'Accuracy'
    )
    lns = lns1 + lns2 + lns3 # Combine labels for both axis
    labs = [l.get_label() for l in lns]
    ax.set_title(
        f'Classification loss when using {loss_fn_name} for regression'
    )
    ax.legend(lns, labs, loc='best')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_ylim([0, 100])
    fig.tight_layout()
    fig.savefig(f'classification-{loss_fn_name}.png')
    plt.close()

    # Plot bbox loss for both training and testing. The loss will be in log
    records = pd.DataFrame(bbox_records)
    records.to_csv(f'regression-record-{loss_fn_name}.csv', index=False)
    master_record[loss_fn_name]['regerssion'] = records
    fig, ax = plt.subplots()
    lns1 = ax.plot(
        records['epoch'], 
        np.log(records['training']), 
        label=f'Training Loss'
    )
    lns2 = ax.plot(
        records['epoch'], 
        np.log(records['testing']), 
        label=f'Testing Loss'
    )
    ax2 = ax.twinx() # IoU will be on the second y-axis
    lns3 = ax2.plot(
        records['epoch'], 
        records['iou'], 
        'r--',
        label=f'Mean IoU'
    )
    lns = lns1 + lns2 + lns3
    labs = [l.get_label() for l in lns]
    ax.set_title(f'Regression loss in log scale when using {loss_fn_name}')
    ax.legend(lns, labs, loc='best')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Log(Loss)')
    ax2.set_ylabel('Mean IoU')
    ax2.set_ylim([0, 1])
    fig.tight_layout()
    fig.savefig(f'bbox-{loss_fn_name}.png')
    plt.close()
    # Plot bbox loss for both training and testing. Normal scale
    fig, ax = plt.subplots()
    lns1 = ax.plot(
        records['epoch'], 
        records['training'],  # No need to be in log scale
        label=f'Training Loss'
    )
    lns2 = ax.plot(
        records['epoch'], 
        records['testing'], 
        label=f'Testing Loss'
    )
    ax2 = ax.twinx()
    lns3 = ax2.plot(
        records['epoch'], 
        records['iou'], 
        'r--',
        label=f'Mean IoU'
    )
    lns = lns1 + lns2 + lns3
    labs = [l.get_label() for l in lns]
    ax.set_title(
        f'Regression loss when using {loss_fn_name}'
    )
    ax.legend(lns, labs, loc='best')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax2.set_ylabel('Mean IoU')
    ax2.set_ylim([0, 1])
    fig.tight_layout()
    fig.savefig(f'bbox2-{loss_fn_name}.png')
    plt.close()
    # Drawing predicted bboxes with predicted class and IoU
    iou_all = torch.diagonal(generalized_box_iou( # Calculate IoU
        torch.stack(list(map(validate_bbox, pred_bboxes))),
        torch.tensor(gt_bboxes)
    ))
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(20, 15))
    for i in range(3):
        for j in range(3):
            x = sample_images[HW5Dataset.LABELS[i]][j]['bbox']
            index =  sample_images[HW5Dataset.LABELS[i]][j]['index']
            pred_bbox = validate_bbox(pred_bboxes[index])
            gt_bbox = sample_images[HW5Dataset.LABELS[i]][j]['bbox']
            bboxes = [gt_bbox, pred_bbox] # Ground truth and predicted
            iou = iou_all[index] # Along with IoU
            result = draw_bounding_boxes(
                read_image(sample_images[HW5Dataset.LABELS[i]][j]['filename']), 
                torch.stack(bboxes)*256, 
                labels=[ # Green label is IoU. Red label is predicted class
                    f'IoU = {iou:.2f}', 
                    HW5Dataset.LABELS[pred_labels[index]]
                ],
                colors=['green', 'red'], width=5 
            ) # Ground truth will be green. Predicted will be red
            ax[i, j].imshow(tvt.functional.to_pil_image(result))
            ax[i, j].set(yticklabels=[]) # No need axis thicks
            ax[i, j].set(xticklabels=[])
    fig.tight_layout()
    fig.suptitle(f'Sample regression result when using {loss_fn_name}')
    fig.savefig(f'sample-{loss_fn_name}.png')
    plt.close()

# Plot comparision losses and IoU scores
fig, ax = plt.subplots()
lns1 = ax.plot(
    master_record['MSE']['regerssion']['epoch'],
    master_record['MSE']['regerssion']['testing'], 
    label=f'MSE testing Loss'
)
lns2 = ax.plot(
    master_record['G-IoU']['regerssion']['epoch'],
    master_record['G-IoU']['regerssion']['testing'], 
    label=f'G-IoU testing Loss'
)
ax2 = ax.twinx() # IoU will be in the second axis
lns3 = ax2.plot(
    master_record['MSE']['regerssion']['epoch'], 
    master_record['MSE']['regerssion']['iou'], 
    linestyle = 'dashed',
    label=f'MSE mean IoU'
)
lns4 = ax2.plot(
    master_record['G-IoU']['regerssion']['epoch'], 
    master_record['G-IoU']['regerssion']['iou'], 
    linestyle = 'dashed',
    label=f'G-IoU mean IoU'
)
lns = lns1 + lns2 + lns3 + lns4
labs = [l.get_label() for l in lns]
ax.set_title(f'Regression loss')
ax.legend(lns, labs, loc='best')
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss')
ax2.set_ylabel('Mean IoU')
ax2.set_ylim([0, 1])
fig.tight_layout()
fig.savefig(f'regression-loss.png')
plt.close()
# Plot classfication losses and accuaries for both cases for comparision
fig, ax = plt.subplots()
lns1 = ax.plot(
    master_record['MSE']['classification']['epoch'],
    master_record['MSE']['classification']['testing'], 
    label=f'MSE testing Loss'
)
lns2 = ax.plot(
    master_record['G-IoU']['classification']['epoch'],
    master_record['G-IoU']['classification']['testing'], 
    label=f'G-IoU testing Loss'
)
ax2 = ax.twinx()
lns3 = ax2.plot(
    master_record['MSE']['classification']['epoch'], 
    master_record['MSE']['classification']['accuracy']*100, 
    linestyle = 'dashed',
    label=f'MSE Accuracy'
)
lns4 = ax2.plot(
    master_record['G-IoU']['classification']['epoch'], 
    master_record['G-IoU']['classification']['accuracy']*100, 
    linestyle = 'dashed',
    label=f'G-IoU Accuracy'
)
lns = lns1 + lns2 + lns3 + lns4
labs = [l.get_label() for l in lns]
ax.set_title(f'Classification loss')
ax.legend(lns, labs, loc='best')
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss')
ax2.set_ylabel('Accuracy (%)')
ax2.set_ylim([0, 100])
fig.tight_layout()
fig.savefig(f'classification-loss.png')
plt.close()