import torch
import torch.nn.functional as F
import torchvision.transforms as tvt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary

import re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from sklearn import metrics
from tqdm import tqdm

class HW4Net1(nn.Module):
    '''
    Net 1 with no padding as stated in the hw instruction
    '''
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(6272, 64)
        self.fc2 = nn.Linear(64, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class HW4Net2(HW4Net1):
    '''
    Add padding so the size does not shink during convu layer.
    Most of setting is identical to Net 1.
    '''
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(8192, 64)

class HW4Net3(HW4Net2):
    '''
    Add 10 more layers to the Net 2.
    These layers don't change tensor shape
    '''
    def __init__(self) -> None:
        super().__init__()
        self.conv01 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv02 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv03 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv04 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv05 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv06 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv07 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv08 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv09 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv10 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv11 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv12 = nn.Conv2d(32, 32, 3, padding=1)
        
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv01(x))
        x = F.relu(self.conv02(x))
        x = F.relu(self.conv03(x))
        x = F.relu(self.conv04(x))
        x = F.relu(self.conv05(x))
        x = F.relu(self.conv06(x))
        x = F.relu(self.conv07(x))
        x = F.relu(self.conv08(x))
        x = F.relu(self.conv09(x))
        x = F.relu(self.conv10(x))
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class HW4Dataset(Dataset):
    LABELS = ['airplane', 'bus', 'cat', 'dog', 'pizza']
    def __init__(self, path, dataset) -> None:
        super().__init__()
        # define a folder
        self.folder = Path('/home/tam') / path
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


def train(dataloader, model, loss_fn, optimizer):
    '''
    This is a traning loop. It will return a training loss at the end.
    '''
    model.train() # Set to training mode
    total_loss = 0
    for _, data in enumerate(dataloader):
        images = data[0].to(device)
        labels = data[1].to(device)
        prediction = model(images)
        loss = loss_fn(prediction, labels)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    total_loss /= len(dataloader)
        
    return total_loss

def test(dataloader, model, loss_fn):
    '''
    This is eval/testing mode. It will return testing loss and predictions.
    '''
    model.eval()
    test_loss = 0
    predictions = []
    with torch.no_grad():
        for data in dataloader:
            images = data[0].to(device)
            labels = data[1].to(device)
            prediction = model(images)
            test_loss += loss_fn(prediction, labels).item()
            predictions += prediction.cpu().argmax(axis=1)
    test_loss /= len(dataloader)

    return test_loss, predictions

path = Path('/home/tam/git/ece60146') # Define dataset location.
# Use cuda 1 because other people in my lab usually use cuda 0
device = "cuda:1" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

epochs = 30 
batch_size = 64 
# Load training dataset
training_dataset = HW4Dataset('git/ece60146/data/hw4_dataset', 'train')
train_dataloader = DataLoader(
    training_dataset, 
    batch_size=batch_size, 
    shuffle=True,
    num_workers=8
)
# Load evalution dataset
validation_dataset = HW4Dataset('git/ece60146/data/hw4_dataset', 'val')
val_dataloader = DataLoader(
    validation_dataset, 
    batch_size=batch_size, 
    shuffle=False, # No need to shuffle. Able to reuse ground truth list
    num_workers=8
)
# Extract ground truth labels. Able to reuse for all models
ground_truth = []
for data in val_dataloader:
    labels = data[1]
    ground_truth += labels.tolist()
# Create all 3 models
net1 = HW4Net1().to(device)
net2 = HW4Net2().to(device)
net3 = HW4Net3().to(device)

master_record = {}

for model_index, model in enumerate([net1, net2, net3]):
    # All models go through the same procedure
    model_index += 1
    print(f'Start training Net {model_index}')
    summary(model, input_size=(batch_size, 3, 64, 64)) # Recheck model params
    criterion = nn.CrossEntropyLoss() # Define new loss function
    optimizer = torch.optim.Adam( # Create new optiimizer for each model
        model.parameters(), 
        lr=1e-3,
        betas=(0.9, 0.99)
    )

    records = []
    for epoch in tqdm(range(epochs)):
        # Training / testing cycle
        record = {'epoch': epoch}
        train_loss = train(train_dataloader, model, criterion, optimizer)
        val_loss, predictions = test(val_dataloader, model, criterion)
        record['training'] = train_loss
        record['validation'] = val_loss
        record['accuracy'] = metrics.accuracy_score(ground_truth, predictions)
        records.append(record) # Record per iteration

    print(metrics.classification_report(
        ground_truth,
        predictions,
        target_names=HW4Dataset.LABELS
    )) # Final result
    # Plot confusion matrix
    plot = metrics.ConfusionMatrixDisplay.from_predictions(
        ground_truth,
        predictions,
        display_labels=HW4Dataset.LABELS,
        colorbar=False
    )
    plot.plot()
    plt.tight_layout()
    plt.savefig(path / f'confusion_net{model_index}')
    # Plot training loss, evaluation loss, and accuracy
    records = pd.DataFrame(records)
    fig, ax = plt.subplots()
    lns1 = ax.plot(
        records['epoch'], 
        records['training'], 
        label=f'Net {model_index} Training Loss'
    )
    lns2 = ax.plot(
        records['epoch'], 
        records['validation'], 
        label=f'Net {model_index} Validation Loss'
    )
    ax2 = ax.twinx()
    lns3 = ax2.plot(
        records['epoch'], 
        records['accuracy']*100, 
        'r--',
        label=f'Net {model_index} Accuracy'
    )

    lns = lns1 + lns2 + lns3
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='best')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_ylim([0, 100])
    fig.tight_layout()
    fig.savefig(path / f'net{model_index}')
    plt.close()
    # Save the training record
    master_record[f'Net{model_index}'] = records

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
for model_index in range(1, 4):
    records = master_record[f'Net{model_index}']
    ax1.plot( # Plot training loss for all models
        records['epoch'], 
        records['training'], 
        label=f'Net {model_index}'
    )

    ax2.plot( # Plot accuracy for all models
        records['epoch'], 
        records['accuracy']*100, 
        label=f'Net {model_index} Accuracy'
    )

ax1.legend(loc='best')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Loss')
ax1.set_title('Trainging Loss')
fig1.savefig(path / 'training_loss.png')

ax2.legend(loc='best')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Accuracy (%)')
ax2.set_ylim(0, 100)
ax2.set_title('Prediction Accuracy')
fig2.savefig(path / 'accuracy.png')
