import torch
from torch import nn
from network import Transformer
from dataset import HW4Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt

'''
----------------------- Train / Test -------------------------------------------
'''
def train(dataloader, model, loss_fn, optimizer):
    '''
    This is a traning loop. It will return a training loss at the end.
    '''
    model.train() # Set to training mode
    total_loss = 0
    for data in tqdm(dataloader):
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
'''
------------------------ Main --------------------------------------------------
'''
device = "cuda:0" if torch.cuda.is_available() else "cpu"
epochs = 10
batch_size = 16 
n_worker = 2
# Load training dataset
training_dataset = HW4Dataset('train')
train_dataloader = DataLoader(
    training_dataset, 
    batch_size=batch_size, 
    shuffle=True,
    num_workers=n_worker
)
# Load evalution dataset
validation_dataset = HW4Dataset('val')
val_dataloader = DataLoader(
    validation_dataset, 
    batch_size=batch_size, 
    shuffle=False, # No need to shuffle. Able to reuse ground truth list
    num_workers=n_worker
)
# Extract ground truth labels.
ground_truth = []
for data in val_dataloader:
    labels = data[1]
    ground_truth += labels.tolist()
patch_width = 16
model = Transformer(
    embedded_size=128,
    patch_width=patch_width,
    n_patch=1 + (64//patch_width)**2,
    n_encoder=1,
    n_head=4,
).to(device)
criterion = nn.CrossEntropyLoss() # Define new loss function
optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=1e-5,
    betas=(0.9, 0.99)
)
records = [] # Save record of each iteration
for epoch in range(epochs):
    record = {'epoch': epoch+1}
    print(f'{"-"*35}Epoch {epoch+1}{"-"*35}')
    train_loss = train(train_dataloader, model, criterion, optimizer)
    val_loss, predictions = test(val_dataloader, model, criterion)
    record['training'] = train_loss
    record['validation'] = val_loss
    record['accuracy'] = metrics.accuracy_score(ground_truth, predictions)
    records.append(record) # Record per iteration
    print(metrics.classification_report( # Check result
        ground_truth,
        predictions,
        target_names=HW4Dataset.LABELS
    ))
# Plot confusion matrix
plot = metrics.ConfusionMatrixDisplay.from_predictions(
    ground_truth,
    predictions,
    display_labels=HW4Dataset.LABELS,
    colorbar=False
)
plot.plot()
plt.tight_layout()
plt.savefig(f'confusion.png')
# Check result on the terminal
records = pd.DataFrame(records)
print(records)
# Plot training loss, evaluation loss, and accuracy
fig, ax = plt.subplots()
lns1 = ax.plot(
    records['epoch'], 
    records['training'], 
    label=f'Training Loss'
)
lns2 = ax.plot(
    records['epoch'], 
    records['validation'], 
    label=f'Validation Loss'
)
ax2 = ax.twinx()
lns3 = ax2.plot(
    records['epoch'], 
    records['accuracy']*100, 
    'r--',
    label=f'Accuracy'
)
lns = lns1 + lns2 + lns3
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc='best')
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss')
ax2.set_ylabel('Accuracy (%)')
ax2.set_ylim([0, 100])
fig.tight_layout()
fig.savefig(f'record.png')
plt.close()