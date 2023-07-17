import argparse
import torch
import utils
import plot
from pathlib import Path
from typing import Dict
from dataloader import HW6Dataset
from network import HW6Net
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm

IMAGE_SIZE = HW6Dataset.IMAGE_SIZE
LABEL = HW6Dataset.LABELS
ANCHOR_AR = [1/5, 1/3, 1/1, 3/1, 5/1] # Aspect ratios for anchor boxes
'''
--------------------------------------------------------------------------------
Training / testing loop
'''
def calculate_loss(pred_tensor: torch.Tensor, gt_tensor: torch.Tensor,
                   losses: Dict[str, float]):
    '''
    Calculate loss for current batch. Calculate three types losses.
    Only consider the YOLO vectors that thier ground truth have objects
    losses -> dictionary of losses to keep track of each type of losses.
    '''
    # Define loss for backprop for this batch
    loss = torch.tensor(0.0, requires_grad=True).to(device)
    gt_tensor = gt_tensor.view(-1, 8) # View as a tensor of YOLO vector
    pred_tensor = pred_tensor.view(gt_tensor.shape)
    # Only keep the YOLO vectors that ground truth objectness is 1
    indices = gt_tensor[:, 0].nonzero(as_tuple=True)[0]
    pred_tensor = pred_tensor[indices]
    gt_tensor = gt_tensor[indices]
    # Object detection loss (the first element)
    detection_loss = criterion1(
        torch.nn.Sigmoid()(pred_tensor[:, 0]), # Apply sigmoid
        gt_tensor[:, 0]
    )
    loss += detection_loss
    losses['detection'] += detection_loss.item()
    # Object localization loss [dx dy bh bw] (next 4 elements)
    # Apply sigmoid to center coordinates
    pred_tensor[:, 1:3] = torch.nn.Sigmoid()(pred_tensor[:, 1:3])
    localization_loss = criterion2(
        pred_tensor[:, 1:5],
        gt_tensor[:, 1:5]
    )
    loss += localization_loss
    losses['localization'] += localization_loss.item()
    # Object classification loss
    classification_loss = criterion3(
        pred_tensor[:, 5:],
        gt_tensor[:, 5:]
    )
    loss += classification_loss
    losses['classification'] += classification_loss.item()
    
    return loss, losses

def train(dataloader: DataLoader, model: torch.nn.Module, 
          optimizer: torch.optim.Optimizer):
    '''
    This is a traning loop. It will return a training loss at the end.
    '''
    model.train() # Set to training mode 
    # Define loss for debug and tracking progress
    losses = {
        'detection': 0.0,
        'localization': 0.0,
        'classification': 0.0
    }
    for batch, data in enumerate(dataloader):
        images = data['image'].to(device)
        gt_yolo_tensors = data['yolo'].to(device)
        # Get a prediction and reshape to match with ground truth
        prediction = model(images).view(gt_yolo_tensors.shape) 
        loss, losses = calculate_loss(prediction, gt_yolo_tensors, losses)
        # Update parameters
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step() 
        if (batch+1) % 100 == 0: # Tracking learning progress
            utils.print_losses(losses, batch+1)
    # Average loss over all batches
    for loss_type in losses.keys():
        losses[loss_type] /= len(dataloader)

    return losses # Return for ploting 

def test(dataloader: DataLoader, model: torch.nn.Module):
    '''
    This is eval/testing mode. It will return testing losses and predictions.
    '''
    model.eval() # Enter evaluation mode
    # Define loss for debug and tracking progress
    losses = {
        'detection': 0.0,
        'localization': 0.0,
        'classification': 0.0
    }
    with torch.no_grad():
        for data in dataloader:
            images = data['image'].to(device)
            gt_yolo_tensors = data['yolo'].to(device)
            # Get a prediction and reshape to match with ground truth
            prediction = model(images).view(gt_yolo_tensors.shape) 
            _, losses = calculate_loss(prediction, gt_yolo_tensors, losses)
    utils.print_losses(losses, len(dataloader)) # Print loss of all test data
    # Average loss over batches
    for loss_type in losses.keys():
        losses[loss_type] /= len(dataloader)
        
    return losses
'''
--------------------------------------------------------------------------------
'''
# Choose numbers of iteration when execution
parser = argparse.ArgumentParser(description='ECE60146 HW6.')
parser.add_argument(
    '--epoch', '-n', type=int, default=10,
    help='number of training iterations'
)
parser.add_argument(
    '--cell', '-c', type=int, default=5,
    help='number YOLO cell'
)

parser.add_argument(
    '--pretrain', '-p', action='store_true',
    help='use previous train weight'
)

args = vars(parser.parse_args())

path = Path('/home/tam/git/ece60146/data') # Define dataset location.
# Use cuda 1 because other people in my lab usually use cuda 0
if torch.cuda.is_available():
    device = "cuda:1"
    num_workers = 8
    batch_size = 16
else:
    device = 'cpu'
    num_workers = 2
    batch_size = 4 
# Checking message
print(f"Using {device} device with {num_workers} workers")
epochs = args['epoch']
n_cell = args['cell']
# Load training dataset
train_dataset = HW6Dataset(path, 'training', n_cell, ANCHOR_AR)
train_dataloader = DataLoader(
    train_dataset, 
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers
) 
# Load evalution dataset
test_dataset = HW6Dataset(path, 'testing', n_cell, ANCHOR_AR)
test_dataloader = DataLoader(
    test_dataset, 
    batch_size=batch_size, 
    shuffle=False, # No need to shuffle. Able to reuse ground truth list
    num_workers=num_workers
)
if not args['pretrain']:
    # Printout the model summary. Final check before starting training
    model = HW6Net(n_cell, len(ANCHOR_AR)).to(device)
    num_layers = len(list(model.parameters()))
    print(f"The number of layers in the model: {num_layers}")
    with open(f'model-summary.txt', 'w') as fp:
        print(
            str(summary(model, input_size=(batch_size, 3, 256, 256))), file=fp)
        print(f"The number of layers in the model: {num_layers}", file=fp)
    # Start training
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.99))
    criterion1 = torch.nn.BCELoss() # For the first element    
    criterion2 = torch.nn.MSELoss() # For next 4 elements [dx dy bh bw]
    criterion3 = torch.nn.CrossEntropyLoss() # For classicifation
    records = [] # Keep training/testing result
    print(f'Starting training .....')
    for epoch in range(epochs): 
        print(f'Epoch {epoch+1}')
        print('---------------------------training----------------------------')
        utils.print_losses_header()
        train_losses = train(train_dataloader, model, optimizer)
        print('---------------------------testing-----------------------------')
        utils.print_losses_header()
        test_losses = test(test_dataloader, model)
        records.append({
            'epoch': epoch+1,
            'train': train_losses,
            'test': test_losses
        })  
    torch.save(model.state_dict(), 'model-new.pth')
    plot.plot_losses(records)
else: # Use previous train weight
    print('Use previous model for evaluation...')
    model = HW6Net(n_cell, len(ANCHOR_AR)).to(device)
    model.load_state_dict(torch.load('model.pth'))

# Evaluation performance
print(f'Starting evaluation...')
model.eval()
result_path = Path('/home/tam/git/ece60146/data/hw6_dataset/result')
for index in tqdm(range(len(test_dataset))):
    sample = test_dataset[index]
    predictions = plot.get_prediction_from_model(
        sample, model, device, n_cell, ANCHOR_AR
    )
    gt_meta = test_dataset.get_ground_truth(index)
    filename = gt_meta['filename']
    ground_truth = []
    for box, label in zip(gt_meta['bboxes'], gt_meta['labels']):
        ground_truth.append({
            'bbox': box,
            'label': label
        })
    savename = result_path / f'{filename.stem}.png'
    plot.draw_boxes(str(filename), savename, ground_truth,  predictions, LABEL)