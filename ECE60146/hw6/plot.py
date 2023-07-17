import utils
import nms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as tvt
from torchvision.utils import draw_bounding_boxes
from torchvision.io import read_image
from typing import List, Dict

def sum_losses(losses: pd.DataFrame):
    losses['total'] = (losses['detection'] 
                       + losses['localization'] 
                       + losses['classification'])

    return losses

def plot_losses(records: List[Dict[str, float]]):
    '''
    Plot three losses separately and altogether
    '''
    epochs = [record['epoch'] for record in records]
    training_losses = pd.DataFrame([record['train'] for record in records])
    testing_losses = pd.DataFrame([record['test'] for record in records])
    training_losses = sum_losses(training_losses)
    testing_losses = sum_losses(testing_losses)
    for loss_type in training_losses.columns:
        fig, ax = plt.subplots()
        ax.plot(epochs, training_losses[loss_type], label='Training Loss')
        ax.plot(epochs, testing_losses[loss_type], label='Testing Loss')
        ax.set_title(f'{loss_type.title()} loss')
        ax.legend(loc='best')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        fig.tight_layout()
        fig.savefig(f'loss-{loss_type}.png')
        plt.close()

def get_cell_anchor_index(yolo_index: int, n_cell: int, ratios: List[float]):
    '''
    Translate yolo vector index back to cell and anchor box indices.
    '''
    anchor_index = yolo_index % len(ratios)
    anchor_ratio = ratios[anchor_index]
    cell_index = yolo_index // len(ratios)
    row_index = cell_index // n_cell
    column_index = cell_index % n_cell

    return row_index, column_index, anchor_ratio

def get_prediction_from_model(data, model: torch.nn.Module, device: str,
                               n_cell: int, ratios: List[float]):
    '''
    Decode YOLO tensor and return bboxes in x1 y1 x2 y2 format with label.
    '''
    pred_tensor = model(data['image'].unsqueeze(0).to(device))
    pred_tensor = pred_tensor.detach().cpu().view(-1, 8) # Reshape
    objectness = torch.nn.Sigmoid()(pred_tensor[:, 0])
    location = pred_tensor[:, 1:5] # Element 1st to 5th
    location[:, :2] = torch.nn.Sigmoid()(location[:, :2]) # Sigmood for location
    location[:, 2:] = torch.exp(location[:, 2:]) # Exp for scale
    confidence = (pred_tensor[:, 5:]) # Label confidence
    labels = pred_tensor[:, 5:].argmax(axis=1) # Get the box labels
    predictions = []
    for index in range(pred_tensor.shape[0]):
        i, j, _ = get_cell_anchor_index(index, n_cell, ratios)
        loc_info = location[index].tolist()
        # Find the center of the prediction box
        # from the top left of i row j column
        x_center = (i+loc_info[0])/n_cell
        y_center = (j+loc_info[1])/n_cell
        # Height and width are ratio of n_cell's height and width
        height = loc_info[2] / n_cell
        width = loc_info[3] / n_cell
        predictions.append({
            'bbox': utils.validate_bbox(torch.tensor([ # x1 y1 x2 y2 format
                x_center - width/2,
                y_center - height/2,
                x_center + width/2,
                y_center + height/2
            ])),
            'label': labels[index].item(),
            'objectness': objectness[index].item(),
            'confidence': confidence[index, :]
        })
    return nms.refine_prediction(predictions)

def draw_boxes(filename: str, savename: str, gt: List[Dict], 
               pred: List[Dict], labels: List[str]):
    '''
    Plot bounding boxes and labels.
    '''
    image = read_image(filename)
    result = draw_bounding_boxes( # Draw ground turth
        image=image, 
        boxes=torch.stack([box['bbox'] for box in gt])*256, 
        labels=[labels[box['label']] for box in gt],
        colors='green'
    )
    if len(pred) > 0:
        result = draw_bounding_boxes( # Draw prediction
            image=result, 
            boxes=torch.stack([box['bbox'] for box in pred])*256, 
            labels=[labels[box['label']] for box in pred],
            colors='red'
        )
    result = tvt.functional.to_pil_image(result)
    result.save(f'{savename}.png')
