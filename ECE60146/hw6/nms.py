import torch
from torchvision.ops import box_iou

def refine_prediction(predictions):
    '''
    First filter out the boxes that do not meet the cut off objectness and 
    confidence levels. Then use Non-Maximum Surpression to remove overlab boxes.
    '''
    # Set cutoff values
    objectness_cutoff = 0.999999
    confidence_cutoff = 0.9
    filtered_predictions = []
    objectness = torch.tensor([ # Extract objectness values
        prediction['objectness'] for prediction in predictions
    ])
    confidence = torch.stack([ # Extrac confidence values and apply sigmoid
        prediction['confidence'] for prediction in predictions
    ])
    confidence = torch.nn.Sigmoid()(confidence).max(axis=1)[0]
    for index in (objectness > objectness_cutoff).nonzero(as_tuple=True)[0]:
        if confidence[index] < confidence_cutoff: # Filter boxes
            continue
        filtered_predictions.append(predictions[index])
    iou_cutoff = 0.05
    final_predictions = []
    while len(filtered_predictions) > 0: # Keep testing every filter box
        boxes = torch.stack([
            prediction['bbox'] for prediction in filtered_predictions
        ]) # Compose box tensor n x 4
        confidence = torch.stack([
        prediction['confidence'] for prediction in filtered_predictions
        ]) # Confidence of remaining boxes
        confidence = torch.nn.Sigmoid()(confidence).max(axis=1)[0]
        candidate = confidence.argmax(axis=0).item()
        iou = box_iou(boxes, boxes)[candidate, :] # Choose overlab boxes cluster
        clusters = (iou > iou_cutoff).nonzero(as_tuple=True)[0]
        winner = torch.index_select(confidence, 0, clusters).argmax(axis=0)
        final_predictions.append(filtered_predictions[clusters[winner]])
        for index in clusters.sort(descending=True)[0]:
            del filtered_predictions[index] # Del non maximum boxes
            
    return final_predictions