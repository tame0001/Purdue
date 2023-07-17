import numpy as np
import matplotlib.pyplot as plt
from torchinfo import summary
from sklearn import metrics

def model_info(model):
    '''
    Examine the model info
    '''
    n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_layers = len(list(model.parameters()))
    print(f"The number of layers in the model: {num_layers}")
    print(f"The number of learnable parameters in the model: "
        f"{n_param:,}")
    summary(model)

def check_performance(gt, pred, name: str):
    '''
    Record model performance on testing dataset.
    '''
    # Print confusion matrix to the terminal
    print(metrics.confusion_matrix(gt, pred))
    # Print report to the terminal
    report = metrics.classification_report(
        gt, pred, target_names=['Negative', 'Positive']
        )
    print(report)
    with open(f'report-{name.title().replace("_", "")}.txt', 'w+') as fp:
        print(report, file=fp)
    # Plot confustion matrix
    plot = metrics.ConfusionMatrixDisplay.from_predictions(
        gt,
        pred,
        display_labels=['Negative', 'Positive'],
        colorbar=False,
        xticks_rotation='vertical', 
        normalize='true'
    )
    plot.plot()
    plt.title(f'Confusion for {name.title().replace("_", "")}')
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(f'confusion-{name}.png')

def plot_loss(losses, name):
    fig, ax = plt.subplots()
    epochs = (np.arange(len(losses))+1) * 100
    ax.plot(epochs, losses)
    ax.set_title(f'Training loss for {name.title().replace("_", "")}')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    fig.tight_layout()
    fig.savefig(f'loss-{name.title().replace("_", "")}.png')