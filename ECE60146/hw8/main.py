import utils
import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import text_cl, dls
from network import TorchGRUnet, ThirawatGRUNet
from tqdm import tqdm

device = "cuda:1"

def train(net: nn.Module, name: str):
    print_interval = 100
    net = net.to(device)
    ##  Note that the GRUnet now produces the LogSoftmax output:
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(
        net.parameters(), 
        lr=dls.learning_rate
    )
    training_losses = []
    for epoch in range(dls.epochs):  
        net.train()
        running_loss = 0.0
        for i, data in enumerate(text_cl.train_dataloader):    
            review_tensor, sentiment = data['review'], data['sentiment']
            review_tensor = review_tensor.to(device)
            sentiment = sentiment.to(device)
            optimizer.zero_grad()
            hidden = net.init_hidden().to(device)
            for k in range(review_tensor.shape[1]):
                output, hidden = net(torch.unsqueeze(
                    torch.unsqueeze(review_tensor[0, k], 0), 
                    0), 
                    hidden
                )
            loss = criterion(output, torch.argmax(sentiment, 1))
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            if (i+1) % print_interval == 0:    
                avg_loss = running_loss / print_interval
                training_losses.append(avg_loss)
                print(f"[epoch:{epoch+1:2}  iter:{i+1:6,}] loss:{avg_loss:.3f}")
                running_loss = 0.0
        torch.save(net.state_dict(), f'{dls.path_saved_model}-{name}')
        utils.plot_loss(training_losses, name)
        test(model, name)
    print("\nFinished Training\n")

def test(net: nn.Module, name: str):
    net.load_state_dict(torch.load(f'{dls.path_saved_model}-{name}'))
    net.eval()
    pred = []
    gt = []
    with torch.no_grad():
        for data in tqdm(text_cl.test_dataloader):
            review_tensor, sentiment = data['review'], data['sentiment']
            hidden = net.init_hidden().to(device)
            review_tensor = review_tensor.to(device)
            sentiment = sentiment.to(device)
            for k in range(review_tensor.shape[1]):
                output, hidden = net(torch.unsqueeze(
                    torch.unsqueeze(review_tensor[0, k], 0), 0
                    ), hidden
                )
            predicted_idx = torch.argmax(output).item()
            gt_idx = torch.argmax(sentiment).item()
            gt.append(gt_idx)
            pred.append(predicted_idx)
    utils.check_performance(gt, pred, name)

models = {
    'thirawat': ThirawatGRUNet(300, 100, 2, 2),
    'torch': TorchGRUnet(300, 100, 2, 2),
    'torch_bidectional': TorchGRUnet(300, 100, 2, 2, True)
}

for name, model in models.items():
    print(f'{"-"*20}{name.title().replace("_", " ")}{"-"*20}')
    with torch.autograd.set_detect_anomaly(True):
        train(model, name)
