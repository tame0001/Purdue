import torch
import torch.nn as nn
from utils import model_info

class TorchGRUnet(nn.Module):
    """
    GRU network using PyTorch
    """
    def __init__(self, input_size, hidden_size, output_size, 
                 num_layers=1, bidirection=False): 
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirection = bidirection
        if bidirection: # Add option for bidirectional
            self.gru = nn.GRU(
                input_size, 
                hidden_size, 
                num_layers,
                bidirectional=bidirection)
            self.fc = nn.Linear(hidden_size*2, output_size)
        else:
            self.gru = nn.GRU(input_size, hidden_size, num_layers)
            self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out[:,-1]))
        out = self.logsoftmax(out)
        return out, h

    def init_hidden(self, batch_size=1):
        weight = next(self.parameters()).data
        # num_layers, batch_size, hidden_size
        factor = 2 if self.bidirection else 1
        hidden = weight.new(
            self.num_layers*factor, 
            batch_size, 
            self.hidden_size
        ).zero_()

        return hidden

class ThirawatGRUNet(nn.Module):
    '''
    My implimentation of GRU. Only support batch size = 1.
    '''
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.cells = nn.ModuleList()

        self.cells.append(self.GRUCell(
            self.input_size,
            self.hidden_size
        ))
        for _ in range(1, self.num_layers):
            self.cells.append(self.GRUCell(
                self.hidden_size,
                self.hidden_size
        ))
        self.fc = nn.Linear(self.hidden_size, self.output_size)
        self.relu = nn.ReLU()
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x, h):
        hiddens = []
        for layer in range(self.num_layers):
            if layer == 0:
                hidden = self.cells[layer](x[:, 0, :], h[layer, :, :])
            else:
                hidden = self.cells[layer](h[layer-1, :, :], h[layer, :, :])
            hiddens.append(hidden)

        out = self.fc(self.relu(hidden))
        out = self.logsoftmax(out)

        return out, torch.stack(hiddens)

    def init_hidden(self, batch_size=1):
        weight = next(self.parameters()).data
        # num_layers, batch_size, hidden_size
        hidden = weight.new(
            self.num_layers, 
            batch_size, 
            self.hidden_size
        ).zero_()

        return hidden

    class GRUCell(nn.Module):
        def __init__(self, input_size, hidden_size) -> None:
            super().__init__()

            self.input_reset = nn.Linear(input_size, hidden_size)
            self.hidden_reset = nn.Linear(hidden_size, hidden_size)
            self.h1 = nn.Linear(hidden_size, hidden_size)
            self.x1 = nn.Linear(input_size, hidden_size)

            self.input_update = nn.Linear(input_size, hidden_size)
            self.hidden_update = nn.Linear(hidden_size, hidden_size)

        def forward(self, x, h):
            reset_gate = torch.sigmoid(
                self.input_reset(x) 
                + self.hidden_reset(h)
            )
            r = nn.Tanh()((reset_gate * self.h1(h)) + self.x1(x))

            update_gate = torch.sigmoid(
                self.input_update(x) 
                + self.hidden_update(h)
            )
            u = update_gate * h

            out = ((1-update_gate) * r) + u

            return out


if __name__ == '__main__':
    model = TorchGRUnet(300, 100, 2, 2, True)
    model_info(model)
    model = ThirawatGRUNet(300, 100, 2, 2)
    model_info(model)