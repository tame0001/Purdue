import torch.nn.functional as F
from torch import nn
from torchinfo import summary

class Critic(nn.Module):
    '''
    Critic or Discriminator model.
    '''
    def __init__(self, in_channel=3, sigmoid=True) -> None:
        super().__init__()
        self.sigmoid = sigmoid
        # Start with 3x64x64
        n_channel = 64
        model = [
            nn.Conv2d(in_channel, n_channel, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        for _ in range(3):
            out_channel = n_channel*2
            model.extend([
                nn.Conv2d(n_channel,out_channel, 4, 2, 1),
                nn.BatchNorm2d(out_channel),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            n_channel = out_channel
        
        self.model = nn.Sequential(*model) 

        self.fc = nn.Sequential(
            nn.Conv2d(n_channel, 1, 4),
        ) # Final output is only 1 element

    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        if self.sigmoid: # For BCE-GAN
            x = nn.Sigmoid()(x)

        return x
    
class Generator(nn.Module):
    '''
    Generator model.
    '''
    def __init__(self, letent: int, our_channel=3) -> None:
        super().__init__()
        # Start with 100x1x1
        n_channel = 512
        model = [
            nn.ConvTranspose2d(letent, n_channel, 2),
            nn.BatchNorm2d(n_channel),
            nn.ReLU(inplace=True)
        ]
        for _ in range(4):
            out_channel = int(n_channel/2)
            model.extend([
                nn.ConvTranspose2d(n_channel, out_channel, 4, 2, 1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True)
            ])
            n_channel = out_channel
        model.extend([
            nn.ConvTranspose2d(n_channel, our_channel, 4, 2, 1),
            nn.Tanh()
        ])
        self.model = nn.Sequential(*model) # 3x64x64

    def forward(self, x):
        x = self.model(x)

        return x
    
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        if isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        if isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 0.0,0.02)

if __name__ == '__main__':
    print('---------------------------Critic----------------------------------')
    model = Critic()
    num_layers = len(list(model.parameters()))
    print(f"The number of layers in the model: {num_layers}")
    summary(model, input_size=(1, 3, 64, 64))
    print('---------------------------Generator-------------------------------')
    letent = 100
    model = Generator(letent)
    num_layers = len(list(model.parameters()))
    print(f"The number of layers in the model: {num_layers}")
    summary(model, input_size=(1, letent, 1, 1))