import torch
import torch.nn as nn
from ViTHelper import MasterEncoder
from torchinfo import summary

class Transformer(nn.Module):
    def __init__(self, embedded_size, patch_width, n_patch, 
                 n_encoder=2, n_head=2, n_class=5) -> None:
        super().__init__()
        self.embedded_size = embedded_size
        self.patch = nn.Conv2d( # Create patches
            3, embedded_size, kernel_size=patch_width, stride=patch_width
        )
        self.class_token = nn.parameter.Parameter( # Encode patches
            torch.randn(1, 1, embedded_size)
        )
        self.position = nn.parameter.Parameter( # Positional information
         torch.randn(1, n_patch, embedded_size)   
        )
        self.encoder = MasterEncoder(n_patch, embedded_size, n_encoder, n_head)
        self.classifier = nn.Linear(embedded_size, n_class) # The final layer

    def forward(self, image):
        x = self.patch(image)
        x = x.view(x.shape[0], -1, self.embedded_size)
        x = torch.cat((self.class_token.repeat(x.shape[0], 1, 1), x), dim=1)
        x = x + self.position
        x = self.encoder(x)
        x = self.classifier(nn.ReLU(inplace=True)(x[:, 0, :]))

        return x

if __name__ == '__main__':
    n_patch = 17
    embedded_size = 64
    model = MasterEncoder(n_patch, embedded_size, 5, 2)
    # summary(model, input_size=(10, n_patch, embedded_size))

    model = Transformer(embedded_size, 16, n_patch)
    summary(model, input_size=(10, 3, 64, 64))