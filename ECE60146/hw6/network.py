from torch import nn
import torch.nn.functional as F

class DownSampling(nn.Module):
    '''
    This class will half spatial resolutoin and double channels.
    '''
    def __init__(self, n_channel) -> None:
        super().__init__()
        out_ch = n_channel*2
        self.conv1 = nn.Conv2d(n_channel, out_ch, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))

        return x
    
class SkipBlock(nn.Module):
    '''
    This is a skip connection block. It will retain the original size.
    '''
    def __init__(self, n_channel) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(n_channel, n_channel, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(n_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_channel, n_channel, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(n_channel)

    def forward(self, x):
        original = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.relu(x + original)

        return x
    
class DownSamplingWithSkip(DownSampling):
    '''
    This class will half spatial resolutoin and double channels. 
    Convo original tensor then add at the end.
    '''
    def __init__(self, n_channel) -> None:
        super().__init__(n_channel)
        out_ch = n_channel*2
        self.down_conv = nn.Conv2d(n_channel, out_ch, 1, stride=2)
    
    def forward(self, x):
        skip = self.down_conv(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.relu(x + skip)

        return x
    
class HW6Net(nn.Module):
    '''
    Object detection and localization with skip blocks.
    The network could be configured by number of YOLO cells and anchor boxes.
    '''
    def __init__(self, n_cell: int, n_anchor: int) -> None:
        super().__init__()
        # Start with 3x256x256
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 16, 7),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        ] # 16x256x256
        n_channel = 16
        for _ in range(4):
            model.append(DownSampling(n_channel))
            n_channel *= 2
        self.block1 = nn.Sequential(*model)

        model = [] 
        for _ in range(2):
            model.extend([
                SkipBlock(n_channel),
                DownSamplingWithSkip(n_channel)
            ]) # 1024 x 4 x 4
            n_channel *= 2
        self.block2 = nn.Sequential(*model)

        self.fc = nn.Sequential(
            nn.Linear(16384, 8192),
            nn.ReLU(inplace=True),
            nn.Linear(8192, n_cell**2 * n_anchor * 8)
        ) # Final output vector

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x