import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class Encoder(nn.Module):
    def __init__(self, inch, outch):
        super().__init__()
        midch = (inch + outch)//2
        self.conv1 = nn.Conv2d(inch, midch, kernel_size = (3, 3), padding='same')
        self.bn1 = nn.BatchNorm2d(midch)

        self.conv2 = nn.Conv2d(midch, outch, kernel_size = (3, 3), padding='same')
        self.bn2 = nn.BatchNorm2d(outch)

        self.mp = nn.MaxPool2d(kernel_size=(2,2), stride = (2,2))
    
        self.conv3 = nn.Conv2d(outch, outch, kernel_size = (3, 3), padding='same')
        self.bn3 = nn.BatchNorm2d(outch)

    def forward(self, input_tensor, *args, **kwargs):
        x = self.conv1(input_tensor)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.mp(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        return x

class Decoder(nn.Module):
    def __init__(self, inch, outch):
        super().__init__()
        midch = (inch + outch)//2

        self.conv1 = nn.Conv2d(inch, midch, kernel_size = (3, 3), padding='same')
        self.bn1 = nn.BatchNorm2d(midch)

        self.conv2 = nn.Conv2d(midch, outch, kernel_size = (3, 3), padding='same')
        self.bn2 = nn.BatchNorm2d(outch)
        
        self.up = nn.Upsample(scale_factor=2)

        self.conv3 = nn.Conv2d(outch, outch, kernel_size = (3, 3), padding='same')
        self.bn3 = nn.BatchNorm2d(outch)

    def forward(self, input_tensor, *args, **kwargs):
        x = self.conv1(input_tensor)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.up(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        return x

class Residual(nn.Module):
    def __init__(self, inch, hich):
        super().__init__()
        self.conv1 = nn.Conv2d(inch, hich, kernel_size = (3, 3), padding='same')
        self.bn1 = nn.BatchNorm2d(hich)

        self.conv2 = nn.Conv2d(hich, inch, kernel_size = (3, 3), padding='same')
        self.bn2 = nn.BatchNorm2d(inch)

    def forward(self, input_tensor, *args, **kwargs):
        x = self.conv1(input_tensor)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x + input_tensor)
        return x

class Aggregation(nn.Module):
    def __init__(self, inch, outch, hich=None):
        if hich is None:
            hich = outch
        super().__init__()
        self.conv1 = nn.Conv2d(inch, hich, kernel_size = (3, 3), padding='same')
        self.bn1 = nn.BatchNorm2d(hich)
        
        self.up = nn.Upsample(scale_factor=2)

        self.conv2 = nn.Conv2d(hich, outch, kernel_size = (3, 3), padding='same')
        self.bn2 = nn.BatchNorm2d(outch)

    def forward(self, input_tensor, other_tensor, *args, **kwargs):
        x = self.conv1(input_tensor)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x+self.up(other_tensor))
        x = self.bn2(x)
        x = F.relu(x)

        return x

class LaneNet(nn.Module):
    def __init__(self, DEVICE):
        super().__init__()
        self.e1 = Encoder(3, 32)
        self.e2 = Encoder(32, 64)
        
        self.r1 = Residual(64, 128)
        self.r2 = Residual(64, 128)

        self.d1 = Decoder(64, 32)
        self.d2 = Decoder(32, 1)

        self.a1 = Aggregation(3, 32)
        self.a2 = Aggregation(32, 64)
        self.a3 = Aggregation(64, 64)
        self.a4 = Aggregation(64, 32)
        self.a5 = Aggregation(32, 1, hich=32)
        
        self.b1 = Aggregation(32, 64)
        self.b2 = Aggregation(64, 64)
        self.b3 = Aggregation(64, 32, hich=64)
        
        self.DEVICE = DEVICE
        self.to(DEVICE)
        
    def forward(self, input_tensor, *args, **kwargs):
        e1o = self.e1(input_tensor) 
        e2o = self.e2(e1o)
        r1o = self.r1(e2o)
        r2o = self.r2(r1o)
        d1o = self.d1(r2o)
        d2o = self.d2(d1o)

        a1o = self.a1(input_tensor, e1o)
        b1o = self.b1(e1o, e2o)
        a2o = self.a2(a1o, b1o)
        b2o = self.b2(b1o, r1o)
        a3o = self.a3(a2o, b2o)
        b3o = self.b3(b2o, r2o)
        a4o = self.a4(a3o, b3o)
        a5o = self.a5(a4o, d1o)

        return torch.squeeze(torch.sigmoid(d2o + a5o))