import torch.nn as nn


def conv_block(in_channels,out_channels,d):

    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,3,padding=d,dilation=d),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class DilatedCNN(nn.Module):
    def __init__(self):
        super(DilatedCNN,self).__init__()
        self.encoder = nn.Sequential(
            conv_block(1,128,2),
            conv_block(128,128,2),
            conv_block(128,128,2),
            conv_block(128,128,2)
        )
    def forward(self,x):
        (num_samples,seq_len,mel_bins) = x.shape
        x = x.view(-1,1,seq_len,mel_bins)
        x = self.encoder(x)
        return x.view(x.size(0),-1)
