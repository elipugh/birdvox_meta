import torch.nn as nn


def conv_block(in_channels,out_channels):

    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,3,padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.encoder = nn.Sequential(
            conv_block(1,128),
            conv_block(128,128),
            conv_block(128,128),
            conv_block(128,128)
        )
    def forward(self,x):
        (num_samples,seq_len,mel_bins) = x.shape
        x = x.view(-1,1,seq_len,mel_bins)
        x = self.encoder(x)
        return x.view(x.size(0),-1)
