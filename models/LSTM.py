import torch.nn as nn
import torch.nn.functional as F

class PyramidLSTMLayer(nn.Module):
  """A Pyramid LSTM layer is a standard LSTM layer that halves the size 
  of the input in its hidden embeddings.
  """
  def __init__(self, input_dim, hidden_dim, num_layers=1,
                bidirectional=True, dropout=0.):
    super().__init__()
    self.rnn = nn.LSTM(input_dim * 2, hidden_dim, num_layers=num_layers,
                        bidirectional=bidirectional, dropout=dropout,
                        batch_first=True)
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.num_layers = num_layers
    self.bidirectional = bidirectional
    self.dropout = dropout

  def forward(self, inputs):
    batch_size, maxlen, input_dim = inputs.size()
    
    # reduce time resolution?
    inputs = inputs.contiguous().view(batch_size, maxlen // 2, input_dim * 2)
    outputs, hiddens = self.rnn(inputs)
    return outputs, hiddens

class Listener(nn.Module):
  """Stacks 3 layers of PyramidLSTMLayers to reduce resolution 8 times.

  Args:
    input_dim: Number of input features.
    hidden_dim: Number of hidden features.
    num_pyramid_layers: Number of stacked lstm layers. (default: 3)
    dropout: Dropout probability. (default: 0)
  """
  def __init__(
      self, input_dim, hidden_dim, num_pyramid_layers=3, dropout=0., 
      bidirectional=True):
    super().__init__()
    self.rnn_layer0 = PyramidLSTMLayer(input_dim, hidden_dim, num_layers=1,
                                        bidirectional=True, dropout=dropout)
    for i in range(1, num_pyramid_layers):
      setattr(
          self, 
          f'rnn_layer{i}',
          PyramidLSTMLayer(hidden_dim * 2, hidden_dim, num_layers=1,
                            bidirectional=bidirectional, dropout=dropout),
      )
    
    self.num_pyramid_layers = num_pyramid_layers

  def forward(self, inputs):
    outputs, hiddens = self.rnn_layer0(inputs)
    for i in range(1, self.num_pyramid_layers):
      outputs, hiddens = getattr(self, f'rnn_layer{i}')(outputs)
    return outputs, hiddens

class LSTM(nn.Module):
    def __init__(self, layers=1):
        super(LSTM,self).__init__()
        h = 64
        self.encoder = Listener(128, h, num_pyramid_layers=4, dropout=0.1, 
      bidirectional=True)
        self.linear = nn.Linear(4*h,128)

    def forward(self,x):
        (num_samples,seq_len,mel_bins) = x.shape
        x, (h,c) = self.encoder(x[:,1:])
        #x = self.linear(x.reshape(num_samples,-1))
        return x.reshape(num_samples,-1)