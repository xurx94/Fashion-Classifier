import torch.nn as nn

class FashionMNIST(nn.Module):
  def __init__(self,inp_dim,out_dim,hidn_lyrs,neurons_per_lyr):
    super().__init__()
    layers = []

    curr_dim = inp_dim

    for i in range(hidn_lyrs):
      layers.append(nn.Linear(curr_dim, neurons_per_lyr))
      layers.append(nn.BatchNorm1d(neurons_per_lyr))
      layers.append(nn.ReLU())
      layers.append(nn.Dropout(0.45))
      curr_dim = neurons_per_lyr

    layers.append(nn.Linear(curr_dim, out_dim))

    self.model = nn.Sequential(*layers)

  def forward(self,x):
    return self.model(x)
