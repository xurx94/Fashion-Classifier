import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import torch.optim as optim
from Fashion_MNIST_modelCNN import dynamiCNN
from dataset import CustomDataset
from torch.utils.data import DataLoader,Dataset

dfT = pd.read_csv('fashion-mnist_test.csv')

xT = dfT.iloc[:, 1:].to_numpy()
yT = dfT.iloc[:, 0].to_numpy()
xT = xT/255.0


class CustomDataset(Dataset):
  def __init__(self,x,y):
    self.features = torch.tensor(x,dtype=torch.float32).reshape(-1,1,28,28)
    self.labels = torch.tensor(y,dtype=torch.long)

  def __getitem__(self,idx):
    return self.features[idx],self.labels[idx]

  def __len__(self):
    return len(self.features)


T_dataloader = DataLoader(CustomDataset(xT,yT),64,shuffle=False,pin_memory=True,drop_last=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

trainedModel = dynamiCNN(1,28,3,[32,64,128],2,[256,256],10,0.21).to(device)

trainedModel.load_state_dict(torch.load('Fashion-mnistCNN.pth',map_location=device))
trainedModel.to(device)


trainedModel.eval()
tot = 0
corr = 0
with torch.no_grad():
  for batch_x,batch_y in T_dataloader:
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)
    output = trainedModel(batch_x)
    _,prediction = torch.max(output,1)
    tot += batch_y.shape[0]
    corr += (prediction == batch_y).sum().item()
print(f"Model Accuracy: {(corr/tot)*100}% ")