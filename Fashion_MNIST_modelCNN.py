import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CustomDataset(Dataset):
  def __init__(self,x,y):
    self.features = torch.tensor(x,dtype=torch.float32).reshape(-1,1,28,28)
    self.labels = torch.tensor(y,dtype=torch.long)

  def __getitem__(self,idx):
    return self.features[idx],self.labels[idx]

  def __len__(self):
    return len(self.features)

df = pd.read_csv('fashion-mnist_train.csv')
x = df.iloc[:,1:].values
y = df.iloc[:,0].values
x = x/255.0

dfT = pd.read_csv('fashion-mnist_test.csv')
xT = dfT.iloc[:,1:].values
yT = dfT.iloc[:,0].values
xT = xT/255.0

class dynamiCNN(nn.Module):
    def __init__(self, inp_dim, img_size, CNNlayers, CNNout, ANNlayers, ANNneurons, out_dim, drop):
        super().__init__()
        cl = []
        al = []

        current_inp_dim = inp_dim
        for i in range(CNNlayers):
            cl.append(nn.Conv2d(current_inp_dim, CNNout[i], kernel_size=3, padding='same'))
            cl.append(nn.ReLU())
            cl.append(nn.BatchNorm2d(CNNout[i]))
            cl.append(nn.MaxPool2d(kernel_size=2, stride=2))
            current_inp_dim = CNNout[i]

        cl.append(nn.Flatten())
        self.features = nn.Sequential(*cl)

        final_spatial_dim = img_size // (2 ** CNNlayers)
        flattened_size = CNNout[-1] * final_spatial_dim * final_spatial_dim

        for i in range(ANNlayers):
            if i == 0:
                al.append(nn.Linear(flattened_size, ANNneurons[i]))
            else:
                al.append(nn.Linear(ANNneurons[i-1], ANNneurons[i]))
            al.append(nn.ReLU())
            al.append(nn.Dropout(drop))

        if ANNlayers > 0:
            al.append(nn.Linear(ANNneurons[-1], out_dim))
        else:
            al.append(nn.Linear(flattened_size, out_dim))

        self.classifier = nn.Sequential(*al)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
  T_dataloader = DataLoader(CustomDataset(x,y),64,shuffle=True,pin_memory=True,drop_last=True)
  V_dataloader = DataLoader(CustomDataset(xT,yT),64,shuffle=False,pin_memory=True,drop_last=True)

  modelx = dynamiCNN(1,28,3,[32,64,128],2,[256,256],10,0.21).to(device)
  optimizer = optim.Adam(modelx.parameters(),lr=0.000889)
  criterion = nn.CrossEntropyLoss()

  for epoch in range(2):
    epoch_loss = 0
    max_val = 0
    for batch_x,batch_y in T_dataloader:
      modelx.train()
      batch_x = batch_x.to(device)
      batch_y = batch_y.to(device)
      output = modelx(batch_x)

      loss = criterion(output,batch_y)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      epoch_loss += loss.item()

    avg_loss = epoch_loss/len(T_dataloader)
    print(f"Epoch: {epoch+1}, Loss: {avg_loss}")

    modelx.eval()
    tot = 0
    corr = 0
    with torch.no_grad():
      for batch_x,batch_y in V_dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        output = modelx(batch_x)
        _,prediction = torch.max(output,1)
        tot += batch_y.shape[0]
        corr += (prediction == batch_y).sum().item()
    print(f"Validation Accuracy: {(corr/tot)*100}% ")
    if (corr/tot)*100 > max_val:
      max_val = (corr/tot)*100
      # torch.save(modelx.state_dict(),'Fashion-mnistCNN.pth')

