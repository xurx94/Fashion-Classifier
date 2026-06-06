import torch
from torch.utils.data import Dataset,DataLoader

class CustomDataset(Dataset):
  def __init__(self,x,y):
    self.features = torch.tensor(x,dtype=torch.float32)
    self.labels = torch.tensor(y,dtype=torch.long)

  def __getitem__(self,idx):
    return self.features[idx],self.labels[idx]

  def __len__(self):
    return len(self.features)

def load_dataset(x,y,batch_sz,shuff=True,drop_last=False):
  train_data = CustomDataset(x,y)
  train_dataloader = DataLoader(train_data,batch_sz,shuffle=shuff,drop_last=drop_last)
  return train_dataloader